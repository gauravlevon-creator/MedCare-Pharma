"""Seasonal demand signal: label rules, independent recomputation on real data, visibility in decisions."""

import unittest

import numpy as np
import pandas as pd

import config
from database import database as db
from decision_engine.decision_engine import evaluate_sku
from decision_engine.seasonal_signal import (classify_seasonal_signal, compute_seasonal_signals,
                                             signal_for_series)
from tests.helpers import make_temp_db
from tests.test_allocation import flat_forecast_provider

D = config.simulation_timestamp()


def series(season_days_last_year: int, uplift: float, last_flag: int = 0) -> pd.DataFrame:
    dates = pd.date_range(D - pd.Timedelta(days=364), D)
    flag = np.zeros(len(dates), dtype=int)
    analog_start = D + pd.Timedelta(days=1) - pd.Timedelta(days=365)
    in_analog = (dates >= analog_start) & (dates < analog_start + pd.Timedelta(days=season_days_last_year))
    flag[in_analog] = 1
    flag[-1] = last_flag
    demand = np.where(flag == 1, 10 * uplift, 10.0)
    return pd.DataFrame({"date": dates, "demand_qty": demand, "season_flag": flag})


class SeasonalRuleTests(unittest.TestCase):

    def test_labels(self):
        self.assertEqual(classify_seasonal_signal(1.0, 1.10), config.SEASON_SIGNAL_ELEVATED)
        self.assertEqual(classify_seasonal_signal(1.0, 1.01), config.SEASON_SIGNAL_SEASONAL)
        self.assertEqual(classify_seasonal_signal(0.3, 1.30), config.SEASON_SIGNAL_SEASONAL)
        self.assertEqual(classify_seasonal_signal(0.1, 1.30), config.SEASON_SIGNAL_NORMAL)
        self.assertEqual(classify_seasonal_signal(0.0, 1.00), config.SEASON_SIGNAL_NORMAL)

    def test_elevated_series_and_covariate_gap(self):
        s = signal_for_series(series(30, 1.3, last_flag=0), "S", "W", forecast_values=np.full(30, 10.0))
        self.assertEqual(s["seasonal_signal"], config.SEASON_SIGNAL_ELEVATED)
        self.assertEqual(s["analog_season_share"], 1.0)
        self.assertTrue(s["covariate_gap"])
        self.assertIn("Historical seasonal demand pattern", s["seasonal_reason"])
        self.assertIn("Chronos-2 forecast assumes no season", s["seasonal_reason"])
        self.assertNotIn("flu", s["seasonal_reason"].lower())
        self.assertIsNotNone(s["forecast_demand_index"])

    def test_normal_series(self):
        s = signal_for_series(series(0, 1.0), "S", "W")
        self.assertEqual(s["seasonal_signal"], config.SEASON_SIGNAL_NORMAL)
        self.assertFalse(s["covariate_gap"])

    def test_no_gap_when_season_flag_still_on(self):
        s = signal_for_series(series(30, 1.3, last_flag=1), "S", "W")
        self.assertFalse(s["covariate_gap"])


class SeasonalRealDataTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.db_path = make_temp_db()
        cls.signals = compute_seasonal_signals(db_path=cls.db_path)

    def test_every_item_labelled(self):
        self.assertEqual(len(self.signals), 180)
        self.assertTrue(self.signals.seasonal_signal.isin(
            [config.SEASON_SIGNAL_NORMAL, config.SEASON_SIGNAL_SEASONAL, config.SEASON_SIGNAL_ELEVATED]).all())

    def test_m001_w002_recomputed_independently(self):
        dem = pd.read_csv(config.DEMAND_CSV, parse_dates=["date"])
        h = dem[(dem.sku_id == "M001") & (dem.warehouse_id == "W002") & (dem.date <= D)]
        a = h[(h.date >= D + pd.Timedelta(days=1) - pd.Timedelta(days=365))
              & (h.date <= D + pd.Timedelta(days=config.FORECAST_HORIZON_DAYS) - pd.Timedelta(days=365))]
        share, index = a.season_flag.mean(), a.demand_qty.mean() / h.demand_qty.mean()
        row = self.signals[(self.signals.sku_id == "M001") & (self.signals.warehouse_id == "W002")].iloc[0]
        self.assertAlmostEqual(row.analog_season_share, round(share, 3))
        self.assertAlmostEqual(row.analog_demand_index, round(index, 3))
        self.assertEqual(row.seasonal_signal, classify_seasonal_signal(share, index))
        self.assertEqual(bool(row.covariate_gap), share >= config.SEASON_ACTIVE_SHARE and h.season_flag.iloc[-1] == 0)

    def test_signal_visible_in_decisions_without_changing_them(self):
        decisions, _, alerts = evaluate_sku("M001", flat_forecast_provider(self.db_path), db_path=self.db_path)
        row = self.signals[(self.signals.sku_id == "M001") & (self.signals.warehouse_id == "W002")].iloc[0]
        self.assertEqual(decisions["W002"].seasonal_signal, row.seasonal_signal)
        self.assertTrue(decisions["W002"].seasonal_reason)
        if row.seasonal_signal != config.SEASON_SIGNAL_NORMAL:
            msg = [a for a in alerts if a.warehouse_id == "W002" and a.alert_type.startswith("STOCKOUT")]
            if msg:
                self.assertIn("Seasonal demand signal", msg[0].message)


if __name__ == "__main__":
    unittest.main()
