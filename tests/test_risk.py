"""Risk tests: P1 stock-out risk rules and the separate E1 threshold alert."""

import unittest

import config
from decision_engine.risk_engine import classify_risk, e1_threshold_alert


class P1RiskTests(unittest.TestCase):

    def test_high_when_projected_negative(self):
        status, reason = classify_risk(-155.0, safety_stock=151)
        self.assertEqual(status, config.RISK_HIGH)
        self.assertIn("-155.00", reason)

    def test_high_at_exactly_zero(self):
        self.assertEqual(classify_risk(0.0, 50)[0], config.RISK_HIGH)

    def test_medium_below_safety_stock(self):
        self.assertEqual(classify_risk(40.0, 50)[0], config.RISK_MEDIUM)

    def test_medium_at_exactly_safety_stock(self):
        self.assertEqual(classify_risk(50.0, 50)[0], config.RISK_MEDIUM)

    def test_low(self):
        status, reason = classify_risk(51.0, 50)
        self.assertEqual(status, config.RISK_LOW)
        self.assertTrue(reason)

    def test_only_valid_states(self):
        for projected in (-100, 0, 1, 49, 50, 51, 1000):
            self.assertIn(classify_risk(float(projected), 50)[0], config.RISK_LEVELS)


class E1AlertTests(unittest.TestCase):

    def test_alert_at_or_below_threshold(self):
        self.assertTrue(e1_threshold_alert(100, 100)[0])
        self.assertTrue(e1_threshold_alert(36, 453)[0])

    def test_no_alert_above_threshold(self):
        alert, reason = e1_threshold_alert(101, 100)
        self.assertFalse(alert)
        self.assertIn("above", reason)

    def test_e1_independent_of_p1(self):
        # Low P1 risk can coexist with an E1 alert and vice versa.
        self.assertEqual(classify_risk(500.0, 50)[0], config.RISK_LOW)
        self.assertTrue(e1_threshold_alert(90, 100)[0])


if __name__ == "__main__":
    unittest.main()
