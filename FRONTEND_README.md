# MedCare Pharma — Streamlit Dashboard (Gaurav)

Presentation layer for the Phase 9 backend. It shows backend outputs and does not
calculate forecasts, risk, expiry, transfers, reorders or decisions itself.

## Run

From the `medcare/` folder:

```bash
pip install -r requirements.txt -r requirements-frontend.txt
python main.py init-db          # SQLite from data/*.csv
python main.py forecast-all     # real Chronos-2, 180 series -> outputs/forecast_output.csv
python main.py decide-all       # Phase 9 -> outputs/decisions.csv, transfers.csv, alerts.csv
streamlit run streamlit_app.py
```

If the outputs are missing, the app says what to run. The sidebar's
**Backend status** panel shows the file timestamps. It also has **Reload outputs**
and **Run decide-all** (runs `python main.py decide-all`).

## HTML dashboard (one file, opens with a double-click)

`medcare_dashboard.html` is a single self-contained page (Plotly and all data embedded; no server,
no Python, no internet needed to view it). Seven pages: Executive Dashboard, Inventory Monitor,
Demand Forecast (item journey), Sales, Seasonal Insights, Risk & Alerts, Recommendations; light/dark toggle; CSV downloads.

Build it (after `forecast-all` and `decide-all`), from the `medcare/` folder:

    python html_frontend/build_html.py        # writes medcare_dashboard.html
    open medcare_dashboard.html

or double-click **Build and Open Dashboard.command**. Rebuild whenever the backend outputs change.
The page shows backend outputs only; sales figures are totals/averages of `data/sales.csv`
(by date, week, month, year; average per day, per weekday, per month; top SKUs; revenue by warehouse).

### Sales by month / year
Sales page → **📅 Month / Year** tab: pick a year, a month (or "All months"), and a SKU / warehouse
(or all). Only that period is shown: units, revenue, avg per day, avg unit price, change vs the previous
period, daily (or monthly) chart vs demand, split by SKU and by warehouse, CSV download.
**📈 Trends (all data)** keeps the full-history views.

### Seasonal Insights page
Three calendar seasons (India), defined once in `html_frontend/template.html` (`SEASONS`):
* 🤧 Flu & Respiratory — Autumn–Winter: Oct, Nov, Dec, Jan, Feb
* 🌼 Allergy & Hay Fever — Spring–Early Summer: Mar, Apr, May
* 🌧️ Monsoon & Vector-Borne — Rainy months / Late Summer: Jun, Jul, Aug, Sep

For all SKUs/warehouses or one: sales and demand in the season vs the rest of the year (uplift),
monthly averages, SKUs/warehouses with the biggest seasonal uplift, and a **prediction** for the next
occurrence of the season. The prediction is the real Chronos-2 forecast wherever the 30-day forecast
window overlaps the season (Oct 2026 → Flu season), compared with the same days last year, plus the
backend's seasonal demand signal (ELEVATED / SEASONAL / covariate gap) and each item's decision.
Seasons the forecast does not reach yet show last season's actuals, clearly labelled as history.
The data has no disease labels; seasons are calendar months.

## Red alerts by email (gaurav.levon@gmail.com)

`red_alert_mailer.py` emails every RED alert (HIGH severity: `STOCKOUT_RISK_HIGH`, `EXPIRY_RISK_HIGH`)
through the backend's notification engine. Already-emailed alerts are not re-sent; everything is logged
to `outputs/notification_log.csv` and shown on Risk & Alerts → 📧 Email notifications.

1. Gmail needs an **App Password**: Google Account → Security → 2-Step Verification (turn on) → App passwords → create one.
2. `cp email_settings.env.example email_settings.env` and paste the 16-character app password into it.
3. `python red_alert_mailer.py --test` (sends a test email)
4. `python red_alert_mailer.py --run` (runs decide-all and emails the new red alerts), or
   `python red_alert_mailer.py --watch` (leave it running: emails new red alerts the moment `outputs/alerts.csv` changes).

## Streamlit pages

| Page | Source |
|---|---|
| 📊 Executive Dashboard | KPI counts, sales performance (monthly chart, yearly/monthly tables, avg per day) from `decisions.csv` / `alerts.csv`, risk charts, decision trace (defaults to M001 / W002), top recommendations in the alert engine's display order |
| 📦 Inventory Monitor | `decisions.csv`, with SKU, warehouse, risk and action filters; usable vs required stock chart |
| 📈 Demand Forecast | tab 1: history from `database.get_demand_history`, the saved 30-day Chronos-2 forecast (`forecast_output.csv`); next day / 7 / 30 / average are derived from that one forecast. Tab 2 **Sales history**: monthly / yearly totals and average per day from `data/sales.csv`, filterable by SKU and warehouse |
| 🚨 Risk & Alerts | `alerts.csv` (select a row for its backend values), and batch expiry from `expiry_engine.analyse_batches` joined with `transfers.csv` |
| 💡 Recommendations | TRANSFER / REORDER / NO ACTION cards from `decisions.csv` + `transfers.csv` |

## Notes

* **MAE/RMSE** appear only after a real run of `ml/evaluate.py`: click "Run holdout
  evaluation" on the Forecast page, which needs Chronos-2. The result is saved to
  `outputs/model_evaluation.json`. No metrics are hard-coded.
* If `forecast_metadata.json` says the forecast came from something other than
  `Chronos2Pipeline`, a red **STAND-IN FORECAST** banner appears. If the data or config
  changed since the forecast was made, a "stale" warning appears.
* There is no What-if slider, because the backend has no scenario interface.
* There is no DEMAND_SURGE alert. Only the 7 alert types from `alert_engine.py` are shown.
* The views live in `frontend/views/`, not `pages/`. Streamlit would otherwise
  auto-create extra sidebar pages.

## Layout

```
streamlit_app.py            entry point, header, sidebar navigation (5 pages)
.streamlit/config.toml      theme
frontend/utils/data_loader.py   thin adapter over outputs/*.csv + backend functions (cached by file mtime)
frontend/utils/formatting.py    number/badge formatting only
frontend/components/            styles (CSS), kpi_cards, charts, alert_cards, recommendation_cards
frontend/views/                 dashboard, inventory, forecast, risk_alerts, recommendations
```
