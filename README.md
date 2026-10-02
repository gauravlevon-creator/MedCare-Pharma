# MedCare Pharma: Inventory Alerts + Demand Sensing (E1 + P1)

Decision support for supply-chain planners. It combines current inventory
(E1) with a 30-day demand forecast (P1) to recommend TRANSFER, REORDER or
NO_ACTION before a stock-out happens.

**Status:** Phases 1–9 are done, plus the E1 + P1 extensions (sales/receipt/adjustment ledger, seasonal demand signal, notifications): structure, SQLite, Chronos-2 integration,
the inventory calculator, the risk engine and the expiry engine. Allocation,
replenishment and alerts (Phases 8–13) come next.

## Inventory calculations and risk (Phases 5–6)

`decision_engine/inventory_calculator.py`

| Metric | Formula |
|---|---|
| Average daily demand | total historical demand / number of historical days (up to SIMULATION_DATE) |
| Lead-time demand | sum of the first `lead_time_days` Chronos-2 forecast days |
| Required stock | lead-time demand + safety stock |
| Projected inventory | current stock − lead-time demand |
| Days of stock | current stock / average daily demand (`None` if average demand is 0) |
| Excess inventory | current stock − required stock (positive = excess, negative = shortage) |

`decision_engine/risk_engine.py` is the only risk engine. All figures use
**usable stock**: recorded stock minus expired batches. Recorded and expired
stock stay visible for audit only.

* **P1 stock-out risk** (forecast-driven): projected usable inventory =
  usable stock − lead-time demand. ≤ 0 → **HIGH**; ≤ safety stock → **MEDIUM**;
  otherwise **LOW**.
* **E1 inventory alert** (real-time, separate flag): usable stock ≤ min
  threshold → `e1_threshold_alert = True`. On 2026-09-30 this fires for
  126 of 180 items.

`decision_engine.evaluate_inventory_state(sku, warehouse, forecast)` combines
inventory, P1 risk, the E1 alert and the expiry summary into one result.
Run it with `python main.py inventory-state --sku M001 --warehouse W002`.
Risk status is separate from the action (TRANSFER / REORDER / NO_ACTION),
which comes in a later phase.

## Ownership

| Area | Owner | Location |
|---|---|---|
| Source data | Ayush | `data/`, `database/schema.sql` |
| Forecasting | Nishit | `ml/` |
| Business decisions | Team lead | `decision_engine/` (Phase 5+) |
| Integration | Ruthie | `database/database.py`, `integration/`, `main.py` |
| Dashboard | Gaurav | Streamlit (later) |

## Data timelines

```
SOURCE DEMAND DATA:        2025-09-23 → 2026-09-22   (data/medcare_mysql.sql)
DERIVED DEMAND SIMULATION: 2025-10-01 → 2026-09-30
SOURCE SALES DATA:         2025-09-23 → 2026-09-22
DERIVED SALES SIMULATION:  2025-10-01 → 2026-09-30
SIMULATION / AS-OF DATE:   2026-09-30
FORECAST:                  2026-10-01 → 2026-10-30
```

**The original source files (`data/demand.csv`, `data/sales.csv`,
`data/inventory.csv`, `data/batches.csv`) are untouched.** The shifted
datasets are derived copies in `data/derived/`, and `init-db` regenerates
them on every run. Each copy has a metadata JSON recording the shift, row
counts and the source file's SHA-256.

* **Demand** and **sales** are both shifted forward 8 days, so both
  end on the simulation date. Only the date column changes; every other
  value is identical row for row.
* The source demand has 365 days (2025-09-23 .. 2026-09-22), so the demand
  history runs 2025-10-01 .. 2026-09-30.
* Source data comes from `data/medcare_mysql.sql`
  (`python database/import_mysql_dump.py` regenerates the four source CSVs).
* **Side effects of the shift:** weekday labels move by one day (8 days);
  calendar months stay almost aligned with the source.
* Chronos-2 receives plain arrays without timestamps, so its forecast
  values are unchanged; only the dates move.
* **Demand and sales are separate datasets.** Demand drives Chronos-2 and P1
  planning. Sales drives E1 stock consumption only, and is never used for
  forecasting.

## Data (Ayush, master)

* `demand.csv`: 65,700 rows, 30 SKUs × 6 warehouses × 365 days (loaded via the date-shifted copy above)
* `inventory.csv`: 180 rows (current stock equals the sum of each item's batches)
* `batches.csv`: 428 rows, 1–4 per SKU + warehouse (current stock = sum of batches)

`python main.py init-db` rebuilds `database/medcare.db` from these CSVs.
It validates them first, and it replaces the old database only after the
new one has loaded completely.

## Forecasting (Nishit)

* Model: **amazon/chronos-2** (`Chronos2Pipeline`), a pre-trained model
  used as-is, with no fine-tuning.
* Inputs per SKU + warehouse: daily `demand_qty`, plus `promotion_flag`
  and `season_flag` as covariates.
* Point forecast: the 0.5 quantile (median). Negative values are clipped to 0.
* **Future covariates fallback:** there is no promotion or season calendar
  for the forecast period, so each series' last known flag values are carried forward
  for all 30 days. On the last history day (2026-09-30), 12 of 180 series have
  `promotion_flag = 1`, so their forecasts assume a 30-day promotion.
  This is a limitation. The roadmap fix is to use real promotion and
  event calendars.
* The lag and rolling-mean features in `ml/features.py` are **not** model
  inputs.

### Evaluation: scope only

`python main.py evaluate` compares Chronos-2 with a 7-day moving average
on a holdout window (80/20 time split). Nishit reported these results for
**M001/W002 only**:

| Window | Setup | Model | MAE | RMSE |
|---|---|---|---:|---:|
| 7 days | univariate (`compare_baseline.py`) | 7-day MA | 7.39 | 9.27 |
| 7 days | univariate | Chronos-2 | 7.38 | 8.68 |
| 30 days (days 293–322 of the series; 2026-07-20 → 2026-08-18 on the simulation timeline) | with covariates (`evaluate_chronos.py`) | Chronos-2 | 6.76 | 8.11 |

These results cover one series and one window. They do not measure
overall model accuracy.

## Expiry (Phase 7)

`decision_engine/expiry_engine.py`. Ayush's data is not changed.

* **Recorded** = all batches · **Expired** = `expiry_date <= SIMULATION_DATE`
  · **Usable** = `expiry_date > SIMULATION_DATE`. The simulation date is the end
  of that day, so a batch expiring on it cannot serve future demand; it
  counts as expired/unavailable, keeping its NEAR_EXPIRY calendar label. Expired stock
  is visible for audit, but it is never transferred or counted in risk or
  replenishment.
* **Status** (calendar only): `<0` EXPIRED · `0–30` NEAR_EXPIRY ·
  `31–60` EXPIRING_SOON · `>60` NORMAL.
* **Expiry risk** (based on forecast consumption): usable batches are
  consumed earliest-expiry-first. A batch with `d` days to expiry can meet
  forecast demand on days 1..d.
  `expected_consumption = min(qty, forecast demand through expiry − demand used by earlier batches)`
  and `potential_expiry_excess = qty − expected_consumption`. Days beyond
  the 30-day forecast are extended at the mean daily forecast (flagged in
  `consumption_basis`). **HIGH** = excess in a NEAR_EXPIRY batch,
  **MEDIUM** = excess in an EXPIRING_SOON batch, **LOW** = neither.
* **Transfer assumption** (`TRANSFER_TRANSIT_DAYS = 2` in `config.py`): the
  data has no inter-warehouse transit time, and supplier `lead_time_days`
  is not used for transfers. A batch can be transferred only if
  `days_to_expiry > TRANSFER_TRANSIT_DAYS`.
* As of 2026-09-30, 99.5% of recorded units are usable. 7 batches (740 units)
  are expired or expire on the simulation date and count as unavailable; 16 usable
  batches expire within 30 days, 24 in 31–60 days and 381 after 60 days.
* **Supply-shortfall scenario (M004, M020, M030):** the last supplier deliveries
  for these three SKUs were short-shipped, so every non-expired batch of them
  holds 60% of its original quantity (`current_stock` = sum of its batches). The
  whole network is short of these SKUs, so transfers alone cannot cover demand
  and the engine reorders the remaining gap. All other SKUs, and all demand and
  sales data, are unchanged, so the Chronos-2 forecast cache stays valid.

`python main.py expiry --sku M001 --warehouse W002` prints the batch-level table.

## Cross-warehouse allocation (Phase 8)

`decision_engine/allocation_engine.py`: rule-based and deterministic, with
no optimiser and no ML. It runs per SKU across all six warehouses, so the
same stock is never promised twice. Every figure uses usable stock.

* **Shortage** (destination) = `max(0, required_stock − usable_stock)`.
* **Transfer eligibility**: the batch must be usable and have
  `days_to_expiry > TRANSFER_TRANSIT_DAYS` (2, a simulation assumption;
  supplier lead time is never used). Transferred stock can serve
  destination demand on forecast days 3..expiry.
* **Destination consumption before expiry** = destination forecast demand
  from arrival to expiry, minus the destination's own stock and earlier
  transfers that expire first.
* **EXPIRY_PREVENTION** transfer: `min(potential expiry excess, destination
  consumption before expiry, capacity room)`. Allowed even when the source
  is above its required stock.
* **SHORTAGE_EXCESS** transfer: `min(shortage, source excess, batch quantity,
  destination consumption before expiry, capacity room)`. Source excess
  = usable − required − units still forecast to expire unused at the
  source. The source never drops below its required stock.
* **Capacity room** = capacity − recorded stock (expired units still take
  space) − inbound transfers.
* **Order**:
  1. Shortage destinations, HIGH risk first and larger shortage first. For
     each, take expiry-risk batches from other warehouses first, then normal
     excess, earliest expiry first.
  2. Any remaining expiry excess goes to warehouses that can use it before
     it expires. Warehouses that already shipped this SKU out are skipped,
     so stock is never shipped both ways.
* **Actions**:
  * **TRANSFER**: the warehouse receives stock. Any shortage still left
    becomes a supplementary reorder.
  * **REORDER**: shortage with no eligible source; the quantity is the
    shortage rounded up, capped at capacity room.
  * **NO_ACTION**: no shortage and nothing inbound.

`python main.py allocate --sku M001 --warehouse W002` prints the transfers
(batch, quantity, type, expiry, usable stock before and after) and each
warehouse's decision.

## Final decisions and alerts (Phase 9)

`decision_engine/final_decision.py` gives each SKU + warehouse exactly one
action:

* **REORDER**: usable stock is still below required stock after all valid
  transfers. `reorder_quantity = ceil(remaining shortage)`, capped at
  capacity room. If transfers are also planned, `transfer_recommended = True`
  and both facts stay in the detailed result.
* **TRANSFER**: no reorder is needed and the warehouse either receives stock
  or ships an expiry-prevention batch out.
* **NO_ACTION**: no shortage remains and no inbound or expiry-prevention
  transfer involves the warehouse. A warehouse that only ships surplus
  shows `transfer_role = SOURCE`.

P1 risk, the E1 alert and expiry risk are reported as they were before
allocation.

`decision_engine/alert_engine.py` builds alerts from final decisions only:
`STOCKOUT_RISK_HIGH/MEDIUM`, `EXPIRY_RISK_HIGH/MEDIUM`, `E1_THRESHOLD` (kept
separate from P1), `TRANSFER_RECOMMENDED` and `REORDER_RECOMMENDED`. Each
alert carries SKU, warehouse, severity, type, message, a supporting value
and the recommended action.

Display order (presentation only, not a score):
1. HIGH stock-out
2. HIGH expiry
3. MEDIUM stock-out
4. MEDIUM expiry
5. E1 threshold
6. Transfer and reorder (informational)

Within each group, larger values come first.

* `python main.py decide --sku M001 --warehouse W002`: one SKU's decisions and alerts.
* `python main.py decide-all`: all 180 items, written to
  `outputs/decisions.csv`, `transfers.csv` and `alerts.csv` for the dashboard.

## E1 operations: sales, receipts and adjustments

### P1 plan vs E1 operations

The **Phase 9 decision engine is the authoritative P1 planning decision as of
2026-09-30.** The E1 simulation executes that plan and then simulates what
happens operationally over the next 30 days. It never changes the Phase 9
decision engine, and its later reorders are never presented as the
2026-09-30 recommendation.

```
2026-09-30
Initial P1 planning decision (Phase 9: decide / decide-all)
        ↓
TRANSFER / REORDER / NO_ACTION
        ↓
30-day operational simulation (python main.py simulate)
        ↓
Daily SALES consume inventory
        ↓
Open transfers/reorders arrive according to their lead times
        ↓
If stock + open orders reaches minimum threshold
        ↓
Simulation generates an additional operational reorder
        ↓
Continue simulation
```

Every purchase order and receipt carries its origin:

| origin | reference prefix | meaning |
|---|---|---|
| `P1_PLAN` | `P1-PO:` / `P1-TRF:` | the Phase 9 recommendation as of 2026-09-30 |
| `E1_OPERATIONAL_REORDER` | `E1-OPS-PO:` | generated during the simulation by the reorder-point rule; **not** part of the P1 recommendation |

**The E1 reorder-point rule** uses only existing data: thresholds, lead
times, and the historical average demand already used to scale sales. At
the end of each simulated day (from 2026-10-01; 2026-09-30 belongs to the P1
plan), if usable + on order ≤ min_threshold, it orders
`ceil(min_threshold + avg_daily_demand × lead_time_days) − (usable + on order)`,
received after `lead_time_days`.

**Policies:**
* `python main.py simulate` uses **`plan+reorder_point` (default)**: the P1
  plan plus operational reorders.
* `python main.py simulate --policy plan-only` runs the P1 plan only, for
  comparison and testing. `plan` is an alias.

The CLI runs the P1 plan on the saved **real Chronos-2** forecast. It
refuses to run if the forecast cache was produced by anything else
(`--allow-stand-in` exists for testing only, and labels the result).

### Ledger

`operations/ledger.py` keeps a batch-level, auditable inventory ledger.
Every stock change is one row in `inventory_transactions` (SQLite) and in
`outputs/transaction_log.csv`:

| txn_type | subtype | effect |
|---|---|---|
| SALE | – | stock leaves, earliest expiry first; `unfulfilled_quantity` = lost sale |
| RECEIPT | TRANSFER_RECEIPT / SUPPLIER_RECEIPT | stock arrives (reason states P1 plan or E1 operational) |
| ADJUSTMENT | EXPIRY_WRITE_OFF / INVENTORY_CORRECTION | deliberate change, always with a reason |
| TRANSFER_OUT | – | dispatch to another warehouse, paired with the destination's receipt |

**Stock never goes negative.** A sale larger than usable stock is only
partly fulfilled and the rest is logged as unfulfilled. Adjustments larger
than the stock they target are rejected.

### Simulation details (`operations/sales_simulation.py`)

* **Opening write-off:** at the end of 2026-09-30, batches with
  `expiry_date <= 2026-09-30` are written off. They stay visible in the
  ledger.
* **P1 transfers:** dispatched 2026-10-01 and received after
  `TRANSFER_TRANSIT_DAYS` (2), keeping the same batch and expiry.
* **P1 and E1 reorders:** received after the item's `lead_time_days`. A
  supplier batch's expiry is a documented assumption
  (`SUPPLIER_RECEIPT_SHELF_LIFE_DAYS`), because the data has no shelf-life
  information.
* **Requested sales:** for day d, units_sold on d − 365 (the same days last
  year) × (item's mean demand ÷ mean sales). Raw sales run about 2.8× the
  demand that stock and thresholds were sized for. `--scaling raw` is
  available as a stress test.
* **E1 check each day:** usable stock ≤ min_threshold gives a BREACH event
  (when an item enters breach) and a RECOVERED event when it leaves.

### Simulation outputs (for the dashboard)

| file | content |
|---|---|
| `outputs/simulation_summary.json` | sales requested/fulfilled/lost, fill rate, P1 plan vs E1 operational reorders, transfer and supplier receipts, adjustments and expiry write-offs, opening/ending inventory, E1 breaches, stock-outs, forecast source (real Chronos-2 or not) |
| `outputs/simulation_purchase_orders.csv` | every purchase order with `origin` and `trigger` |
| `outputs/transaction_log.csv` | the full ledger |
| `outputs/simulated_inventory_daily.csv` | on-hand, usable, threshold state, requested/sold/lost per item per day |

## Seasonal demand signal (P1 sensing)

`decision_engine/seasonal_signal.py` (`python main.py seasonal-signal`).
**This is not an influenza model;** the data contains no flu information,
so there is no flu label.

For the forecast window, it compares the same period last year:
* **ELEVATED:** season_flag active on ≥ 50% of days and demand ≥ 1.05× the
  item's average.
* **SEASONAL:** season_flag active on ≥ 20% of days.
* **NORMAL:** otherwise.

It also flags a **covariate gap**. season_flag is 0 on 2026-09-30, so
Chronos-2's carry-forward fallback assumes no season, even where last
year's same window was seasonal. The signal appears in `decisions.csv` and
in stock-out alert messages, and is written to
`outputs/seasonal_demand_signal.csv`. It does not change any forecast or
quantity.

## Notifications

`notification_engine/` takes alert-engine output (and E1 breach events) and
delivers the configured types (`NOTIFY_ALERT_TYPES`: HIGH stock-out, E1
threshold, HIGH expiry, reorder) as **one digest email per run**. It logs
one row per alert to the `notifications` table and to
`outputs/notification_log.csv`. Repeat alerts (same item, type and date)
are not logged twice.

Credentials come **only from environment variables**:

```bash
export MEDCARE_NOTIFICATION_MODE=auto     # outbox (default) | email | auto
export SMTP_HOST=smtp.gmail.com SMTP_PORT=587 SMTP_USERNAME=you@example.com
export SMTP_PASSWORD='app-password' ALERT_EMAIL_TO='planner@example.com'
```

The logged status is always what actually happened:
* **SENT:** the SMTP server accepted the message.
* **OUTBOX:** recorded only; nothing was sent.
* **FAILED:** sending was attempted; the error is stored, never the password.

`email` mode refuses to start with incomplete settings, and `auto` falls
back to OUTBOX. SMS and WhatsApp are not implemented.

## Database

SQLite, rebuilt by `init-db` from the source and derived CSVs:

* **Ayush's tables (`schema.sql`, unchanged):** `demand`, `inventory`, `batches`.
* **E1 additions (`schema_extensions.sql`):**
  * `sales_history`, loaded from the derived sales copy
  * `inventory_transactions` (the ledger)
  * `notifications` (the alert and notification log)

## Setup and commands

Run all commands from the `medcare/` folder.

Windows (PowerShell):

```powershell
py -3.13 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Then:

```bash
python main.py init-db                                  # build SQLite
python main.py validate                                 # data checks
python main.py forecast --sku M001 --warehouse W002     # Phase 4 golden forecast
python main.py forecast-all                             # 180 series -> outputs/forecast_output.csv
python main.py evaluate --sku M001 --warehouse W002 --horizon 30
python main.py inventory-state --sku M001 --warehouse W002   # Phase 5-6 checkpoint
python main.py expiry --sku M001 --warehouse W002            # Phase 7 checkpoint
python main.py allocate --sku M001 --warehouse W002          # Phase 8 checkpoint
python main.py decide --sku M001 --warehouse W002            # Phase 9 checkpoint
python main.py decide-all                                    # all 180 items -> outputs/*.csv
python main.py seasonal-signal --sku M001                    # P1 seasonal demand signal
python main.py simulate                                      # E1 30-day sim (default policy plan+reorder_point)
python main.py simulate --policy plan-only                   # comparison: P1 plan only
python main.py simulate --notify                             # also send/record E1 breach notifications
python main.py notify                                        # deliver outputs/alerts.csv (email/outbox)
python -m pytest -v                                     # all tests
```

The first forecast downloads `amazon/chronos-2` from Hugging Face.
`forecast-all` saves its results to `outputs/`. Later commands reuse
them until the data or the config changes. Add `--refresh` to force a
new run.

## Project layout

```
medcare/
├── config.py               central constants (simulation date, horizon, paths, model)
├── main.py                 CLI
├── data/                   source CSVs (demand, inventory, batches, sales) + derived/ shifted copies
├── database/               schema.sql (Ayush), schema_extensions.sql (E1), database.py, simulation_data.py
├── ml/                     model.py, predict.py, data_loader.py, validation.py, evaluate.py, features.py
├── integration/            ml_service.py (forecast access + cache)
├── operations/             ledger.py, sales_simulation.py (E1)
├── notification_engine/    settings.py, engine.py (email / outbox)
├── decision_engine/        inventory_calculator.py, risk_engine.py, expiry_engine.py, allocation_engine.py, final_decision.py, alert_engine.py, seasonal_signal.py, decision_engine.py
├── tests/                  unit + integration tests
└── outputs/                generated: forecast, decisions, transfers, alerts, transaction_log,
                            simulated_inventory_daily, simulation_summary.json, simulation_purchase_orders,
                            notification_log, seasonal_demand_signal
```
