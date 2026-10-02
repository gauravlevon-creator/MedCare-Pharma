"""
MedCare command-line entry point. Run from the medcare/ folder.

    python main.py init-db                          # build SQLite from data/*.csv
    python main.py validate                         # data validation report
    python main.py forecast --sku M001 --warehouse W002
    python main.py forecast-all [--refresh]         # 180 series -> outputs/forecast_output.csv
    python main.py evaluate --sku M001 --warehouse W002 --horizon 30
    python main.py inventory-state --sku M001 --warehouse W002   # Phase 5-6 checkpoint
    python main.py expiry --sku M001 --warehouse W002            # Phase 7 checkpoint
    python main.py allocate --sku M001 --warehouse W002          # Phase 8 checkpoint
    python main.py decide --sku M001 --warehouse W002            # Phase 9 checkpoint
    python main.py decide-all                                    # all 180 items -> outputs/*.csv
    python main.py seasonal-signal --sku M001                    # P1 seasonal demand signal
    python main.py simulate [--notify]                           # E1 30-day sim (default plan+reorder_point)
    python main.py simulate --policy plan-only                   # comparison: P1 plan only
    python main.py notify [--mode outbox|email|auto]             # deliver alerts.csv
"""

from __future__ import annotations

import argparse
import logging
import sys

import config


def cmd_init_db(_args) -> int:
    from database import database as db
    counts = db.init_database()
    print(f"Database created at {config.DATABASE_PATH}")
    for table, n in counts.items():
        print(f"  {table:<10} {n:>6} rows")
    return 0


def cmd_validate(_args) -> int:
    from ml.data_loader import load_demand_data
    from ml.validation import validate_demand_data
    report = validate_demand_data(load_demand_data())
    print("Demand validation OK:", report["ok"])
    for key, value in report["summary"].items():
        print(f"  {key}: {value}")
    for issue in report["issues"]:
        print("  ISSUE:", issue)
    return 0 if report["ok"] else 1


def cmd_forecast(args) -> int:
    from integration.ml_service import get_forecast
    fc = get_forecast(args.sku, args.warehouse, refresh=args.refresh)
    print(f"\n{config.FORECAST_HORIZON_DAYS}-day forecast for {args.sku} / {args.warehouse} "
          f"(simulation date {config.SIMULATION_DATE}, model {config.CHRONOS_MODEL_NAME})")
    shown = fc.copy()
    shown["date"] = shown["date"].dt.strftime("%Y-%m-%d")
    shown["forecast_demand"] = shown["forecast_demand"].round(2)
    print(shown.to_string(index=False))
    print(f"\nTotal forecast demand: {fc['forecast_demand'].sum():.2f} units")
    print(f"Mean daily forecast:   {fc['forecast_demand'].mean():.2f} units/day")
    return 0


def cmd_forecast_all(args) -> int:
    from integration.ml_service import get_all_forecasts
    fc, meta = get_all_forecasts(refresh=args.refresh)
    print(f"Series forecast: {fc[['sku_id', 'warehouse_id']].drop_duplicates().shape[0]}")
    print(f"Rows:            {len(fc)}")
    print(f"Date range:      {fc['date'].min().date()} to {fc['date'].max().date()}")
    print(f"Output:          {config.FORECAST_OUTPUT_CSV}")
    errors = meta.get("errors", {})
    print(f"Errors:          {len(errors)}")
    for key, msg in errors.items():
        print(f"  {key}: {msg}")
    return 1 if errors else 0


def cmd_evaluate(args) -> int:
    from ml.evaluate import evaluate_series
    result = evaluate_series(args.sku, args.warehouse, args.horizon,
                             use_covariates=not args.no_covariates)
    for key, value in result.items():
        print(f"{key:>20}: {value}")
    return 0


def cmd_inventory_state(args) -> int:
    from decision_engine.decision_engine import evaluate_inventory_state, format_state_report
    from integration.ml_service import get_forecast
    forecast = get_forecast(args.sku, args.warehouse, refresh=args.refresh)
    state = evaluate_inventory_state(args.sku, args.warehouse, forecast)
    print(f"\nInventory state (simulation date {config.SIMULATION_DATE}, "
          f"forecast {config.CHRONOS_MODEL_NAME})")
    print(format_state_report(state))
    print("\nForecast days used for lead-time demand:")
    used = forecast.sort_values("date").head(state["lead_time_days"])
    for row in used.itertuples(index=False):
        print(f"  {row.date:%Y-%m-%d}  {row.forecast_demand:8.2f}")
    return 0


def cmd_expiry(args) -> int:
    from decision_engine.expiry_engine import evaluate_expiry
    from integration.ml_service import get_forecast
    forecast = get_forecast(args.sku, args.warehouse, refresh=args.refresh)
    r = evaluate_expiry(args.sku, args.warehouse, forecast)
    print(f"\nExpiry analysis {r.sku_id} / {r.warehouse_id} as of {r.as_of_date} "
          f"(usable = expiry_date > as-of date; transit assumption {config.TRANSFER_TRANSIT_DAYS} days)")
    print(f"{'batch_id':<8} {'qty':>5} {'expiry':<10} {'days':>5} {'status':<13} "
          f"{'fc_to_expiry':>12} {'consumed':>9} {'excess':>8} {'xfer_ok':<7} basis")
    for d in r.batch_details:
        fc = "-" if d["forecast_demand_through_expiry"] is None else f"{d['forecast_demand_through_expiry']:.2f}"
        print(f"{d['batch_id']:<8} {d['quantity']:>5} {d['expiry_date']:<10} {d['days_to_expiry']:>5} "
              f"{d['expiry_status']:<13} {fc:>12} {d['expected_consumption']:>9.2f} "
              f"{d['potential_expiry_excess']:>8.2f} {str(d['transfer_eligible']):<7} {d['consumption_basis'] or '-'}")
    for label, value in [("Total recorded", r.total_recorded_quantity),
                         ("Expired", r.expired_quantity), ("Usable", r.usable_quantity),
                         ("Near expiry (0-30d)", r.near_expiry_quantity),
                         ("Expiring soon (31-60d)", r.expiring_soon_quantity),
                         ("Normal (>60d)", r.normal_quantity),
                         ("Potential expiry excess", r.potential_expiry_excess),
                         ("  of which near expiry", r.near_expiry_excess),
                         ("Transfer-eligible qty", r.transfer_eligible_quantity),
                         ("Expiry risk", r.expiry_risk), ("Reason", r.expiry_reason)]:
        print(f"{label + ':':<25} {value}")
    return 0


def cmd_allocate(args) -> int:
    from decision_engine.allocation_engine import allocate_sku
    r = allocate_sku(args.sku)
    print(f"\nCross-warehouse allocation for {r.sku_id} (simulation date {config.SIMULATION_DATE}; "
          f"transfer transit assumption {config.TRANSFER_TRANSIT_DAYS} days)")
    transfers = [t for t in r.transfers
                 if args.warehouse in (None, t.destination_warehouse, t.source_warehouse)]
    print(f"\nTransfers ({len(transfers)}):")
    for t in transfers:
        print(f"  {t.source_warehouse} -> {t.destination_warehouse}  batch {t.batch_id}  "
              f"qty {t.transfer_quantity}  [{t.transfer_type}]")
        print(f"      expiry {t.expiry_date} ({t.days_to_expiry} days); destination consumption "
              f"before expiry {t.destination_consumption_before_expiry:,.2f}")
        print(f"      source usable {t.source_usable_before} -> {t.source_usable_after}; "
              f"destination usable {t.destination_usable_before} -> {t.destination_usable_after}")
        print(f"      reason: {t.reason}")
    print("\nDecisions:")
    for w, d in r.decisions.items():
        if args.warehouse and w != args.warehouse:
            continue
        qty = (f" reorder {d.reorder_quantity}" if d.reorder_quantity else "") + \
              (f" supplementary reorder {d.supplementary_reorder_quantity}" if d.supplementary_reorder_quantity else "")
        print(f"  {w}: {d.recommended_action:<9} P1 {d.risk_status:<6} usable {d.usable_stock_before} -> "
              f"{d.usable_stock_after} required {d.required_stock:,.2f} shortage {d.shortage_before:,.2f}"
              f" -> {d.remaining_shortage:,.2f}{qty}")
        print(f"      {d.reason}")
    return 0


def cmd_decide(args) -> int:
    from decision_engine.decision_engine import evaluate_sku
    decisions, transfers, alerts = evaluate_sku(args.sku)
    print(f"\nFinal decisions for {args.sku} (simulation date {config.SIMULATION_DATE})")
    for w, d in decisions.items():
        if args.warehouse and w != args.warehouse:
            continue
        print(f"\n  {w}: {d.final_action}  (transfer_recommended={d.transfer_recommended}, "
              f"role={d.transfer_role or '-'}, reorder={d.reorder_quantity})")
        print(f"      P1 before allocation: {d.risk_status} | E1 alert: {d.e1_threshold_alert} | "
              f"expiry risk: {d.expiry_risk} (excess {d.potential_expiry_excess:,.2f})")
        print(f"      usable {d.usable_stock_before} -> {d.usable_stock_after}, required "
              f"{d.required_stock:,.2f}, shortage {d.shortage_before:,.2f} -> {d.remaining_shortage:,.2f}")
        print(f"      in {d.transfer_in_quantity} from {d.source_warehouses or '-'}; out "
              f"{d.transfer_out_quantity} to {d.destination_warehouses or '-'}")
        print(f"      reason: {d.reason}")
    shown = [a for a in alerts if not args.warehouse or a.warehouse_id == args.warehouse]
    print(f"\nAlerts ({len(shown)}), in display order:")
    for a in shown:
        print(f"  [{a.severity:<6}] {a.alert_type:<21} {a.warehouse_id}  {a.value_label}={a.value:,.2f}  "
              f"-> {a.recommended_action}")
        print(f"      {a.message}")
    return 0


def cmd_decide_all(args) -> int:
    from decision_engine.decision_engine import evaluate_all
    decisions, transfers, alerts = evaluate_all()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(config.OUTPUT_DIR / "decisions.csv", index=False)
    transfers.to_csv(config.OUTPUT_DIR / "transfers.csv", index=False)
    alerts.to_csv(config.OUTPUT_DIR / "alerts.csv", index=False)
    print(f"Items evaluated:  {len(decisions)}")
    print(f"Final actions:    {decisions['final_action'].value_counts().to_dict()}")
    print(f"Transfers:        {len(transfers)}")
    print(f"Alerts by type:   {alerts['alert_type'].value_counts().to_dict() if len(alerts) else {}}")
    print(f"Written to:       {config.OUTPUT_DIR}/decisions.csv, transfers.csv, alerts.csv")
    print(f"\nTop {args.top} alerts:")
    for a in alerts.head(args.top).itertuples():
        print(f"  [{a.severity:<6}] {a.alert_type:<21} {a.sku_id}/{a.warehouse_id}  "
              f"{a.value_label}={a.value:,.2f} -> {a.recommended_action}")
    return 0


def _saved_forecasts_or_none():
    """Saved Chronos-2 forecast if the cache is valid, else None (never triggers a model run)."""
    from integration import ml_service
    from ml import data_loader
    cached = ml_service._load_cache(ml_service._expected_metadata(data_loader.load_demand_data()))
    return cached[0] if cached else None


def cmd_seasonal(args) -> int:
    from decision_engine.seasonal_signal import compute_seasonal_signals
    fc = _saved_forecasts_or_none()
    sig = compute_seasonal_signals(forecasts=fc)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sig.to_csv(config.SEASONAL_SIGNAL_CSV, index=False)
    print(f"Seasonal demand signal for {sig.window_start.iloc[0]}..{sig.window_end.iloc[0]} "
          f"(historical season_flag pattern; not an influenza model)")
    print(f"  {sig.seasonal_signal.value_counts().to_dict()}; covariate gap on {int(sig.covariate_gap.sum())} items"
          f"{'' if fc is not None else ' (no saved forecast: forecast_demand_index left empty)'}")
    if args.sku:
        for r in sig[sig.sku_id == args.sku].itertuples():
            print(f"  {r.sku_id}/{r.warehouse_id}: {r.seasonal_signal} | {r.seasonal_reason}")
    print(f"Written to {config.SEASONAL_SIGNAL_CSV}")
    return 0


def cmd_simulate(args) -> int:
    """
    E1 30-day operational simulation.
    1. Load the saved/real Chronos-2 forecast and run the Phase 9 P1 decision as of
       SIMULATION_DATE (authoritative; not modified by the simulation).
    2. Simulate daily sales, P1 transfer/reorder receipts, expiry write-offs and, with
       the default policy plan+reorder_point, additional E1 operational reorders.
    """
    from decision_engine.decision_engine import evaluate_all, forecast_provider_from_frame
    from integration.ml_service import get_all_forecasts
    from notification_engine import NotificationEngine, export_notification_log
    from operations.sales_simulation import (e1_alerts_from_events, resolve_policy, run_simulation,
                                             write_summary)
    policy = resolve_policy(args.policy)
    decisions = transfers = None
    source = {"note": "no P1 plan used (--no-plan)"}
    if not args.no_plan:
        forecasts, meta = get_all_forecasts()
        pipeline = meta.get("pipeline", "unknown (forecast cache predates pipeline tracking)")
        is_real = pipeline == "Chronos2Pipeline"
        source = {"model_name": meta.get("model_name"), "pipeline": pipeline,
                  "is_real_chronos2": is_real, "generated_at_utc": meta.get("generated_at_utc"),
                  "forecast_rows": meta.get("rows"), "forecast_errors": len(meta.get("errors", {}))}
        if not is_real and not args.allow_stand_in:
            print(f"Refusing to simulate: the saved forecast was produced by '{pipeline}', not the real "
                  f"Chronos-2 model. Run 'python main.py forecast-all --refresh' first "
                  f"(or pass --allow-stand-in for testing only).")
            return 2
        decisions, transfers, _ = evaluate_all(forecast_provider_from_frame(forecasts))
    result = run_simulation(days=args.days, decisions=decisions, transfers=transfers,
                            scaling=args.scaling, replenishment_policy=policy, persist=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.transactions.to_csv(config.TRANSACTION_LOG_CSV, index=False)
    result.daily.to_csv(config.SIMULATED_INVENTORY_CSV, index=False)
    summary = write_summary(result, forecast_source=source)
    _print_simulation_summary(summary)
    if result.reconciliation_problems:
        print("  RECONCILIATION PROBLEMS:", result.reconciliation_problems[:5])
    print(f"\nWritten: {config.SIMULATION_SUMMARY_JSON.name}, {config.SIMULATION_PURCHASE_ORDERS_CSV.name}, "
          f"{config.TRANSACTION_LOG_CSV.name}, {config.SIMULATED_INVENTORY_CSV.name}; ledger in SQLite "
          f"inventory_transactions (run_id {result.run_id})")
    if args.notify:
        res = NotificationEngine().notify(e1_alerts_from_events(result.threshold_events, decisions))
        export_notification_log()
        print(f"E1 threshold notifications: {res.notified} logged as {res.status} via {res.channel} "
              f"({res.skipped_duplicates} duplicates skipped){' - ' + res.error if res.error else ''}")
    return 0 if not result.reconciliation_problems else 1


def _print_simulation_summary(s: dict) -> None:
    fs = s.get("forecast_source", {})
    print(f"\nE1 operational simulation {s['period']} ({s['days']} days), policy {s['replenishment_policy']}, "
          f"sales scaling {s['sales_scaling']}")
    if "pipeline" in fs:
        tag = "REAL Chronos-2" if fs.get("is_real_chronos2") else "STAND-IN (not a real result)"
        print(f"P1 plan forecast: {fs.get('model_name')} via {fs['pipeline']} [{tag}], "
              f"generated {fs.get('generated_at_utc')}")
    sa, p1, e1 = s["sales"], s["p1_plan_2026_09_30"], s["e1_operational_reorders"]
    rc, ad, inv, th, so = s["receipts"], s["adjustments"], s["inventory"], s["e1_threshold"], s["stockouts"]
    rows = [
        ("Sales requested (units)", sa["units_requested"]),
        ("Sales fulfilled (units)", sa["units_fulfilled"]),
        ("Lost sales (units)", sa["units_lost"]),
        ("Fill rate", f"{sa['fill_rate']:.2%}" if sa["fill_rate"] is not None else "n/a"),
        ("P1 plan (2026-09-30): transfers dispatched", f"{p1['transfers_dispatched']} ({p1['transfer_units_dispatched']} units)"),
        ("P1 plan (2026-09-30): reorders placed", f"{p1['reorders_placed']} ({p1['reorder_units']} units)"),
        ("E1 operational reorders (simulation)", f"{e1['orders_placed']} ({e1['units_ordered']} units, {e1['items_reordered']} items)"),
        ("Transfer receipts", f"{rc['transfer_receipts']} ({rc['transfer_units_received']} units)"),
        ("Supplier receipts: P1 plan / E1 operational", f"{rc['supplier_receipts_p1_plan']} / {rc['supplier_receipts_e1_operational']} ({rc['supplier_units_received']} units)"),
        ("Still in transit at end", f"{rc['still_in_transit_at_end']} ({rc['units_in_transit_at_end']} units)"),
        ("Adjustments", f"{ad['count']} (net {ad['units_net']} units)"),
        ("Expired units written off: opening / during", f"{ad['expired_units_written_off_opening']} / {ad['expired_units_written_off_during_simulation']}"),
        ("Inventory on hand: opening -> ending", f"{inv['opening_on_hand_units']} -> {inv['ending_on_hand_units']} (ending usable {inv['ending_usable_units']})"),
        ("E1 breaches: items at start / at end", f"{th['items_in_breach_at_start']} / {th['items_in_breach_at_end']}"),
        ("E1 breach events (new during sim) / recoveries", f"{th['breach_events']} ({th['new_breaches_during_simulation']}) / {th['recoveries']}"),
        ("Stock-outs: item-days with lost sales / items", f"{so['item_days_with_lost_sales']} / {so['items_with_any_lost_sales']}"),
        ("Ledger reconciled", s["reconciled"]),
    ]
    for label, value in rows:
        print(f"  {label + ':':<50} {value}")
    if so["top_items_by_lost_units"]:
        print("  Top lost-sales items: " + ", ".join(f"{x['sku_id']}/{x['warehouse_id']} ({x['units_lost']})"
                                                     for x in so["top_items_by_lost_units"][:5]))


def cmd_notify(args) -> int:
    import pandas as pd
    from notification_engine import NotificationEngine, export_notification_log
    if not config.OUTPUT_DIR.joinpath("alerts.csv").exists():
        print("outputs/alerts.csv not found. Run: python main.py decide-all")
        return 1
    alerts = pd.read_csv(config.OUTPUT_DIR / "alerts.csv").to_dict("records")
    engine = NotificationEngine(mode=args.mode)
    res = engine.notify(alerts)
    export_notification_log()
    print(f"Mode: {res.mode}; channel: {res.channel}; status: {res.status}")
    print(f"Notified: {res.notified}; duplicates skipped: {res.skipped_duplicates}; "
          f"alert types not configured for notification: {res.skipped_types}")
    if res.status == "OUTBOX":
        print("Recorded in the outbox only. No email was sent.")
    if res.error:
        print(f"Detail: {res.error}")
    print(f"Log: {config.NOTIFICATION_LOG_CSV}")
    return 0 if res.status != "FAILED" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MedCare inventory decision support")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db").set_defaults(func=cmd_init_db)
    sub.add_parser("validate").set_defaults(func=cmd_validate)

    p = sub.add_parser("forecast")
    p.add_argument("--sku", required=True)
    p.add_argument("--warehouse", required=True)
    p.add_argument("--refresh", action="store_true", help="ignore the saved forecast")
    p.set_defaults(func=cmd_forecast)

    p = sub.add_parser("forecast-all")
    p.add_argument("--refresh", action="store_true")
    p.set_defaults(func=cmd_forecast_all)

    p = sub.add_parser("inventory-state", help="Phase 5-6: inventory calculations + risk")
    p.add_argument("--sku", required=True)
    p.add_argument("--warehouse", required=True)
    p.add_argument("--refresh", action="store_true", help="ignore the saved forecast")
    p.set_defaults(func=cmd_inventory_state)

    p = sub.add_parser("expiry", help="Phase 7: batch-level expiry analysis")
    p.add_argument("--sku", required=True)
    p.add_argument("--warehouse", required=True)
    p.add_argument("--refresh", action="store_true", help="ignore the saved forecast")
    p.set_defaults(func=cmd_expiry)

    p = sub.add_parser("allocate", help="Phase 8: cross-warehouse allocation for one SKU")
    p.add_argument("--sku", required=True)
    p.add_argument("--warehouse", help="only show transfers/decisions involving this warehouse")
    p.set_defaults(func=cmd_allocate)

    p = sub.add_parser("decide", help="Phase 9: final actions + alerts for one SKU")
    p.add_argument("--sku", required=True)
    p.add_argument("--warehouse", help="only show this warehouse")
    p.set_defaults(func=cmd_decide)

    p = sub.add_parser("decide-all", help="Phase 9: final actions + alerts for all 180 items")
    p.add_argument("--top", type=int, default=15)
    p.set_defaults(func=cmd_decide_all)

    p = sub.add_parser("seasonal-signal", help="P1 seasonal demand signal for the forecast window")
    p.add_argument("--sku", help="print reasons for this SKU")
    p.set_defaults(func=cmd_seasonal)

    p = sub.add_parser(
        "simulate",
        help="E1 30-day operational simulation (default policy: plan+reorder_point)",
        description=("Runs the Phase 9 P1 decision as of the simulation date (authoritative, unchanged), "
                     "then simulates 30 days of sales, P1 transfer/reorder receipts, expiry write-offs and, "
                     "by default, additional E1 operational reorders when usable stock + open orders reach "
                     "the minimum threshold. Use --policy plan-only to compare without operational reorders."))
    p.add_argument("--days", type=int, default=config.E1_SIMULATION_DAYS)
    p.add_argument("--policy", choices=["plan+reorder_point", "plan-only", "plan"],
                   default=config.E1_REPLENISHMENT_POLICY,
                   help="plan+reorder_point (default) | plan-only (P1 plan only; 'plan' is an alias)")
    p.add_argument("--scaling", choices=["demand_calibrated", "raw"], default=config.SALES_SIMULATION_SCALING)
    p.add_argument("--no-plan", action="store_true", help="sales and expiry only; no Phase 9 transfers/reorders")
    p.add_argument("--notify", action="store_true", help="send/record E1 threshold breach notifications")
    p.add_argument("--allow-stand-in", action="store_true",
                   help="testing only: allow a non-Chronos forecast cache (result is labelled STAND-IN)")
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("notify", help="Deliver outputs/alerts.csv via email or record in the outbox")
    p.add_argument("--mode", choices=["outbox", "email", "auto"], default=None,
                   help="overrides MEDCARE_NOTIFICATION_MODE")
    p.set_defaults(func=cmd_notify)

    p = sub.add_parser("evaluate")
    p.add_argument("--sku", default="M001")
    p.add_argument("--warehouse", default="W002")
    p.add_argument("--horizon", type=int, default=config.FORECAST_HORIZON_DAYS)
    p.add_argument("--no-covariates", action="store_true")
    p.set_defaults(func=cmd_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        return args.func(args)
    except Exception as exc:  # report cleanly at the CLI boundary
        logging.getLogger("main").error("%s: %s", type(exc).__name__, exc)
        if args.verbose:
            raise
        return 2


if __name__ == "__main__":
    sys.exit(main())
