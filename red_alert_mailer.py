"""
Red-alert mailer: emails every RED (HIGH severity) alert as soon as the Phase 9
alert engine produces it.

Red alerts = severity HIGH from decision_engine/alert_engine.py:
    STOCKOUT_RISK_HIGH, EXPIRY_RISK_HIGH

It reuses the backend's own notification engine (notification_engine/), so it
never changes any alert or decision. Delivery is a digest email per run through
Gmail SMTP. Every alert is logged (SQLite `notifications` + outputs/notification_log.csv)
and an alert that was already emailed (same SKU, warehouse, type and date) is not
sent again.

Setup (one time) - create a Gmail App Password (Google Account -> Security ->
2-Step Verification -> App passwords), then either export the variables or put
them in medcare/email_settings.env (never commit that file):

    SMTP_USERNAME=gaurav.levon@gmail.com
    SMTP_PASSWORD=xxxx xxxx xxxx xxxx          (the 16-character app password)
    ALERT_EMAIL_TO=gaurav.levon@gmail.com       (default if not set)

Usage (from the medcare/ folder):
    python red_alert_mailer.py --test          send one test email to check the settings
    python red_alert_mailer.py                 email the red alerts in outputs/alerts.csv now
    python red_alert_mailer.py --run           run `main.py decide-all`, then email new red alerts
    python red_alert_mailer.py --watch         keep running; email new red alerts the moment
                                               outputs/alerts.csv changes (Ctrl+C to stop)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DEFAULT_RECIPIENT = "gaurav.levon@gmail.com"
SETTINGS_FILE = ROOT / "email_settings.env"
RED_TYPES = ("STOCKOUT_RISK_HIGH", "EXPIRY_RISK_HIGH")


def load_settings_file() -> None:
    """Read KEY=VALUE lines from email_settings.env into the environment (env vars win)."""
    if not SETTINGS_FILE.exists():
        return
    for line in SETTINGS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def prepare_env() -> None:
    load_settings_file()
    os.environ.setdefault("SMTP_HOST", "smtp.gmail.com")
    os.environ.setdefault("SMTP_PORT", "587")
    os.environ.setdefault("ALERT_EMAIL_TO", DEFAULT_RECIPIENT)
    os.environ["MEDCARE_NOTIFICATION_MODE"] = "email"


def engine():
    from notification_engine import NotificationEngine
    from notification_engine.settings import NotificationConfigError
    try:
        return NotificationEngine(mode="email")
    except NotificationConfigError as exc:
        print(f"\nEmail is not set up yet: {exc}\n"
              f"Put your Gmail app password in {SETTINGS_FILE.name} (see the top of this file), then run again.\n")
        sys.exit(2)


def red_alerts() -> list[dict]:
    import config
    import pandas as pd
    path = config.OUTPUT_DIR / "alerts.csv"
    if not path.exists() or not path.stat().st_size:
        print("outputs/alerts.csv not found. Run: python main.py decide-all")
        return []
    df = pd.read_csv(path, dtype={"sku_id": str, "warehouse_id": str})
    df = df[(df["severity"] == "HIGH") & (df["alert_type"].isin(RED_TYPES))]
    return df.to_dict("records")


def send_red_alerts() -> None:
    from notification_engine import export_notification_log
    alerts = red_alerts()
    if not alerts:
        print("No red alerts to send.")
        return
    res = engine().notify(alerts)
    export_notification_log()
    stamp = datetime.now().strftime("%H:%M:%S")
    if res.status == "SENT":
        print(f"[{stamp}] Emailed {res.notified} new red alert(s) to {os.environ['ALERT_EMAIL_TO']} "
              f"({res.skipped_duplicates} already sent before).")
    elif res.status == "NOTHING_TO_SEND":
        print(f"[{stamp}] No new red alerts ({res.skipped_duplicates} were already emailed).")
    else:
        print(f"[{stamp}] Email {res.status}: {res.error}")


def send_test() -> None:
    e = engine()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    e._send("[MedCare] Test email - red alert notifications are set up",
            f"This is a test from MedCare red_alert_mailer.py ({now} UTC).\n"
            f"Red alerts (HIGH stock-out and HIGH expiry risk) will be sent to {', '.join(e.settings.recipients)}.")
    print(f"Test email sent to {', '.join(e.settings.recipients)}. Check the inbox (and the spam folder).")


def run_decide_all() -> bool:
    print("Running python main.py decide-all ...")
    proc = subprocess.run([sys.executable, "main.py", "decide-all", "--top", "5"], cwd=ROOT)
    return proc.returncode == 0


def watch(interval: float) -> None:
    import config
    path = config.OUTPUT_DIR / "alerts.csv"
    engine()  # fail fast if email is not configured
    print(f"Watching {path} - new red alerts are emailed as soon as it changes. Press Ctrl+C to stop.")
    last = None
    while True:
        try:
            m = path.stat().st_mtime if path.exists() else None
            if m is not None and m != last:
                time.sleep(1)  # let decide-all finish writing the file
                send_red_alerts()
                last = path.stat().st_mtime
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopped.")
            return


def main() -> int:
    p = argparse.ArgumentParser(description="Email MedCare red (HIGH) alerts")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--test", action="store_true", help="send one test email")
    g.add_argument("--run", action="store_true", help="run decide-all, then email new red alerts")
    g.add_argument("--watch", action="store_true", help="email new red alerts whenever alerts.csv changes")
    p.add_argument("--interval", type=float, default=10.0, help="seconds between checks in --watch mode")
    a = p.parse_args()
    prepare_env()
    if a.test:
        send_test()
    elif a.watch:
        watch(a.interval)
    elif a.run:
        if not run_decide_all():
            print("decide-all failed; nothing emailed.")
            return 1
        send_red_alerts()
    else:
        send_red_alerts()
    return 0


if __name__ == "__main__":
    sys.exit(main())
