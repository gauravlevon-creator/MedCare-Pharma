"""
Notification engine: Alert Engine -> Notification Engine -> Email / Outbox.

* Consumes alerts from decision_engine.alert_engine (Alert objects or their
  dicts) and E1 threshold events from the E1 simulation (dicts). Only types
  in config.NOTIFY_ALERT_TYPES are notified: HIGH stock-out risk, E1
  threshold breach, HIGH expiry risk and reorder recommendation.
* Each notify() call sends ONE digest email (not one email per alert) and
  writes one log row per alert.
* Status is always what actually happened:
      SENT    the SMTP server accepted the digest message
      OUTBOX  recorded only; nothing was sent (outbox mode, or auto mode
              without complete SMTP settings)
      FAILED  sending was attempted and failed (error stored, never the password)
* De-duplication: an alert with the same (sku, warehouse, alert_type,
  alert_date) that is already logged as SENT or OUTBOX is not logged again.
* SMS / WhatsApp are not implemented. They would be further Channel
  implementations; no paid API is required or faked.
"""

from __future__ import annotations

import smtplib
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from email.utils import make_msgid
from typing import Callable, Iterable, Mapping

import pandas as pd

import config
from database import database as db
from notification_engine.settings import (MODE_AUTO, MODE_EMAIL, MODE_OUTBOX, EmailSettings,
                                          NotificationConfigError, load_email_settings, load_mode)

STATUS_SENT = "SENT"
STATUS_OUTBOX = "OUTBOX"
STATUS_FAILED = "FAILED"
CHANNEL_EMAIL = "EMAIL"
CHANNEL_OUTBOX = "OUTBOX"


@dataclass
class NotifyResult:
    mode: str
    channel: str
    status: str
    notified: int
    skipped_duplicates: int
    skipped_types: int
    delivery_ref: str | None
    error: str | None
    rows: list[dict]


def _as_dict(alert) -> dict:
    d = alert.to_dict() if hasattr(alert, "to_dict") else dict(alert)
    d.setdefault("alert_date", config.SIMULATION_DATE)
    d.setdefault("recommended_action", "")
    d.setdefault("severity", "")
    return d


def build_digest(alerts: list[dict], generated_at: str) -> tuple[str, str]:
    """Subject and plain-text body of the digest email."""
    counts = pd.Series([a["alert_type"] for a in alerts]).value_counts().to_dict() if alerts else {}
    subject = (f"[MedCare] {len(alerts)} inventory alert(s) as of {config.SIMULATION_DATE} (simulation)")
    lines = [
        "MedCare Pharma inventory alerts (hackathon simulation)",
        f"Simulation date: {config.SIMULATION_DATE}   Generated (UTC): {generated_at}",
        "Counts: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
        "",
    ]
    for a in alerts:
        lines.append(f"[{a['severity']}] {a['alert_type']}  {a['sku_id']} / {a['warehouse_id']}  "
                     f"(date {a['alert_date']}) -> {a['recommended_action']}")
        lines.append(f"    {a['message']}")
    lines += ["", "This message was generated automatically from simulated data."]
    return subject, "\n".join(lines)


class NotificationEngine:
    def __init__(self, mode: str | None = None, env: Mapping[str, str] | None = None,
                 smtp_factory: Callable | None = None, db_path=None, timeout: float = 15.0):
        self.mode = mode or load_mode(env)
        if self.mode not in (MODE_OUTBOX, MODE_EMAIL, MODE_AUTO):
            raise NotificationConfigError(f"Unknown notification mode {self.mode!r}")
        self.settings: EmailSettings = load_email_settings(env)
        self.smtp_factory = smtp_factory or (smtplib.SMTP_SSL if self.settings.port == 465 else smtplib.SMTP)
        self.db_path = db_path
        self.timeout = timeout
        if self.mode == MODE_EMAIL:
            problems = self.settings.validate()
            if problems:
                raise NotificationConfigError("Email mode requested but SMTP settings are invalid: "
                                              + "; ".join(problems))

    @property
    def will_send_email(self) -> bool:
        return self.mode == MODE_EMAIL or (self.mode == MODE_AUTO and self.settings.is_complete)

    def _already_logged(self) -> set[tuple]:
        try:
            log = db.get_notifications(db_path=self.db_path)
        except db.DatabaseError:
            return set()
        log = log[log.status.isin([STATUS_SENT, STATUS_OUTBOX])]
        return set(zip(log.sku_id, log.warehouse_id, log.alert_type, log.alert_date))

    def _send(self, subject: str, body: str) -> str:
        s = self.settings
        msg = EmailMessage()
        msg["Subject"], msg["From"], msg["To"] = subject, s.sender, ", ".join(s.recipients)
        msg["Message-ID"] = make_msgid(domain="medcare.local")
        msg.set_content(body)
        with self.smtp_factory(s.host, s.port, timeout=self.timeout) as smtp:
            if s.use_tls and self.smtp_factory is not smtplib.SMTP_SSL:
                smtp.starttls()
            smtp.login(s.username, s.password)
            smtp.send_message(msg)
        return msg["Message-ID"]

    def notify(self, alerts: Iterable, persist: bool = True) -> NotifyResult:
        """Filter, de-duplicate, deliver (or record) and log the alerts."""
        all_alerts = [_as_dict(a) for a in alerts]
        wanted = [a for a in all_alerts if a["alert_type"] in config.NOTIFY_ALERT_TYPES]
        seen = self._already_logged()
        fresh, dup = [], 0
        for a in wanted:
            key = (a["sku_id"], a["warehouse_id"], a["alert_type"], str(a["alert_date"]))
            if key in seen:
                dup += 1
                continue
            seen.add(key)
            fresh.append(a)

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        channel = CHANNEL_EMAIL if self.will_send_email else CHANNEL_OUTBOX
        status, ref, error = STATUS_OUTBOX, None, None
        if fresh and self.will_send_email:
            subject, body = build_digest(fresh, now)
            try:
                ref, status = self._send(subject, body), STATUS_SENT
            except Exception as exc:  # network/auth errors: record honestly, never the password
                status, error = STATUS_FAILED, f"{type(exc).__name__}: {str(exc)[:300]}"
        elif fresh and self.mode == MODE_AUTO:
            error = "SMTP not configured; recorded in outbox only (not sent)"

        recipient = ", ".join(self.settings.recipients) or None
        rows = []
        for a in fresh:
            rows.append({
                "notification_id": uuid.uuid4().hex, "created_at_utc": now,
                "alert_date": str(a["alert_date"]), "sku_id": a["sku_id"], "warehouse_id": a["warehouse_id"],
                "alert_type": a["alert_type"], "severity": a["severity"],
                "recommended_action": a["recommended_action"], "channel": channel,
                "recipient": recipient,   # intended recipients; in OUTBOX mode nothing was sent
                "subject": f"{a['alert_type']}: {a['sku_id']} / {a['warehouse_id']}",
                "message": a["message"], "status": status, "delivery_ref": ref, "error": error,
            })
        if persist and rows:
            db.save_notifications(rows, db_path=self.db_path)
        return NotifyResult(mode=self.mode, channel=channel, status=status if fresh else "NOTHING_TO_SEND",
                            notified=len(rows), skipped_duplicates=dup,
                            skipped_types=len(all_alerts) - len(wanted), delivery_ref=ref,
                            error=error, rows=rows)


def export_notification_log(path=None, db_path=None) -> pd.DataFrame:
    """Write the full notification log (from SQLite) to outputs/notification_log.csv."""
    log = db.get_notifications(db_path=db_path)
    out = path or config.NOTIFICATION_LOG_CSV
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log.to_csv(out, index=False)
    return log
