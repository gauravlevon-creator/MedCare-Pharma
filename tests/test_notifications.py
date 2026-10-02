"""Notification engine: filtering, outbox mode, email config validation, SMTP send/failure, log, de-duplication."""

import unittest

import config
from database import database as db
from decision_engine import alert_engine as al
from notification_engine import NotificationConfigError, NotificationEngine, export_notification_log
from notification_engine.engine import STATUS_FAILED, STATUS_OUTBOX, STATUS_SENT
from notification_engine.settings import load_email_settings, load_mode
from tests.helpers import make_temp_db

GOOD_ENV = {"SMTP_HOST": "smtp.example.com", "SMTP_PORT": "587", "SMTP_USERNAME": "alerts@example.com",
            "SMTP_PASSWORD": "s3cret-app-password", "ALERT_EMAIL_TO": "planner@example.com, lead@example.com"}


def alert(t, sku="M001", wh="W002", date=None):
    return {"sku_id": sku, "warehouse_id": wh, "alert_type": t, "severity": al.SEVERITY.get(t, "INFO"),
            "message": f"{t} message", "recommended_action": "REORDER",
            "alert_date": date or config.SIMULATION_DATE}


class FakeSMTP:
    sent = []

    def __init__(self, host, port, timeout=None, fail=False):
        self.host, self.port, self.fail = host, port, fail

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def starttls(self):
        pass

    def login(self, user, password):
        if self.fail:
            raise ConnectionRefusedError("server unavailable")

    def send_message(self, msg):
        FakeSMTP.sent.append(msg)


class SettingsTests(unittest.TestCase):

    def test_valid_settings(self):
        s = load_email_settings(GOOD_ENV)
        self.assertEqual(s.validate(), [])
        self.assertEqual(s.recipients, ["planner@example.com", "lead@example.com"])
        self.assertEqual(s.sender, "alerts@example.com")
        self.assertNotIn("s3cret", repr(s))

    def test_invalid_settings_reported(self):
        bad = dict(GOOD_ENV, SMTP_PORT="abc", ALERT_EMAIL_TO="not-an-email")
        del bad["SMTP_PASSWORD"]
        problems = " | ".join(load_email_settings(bad).validate())
        for text in ("SMTP_PORT", "SMTP_PASSWORD", "invalid addresses"):
            self.assertIn(text, problems)
        self.assertTrue(load_email_settings({}).validate())

    def test_mode(self):
        self.assertEqual(load_mode({}), "outbox")
        self.assertEqual(load_mode({config.NOTIFICATION_MODE_ENV: "AUTO"}), "auto")
        with self.assertRaises(NotificationConfigError):
            load_mode({config.NOTIFICATION_MODE_ENV: "sms"})

    def test_email_mode_refuses_invalid_config(self):
        with self.assertRaises(NotificationConfigError):
            NotificationEngine(mode="email", env={})


class EngineTests(unittest.TestCase):

    def setUp(self):
        self.db_path = make_temp_db()
        FakeSMTP.sent = []

    def test_only_configured_types_notified(self):
        eng = NotificationEngine(mode="outbox", env={}, db_path=self.db_path)
        res = eng.notify([alert("STOCKOUT_RISK_HIGH"), alert("E1_THRESHOLD"), alert("EXPIRY_RISK_HIGH", wh="W004"),
                          alert("REORDER_RECOMMENDED"), alert("STOCKOUT_RISK_MEDIUM"), alert("TRANSFER_RECOMMENDED")])
        self.assertEqual(res.notified, 4)
        self.assertEqual(res.skipped_types, 2)
        self.assertEqual({r["alert_type"] for r in res.rows}, set(config.NOTIFY_ALERT_TYPES))

    def test_outbox_mode_records_but_never_claims_sent(self):
        eng = NotificationEngine(mode="outbox", env=GOOD_ENV, smtp_factory=FakeSMTP, db_path=self.db_path)
        res = eng.notify([alert("STOCKOUT_RISK_HIGH")])
        self.assertEqual((res.status, res.channel), (STATUS_OUTBOX, "OUTBOX"))
        self.assertEqual(FakeSMTP.sent, [])
        log = db.get_notifications(db_path=self.db_path)
        self.assertEqual(log.status.tolist(), [STATUS_OUTBOX])
        self.assertFalse((log.status == STATUS_SENT).any())

    def test_auto_mode_without_credentials_falls_back_to_outbox(self):
        eng = NotificationEngine(mode="auto", env={}, smtp_factory=FakeSMTP, db_path=self.db_path)
        res = eng.notify([alert("E1_THRESHOLD")])
        self.assertEqual(res.status, STATUS_OUTBOX)
        self.assertIn("not sent", res.error)
        self.assertEqual(FakeSMTP.sent, [])

    def test_email_mode_sends_one_digest(self):
        eng = NotificationEngine(mode="email", env=GOOD_ENV, smtp_factory=FakeSMTP, db_path=self.db_path)
        res = eng.notify([alert("STOCKOUT_RISK_HIGH"), alert("E1_THRESHOLD"), alert("REORDER_RECOMMENDED")])
        self.assertEqual((res.status, res.channel, res.notified), (STATUS_SENT, "EMAIL", 3))
        self.assertEqual(len(FakeSMTP.sent), 1)
        msg = FakeSMTP.sent[0]
        self.assertEqual(msg["To"], "planner@example.com, lead@example.com")
        self.assertIn("STOCKOUT_RISK_HIGH", msg.get_content())
        log = db.get_notifications(db_path=self.db_path)
        self.assertTrue((log.status == STATUS_SENT).all())
        self.assertTrue((log.delivery_ref == msg["Message-ID"]).all())

    def test_smtp_failure_logged_without_password(self):
        failing = lambda h, p, timeout=None: FakeSMTP(h, p, timeout, fail=True)  # noqa: E731
        eng = NotificationEngine(mode="email", env=GOOD_ENV, smtp_factory=failing, db_path=self.db_path)
        res = eng.notify([alert("STOCKOUT_RISK_HIGH")])
        self.assertEqual(res.status, STATUS_FAILED)
        log = db.get_notifications(db_path=self.db_path)
        self.assertEqual(log.status.tolist(), [STATUS_FAILED])
        self.assertFalse(log.astype(str).apply(lambda c: c.str.contains("s3cret")).any().any())

    def test_duplicates_not_logged_twice_but_new_dates_are(self):
        eng = NotificationEngine(mode="outbox", env={}, db_path=self.db_path)
        eng.notify([alert("E1_THRESHOLD")])
        again = eng.notify([alert("E1_THRESHOLD"), alert("E1_THRESHOLD", date="2026-10-05")])
        self.assertEqual((again.notified, again.skipped_duplicates), (1, 1))
        self.assertEqual(len(db.get_notifications(db_path=self.db_path)), 2)

    def test_accepts_alert_engine_objects_and_exports_log(self):
        a = al.Alert("M001", "W002", "HIGH", al.STOCKOUT_RISK_HIGH, "msg", 10.0, "u", "REORDER", 1)
        eng = NotificationEngine(mode="outbox", env={}, db_path=self.db_path)
        eng.notify([a])
        tmp = self.db_path.parent / "notification_log.csv"
        log = export_notification_log(path=tmp, db_path=self.db_path)
        self.assertTrue(tmp.exists())
        self.assertEqual(log.alert_date.tolist(), [config.SIMULATION_DATE])


if __name__ == "__main__":
    unittest.main()
