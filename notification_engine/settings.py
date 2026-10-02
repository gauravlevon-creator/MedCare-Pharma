"""
Notification settings, read ONLY from environment variables.

Nothing here is hard-coded, and the password is never logged, printed or
stored in the notification log.

    MEDCARE_NOTIFICATION_MODE  outbox | email | auto        (default: outbox)
    SMTP_HOST                  e.g. smtp.gmail.com
    SMTP_PORT                  e.g. 587
    SMTP_USERNAME
    SMTP_PASSWORD              an app password, never an account password
    ALERT_EMAIL_TO             comma-separated recipients
    ALERT_EMAIL_FROM           optional, defaults to SMTP_USERNAME
    SMTP_USE_TLS               true | false                  (default: true, STARTTLS)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Mapping

import config

MODE_OUTBOX = "outbox"
MODE_EMAIL = "email"
MODE_AUTO = "auto"
MODES = (MODE_OUTBOX, MODE_EMAIL, MODE_AUTO)

_EMAIL_RE = re.compile(r"^[^@\s,;]+@[^@\s,;]+\.[^@\s,;]+$")


class NotificationConfigError(Exception):
    """Email mode was requested but the SMTP settings are missing or invalid."""


@dataclass
class EmailSettings:
    host: str = ""
    port: int | None = None
    username: str = ""
    password: str = field(default="", repr=False)   # excluded from repr/logging
    sender: str = ""
    recipients: list[str] = field(default_factory=list)
    use_tls: bool = True
    port_raw: str = ""

    def validate(self) -> list[str]:
        """Human-readable problems; an empty list means the settings are usable."""
        problems = []
        if not self.host:
            problems.append("SMTP_HOST is not set")
        if self.port is None:
            problems.append(f"SMTP_PORT is not a valid port ({self.port_raw!r})" if self.port_raw
                            else "SMTP_PORT is not set")
        elif not 1 <= self.port <= 65535:
            problems.append(f"SMTP_PORT {self.port} is out of range")
        if not self.username:
            problems.append("SMTP_USERNAME is not set")
        if not self.password:
            problems.append("SMTP_PASSWORD is not set")
        if not self.recipients:
            problems.append("ALERT_EMAIL_TO is not set")
        bad = [r for r in self.recipients if not _EMAIL_RE.match(r)]
        if bad:
            problems.append(f"ALERT_EMAIL_TO has invalid addresses: {', '.join(bad)}")
        if self.sender and not _EMAIL_RE.match(self.sender):
            problems.append(f"ALERT_EMAIL_FROM / SMTP_USERNAME is not an email address: {self.sender}")
        return problems

    @property
    def is_complete(self) -> bool:
        return not self.validate()


def load_email_settings(env: Mapping[str, str] | None = None) -> EmailSettings:
    env = os.environ if env is None else env
    port_raw = (env.get("SMTP_PORT") or "").strip()
    try:
        port = int(port_raw) if port_raw else None
    except ValueError:
        port = None
    username = (env.get("SMTP_USERNAME") or "").strip()
    return EmailSettings(
        host=(env.get("SMTP_HOST") or "").strip(),
        port=port, port_raw=port_raw, username=username,
        password=env.get("SMTP_PASSWORD") or "",
        sender=(env.get("ALERT_EMAIL_FROM") or username).strip(),
        recipients=[r.strip() for r in (env.get("ALERT_EMAIL_TO") or "").split(",") if r.strip()],
        use_tls=(env.get("SMTP_USE_TLS", "true").strip().lower() not in ("0", "false", "no")),
    )


def load_mode(env: Mapping[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    mode = (env.get(config.NOTIFICATION_MODE_ENV) or config.DEFAULT_NOTIFICATION_MODE).strip().lower()
    if mode not in MODES:
        raise NotificationConfigError(f"{config.NOTIFICATION_MODE_ENV} must be one of {MODES}, got {mode!r}")
    return mode
