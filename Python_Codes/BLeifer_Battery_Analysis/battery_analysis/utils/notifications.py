from __future__ import annotations

"""Utility helpers for sending completion notifications."""

import json
import smtplib
from email.message import EmailMessage
from urllib import request

from .config import load_config

__all__ = ["send"]

CONFIG = load_config()


def _post_json(url: str, payload: dict) -> None:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        request.urlopen(req, timeout=5)
    except Exception:
        pass


def send(message: str, *, subject: str = "Import job notification") -> None:
    """Send ``message`` via configured Slack webhook and/or email."""

    webhook = CONFIG.get("slack_webhook")
    if webhook:
        _post_json(webhook, {"text": message})

    recipients = CONFIG.get("email_recipients") or []
    smtp_host = CONFIG.get("smtp_host")
    sender = CONFIG.get("email_sender")
    if recipients and smtp_host and sender:
        email = EmailMessage()
        email["Subject"] = subject
        email["From"] = sender
        email["To"] = ", ".join(recipients)
        email.set_content(message)
        try:
            with smtplib.SMTP(smtp_host) as srv:
                srv.send_message(email)
        except Exception:
            pass
