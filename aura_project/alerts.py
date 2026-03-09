"""Alerting utilities: console, audio, and Twilio WhatsApp."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass

from playsound import playsound
from twilio.rest import Client


@dataclass
class TwilioConfig:
    account_sid: str
    auth_token: str
    from_whatsapp: str
    to_whatsapp: str


class AlertManager:
    """Dispatch critical alerts over multiple channels."""

    def __init__(
        self,
        default_alarm_file: str = "alert.wav",
        sound_map: dict[str, str] | None = None,
    ) -> None:
        self.default_alarm_file = default_alarm_file
        self.sound_map = sound_map or {}

    def _play_file_if_exists(self, sound_file: str) -> bool:
        if not sound_file or not os.path.exists(sound_file):
            return False
        threading.Thread(target=playsound, args=(sound_file,), kwargs={"block": True}, daemon=True).start()
        return True

    def play_alarm(self, alert_type: str = "default") -> None:
        """Play local alarm asynchronously; fallback to terminal bell for system sound."""
        sound_file = self.sound_map.get(alert_type, self.default_alarm_file)
        played = self._play_file_if_exists(sound_file)
        if not played:
            print("\a", end="", flush=True)

    def console_alert(self, message: str) -> None:
        """Emit alert to terminal output."""
        print(f"[AURA ALERT] {message}")

    def send_whatsapp(self, message: str, config: TwilioConfig) -> str:
        """Send WhatsApp message via Twilio."""
        client = Client(config.account_sid, config.auth_token)
        msg = client.messages.create(body=message, from_=config.from_whatsapp, to=config.to_whatsapp)
        return msg.sid

    @staticmethod
    def from_env() -> TwilioConfig | None:
        """Load Twilio config from env vars if provided."""
        sid = os.getenv("TWILIO_ACCOUNT_SID")
        token = os.getenv("TWILIO_AUTH_TOKEN")
        from_wa = os.getenv("TWILIO_WHATSAPP_FROM")
        to_wa = os.getenv("TWILIO_WHATSAPP_TO")
        if sid and token and from_wa and to_wa:
            return TwilioConfig(account_sid=sid, auth_token=token, from_whatsapp=from_wa, to_whatsapp=to_wa)
        return None
