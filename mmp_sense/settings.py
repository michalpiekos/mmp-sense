"""The user's settings, kept in ~/.config/mmp-sense/settings.json."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

__all__ = ["Settings", "config_dir"]


def config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base) / "mmp-sense"


@dataclass
class Settings:
    #: The srv-senselab a workspace is synchronized with, e.g. http://host:8080.
    server: str = ""
    #: A relay to watch live streams on, HOST:PORT; by default the server's host, port 8091.
    relay: str = ""
    #: Where File > Receive streams takes node streams, and where it relays them.
    receive_bind: str = "0.0.0.0:8090"
    receive_relay: str = "127.0.0.1:8091"
    #: Seconds of each live stream kept.
    live_history_s: float = 120.0

    @classmethod
    def path(cls) -> Path:
        return config_dir() / "settings.json"

    @classmethod
    def load(cls) -> "Settings":
        try:
            data = json.loads(cls.path().read_text())
        except (OSError, ValueError):
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known}) if isinstance(data, dict) else cls()

    def save(self) -> None:
        self.server = self.server.strip().rstrip("/")
        self.receive_bind = self.receive_bind.strip() or "0.0.0.0:8090"
        self.receive_relay = self.receive_relay.strip() or "127.0.0.1:8091"
        self.relay = self.relay.strip().rstrip("/")
        self.live_history_s = min(600.0, max(1.0, float(self.live_history_s)))
        path = self.path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2) + "\n")
        os.replace(temporary, path)

    def relay_address(self) -> str:
        """The relay to watch: the one set, else the server's host on port 8091."""
        if self.relay:
            return self.relay
        if self.server:
            from urllib.parse import urlsplit

            host = urlsplit(self.server if "://" in self.server else "http://" + self.server).hostname
            if host:
                return f"{host}:8091"
        return "127.0.0.1:8091"
