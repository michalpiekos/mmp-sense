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
    #: A file holding the source token for that server (`sample source create
    #: --token-file`); when set, the token is read from it and not kept elsewhere.
    token_file: str = ""
    #: A gateway relay to watch live streams on, HOST:PORT; by default the server's host, port 8091.
    relay: str = ""
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
        self.token_file = self.token_file.strip()
        self.relay = self.relay.strip().rstrip("/")
        self.live_history_s = min(600.0, max(1.0, float(self.live_history_s)))
        path = self.path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2) + "\n")
        os.replace(temporary, path)

    def read_token(self) -> str:
        """The token in `token_file`; OSError or ValueError when it cannot be had."""
        path = Path(self.token_file).expanduser()
        token = path.read_text().strip()
        if not token:
            raise ValueError(f"{path} is empty")
        return token

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

    def token(self, server: str | None = None) -> str | None:
        """The source token: from the token file, else the one remembered for the server."""
        if self.token_file:
            try:
                return self.read_token()
            except (OSError, ValueError):
                return None
        from .workspace.sync import load_token

        return load_token(server or self.server) if (server or self.server) else None
