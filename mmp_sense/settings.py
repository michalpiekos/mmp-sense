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
