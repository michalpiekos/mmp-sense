"""A workspace: recordings and what is known about them, kept in a folder.

    field-day/
        workspace.json            the workspace, its recordings and their metadata
        recordings/<id>/<name>    each recording's own bytes, as they were opened

A workspace starts untitled, in memory. A recording opened into it is read
from where it lies until it is saved, when its bytes are copied in unchanged.
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..io.files import kind_of
from ..io.session_export import is_session_folder

__all__ = ["Workspace", "Recording", "SyncState", "WorkspaceError", "is_workspace",
           "WORKSPACE_FILE", "FORMAT"]

WORKSPACE_FILE = "workspace.json"
FORMAT = "mmp-sense-workspace"
SCHEMA_VERSION = 1
RECORDINGS = "recordings"
#: Nanosecond times after 2001 are taken as UTC; earlier ones are a device's own clock.
_EPOCH_NS_FROM = 978_307_200 * 10**9


class WorkspaceError(ValueError):
    """A folder that cannot be a workspace, or a step that needs one on disk."""


def is_workspace(path) -> bool:
    return (Path(path) / WORKSPACE_FILE).is_file()


def _new_id() -> str:
    return str(uuid.uuid4())


def utc_text(when: datetime.datetime) -> str:
    return when.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@dataclass
class SyncState:
    """What srv-senselab has of a recording, and the IDs it is known by there."""

    capture_session_id: str = field(default_factory=_new_id)
    sample_id: str = field(default_factory=_new_id)
    declared: bool = False
    uploaded: bool = False
    closed: bool = False
    #: The last context revision sent, and what it said; `context_sent` once acknowledged.
    context_revision: int = 0
    context_tried: dict | None = None
    context_sent: dict | None = None


@dataclass
class Recording:
    """One opened file, or one srv-senselab export folder."""

    name: str
    kind: str
    id: str = field(default_factory=_new_id)
    title: str = ""
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    started_at_utc: str = ""
    duration_s: float | None = None
    #: How the file is read: read_file's topics, rate, scale and units.
    options: dict = field(default_factory=dict)
    #: "file", or "server" for an export: already on srv-senselab, so never sent.
    origin: str = "file"
    #: The capture session an export came from.
    server_session_id: str | None = None
    sync: SyncState = field(default_factory=SyncState)
    #: Where the bytes are read from until saved into a workspace.
    source: Path | None = field(default=None, compare=False)

    @property
    def saved(self) -> bool:
        return self.source is None

    @property
    def label(self) -> str:
        return self.title or self.name

    def context(self) -> dict:
        """The title, tags and notes as srv-senselab's session context holds them."""
        return {"title": self.title, "tags": sorted(set(self.tags)), "capture_notes": self.notes}

    def to_json(self) -> dict:
        data = asdict(self)
        data.pop("source")
        return data

    @classmethod
    def from_json(cls, data: dict) -> "Recording":
        data = dict(data)
        sync = SyncState(**data.pop("sync", {}))
        return cls(**data, sync=sync)


class Workspace:
    """Recordings and their metadata; on disk once it has a folder."""

    def __init__(self, *, title: str = "", notes: str = "", server: str = "",
                 folder: Path | None = None, id: str | None = None,
                 recordings: list[Recording] | None = None) -> None:
        self.id = id or _new_id()
        self.title = title
        self.notes = notes
        self.server = server
        self.folder = Path(folder) if folder is not None else None
        self.recordings: list[Recording] = list(recordings or [])

    # -- making and opening ---------------------------------------------

    @classmethod
    def untitled(cls) -> "Workspace":
        return cls()

    @classmethod
    def create(cls, folder, title: str = "") -> "Workspace":
        """A new, empty workspace in `folder`, which must not hold anything yet."""
        workspace = cls(title=title or Path(folder).name)
        workspace._claim(Path(folder))
        workspace.save()
        return workspace

    @classmethod
    def open(cls, folder) -> "Workspace":
        folder = Path(folder)
        path = folder / WORKSPACE_FILE
        try:
            data = json.loads(path.read_text())
        except FileNotFoundError:
            raise WorkspaceError(f"{folder}: not a workspace; it holds no {WORKSPACE_FILE}") from None
        except (OSError, ValueError) as exc:
            raise WorkspaceError(f"{path}: {exc}") from None
        if data.get("format") != FORMAT or data.get("schema_version") != SCHEMA_VERSION:
            raise WorkspaceError(f"{path}: not an mmp-sense workspace of schema {SCHEMA_VERSION}")
        recordings = [Recording.from_json(r) for r in data.get("recordings", [])]
        return cls(id=data["id"], title=data.get("title", ""), notes=data.get("notes", ""),
                   server=data.get("server", ""), folder=folder, recordings=recordings)

    def settle(self, folder, title: str = "") -> None:
        """Put this untitled workspace in `folder`, its recordings still unsaved."""
        if self.folder is not None:
            raise WorkspaceError(f"the workspace is in {self.folder} already")
        self._claim(Path(folder))
        self.title = title or self.folder.name
        self.save()

    def _claim(self, folder: Path) -> None:
        if folder.exists() and (not folder.is_dir() or any(folder.iterdir())):
            raise WorkspaceError(f"{folder}: not empty; a workspace needs a new or empty folder")
        folder.mkdir(parents=True, exist_ok=True)
        self.folder = folder

    # -- recordings -------------------------------------------------------

    def add(self, path, series=None, **options) -> Recording:
        """Take in the file or export folder at `path`. `series` is what it
        was read as, when the caller has read it already; `options` are how."""
        path = Path(path).resolve()
        if series is None:
            from ..io.files import read_file

            series = read_file(path, **options)
        kind = kind_of(path)
        recording = Recording(name=path.name, kind=kind, title=path.stem if path.is_file() else path.name,
                              options={k: v for k, v in options.items() if v not in (None, [], ())},
                              source=path)
        recording.started_at_utc = _started_at(path, series)
        recording.duration_s = _duration(series)
        if kind == "session":
            recording.origin = "server"
            recording.server_session_id = series[0].metadata.get("capture_session_id") if series else None
        self.recordings.append(recording)
        return recording

    def get(self, recording_id: str) -> Recording:
        for recording in self.recordings:
            if recording.id == recording_id:
                return recording
        raise KeyError(recording_id)

    def path_of(self, recording: Recording) -> Path:
        """Where the recording is read from: its copy here once saved."""
        if recording.source is not None:
            return recording.source
        return self._home(recording) / recording.name

    def _home(self, recording: Recording) -> Path:
        if self.folder is None:
            raise WorkspaceError("the workspace is untitled; create it in a folder first")
        return self.folder / RECORDINGS / recording.id

    # -- saving -------------------------------------------------------------

    def save_recording(self, recording: Recording) -> None:
        """Copy the recording's bytes in, unchanged, and write the workspace."""
        if recording.source is not None:
            self._copy_in(recording, recording.source)
            recording.source = None
        self.save()

    def save(self) -> None:
        """Write workspace.json: the workspace and its saved recordings."""
        if self.folder is None:
            raise WorkspaceError("the workspace is untitled; create it in a folder first")
        data = {
            "format": FORMAT,
            "schema_version": SCHEMA_VERSION,
            "id": self.id,
            "title": self.title,
            "notes": self.notes,
            "server": self.server,
            "recordings": [r.to_json() for r in self.recordings if r.saved],
        }
        path = self.folder / WORKSPACE_FILE
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(data, indent=2) + "\n")
        os.replace(temporary, path)

    def save_all(self) -> None:
        for recording in self.recordings:
            if recording.source is not None:
                self._copy_in(recording, recording.source)
                recording.source = None
        self.save()

    def save_as(self, folder) -> None:
        """Write all of it to `folder` and work there from now on, as a new
        workspace: new IDs throughout, and nothing of it sent to a server yet."""
        folder = Path(folder)
        if self.folder is not None and folder.resolve() == self.folder.resolve():
            raise WorkspaceError(f"{folder}: that is this workspace")
        sources = [self.path_of(r) for r in self.recordings]
        self._claim(folder)
        self.id = _new_id()
        for recording, source in zip(self.recordings, sources):
            recording.id = _new_id()
            recording.sync = SyncState()
            self._copy_in(recording, source)
            recording.source = None
        self.save()

    def _copy_in(self, recording: Recording, source: Path) -> None:
        home = self._home(recording)
        home.mkdir(parents=True, exist_ok=True)
        target = home / recording.name
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
            # An export folder is read with the manifest beside it.
            manifest = source.parent / "manifest.json"
            if is_session_folder(source) and manifest.is_file():
                shutil.copy2(manifest, home / "manifest.json")
        else:
            shutil.copy2(source, target)

    @property
    def unsaved(self) -> list[Recording]:
        return [r for r in self.recordings if not r.saved]


def _started_at(path: Path, series) -> str:
    for one in series:
        started = one.metadata.get("started_at_utc")
        if started:
            return str(started)
    for one in series:
        epoch = one.metadata.get("epoch_ns")
        if isinstance(epoch, int) and epoch >= _EPOCH_NS_FROM:
            return utc_text(datetime.datetime.fromtimestamp(epoch / 1e9, datetime.timezone.utc))
    return utc_text(datetime.datetime.fromtimestamp(path.stat().st_mtime, datetime.timezone.utc))


def _duration(series) -> float | None:
    spans = []
    for one in series:
        if not one.n_samples:
            continue
        rate = getattr(one, "sample_rate", None)
        start = float(one.time[0])
        end = float(one.time[-1]) + (1.0 / rate if rate else 0.0)
        spans.append((start, end))
    if not spans:
        return None
    return max(end for _, end in spans) - min(start for start, _ in spans)
