"""Sending a workspace's recordings to srv-senselab (its HTTP API /v1).

Each recording is one session. A session folder (a received stream, an export) is
sent as the session it is, with its own session.json; any other recording becomes a
session holding its one file. Its title, tags and notes go in the manifest, and a
changed context is sent as the next revision. Every step is idempotent on the
server, so a Synchronize that stopped part way is simply run again: an upload
resumes from the offset the server has. No credentials: run the server on a LAN or
behind WireGuard.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path

from .model import Recording, Workspace, WorkspaceError

__all__ = ["synchronize", "SyncResult", "ApiError", "Client", "manifest_of", "PART_BYTES"]

API = "/v1"
PART_BYTES = 8 << 20
#: Recording ids that are not UUIDs get a stable one in this namespace.
NAMESPACE = uuid.UUID("6f1c4c1e-9a52-4a8e-9a39-3d0f3b8a0c57")
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_SAFE = re.compile(r"[^A-Za-z0-9_.+-]")


class ApiError(Exception):
    def __init__(self, status: int, body) -> None:
        self.status = status
        body = body if isinstance(body, dict) else {}
        self.code = body.get("error")
        self.body = body
        super().__init__(f"HTTP {status} {self.code}: {body.get('message', '')}")


class Client:
    """The few calls a workspace needs, retried when the server cannot be reached."""

    def __init__(self, server: str, *, attempts: int = 4, timeout: float = 60.0, pause=time.sleep) -> None:
        self.server = server.rstrip("/")
        self.base = self.server + API
        self.attempts = attempts
        self.timeout = timeout
        self._pause = pause

    def call(self, method: str, path: str, body=None, raw: bytes | None = None, headers: dict | None = None):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        headers = dict(headers or {})
        if data is not None:
            headers.setdefault("Content-Type", "application/octet-stream" if raw is not None else "application/json")
        for attempt in range(self.attempts):
            request = urllib.request.Request(self.base + path, data=data, method=method, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read() or b"null")
            except urllib.error.HTTPError as exc:
                try:
                    parsed = json.loads(exc.read() or b"{}")
                except ValueError:
                    parsed = {}
                raise ApiError(exc.code, parsed) from None
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                if attempt == self.attempts - 1:
                    raise
                self._pause(2 ** attempt)


@dataclass
class SyncResult:
    recording: Recording
    outcome: str        # sent, context, up to date, skipped, failed
    message: str = ""
    #: The HTTP status of a failure the server answered.
    status: int | None = None


def synchronize(workspace: Workspace, client: Client, progress=None) -> list[SyncResult]:
    """Send every saved recording that is not on the server yet, and every changed
    context; writes the workspace after each recording so what was sent is
    remembered. `progress(recording, text)` hears each step."""
    if workspace.folder is None:
        raise WorkspaceError("the workspace is untitled; create it in a folder first")
    tell = progress or (lambda _r, _t: None)
    results = []
    unreachable = None          # why the rest is not tried
    for recording in workspace.recordings:
        if unreachable is not None:
            results.append(SyncResult(recording, "failed", f"not tried: {unreachable}"))
            continue
        if not recording.saved:
            results.append(SyncResult(recording, "skipped", "not saved"))
            continue
        try:
            results.append(SyncResult(recording, _send(workspace, recording, client, tell)))
        except urllib.error.URLError as exc:
            unreachable = f"{client.server} cannot be reached ({exc.reason})"
            results.append(SyncResult(recording, "failed", unreachable))
        except ApiError as exc:
            results.append(SyncResult(recording, "failed", str(exc), exc.status))
        except (OSError, ValueError) as exc:
            results.append(SyncResult(recording, "failed", str(exc)))
        finally:
            workspace.save()
    return results


def _send(workspace: Workspace, recording: Recording, client: Client, tell) -> str:
    sync = recording.sync
    context = recording.context()
    if sync.context_sent == context:
        return "up to date"
    new = sync.context_sent is None
    if context != sync.context_tried:
        if sync.context_tried is not None:
            sync.revision += 1
        sync.context_tried = context
    manifest, folder = manifest_of(workspace, recording)
    tell(recording, "sending the session")
    sent = {**manifest, "files": [{k: v for k, v in f.items() if k != "source"} for f in manifest["files"]]}
    answer = client.call("PUT", f"/sessions/{manifest['id']}", sent)
    by_sha = {f["sha256"]: f for f in manifest["files"]}
    for sha in answer.get("missing", []):
        entry = by_sha.get(sha)
        if entry is None:
            continue                                # a file an earlier revision listed; not this one's to send
        _upload(client, folder / entry["source"], sha, entry["size"], recording, tell)
    missing = client.call("GET", f"/sessions/{manifest['id']}").get("missing", [])
    if any(sha in by_sha for sha in missing):
        raise ValueError(f"the server still lacks {len(missing)} file(s)")
    sync.context_sent = context
    return "sent" if new else "context"


def _upload(client: Client, path: Path, sha: str, size: int, recording: Recording, tell) -> None:
    state = client.call("GET", f"/uploads/{sha}")
    offset = state.get("offset", 0)
    with open(path, "rb") as file:
        while not state.get("complete"):
            tell(recording, f"uploading {path.name}: {offset * 100 // max(size, 1)}%")
            file.seek(offset)
            chunk = file.read(PART_BYTES)
            span = f"bytes {offset}-{offset + len(chunk) - 1}/{size}" if chunk else f"bytes */{size}"
            try:
                state = client.call("PUT", f"/uploads/{sha}", raw=chunk, headers={"Content-Range": span})
            except ApiError as exc:
                if exc.status != 409 or "offset" not in exc.body:
                    raise
                state = exc.body
            offset = state.get("offset", 0)


def session_id_of(recording: Recording) -> str:
    """The session a recording is on the server: its own once set, else the recording's id
    when that is a UUID, else a stable one made from it."""
    if not recording.sync.session_id:
        rid = recording.id.lower()
        recording.sync.session_id = rid if _UUID.fullmatch(rid) else str(uuid.uuid5(NAMESPACE, rid))
    return recording.sync.session_id


def manifest_of(workspace: Workspace, recording: Recording) -> tuple[dict, Path]:
    """The session.json to send for a recording, and the folder its files are read from.
    Each file entry carries `source`, its name in that folder."""
    from .. import __version__

    path = workspace.path_of(recording)
    sid = session_id_of(recording)
    revised = int(time.time() * 1000)
    if recording.origin == "session":
        from ..io.session_folder import SESSION_FILE

        m = json.loads((path / SESSION_FILE).read_text())
        base = int(m.get("revision", 0))
        if recording.sync.revision < base:
            recording.sync.revision = base
        changed = (m.get("title", ""), sorted(set(m.get("tags") or [])), m.get("notes", "")) != (
            recording.title, sorted(set(recording.tags)), recording.notes)
        if changed and recording.sync.revision == base:
            recording.sync.revision = base + 1
        m.update(id=sid, title=recording.title, tags=sorted(set(recording.tags)), notes=recording.notes,
                 revision=recording.sync.revision,
                 revised_utc_ms=revised if recording.sync.revision != base else m.get("revised_utc_ms", 0))
        m["files"] = [{**f, "source": f["name"]} for f in m.get("files") or []]
        return m, path
    name = _SAFE.sub("_", recording.name).lstrip(".") or "recording"
    entry = {"name": name, "source": path.name, "sha256": _sha256(path), "size": path.stat().st_size,
             "media_type": _media_type(recording, path)}
    if name != recording.name:
        entry["original_name"] = recording.name
    m = {
        "format": "mmp-session", "version": 1, "id": sid, "origin": "mmp-sense",
        "title": recording.title, "tags": sorted(set(recording.tags)), "notes": recording.notes,
        "revision": recording.sync.revision, "revised_utc_ms": revised,
        "started_utc_ns": _utc_ns(recording.started_at_utc),
        "duration_s": recording.duration_s,
        "device": {"software": "mmp-sense", "version": __version__},
        "workspace": {"id": workspace.id, "recording_id": recording.id},
        "read_options": recording.options,
        "files": [entry],
    }
    if recording.meta:
        m["meta"] = recording.meta
    return m, path.parent


_TYPES = {".wav": "audio/wav", ".mcap": "application/x-mcap", ".m4a": "audio/mp4", ".aac": "audio/aac",
          ".3gp": "audio/3gpp", ".amr": "audio/amr", ".mp3": "audio/mpeg", ".ogg": "audio/ogg",
          ".opus": "audio/ogg", ".flac": "audio/flac", ".webm": "audio/webm", ".csv": "text/csv",
          ".tsv": "text/tab-separated-values", ".npy": "application/x-npy", ".npz": "application/x-npz"}


def _media_type(recording: Recording, path: Path) -> str:
    return _TYPES.get(path.suffix.lower(), "text/plain" if recording.kind == "text" else "application/octet-stream")


def _utc_ns(text: str) -> int | None:
    if not text:
        return None
    try:
        when = datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=datetime.timezone.utc)
    return int(when.timestamp()) * 1_000_000_000 + when.microsecond * 1000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
