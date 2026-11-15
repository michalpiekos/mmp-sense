"""Sending a workspace's recordings to srv-senselab, over ingestion protocol 0.1.

Each recording is a capture session of its own holding one sample: the file
as it is. A recording's title, tags and notes are its session's context.
Every step is idempotent on the server, so a Synchronize that stopped part
way is simply run again: an upload resumes from the part the server has.

The token is a source token (`sample source create --type import`). It is
kept in the user's config, never in the workspace.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .model import Recording, Workspace, WorkspaceError

__all__ = ["synchronize", "SyncResult", "ApiError", "Client", "load_token", "store_token", "forget_token",
           "credentials_path", "declaration_of"]

PROTOCOL = "/v0.1"


class ApiError(Exception):
    def __init__(self, status: int, body) -> None:
        self.status = status
        error = body.get("error", {}) if isinstance(body, dict) else {}
        self.code = error.get("code")
        self.retryable = bool(error.get("retryable", False))
        self.details = error.get("details") or {}
        super().__init__(f"HTTP {status} {self.code}: {error.get('message', body)}")


class Client:
    """The few protocol calls a workspace needs, retried where the server says so."""

    def __init__(self, server: str, token: str, *, attempts: int = 4, timeout: float = 60.0,
                 pause=time.sleep) -> None:
        self.base = server.rstrip("/") + PROTOCOL
        self._token = token
        self.attempts = attempts
        self.timeout = timeout
        self._pause = pause

    def call(self, method: str, path: str, body=None, raw: bytes | None = None):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        headers = {"Authorization": f"Bearer {self._token}"}
        if data is not None:
            headers["Content-Type"] = "application/octet-stream" if raw is not None else "application/json"
        for attempt in range(self.attempts):
            request = urllib.request.Request(self.base + path, data=data, method=method, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read() or b"null")
            except urllib.error.HTTPError as exc:
                payload = exc.read()
                try:
                    parsed = json.loads(payload)
                except ValueError:
                    parsed = {"error": {"message": payload.decode(errors="replace")[:200]}}
                error = ApiError(exc.code, parsed)
                if not error.retryable or attempt == self.attempts - 1:
                    raise error from None
                wait = exc.headers.get("Retry-After")
                self._pause(float(wait) if wait else 2 ** attempt)
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
    """Send every saved recording that is not on the server yet, and every
    changed context; writes the workspace after each recording so what was
    sent is remembered. `progress(recording, text)` hears each step."""
    if workspace.folder is None:
        raise WorkspaceError("the workspace is untitled; create it in a folder first")
    tell = progress or (lambda _r, _t: None)
    results = []
    unreachable = None          # why the rest is not tried
    for recording in workspace.recordings:
        if unreachable is not None:
            results.append(SyncResult(recording, "failed", f"not tried: {unreachable}"))
            continue
        if recording.origin == "server":
            results.append(SyncResult(recording, "skipped", "already on the server"))
            continue
        if not recording.saved:
            results.append(SyncResult(recording, "skipped", "not saved"))
            continue
        try:
            outcome = _send(workspace, recording, client, tell)
            results.append(SyncResult(recording, outcome))
        except urllib.error.URLError as exc:
            unreachable = f"{client.base} cannot be reached ({exc.reason})"
            results.append(SyncResult(recording, "failed", unreachable))
        except ApiError as exc:
            results.append(SyncResult(recording, "failed", str(exc), exc.status))
            if exc.status == 401:
                unreachable = "the server does not accept the token"
        except (OSError, ValueError) as exc:
            results.append(SyncResult(recording, "failed", str(exc)))
        finally:
            workspace.save()
    return results


def _send(workspace: Workspace, recording: Recording, client: Client, tell) -> str:
    sync = recording.sync
    session = sync.capture_session_id
    path = workspace.path_of(recording)
    new = not sync.closed
    if not sync.declared:
        tell(recording, "declaring the session")
        client.call("PUT", f"/capture-sessions/{session}", _session(workspace, recording))
        sync.declared = True
    sent_context = _send_context(recording, client, tell)
    if not sync.uploaded:
        declaration = declaration_of(recording, path)
        tell(recording, "uploading")
        upload = client.call("POST", "/uploads", declaration)
        if upload["state"] != "received":
            upload = _send_parts(client, upload, path, recording, tell)
            upload = client.call("POST", f"/uploads/{upload['upload_id']}/complete")
        sync.uploaded = upload["state"] == "received"
        if not sync.uploaded:
            raise ValueError(f"the upload ended {upload['state']}")
    if not sync.closed:
        tell(recording, "closing the session")
        state = client.call("POST", f"/capture-sessions/{session}/close",
                            {"duration_ns": _duration_ns(recording), "sample_ids": [sync.sample_id]})
        if state.get("missing_samples"):
            raise ValueError(f"the server is missing {state['missing_samples']}")
        sync.closed = True
    if new:
        return "sent"
    return "context" if sent_context else "up to date"


def _send_context(recording: Recording, client: Client, tell) -> bool:
    sync = recording.sync
    context = recording.context()
    if context == sync.context_sent:
        return False
    if context != sync.context_tried:
        sync.context_revision += 1
        sync.context_tried = context
    tell(recording, "sending the title, tags and notes")
    client.call("PUT", f"/capture-sessions/{sync.capture_session_id}/context",
                {"revision": sync.context_revision, "tag_policy_version": 1, **context})
    sync.context_sent = context
    return True


def _send_parts(client: Client, upload: dict, path: Path, recording: Recording, tell) -> dict:
    size = upload["part_size_bytes"]
    with open(path, "rb") as file:
        while upload["next_part"] < upload["total_parts"]:
            part = upload["next_part"]
            tell(recording, f"uploading part {part + 1} of {upload['total_parts']}")
            file.seek(part * size)
            try:
                upload = client.call("PUT", f"/uploads/{upload['upload_id']}/parts/{part}",
                                     raw=file.read(size))
            except ApiError as exc:
                if exc.code != "part_out_of_order":
                    raise
                upload = client.call("GET", f"/uploads/{upload['upload_id']}")
    return upload


def _session(workspace: Workspace, recording: Recording) -> dict:
    from .. import __version__

    return {
        "schema_version": 1,
        "capture_session_id": recording.sync.capture_session_id,
        "started_at_utc": recording.started_at_utc,
        "clock": {"source": "unknown"},
        "trigger": {"type": "manual", "reason": "mmp-sense workspace"},
        "device": {"software": "mmp-sense", "version": __version__},
        "metadata": {"workspace_id": workspace.id, "recording_id": recording.id,
                     "file_name": recording.name},
    }


def _duration_ns(recording: Recording) -> int:
    return int(round((recording.duration_s or 0.0) * 1e9))


_AUDIO_TYPES = {".m4a": "audio/mp4", ".aac": "audio/aac", ".3gp": "audio/3gpp", ".amr": "audio/amr",
                ".mp3": "audio/mpeg", ".ogg": "audio/ogg", ".opus": "audio/ogg",
                ".flac": "audio/flac", ".webm": "audio/webm"}
_TEXT_TYPES = {".csv": "text/csv", ".tsv": "text/tab-separated-values"}


def declaration_of(recording: Recording, path: Path) -> dict:
    """The sample declaration of a recording's one file."""
    from ..io.audio import SENSOR_RATE_BELOW, wav_format

    suffix = path.suffix.lower()
    container = suffix.lstrip(".") or "bin"
    encoding: dict = {"container": container}
    if recording.kind == "wav":
        rate, channels = wav_format(path) or (0, 0)
        sensor = rate < SENSOR_RATE_BELOW or recording.options.get("scale") is not None
        media_type, modality = "audio/wav", "other" if sensor else "audio"
        encoding = {"container": "wav", "sample_rate_hz": rate, "channels": channels}
    elif recording.kind == "mcap":
        media_type, modality = "application/x-mcap", "multi"
        encoding = {"container": "mcap"}
    elif recording.kind == "audio":
        media_type, modality = _AUDIO_TYPES.get(suffix, "application/octet-stream"), "audio"
    elif recording.kind == "array":
        media_type, modality = f"application/x-{container}", "other"
    else:
        media_type, modality = _TEXT_TYPES.get(suffix, "text/plain"), "other"
    return {
        "schema_version": 1,
        "sample_id": recording.sync.sample_id,
        "capture_session_id": recording.sync.capture_session_id,
        "sequence_number": 1,
        "modality": modality,
        "media_type": media_type,
        "capture": {"start_offset_ns": 0, "duration_ns": _duration_ns(recording)},
        "encoding": encoding,
        "payload": {"bytes": path.stat().st_size, "sha256": _sha256(path)},
        "complete": True,
        "metadata": {"file_name": recording.name, "read_options": recording.options},
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# -- the token ----------------------------------------------------------------

def credentials_path() -> Path:
    from ..settings import config_dir

    return config_dir() / "credentials.json"


def load_token(server: str) -> str | None:
    try:
        tokens = json.loads(credentials_path().read_text())
    except (OSError, ValueError):
        return None
    return tokens.get(server.rstrip("/")) if isinstance(tokens, dict) else None


def forget_token(server: str) -> None:
    """Drop the token kept for `server`, so the next Synchronize asks for one."""
    try:
        tokens = json.loads(credentials_path().read_text())
    except (OSError, ValueError):
        return
    if isinstance(tokens, dict) and tokens.pop(server.rstrip("/"), None) is not None:
        _write_tokens(tokens)


def store_token(server: str, token: str) -> None:
    """Keep `token` for `server`, readable by the user alone."""
    path = credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        tokens = json.loads(path.read_text())
        if not isinstance(tokens, dict):
            tokens = {}
    except (OSError, ValueError):
        tokens = {}
    tokens[server.rstrip("/")] = token
    _write_tokens(tokens)


def _write_tokens(tokens: dict) -> None:
    path = credentials_path()
    temporary = path.with_suffix(".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as file:
        json.dump(tokens, file, indent=2)
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
