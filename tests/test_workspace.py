"""Workspaces: recordings kept whole with their metadata, copied whole, and
sent to srv-senselab."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.io import wavfile

from senselab import Senselab
from test_files import START, json_schema, tone, write_mcap  # noqa: F401
from test_session_folder import SESSION, folder  # noqa: F401
from mmp_sense.workspace import Workspace, WorkspaceError, is_workspace
from mmp_sense.workspace import sync


@pytest.fixture
def files(tmp_path):
    where = tmp_path / "in"
    where.mkdir()
    wav = where / "knock.wav"
    wavfile.write(wav, 16_000, (np.sin(np.arange(16_000) / 5) * 9000).astype(np.int16))
    mcap = where / "walk.mcap"
    write_mcap(mcap, {"/imu0/accel": (json_schema("/imu0/accel", "xyz", "m/s^2"), tone())})
    return wav, mcap


def _client(server, fail=None):
    """A client of `server`; `fail` is a set of (method, path fragment) to fail once with a 500."""
    client = sync.Client(server.url, pause=lambda _s: None)
    if fail is not None:
        call = client.call

        def failing(method, path, *args, **kwargs):
            for key in list(fail):
                if key[0] == method and key[1] in path and (len(key) < 3 or key[2](kwargs)):
                    fail.discard(key)
                    raise sync.ApiError(500, {"error": "error", "message": "injected"})
            return call(method, path, *args, **kwargs)
        client.call = failing
    return client


def test_a_saved_workspace_holds_the_files_whole_and_opens_the_same(tmp_path, files):
    wav, mcap = files
    workspace = Workspace.create(tmp_path / "ws", "Field day")
    a = workspace.add(wav)
    b = workspace.add(mcap)
    a.tags = ["kind:knock"]
    a.notes = "on the door"
    assert workspace.unsaved == [a, b]
    workspace.save_all()

    assert is_workspace(tmp_path / "ws")
    assert workspace.path_of(a).read_bytes() == wav.read_bytes()
    assert workspace.path_of(b).read_bytes() == mcap.read_bytes()
    again = Workspace.open(tmp_path / "ws")
    assert again.title == "Field day" and again.id == workspace.id
    assert [r.to_json() for r in again.recordings] == [a.to_json(), b.to_json()]
    assert again.recordings[1].started_at_utc == "2026-09-21T14:13:20.000000Z"
    assert again.recordings[0].duration_s == pytest.approx(1.0)


def test_save_workspace_writes_only_saved_recordings(tmp_path, files):
    workspace = Workspace.create(tmp_path / "ws")
    kept = workspace.add(files[0])
    workspace.save_recording(kept)
    workspace.add(files[1])
    workspace.save()
    assert [r.id for r in Workspace.open(tmp_path / "ws").recordings] == [kept.id]


def test_an_untitled_workspace_is_settled_into_a_folder(tmp_path, files):
    workspace = Workspace.untitled()
    recording = workspace.add(files[0])
    with pytest.raises(WorkspaceError):
        workspace.save_recording(recording)
    workspace.settle(tmp_path / "ws", "Later")
    workspace.save_recording(recording)
    assert Workspace.open(tmp_path / "ws").recordings[0].id == recording.id


def test_a_workspace_needs_an_empty_folder(tmp_path):
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "x").write_text("x")
    with pytest.raises(WorkspaceError, match="not empty"):
        Workspace.create(tmp_path / "full")
    with pytest.raises(WorkspaceError, match="not a workspace"):
        Workspace.open(tmp_path)


def test_save_as_is_an_independent_copy(tmp_path, files):
    first = Workspace.create(tmp_path / "one", "One")
    saved = first.add(files[0])
    first.save_recording(saved)
    first.add(files[1])
    original = {"id": first.id, "recordings": [(r.id, sync.session_id_of(r)) for r in first.recordings]}
    on_disk = (tmp_path / "one" / "workspace.json").read_text()

    first.save_as(tmp_path / "two")

    assert first.folder == tmp_path / "two" and first.id != original["id"]
    assert not first.unsaved
    for recording, (rid, session) in zip(first.recordings, original["recordings"]):
        assert recording.id != rid
        assert sync.session_id_of(recording) != session
        assert recording.sync.context_sent is None
    copy = Workspace.open(tmp_path / "two")
    assert len(copy.recordings) == 2
    assert copy.path_of(copy.recordings[1]).read_bytes() == files[1].read_bytes()
    assert (tmp_path / "one" / "workspace.json").read_text() == on_disk
    with pytest.raises(WorkspaceError):
        first.save_as(tmp_path / "two")


def test_a_session_folder_is_kept_whole_with_its_context(tmp_path, folder):
    workspace = Workspace.create(tmp_path / "ws")
    recording = workspace.add(folder)
    assert recording.origin == "session" and recording.sync.session_id == SESSION
    assert recording.title.startswith("tetra1 ")
    workspace.save_recording(recording)
    from mmp_sense.io.files import read_file

    names = read_file(workspace.path_of(recording))[0].channels
    assert names == ["S", "N", "W", "E"]


def test_a_received_folder_is_moved_in_not_copied(tmp_path, folder):
    workspace = Workspace.create(tmp_path / "ws")
    recording = workspace.adopt(folder)
    assert recording.saved and not folder.exists()
    assert (workspace.path_of(recording) / "session.json").is_file()
    assert Workspace.open(tmp_path / "ws").recordings[0].id == recording.id


def test_synchronize_sends_each_recording_once(tmp_path, files, folder):
    workspace = Workspace.create(tmp_path / "ws", "Field day")
    wav = workspace.add(files[0])
    wav.title, wav.tags, wav.notes = "Knock", ["kind:knock", "env:door"], "three knocks"
    mcap = workspace.add(files[1])
    received = workspace.add(folder)
    workspace.save_all()
    unsaved_later = workspace.add(files[0])

    with Senselab(tmp_path / "srv") as server:
        results = sync.synchronize(workspace, _client(server))
        outcome = {r.recording.id: r.outcome for r in results}
        assert outcome == {wav.id: "sent", mcap.id: "sent", received.id: "sent", unsaved_later.id: "skipped"}
        assert wav.sync.session_id == wav.id
        m = server.manifest(wav.id)
        assert (m["title"], m["tags"], m["notes"], m["revision"]) == ("Knock", ["env:door", "kind:knock"], "three knocks", 0)
        assert m["origin"] == "mmp-sense" and m["workspace"]["id"] == workspace.id
        (entry,) = m["files"]
        assert entry["media_type"] == "audio/wav" and "source" not in entry
        assert server.file(entry["sha256"]) == files[0].read_bytes()
        assert server.manifest(mcap.id)["files"][0]["media_type"] == "application/x-mcap"
        r = server.manifest(SESSION)
        assert r["origin"] == "stream" and {f["name"] for f in r["files"]} == {"mics.wav", "timing.jsonl"}
        assert server.store.session(SESSION)["missing"] == []

        server.calls.clear()
        again = sync.synchronize(Workspace.open(tmp_path / "ws"), _client(server))
        assert [r.outcome for r in again] == ["up to date"] * 3, "the unsaved one was never written"
        assert server.calls == []

        reopened = Workspace.open(tmp_path / "ws")
        reopened.get(wav.id).notes = "four knocks"
        reopened.get(received.id).title = "LOT3AN"
        results = {r.recording.id: r.outcome for r in sync.synchronize(reopened, _client(server))}
        assert results[wav.id] == "context" and results[received.id] == "context"
        assert (server.manifest(wav.id)["revision"], server.manifest(wav.id)["notes"]) == (1, "four knocks")
        assert (server.manifest(SESSION)["revision"], server.manifest(SESSION)["title"]) == (1, "LOT3AN")


def test_an_interrupted_upload_resumes_where_it_stopped(tmp_path, files, monkeypatch):
    monkeypatch.setattr(sync, "PART_BYTES", 4096)
    workspace = Workspace.create(tmp_path / "ws")
    recording = workspace.add(files[0])
    workspace.save_all()
    with Senselab(tmp_path / "srv") as server:
        fail = {("PUT", "/uploads/", lambda kw: kw["headers"]["Content-Range"].startswith("bytes 12288-"))}
        first = sync.synchronize(workspace, _client(server, fail))
        assert first[0].outcome == "failed" and "injected" in first[0].message
        assert recording.sync.context_sent is None
        server.calls.clear()
        second = sync.synchronize(Workspace.open(tmp_path / "ws"), _client(server))
        assert second[0].outcome == "sent"
        puts = [path for method, path in server.calls if method == "PUT" and "/uploads/" in path]
        assert len(puts) == -(-(files[0].stat().st_size - 12288) // 4096), "only the rest is sent"
        sha = server.manifest(recording.id)["files"][0]["sha256"]
        assert server.file(sha) == files[0].read_bytes()


def test_a_failed_session_put_is_retried_at_its_revision(tmp_path, files):
    workspace = Workspace.create(tmp_path / "ws")
    recording = workspace.add(files[0])
    workspace.save_all()
    with Senselab(tmp_path / "srv") as server:
        assert sync.synchronize(workspace, _client(server, {("PUT", "/sessions/")}))[0].outcome == "failed"
        assert sync.synchronize(workspace, _client(server))[0].outcome == "sent"
        assert server.manifest(recording.id)["revision"] == 0


def test_a_copy_is_sent_as_sessions_of_its_own(tmp_path, files, folder):
    workspace = Workspace.create(tmp_path / "ws")
    workspace.add(files[0])
    workspace.add(folder)
    workspace.save_all()
    with Senselab(tmp_path / "srv") as server:
        sync.synchronize(workspace, _client(server))
        workspace.save_as(tmp_path / "copy")
        assert [r.outcome for r in sync.synchronize(workspace, _client(server))] == ["sent", "sent"]
        assert server.store.status()["sessions"] == 4
        assert server.manifest(SESSION)["revision"] == 0, "the original is not contested"


def test_an_unreachable_server_is_tried_once(tmp_path, files):
    workspace = Workspace.create(tmp_path / "ws")
    workspace.add(files[0])
    workspace.add(files[1])
    workspace.save_all()
    client = sync.Client("http://127.0.0.1:9", attempts=1)
    results = sync.synchronize(workspace, client)
    assert [r.outcome for r in results] == ["failed", "failed"]
    assert "cannot be reached" in results[0].message and "not tried" in results[1].message


def test_a_file_name_the_server_refuses_is_made_safe(tmp_path):
    path = tmp_path / "my knock (1).wav"
    wavfile.write(path, 16_000, np.zeros(1600, dtype=np.int16))
    workspace = Workspace.create(tmp_path / "ws")
    recording = workspace.add(path)
    workspace.save_all()
    manifest, _ = sync.manifest_of(workspace, recording)
    assert manifest["files"][0]["name"] == "my_knock__1_.wav"
    assert manifest["files"][0]["original_name"] == "my knock (1).wav"
    with Senselab(tmp_path / "srv") as server:
        assert sync.synchronize(workspace, _client(server))[0].outcome == "sent"


def test_an_untitled_workspace_cannot_be_synchronized(files):
    workspace = Workspace.untitled()
    workspace.add(files[0])
    with pytest.raises(WorkspaceError):
        sync.synchronize(workspace, sync.Client("http://127.0.0.1:9"))
