"""Workspaces: recordings kept whole with their metadata, copied whole, and
sent to srv-senselab."""

from __future__ import annotations

import json
import stat

import numpy as np
import pytest
from scipy.io import wavfile

from fake_senselab import TOKEN, FakeSenselab
from test_files import START, json_schema, tone, write_mcap
from test_session_export import SESSION, export  # noqa: F401
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


def _client(fake):
    return sync.Client(fake.url, TOKEN, pause=lambda _s: None)


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
    original = {"id": first.id, "recordings": [(r.id, r.sync.capture_session_id, r.sync.sample_id)
                                                for r in first.recordings]}
    on_disk = (tmp_path / "one" / "workspace.json").read_text()

    first.save_as(tmp_path / "two")

    assert first.folder == tmp_path / "two" and first.id != original["id"]
    assert not first.unsaved
    for recording, (rid, session, sample) in zip(first.recordings, original["recordings"]):
        assert recording.id != rid
        assert recording.sync.capture_session_id != session and recording.sync.sample_id != sample
        assert not recording.sync.declared
    copy = Workspace.open(tmp_path / "two")
    assert len(copy.recordings) == 2
    assert copy.path_of(copy.recordings[1]).read_bytes() == files[1].read_bytes()
    assert (tmp_path / "one" / "workspace.json").read_text() == on_disk
    with pytest.raises(WorkspaceError):
        first.save_as(tmp_path / "two")


def test_an_export_folder_is_kept_with_its_manifest_and_marked_as_the_servers(tmp_path, export):
    workspace = Workspace.create(tmp_path / "ws")
    recording = workspace.add(export)
    assert recording.origin == "server" and recording.server_session_id == SESSION
    workspace.save_recording(recording)
    from mmp_sense.io.files import read_file

    names = read_file(workspace.path_of(recording))[0].channels
    assert names == ["S", "N", "W", "E"]


def test_synchronize_sends_each_recording_once(tmp_path, files, export):
    workspace = Workspace.create(tmp_path / "ws", "Field day")
    wav = workspace.add(files[0])
    wav.title, wav.tags, wav.notes = "Knock", ["kind:knock", "env:door"], "three knocks"
    mcap = workspace.add(files[1])
    server = workspace.add(export)
    unsaved_later = None
    workspace.save_all()
    unsaved_later = workspace.add(files[0])

    with FakeSenselab(part_size=4096) as fake:
        results = sync.synchronize(workspace, _client(fake))
        outcome = {r.recording.id: r.outcome for r in results}
        assert outcome == {wav.id: "sent", mcap.id: "sent", server.id: "skipped",
                           unsaved_later.id: "skipped"}
        sample = fake.samples[wav.sync.sample_id]
        assert sample["data"] == files[0].read_bytes()
        declaration = sample["declaration"]
        assert declaration["media_type"] == "audio/wav"
        assert declaration["encoding"] == {"container": "wav", "sample_rate_hz": 16_000, "channels": 1}
        assert fake.samples[mcap.sync.sample_id]["declaration"]["media_type"] == "application/x-mcap"
        session = fake.sessions[wav.sync.capture_session_id]
        assert session["metadata"]["workspace_id"] == workspace.id
        assert fake.contexts[wav.sync.capture_session_id] == {
            "revision": 1, "tag_policy_version": 1, "title": "Knock",
            "tags": ["env:door", "kind:knock"], "capture_notes": "three knocks"}
        assert set(fake.closed) == {wav.sync.capture_session_id, mcap.sync.capture_session_id}

        fake.calls.clear()
        again = sync.synchronize(Workspace.open(tmp_path / "ws"), _client(fake))
        assert {r.outcome for r in again} == {"up to date", "skipped"}
        assert fake.calls == []

        reopened = Workspace.open(tmp_path / "ws")
        reopened.get(wav.id).notes = "four knocks"
        results = sync.synchronize(reopened, _client(fake))
        assert {r.recording.id: r.outcome for r in results}[wav.id] == "context"
        assert fake.contexts[wav.sync.capture_session_id]["revision"] == 2
        assert fake.contexts[wav.sync.capture_session_id]["capture_notes"] == "four knocks"


def test_an_interrupted_upload_resumes_where_it_stopped(tmp_path, files):
    workspace = Workspace.create(tmp_path / "ws")
    recording = workspace.add(files[0])
    workspace.save_all()
    with FakeSenselab(part_size=4096) as fake:
        fake.fail_once.add(("PUT", r"/uploads/[^/]+/parts/3"))
        first = sync.synchronize(workspace, _client(fake))
        assert first[0].outcome == "failed" and "injected" in first[0].message
        assert not recording.sync.uploaded and recording.sync.declared
        fake.calls.clear()
        second = sync.synchronize(Workspace.open(tmp_path / "ws"), _client(fake))
        assert second[0].outcome == "sent"
        parts = [path for method, path in fake.calls if "/parts/" in path]
        assert parts[0].endswith("/parts/3")
        assert fake.samples[recording.sync.sample_id]["data"] == files[0].read_bytes()


def test_a_failed_context_is_retried_at_its_revision(tmp_path, files):
    workspace = Workspace.create(tmp_path / "ws")
    recording = workspace.add(files[0])
    workspace.save_all()
    with FakeSenselab() as fake:
        fake.fail_once.add(("PUT", r"/capture-sessions/[^/]+/context"))
        assert sync.synchronize(workspace, _client(fake))[0].outcome == "failed"
        assert sync.synchronize(workspace, _client(fake))[0].outcome == "sent"
        assert fake.contexts[recording.sync.capture_session_id]["revision"] == 1


def test_a_copy_is_sent_as_sessions_of_its_own(tmp_path, files):
    workspace = Workspace.create(tmp_path / "ws")
    workspace.add(files[0])
    workspace.save_all()
    with FakeSenselab() as fake:
        sync.synchronize(workspace, _client(fake))
        workspace.save_as(tmp_path / "copy")
        assert sync.synchronize(workspace, _client(fake))[0].outcome == "sent"
        assert len(fake.sessions) == 2 and len(fake.samples) == 2


def test_an_unreachable_server_is_tried_once(tmp_path, files):
    workspace = Workspace.create(tmp_path / "ws")
    workspace.add(files[0])
    workspace.add(files[1])
    workspace.save_all()
    client = sync.Client("http://127.0.0.1:9", TOKEN, attempts=1)
    results = sync.synchronize(workspace, client)
    assert [r.outcome for r in results] == ["failed", "failed"]
    assert "cannot be reached" in results[0].message and "not tried" in results[1].message


def test_a_sensors_wav_is_not_declared_as_sound(tmp_path):
    path = tmp_path / "accel.wav"
    wavfile.write(path, 400, np.zeros((400, 3), dtype=np.int16))
    workspace = Workspace.untitled()
    recording = workspace.add(path)
    declaration = sync.declaration_of(recording, path)
    assert declaration["modality"] == "other" and declaration["media_type"] == "audio/wav"
    assert declaration["encoding"]["channels"] == 3


def test_an_untitled_workspace_cannot_be_synchronized(files):
    workspace = Workspace.untitled()
    workspace.add(files[0])
    with pytest.raises(WorkspaceError):
        sync.synchronize(workspace, sync.Client("http://127.0.0.1:9", TOKEN))


def test_the_token_is_kept_for_the_user_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert sync.load_token("http://srv:8080") is None
    sync.store_token("http://srv:8080/", "mmps_a")
    sync.store_token("http://other", "mmps_b")
    assert sync.load_token("http://srv:8080") == "mmps_a"
    path = sync.credentials_path()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_text()) == {"http://srv:8080": "mmps_a", "http://other": "mmps_b"}
