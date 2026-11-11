"""Opening a session srv-senselab exported with `sample export --raw`: the mics
by their names, on the session's time, and the recording's clocks."""

from __future__ import annotations

import json
import uuid

import numpy as np
import pytest
from mcap.writer import Writer
from scipy.io import wavfile

import mmp_sense
from mmp_sense.io.files import kind_of, read_file
from mmp_sense.mcp import tools

RATE = 48_000
SESSION = "08999e46-52a8-4ab0-81ec-208243af8193"
CLOCKS = [{"id": 0, "name": "mono", "kind": "monotonic", "hz": 1e6, "bits": None, "timeline": True, "source": None, "domain": None},
          {"id": 1, "name": "i2s", "kind": "sample", "hz": 48000.0, "bits": None, "timeline": False, "source": None, "domain": None},
          {"id": 2, "name": "utc", "kind": "utc", "hz": 1e9, "bits": None, "timeline": False, "source": "ntp", "domain": None}]
UTC0 = 1_794_409_391_295_124_000


def _mcap(path, messages):
    with open(path, "wb") as f:
        w = Writer(f)
        w.start()
        channels = {}
        for seq, (topic, t_ns, body) in enumerate(messages):
            if topic not in channels:
                schema = w.register_schema(name="dev.sensetool.X.v1", encoding="jsonschema", data=b"{}")
                channels[topic] = w.register_channel(topic=topic, message_encoding="json", schema_id=schema)
            w.add_message(channels[topic], log_time=t_ns, publish_time=t_ns, sequence=seq, data=json.dumps(body).encode())
        w.finish()


@pytest.fixture
def export(tmp_path):
    """One second of four mics, with 100 ms lost after 0.5 s, and its clocks."""
    folder = tmp_path / SESSION
    (folder / "samples").mkdir(parents=True)
    wav_id, mcap_id = str(uuid.uuid4()), str(uuid.uuid4())
    frames = np.arange(43_200)
    values = np.stack([(frames % 1000) * (c + 1) for c in range(4)], axis=1).astype(np.int16)
    wavfile.write(folder / "samples" / f"{wav_id}.wav", RATE, values)
    timing = [("/audio/timing", 0, {"audio_sample_id": wav_id, "first_frame": 0, "frames": 24_000, "session_ns": 0}),
              ("/audio/timing", 600_000_000, {"audio_sample_id": wav_id, "first_frame": 24_000, "frames": 19_200,
                                              "session_ns": 600_000_000, "discontinuity": True})]
    observations = [("/clock/observation", s * 1_000_000_000, {
        "schema_version": 1, "kind": "device", "pps": False, "gateway_utc_ns": UTC0 + s * 1_000_000_000 + 9_000_000,
        "readings": [{"clock": "utc", "id": 2, "value": UTC0 + s * 1_000_000_000, "uncertainty_ns": 20_000_000},
                     {"clock": "mono", "id": 0, "value": 14_224_454 + s * 1_000_000, "uncertainty_ns": 89_500}]}) for s in (0, 1)]
    observations.append(("/clock/observation", 500_000_000, {
        "schema_version": 1, "kind": "pong", "ping_id": 1, "gateway_send_utc_ns": UTC0 + 400_000_000,
        "gateway_recv_utc_ns": UTC0 + 420_000_000, "readings": [{"clock": "mono", "id": 0, "value": 14_734_454, "uncertainty_ns": 0}]}))
    _mcap(folder / "samples" / f"{mcap_id}.mcap", timing + observations)
    (folder / "capture_session.json").write_text(json.dumps({
        "id": SESSION, "started_at": "2026-11-11T15:43:11.295124Z", "capture_config": {"clocks": CLOCKS, "stream_format": 2},
        "sensors": [{"sensor_key": "mics", "modality": "audio", "channels": [{"name": n} for n in "SNWE"]}]}))
    wav = {"sample_id": wav_id, "modality": "audio", "sensor_key": "mics", "media_type": "audio/wav",
           "capture": {"start_offset_ns": 0, "duration_ns": 1_000_000_000},
           "encoding": {"container": "wav", "codec": "pcm_s16le", "sample_rate_hz": RATE, "channels": 4,
                        "channel_names": ["S", "N", "W", "E"], "first_frame": 0, "stream": "mics"}}
    mcap = {"sample_id": mcap_id, "modality": "timing", "sensor_key": None, "media_type": "application/x-mcap",
            "capture": {"start_offset_ns": 0, "duration_ns": 1_000_000_000}, "encoding": {"container": "mcap"}}
    (tmp_path / "manifest.json").write_text(json.dumps([{"capture_session_id": SESSION, "samples": [
        {"path": f"samples/{wav_id}.wav", "sample": wav}, {"path": f"samples/{mcap_id}.mcap", "sample": mcap}]}]))
    return folder


def test_the_folder_is_a_session(export):
    assert kind_of(export) == "session"
    (series,) = read_file(export)
    assert series.name == "mics"
    assert series.channels == ["S", "N", "W", "E"]
    assert series.sample_rate == RATE


def test_lost_frames_stay_a_gap_at_their_time(export):
    (series,) = read_file(export)
    t = series.time
    assert series.n_samples == 43_200, "every frame, none made up"
    jumps = np.flatnonzero(np.diff(t) > 1.5 / RATE)
    assert list(jumps) == [23_999], "one gap"
    assert t[23_999] == pytest.approx(23_999 / RATE)
    assert t[24_000] == pytest.approx(0.6), "frames after the loss are at the time the timing point gives"
    assert series.values[24_000, 0] == 24_000 % 1000 / 32768.0
    assert t[-1] == pytest.approx(0.6 + 19_199 / RATE)


def test_the_session_names_the_mics_and_holds_the_clocks(export):
    session = mmp_sense.load(export)
    names = session.signals.names
    for mic in "SNWE":
        assert any(n.split(":")[-1] == mic for n in names), names
    clocks = session.clocks()
    assert list(clocks["name"]) == ["mono", "i2s", "utc"]
    assert clocks.loc[clocks["timeline"], "name"].item() == "mono"
    obs = session.clock_observations()
    assert len(obs) == 5, "two readings each of two observations, and a pong's one"
    utc = obs[(obs["kind"] == "device") & (obs["clock"] == "utc")]
    assert list(utc["value"]) == [UTC0, UTC0 + 1_000_000_000], "64-bit ns kept exactly"
    assert list(utc["time"]) == [0.0, 1.0]
    pong = obs[obs["kind"] == "pong"].iloc[0]
    assert (pong["gateway_send_utc_ns"], pong["gateway_recv_utc_ns"]) == (UTC0 + 400_000_000, UTC0 + 420_000_000)


def test_the_assistant_reads_the_same_clocks(export):
    session = mmp_sense.load(export)
    assert [c["name"] for c in tools.call(session, "get_clocks")["clocks"]] == ["mono", "i2s", "utc"]
    found = tools.call(session, "get_clock_observations", {"clock": "utc"})
    assert found["rows"] == 2
    assert found["observations"][0]["value"] == UTC0


def test_other_files_have_no_clocks(tmp_path):
    path = tmp_path / "clip.wav"
    wavfile.write(path, 16_000, np.zeros(1600, dtype=np.int16))
    session = mmp_sense.load(path)
    assert session.clocks().empty
    assert session.clock_observations().empty


def test_a_folder_without_its_manifest_says_so(export):
    (export.parent / "manifest.json").unlink()
    with pytest.raises(ValueError, match="manifest.json"):
        read_file(export)
