"""Opening a session folder as mmp-stream's receiver writes it (and srv-senselab exports it):
the mics by their names, on the session's time, and the recording's clocks."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import mmp_stream
import numpy as np
import pytest

import mmp_sense
from mmp_sense.io.files import kind_of, read_file
from mmp_sense.io.session_folder import read_session_folder
from mmp_sense.mcp import tools

RATE = 48_000
SESSION = "08999e46-52a8-4ab0-81ec-208243af8193"
UTC0 = 1_794_409_391_295_124_000
MONO0 = 14_224_454


def _block(first: int, frames: int) -> bytes:
    k = first + np.arange(frames)
    values = np.stack([(k % 1000) * (c + 1) for c in range(4)], axis=1).astype("<i2")
    return struct.pack("<Qq", first, MONO0 + round(first / RATE * 1e6)) + values.tobytes()


def _write(out, blocks) -> object:
    hello = mmp_stream.check_hello(mmp_stream.profile("tetrahedral", "tetra1")[0])
    rec = mmp_stream.Recording(out, SESSION, hello, "test")
    for s in (0, 1):
        rec.take("C", 0, 0, bytes([0]) + mmp_stream.encode_readings(
            [(3, UTC0 + s * 1_000_000_000, 20_000_000), (0, MONO0 + s * 1_000_000, 89_500)]), {})
    for first, frames in blocks:
        rec.take("D", 0, 1, _block(first, frames), {})
    return rec.close("clean")


@pytest.fixture
def folder(tmp_path):
    """One second of four mics in 100 ms blocks, with 100 ms lost after 0.5 s, and its clocks."""
    blocks = [(f, 4800) for f in range(0, 24_000, 4800)] + [(f, 4800) for f in range(28_800, 48_000, 4800)]
    return _write(tmp_path, blocks)


def test_the_folder_is_a_session(folder):
    assert kind_of(folder) == "session"
    (series,) = read_file(folder)
    assert series.name == "mics"
    assert series.channels == ["S", "N", "W", "E"]
    assert series.sample_rate == RATE
    assert series.metadata["session_id"] == SESSION
    assert series.metadata["session_title"].startswith("tetra1 ")


def test_lost_frames_stay_a_gap_at_their_time(folder):
    (series,) = read_file(folder)
    t = series.time
    assert series.n_samples == 43_200, "every frame, none made up"
    jumps = np.flatnonzero(np.diff(t) > 1.5 / RATE)
    assert list(jumps) == [23_999], "one gap"
    assert t[23_999] == pytest.approx(23_999 / RATE)
    assert t[24_000] == pytest.approx(0.6), "frames after the loss are at the time the timing point gives"
    assert series.values[24_000, 0] == 28_800 % 1000 / 32768.0
    assert t[-1] == pytest.approx(0.6 + 19_199 / RATE)


def test_the_session_names_the_mics_and_holds_the_clocks(folder):
    session = mmp_sense.load(folder)
    for mic in "SNWE":
        assert any(n.split(":")[-1] == mic for n in session.signals.names)
    clocks = session.clocks()
    assert list(clocks["name"]) == ["mono", "i2s", "utc"]
    assert clocks.loc[clocks["timeline"].astype(bool), "name"].item() == "mono"
    obs = session.clock_observations()
    device = obs[obs["kind"] == "device"]
    assert len(device) == 4, "two readings each of two observations"
    utc = device[device["clock"] == "utc"]
    assert list(utc["value"]) == [UTC0, UTC0 + 1_000_000_000], "64-bit ns kept exactly"
    assert list(utc["time"]) == pytest.approx([0.0, 1.0])
    assert (obs["kind"] == "arrival").any()


def test_the_assistant_reads_the_same_clocks(folder):
    session = mmp_sense.load(folder)
    assert [c["name"] for c in tools.call(session, "get_clocks")["clocks"]] == ["mono", "i2s", "utc"]
    found = tools.call(session, "get_clock_observations", {"clock": "utc"})
    assert found["rows"] == 2
    assert found["observations"][0]["value"] == UTC0


def test_a_rolled_over_stream_reads_as_one(tmp_path, monkeypatch):
    monkeypatch.setattr(mmp_stream, "MAX_FILE_BYTES", 44 + 8 * 4800 * 2 + 100)
    folder = _write(tmp_path, [(f, 4800) for f in range(0, 24_000, 4800)])
    names = [f["name"] for f in json.loads((folder / "session.json").read_text())["files"]]
    assert "mics-001.wav" in names
    (series,) = read_file(folder)
    assert series.n_samples == 24_000
    assert np.all(np.diff(series.time) > 0)
    assert series.time[-1] == pytest.approx(23_999 / RATE)


def test_a_sensor_stream_is_in_its_units(tmp_path):
    hello = mmp_stream.check_hello(mmp_stream.profile("imu", "n")[0])
    rec = mmp_stream.Recording(tmp_path, SESSION, hello, "test")
    counts = np.tile(np.array([100, -200, 300], "<i2"), (20, 1))
    rec.take("D", 0, 2, struct.pack("<Qq", 0, 0) + counts.tobytes(), {})
    (series,) = read_file(rec.close("clean"))
    assert series.name == "accelerometer"
    assert series.channels == ["x", "y", "z"]
    assert series.units == "m/s^2"
    assert series.values[0] == pytest.approx([100 * 0.0023942, -200 * 0.0023942, 300 * 0.0023942])


def test_other_files_have_no_clocks(tmp_path):
    from scipy.io import wavfile

    path = tmp_path / "clip.wav"
    wavfile.write(path, 16_000, np.zeros(1600, dtype=np.int16))
    session = mmp_sense.load(path)
    assert session.clocks().empty
    assert session.clock_observations().empty


def test_jittery_timing_points_keep_time_increasing():
    """tetra1 stamps its points when a block arrives, so they jitter by a few ms around 48 kHz;
    LOT3AN on 2026-11-15 had a point 2.1 ms earlier than its predecessor's run predicted."""
    from mmp_sense.io import session_folder as sf

    points = [(0, 0.0, False), (12_000, 0.250001, False), (24_480, 0.51186, False), (36_480, 0.7598, False)]
    t = sf._frame_times(0, 48_000, RATE, 0.0, points)
    assert np.all(np.diff(t) >= 0), "time never steps back"
    assert t[24_480] == pytest.approx(0.51186) and t[36_480] == pytest.approx(0.7598)
    assert t[-1] == pytest.approx(0.7598 + 11_519 / RATE), "past the last point, at the rate"


PHONE_FIXTURE = Path(__file__).resolve().parents[2] / "mmp-sample" / "fixtures" / "session-48k"


@pytest.mark.skipif(not PHONE_FIXTURE.is_dir(), reason="mmp-sample's phone fixture is not beside this repo")
def test_phone_session_merges_each_topic_across_chunks():
    by_name = {s.name: s for s in read_session_folder(PHONE_FIXTURE)}
    assert set(by_name) == {"microphone", "/location/gnss", "/sensor/accelerometer", "/sensor/gyroscope", "/sensor/magnetometer"}
    accel = by_name["/sensor/accelerometer"]
    assert len(accel.time) == 100 and accel.time[0] == pytest.approx(0.0) and accel.time[-1] == pytest.approx(4.95)
    assert np.all(np.diff(accel.time) > 0)
    assert by_name["microphone"].time[-1] == pytest.approx(5.0, abs=1e-3)
