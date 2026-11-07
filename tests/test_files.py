"""Opening single data files: each format, told apart by its own bytes.

The MCAP files are written here two ways: with the `mcap` library, as a
recorder that keeps a summary writes them, and record by record as the
mmp-sample phone app does for srv-senselab -- no summary, and no footer
at all when the app never closed the file.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest
from mcap.writer import Writer
from scipy.io import wavfile

import mmp_sense
from mmp_sense.io.files import FileFormatError, kind_of, read_file
from mmp_sense.io.mcap import plottable_topics, read_mcap, topics_in


# -- MCAP as a recorder with a summary writes it --------------------------


def json_schema(topic, components, unit):
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": topic,
        "type": "object",
        "properties": {c: {"type": "number", "unit": unit} for c in components},
        "required": list(components),
    }
    return json.dumps(schema).encode()


def write_mcap(path, streams, *, chunked=True, statistics=True, encoding="json",
               chunk_size=1 << 20):
    """streams: {topic: (schema bytes, [(time_ns, message dict), ...])}"""
    with open(path, "wb") as file:
        writer = Writer(file, use_chunking=chunked, use_statistics=statistics,
                        chunk_size=chunk_size)
        writer.start()
        for topic, (schema, messages) in streams.items():
            schema_id = writer.register_schema(topic, "jsonschema", schema)
            channel = writer.register_channel(topic, encoding, schema_id)
            for t, message in messages:
                writer.add_message(channel, log_time=t, publish_time=t,
                                   data=json.dumps(message).encode())
        writer.finish()
    return path


START = 1_790_000_000_000_000_000


def tone(count=200, rate=100.0, frequency=7.0):
    return [(START + int(i * 1e9 / rate),
             {"x": float(np.sin(2 * np.pi * frequency * i / rate)),
              "y": float(np.cos(2 * np.pi * frequency * i / rate)), "z": 9.81})
            for i in range(count)]


@pytest.fixture
def imu_mcap(tmp_path):
    gyro = [(START + 250_000_000 + i * 20_000_000, {"x": 0.1 * i, "y": 0.0, "z": 0.0})
            for i in range(50)]
    return write_mcap(tmp_path / "imu.mcap", {
        "/imu0/accel": (json_schema("/imu0/accel", "xyz", "m/s^2"), tone()),
        "/imu0/gyro": (json_schema("/imu0/gyro", "xyz", "rad/s"), gyro),
        "/imu0/temp": (json_schema("/imu0/temp", ["t"], "degC"), []),
    })


def test_a_recording_lists_its_topics(imu_mcap):
    topics = {t.topic: t for t in topics_in(imu_mcap)}
    assert topics["/imu0/accel"].components == ["x", "y", "z"]
    assert topics["/imu0/accel"].unit == "m/s^2"
    assert topics["/imu0/accel"].message_count == 200
    assert plottable_topics(imu_mcap) == ["/imu0/accel", "/imu0/gyro"]


def test_what_was_written_is_what_comes_back(imu_mcap):
    series = read_mcap(imu_mcap, "/imu0/accel")
    assert series.channels == ["x", "y", "z"] and series.units == "m/s^2"
    assert series.n_samples == 200
    assert series.values[10] == pytest.approx(
        [np.sin(2 * np.pi * 0.7), np.cos(2 * np.pi * 0.7), 9.81])
    assert np.diff(series.time).mean() == pytest.approx(0.01)
    assert series.metadata["epoch_ns"] == START


def test_a_file_with_several_topics_will_not_guess_which_one(imu_mcap):
    with pytest.raises(ValueError, match="name the one to read"):
        read_mcap(imu_mcap)


def test_a_topic_that_was_never_sent_reads_as_empty(imu_mcap):
    series = read_mcap(imu_mcap, "/imu0/temp")
    assert series.n_samples == 0 and series.channels == ["t"]


def test_a_whole_recording_opens_as_a_stream_per_topic_on_one_clock(imu_mcap):
    session = mmp_sense.load(imu_mcap)
    assert list(session.streams) == ["accel", "gyro"]
    assert len(session.streams["accel"].buffer) == 200
    assert session.streams["gyro"].buffer.t_start == pytest.approx(0.25, abs=1e-6)
    assert session.values("gyro:x").iloc[-1] == pytest.approx(4.9)


def test_one_topic_can_be_picked(imu_mcap):
    session = mmp_sense.load(imu_mcap, "/imu0/accel")
    assert session.dataset == "/imu0/accel"
    assert session.spectrum_summary("x")["dominant_hz"] == pytest.approx(7.0, abs=0.5)


def test_an_unchunked_file_reads_the_same(tmp_path):
    path = write_mcap(tmp_path / "flat.mcap",
                      {"/a/accel": (json_schema("/a/accel", "xyz", "g"), tone(50))},
                      chunked=False)
    assert read_mcap(path).n_samples == 50


def test_a_recorded_file_cut_short_reads_up_to_the_cut(tmp_path):
    whole = write_mcap(tmp_path / "whole.mcap",
                       {"/a/accel": (json_schema("/a/accel", "xyz", "g"), tone(2000))},
                       chunk_size=4096).read_bytes()
    (tmp_path / "torn.mcap").write_bytes(whole[: len(whole) // 2])
    series = read_mcap(tmp_path / "torn.mcap")
    assert 100 < series.n_samples < 2000
    assert series.values[:, 2] == pytest.approx(9.81)


def test_a_summary_without_statistics_still_counts(tmp_path):
    path = write_mcap(tmp_path / "s.mcap",
                      {"/a/accel": (json_schema("/a/accel", "xyz", "g"), tone(100))},
                      statistics=False)
    assert topics_in(path)[0].message_count == 100


def test_a_read_that_fails_partway_does_not_repeat_itself(tmp_path, monkeypatch):
    from mcap.reader import SeekingReader

    path = write_mcap(tmp_path / "s.mcap",
                      {"/a/accel": (json_schema("/a/accel", "xyz", "g"), tone(500))},
                      chunk_size=2048)
    original = SeekingReader.iter_messages

    def failing(self, *args, **kwargs):
        for n, item in enumerate(original(self, *args, **kwargs)):
            if n == 120:
                raise ValueError("a corrupt chunk")
            yield item

    monkeypatch.setattr(SeekingReader, "iter_messages", failing)
    assert read_mcap(path).n_samples == 120


def test_every_topic_is_read_in_one_pass(tmp_path, phone, monkeypatch):
    from mmp_sense.io import mcap

    path = tmp_path / "sample.mcap"
    path.write_bytes(phone.close())
    passes = []
    original = mcap._scan
    monkeypatch.setattr(mcap, "_scan", lambda data: passes.append(1) or original(data))
    assert len(read_file(path)) == 3
    assert len(passes) == 2                                 # what it holds, then the messages


def test_a_recording_with_nothing_to_plot_says_so(tmp_path):
    notes = json.dumps({"properties": {"text": {"type": "string"}}}).encode()
    path = write_mcap(tmp_path / "notes.mcap",
                      {"/notes": (notes, [(START, {"text": "hello"})]),
                       "/raw": (b"", [(START, {"x": 1.0})])}, encoding="protobuf")
    with pytest.raises(ValueError, match="holds numbers to plot"):
        mmp_sense.load(path)
    with pytest.raises(ValueError, match="no numbers to plot"):
        read_mcap(path, "/notes")


# -- MCAP as srv-senselab's phone app writes it ---------------------------


class PhoneMcap:
    """The mmp-sample writer, record by record: every topic declared up
    front, messages as they come, and a footer with no summary."""

    TOPICS = {
        "/audio/timing": "AudioTiming",
        "/sensor/accelerometer": "SensorSample",
        "/sensor/gyroscope": "SensorSample",
        "/sensor/magnetometer": "SensorSample",
        "/location/gnss": "LocationSample",
        "/annotation/marker": "Annotation",
    }

    def __init__(self):
        self.out = bytearray(b"\x89MCAP0\r\n")
        self.ids = {}
        self.record(1, self.str("") + self.str("mmp-sample/1"))
        for i, (topic, name) in enumerate(self.TOPICS.items(), start=1):
            self.ids[topic] = i
            schema = json.dumps({"title": f"dev.sensetool.{name}.v1",
                                 "properties": {"schema_version": {"const": 1}}}).encode()
            self.record(3, struct.pack("<H", i) + self.str(f"dev.sensetool.{name}.v1")
                        + self.str("jsonschema") + struct.pack("<I", len(schema)) + schema)
            meta = self.str("clock") + self.str("session_relative_boottime_ns")
            self.record(4, struct.pack("<HH", i, i) + self.str(topic) + self.str("json")
                        + struct.pack("<I", len(meta)) + meta)
        self.sequence = 0

    @staticmethod
    def str(text):
        raw = text.encode()
        return struct.pack("<I", len(raw)) + raw

    def record(self, op, payload):
        self.out += struct.pack("<BQ", op, len(payload)) + payload

    def message(self, topic, t, body):
        self.record(5, struct.pack("<HIQQ", self.ids[topic], self.sequence, t, t)
                    + json.dumps(body).encode())
        self.sequence += 1

    def close(self):
        self.record(15, struct.pack("<I", 0))
        self.record(2, struct.pack("<QQI", 0, 0, 0))
        self.out += b"\x89MCAP0\r\n"
        return bytes(self.out)


@pytest.fixture
def phone():
    writer = PhoneMcap()
    for i in range(100):
        t = 1_000_000_000 + i * 20_000_000
        writer.message("/sensor/accelerometer", t,
                       {"schema_version": 1, "elapsed_ns": t, "accuracy": 3,
                        "values": [0.0, 0.1 * i, 9.81]})
        if i % 2 == 0:
            writer.message("/sensor/gyroscope", t,
                           {"schema_version": 1, "elapsed_ns": t, "accuracy": 3,
                            "values": [0.01, 0.0, -0.01]})
        if i % 10 == 0:
            writer.message("/audio/timing", t,
                           {"schema_version": 1, "audio_sample_id": "a", "first_frame": i,
                            "frames": 480, "sample_rate_hz": 48000, "session_ns": t})
    for i in range(3):
        writer.message("/location/gnss", 1_000_000_000 + i * 1_000_000_000,
                       {"schema_version": 1, "latitude": 51.1 + i * 1e-4, "longitude": 17.0,
                        "altitude_m": None, "accuracy_m": 4.0, "speed_mps": None,
                        "bearing_deg": None, "provider": "gps", "fix_utc_ms": 1, "mock": False})
    return writer


def test_a_phone_recording_opens_its_sensors_and_nothing_else(tmp_path, phone):
    path = tmp_path / "sample.mcap"
    path.write_bytes(phone.close())
    assert plottable_topics(path) == ["/location/gnss", "/sensor/accelerometer",
                                      "/sensor/gyroscope"]
    session = mmp_sense.load(path)
    assert list(session.streams) == ["gnss", "accelerometer", "gyroscope"]
    accel = session.streams["accelerometer"]
    assert accel.buffer.channels == ["x", "y", "z"]
    assert len(accel.buffer) == 100
    assert session.values("accelerometer:y").iloc[-1] == pytest.approx(9.9)


def test_phone_sensors_carry_si_units(tmp_path, phone):
    path = tmp_path / "sample.mcap"
    path.write_bytes(phone.close())
    accel = read_mcap(path, "/sensor/accelerometer")
    assert accel.units == "m/s^2"
    assert accel.time[1] == pytest.approx(0.02)
    assert read_mcap(path, "/sensor/gyroscope").units == "rad/s"
    gnss = read_mcap(path, "/location/gnss")
    assert gnss.channels[:2] == ["latitude", "longitude"]
    assert gnss.values[2, 0] == pytest.approx(51.1002)
    assert np.isnan(gnss.values[0, 2])                      # no altitude


def test_a_recording_the_phone_never_closed_still_reads(tmp_path, phone):
    whole = phone.close()
    cut = whole[: len(whole) // 2 + 7]                      # no footer, a torn message
    path = tmp_path / "torn.mcap"
    path.write_bytes(cut)
    accel = read_mcap(path, "/sensor/accelerometer")
    assert 20 < accel.n_samples < 100
    assert accel.values[5, 1] == pytest.approx(0.5)


# -- WAV ----------------------------------------------------------------


def test_a_sensors_wav_opens_in_counts_at_its_own_rate(tmp_path):
    counts = np.array([[100, -200, 4096]] * 400, dtype=np.int16)
    wavfile.write(tmp_path / "accelerometer.wav", 400, counts)
    (series,) = read_file(tmp_path / "accelerometer.wav")
    assert series.channels == ["x", "y", "z"]
    assert series.units == "counts"
    assert series.sample_rate == 400
    assert series.values[0].tolist() == [100, -200, 4096]


def test_a_sensors_wav_with_a_scale_is_in_units(tmp_path):
    counts = np.array([[100, -200, 4096]] * 400, dtype=np.int16)
    wavfile.write(tmp_path / "a.wav", 400, counts)
    session = mmp_sense.load(tmp_path / "a.wav", scale=0.0024, units="m/s^2")
    assert session.buffer.channels == ["x", "y", "z"]
    assert session.values("z").iloc[0] == pytest.approx(4096 * 0.0024)
    assert session.buffer.rate == pytest.approx(400)


def test_a_scale_makes_any_wav_a_sensors(tmp_path):
    wavfile.write(tmp_path / "fast.wav", 16000, np.full((1600, 3), 10, dtype=np.int16))
    (series,) = read_file(tmp_path / "fast.wav", scale=2.0, units="g")
    assert series.channels == ["x", "y", "z"] and series.units == "g"
    assert series.values[0].tolist() == [20, 20, 20]


def test_sound_stays_sound(tmp_path):
    t = np.arange(48000) / 48000
    wavfile.write(tmp_path / "s.wav", 48000, (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32))
    (series,) = read_file(tmp_path / "s.wav")
    assert series.channels == ["audio"] and series.sample_rate == 48000


def test_a_24_bit_multichannel_wav_keeps_its_rate_and_channels(tmp_path):
    rate = 44100
    frames = np.zeros((rate // 10, 4), dtype=np.int32)
    frames[:, 2] = 1 << 30
    raw = (frames >> 8).astype("<i4").view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    fmt = struct.pack("<HHIIHH", 1, 4, rate, rate * 12, 12, 24)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(raw)) + raw
    (tmp_path / "multi.wav").write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
    (series,) = read_file(tmp_path / "multi.wav")
    assert series.channels == ["ch0", "ch1", "ch2", "ch3"]
    assert series.sample_rate == rate
    assert series.values[0, 2] == pytest.approx(0.5)


# -- delimited text -----------------------------------------------------


def test_srv_senselab_clip_csv_is_in_seconds_and_si_units(tmp_path):
    path = tmp_path / "1a2b3c4d_accelerometer_10.000-12.000.csv"
    rows = "\n".join(f"{10_000_000_000 + i * 5_000_000},0.1,0.2,9.81" for i in range(400))
    path.write_text("t_ns,x,y,z\n" + rows + "\n")
    (series,) = read_file(path)
    assert series.channels == ["x", "y", "z"] and series.units == "m/s^2"
    assert series.time[0] == pytest.approx(10.0)
    assert series.rate == pytest.approx(200)


def test_srv_senselab_gnss_csv_keeps_its_gaps(tmp_path):
    path = tmp_path / "1a2b3c4d_gnss_0.000-3.000.csv"
    path.write_text("t_ns,latitude,longitude,altitude_m,accuracy_m,speed_mps,bearing_deg\n"
                    "0,51.1,17.0,,4.0,,\n1000000000,51.2,17.0,,4.0,,\n")
    (series,) = read_file(path)
    assert series.channels[:2] == ["latitude", "longitude"]
    assert np.isnan(series.values[0, 2])


@pytest.mark.parametrize("sep", ["\t", ";", " "])
def test_the_separator_is_sniffed(tmp_path, sep):
    path = tmp_path / "table.txt"
    path.write_text(sep.join(["time", "a", "b"]) + "\n"
                    + "\n".join(sep.join([str(i / 100), str(i), str(-i)]) for i in range(50)))
    (series,) = read_file(path)
    assert series.channels == ["a", "b"]
    assert series.values[49].tolist() == [49, -49]


def test_rows_are_put_in_time_order_and_untimed_ones_dropped(tmp_path):
    path = tmp_path / "shuffled.csv"
    path.write_text("time,v\n0.2,2\n0.0,0\n,9\n0.1,1\n")
    (series,) = read_file(path)
    assert series.time.tolist() == [0.0, 0.1, 0.2]
    assert series.values[:, 0].tolist() == [0, 1, 2]


def test_a_timestamp_too_large_for_seconds_is_an_epochs_milliseconds(tmp_path):
    path = tmp_path / "android.csv"
    path.write_text("timestamp,ax\n" + "\n".join(f"{1_760_000_000_000 + 5 * i},{i}"
                                                  for i in range(200)))
    (series,) = read_file(path)
    assert series.time[-1] == pytest.approx(199 * 0.005)
    assert series.rate == pytest.approx(200)


def test_dates_are_a_time_column_too(tmp_path):
    path = tmp_path / "logger.csv"
    path.write_text("time,temp\n2026-10-13T08:00:00Z,20.5\n2026-10-13T08:00:30Z,20.7\n")
    (series,) = read_file(path)
    assert series.time.tolist() == [0.0, 30.0]


def test_a_semicolon_table_may_use_decimal_commas(tmp_path):
    path = tmp_path / "eu.csv"
    path.write_text("time;v\n0,0;1,5\n0,1;2,5\n")
    (series,) = read_file(path)
    assert series.time.tolist() == [0.0, 0.1]
    assert series.values[:, 0].tolist() == [1.5, 2.5]


def test_a_table_of_time_alone_is_said_plainly(tmp_path):
    path = tmp_path / "times.csv"
    path.write_text("time\n0\n1\n")
    with pytest.raises(ValueError, match="times.csv: no numeric columns"):
        read_file(path)


def test_an_empty_file_is_said_plainly(tmp_path):
    for name in ("e.csv", "e.npy", "e"):
        (tmp_path / name).write_bytes(b"")
        with pytest.raises(FileFormatError, match="empty"):
            read_file(tmp_path / name)


def test_a_table_with_no_time_takes_a_rate(tmp_path):
    path = tmp_path / "samples.csv"
    path.write_text("ax,ay,note\n" + "\n".join(f"{i},{-i},x" for i in range(100)))
    with pytest.raises(KeyError, match="give the sample rate"):
        read_file(path)
    (series,) = read_file(path, rate=50)
    assert series.channels == ["ax", "ay"]                 # notes are not channels
    assert series.time[-1] == pytest.approx(99 / 50)


def test_an_epoch_clock_counts_from_its_first_sample(tmp_path):
    path = tmp_path / "epoch.csv"
    path.write_text("time_ns,v\n" + "\n".join(f"{START + i * 1000},{i}" for i in range(10)))
    (series,) = read_file(path)
    assert series.time[0] == 0 and series.time[-1] == pytest.approx(9e-6)
    assert series.metadata["epoch"] == START


# -- NumPy --------------------------------------------------------------


def test_an_npy_needs_its_rate(tmp_path):
    np.save(tmp_path / "a.npy", np.ones((300, 2)))
    with pytest.raises(ValueError, match="give the sample rate"):
        read_file(tmp_path / "a.npy")
    session = mmp_sense.load(tmp_path / "a.npy", sample_rate=100)
    assert session.buffer.channels == ["ch0", "ch1"]
    assert session.buffer.rate == pytest.approx(100)


def test_an_npz_names_its_channels_and_may_carry_its_rate(tmp_path):
    np.savez(tmp_path / "b.npz", fs=250.0, accel=np.zeros((500, 3)), temp=np.ones(500))
    (series,) = read_file(tmp_path / "b.npz")
    assert series.channels == ["accel0", "accel1", "accel2", "temp"]
    assert series.sample_rate == 250


def test_an_npz_may_carry_its_times(tmp_path):
    np.savez(tmp_path / "c.npz", t=np.array([0.0, 0.5, 2.0]), v=np.array([1, 2, 3]))
    (series,) = read_file(tmp_path / "c.npz")
    assert series.time.tolist() == [0.0, 0.5, 2.0]


def test_pickled_arrays_are_refused(tmp_path):
    np.save(tmp_path / "evil.npy", np.array([{"a": 1}], dtype=object), allow_pickle=True)
    with pytest.raises(ValueError, match="safely"):
        read_file(tmp_path / "evil.npy", rate=1)


# -- telling formats apart ----------------------------------------------


def test_a_format_is_known_by_its_bytes_not_its_name(tmp_path, imu_mcap):
    renamed = tmp_path / "recording.bin"
    renamed.write_bytes(imu_mcap.read_bytes())
    assert kind_of(renamed) == "mcap"
    wavfile.write(tmp_path / "x.dat", 400, np.zeros((10, 3), dtype=np.int16))
    assert kind_of(tmp_path / "x.dat") == "wav"
    np.save(tmp_path / "arr", np.zeros(3))
    (tmp_path / "arr.npy").rename(tmp_path / "arr.raw")
    assert kind_of(tmp_path / "arr.raw") == "array"


def test_a_wav_under_another_name_needs_no_ffmpeg(tmp_path, monkeypatch):
    import shutil

    wavfile.write(tmp_path / "clip.dat", 16000, np.zeros(1600, dtype=np.int16))
    monkeypatch.setattr(shutil, "which", lambda _tool: None)
    (series,) = read_file(tmp_path / "clip.dat")
    assert series.channels == ["audio"] and series.metadata["decoder"] == "wav"


def test_an_unknown_binary_is_said_plainly(tmp_path):
    (tmp_path / "blob.bin").write_bytes(bytes(range(256)) * 4)
    with pytest.raises(FileFormatError, match="not a format mmp-sense reads"):
        read_file(tmp_path / "blob.bin")


def test_topics_are_an_mcaps_only(tmp_path):
    wavfile.write(tmp_path / "s.wav", 8000, np.zeros(800, dtype=np.int16))
    with pytest.raises(ValueError, match="topics are an MCAP's"):
        mmp_sense.load(tmp_path / "s.wav", topics=["/a"])


# -- every way in -------------------------------------------------------


def test_the_window_opens_a_file_named_on_its_own(tmp_path, phone):
    from mmp_sense.ui.app import build_session, parse_args

    path = tmp_path / "sample.mcap"
    path.write_bytes(phone.close())
    session = build_session(parse_args([str(path)]))
    assert list(session.streams) == ["gnss", "accelerometer", "gyroscope"]
    session = build_session(parse_args(["--file", str(path), "--topic",
                                        "/sensor/gyroscope"]))
    assert session.dataset == "/sensor/gyroscope"


def test_the_window_takes_a_sensor_wavs_scale(tmp_path):
    from mmp_sense.ui.app import build_session, parse_args

    wavfile.write(tmp_path / "g.wav", 200, np.full((400, 3), 1000, dtype=np.int16))
    session = build_session(parse_args([str(tmp_path / "g.wav"), "--scale", "0.001",
                                        "--units", "rad/s"]))
    assert session.values("x").iloc[0] == pytest.approx(1.0)


def test_a_file_is_named_once(tmp_path):
    from mmp_sense.ui.app import parse_args

    with pytest.raises(SystemExit):
        parse_args(["a.csv", "--file", "b.csv"])


def test_the_window_shows_a_phone_recording(tmp_path, phone, qapp):
    from mmp_sense.ui.app import build_session, parse_args
    from mmp_sense.ui.main_window import MainWindow

    path = tmp_path / "sample.mcap"
    path.write_bytes(phone.close())
    window = MainWindow(build_session(parse_args([str(path)])))
    window.tick()
    assert window.windowTitle().startswith("mmp-sense -- ")
    assert len(window.session.streams["accelerometer"].buffer) == 100
    window.close()


def test_the_mcp_server_reads_what_the_window_reads(tmp_path):
    from mmp_sense.mcp.server import _own_session

    rate = 1000
    t = np.arange(120 * rate) / rate                        # longer than --buffer
    wavfile.write(tmp_path / "v.wav", 16000, np.zeros(16000, dtype=np.float32))
    np.save(tmp_path / "long.npy", np.sin(2 * np.pi * 5 * t))

    class Args:
        path, file, topic = str(tmp_path / "long.npy"), None, []
        scale, units, buffer = None, None, 60.0

    Args.rate = rate
    backend = _own_session(Args)
    try:
        backend.session.poll()
        assert len(backend.session.buffer) == len(t)        # held whole
    finally:
        backend.close()
    Args.path, Args.rate = str(tmp_path / "v.wav"), None
    backend = _own_session(Args)
    try:
        assert backend.session.buffer.channels == ["audio"]
    finally:
        backend.close()


def test_the_mcp_server_says_what_is_wrong_with_a_file(tmp_path, capsys):
    from mmp_sense.mcp.server import main

    (tmp_path / "x.csv").write_text("a,b\n1,2\n")
    assert main([str(tmp_path / "x.csv")]) == 1
    assert "give the sample rate" in capsys.readouterr().err
