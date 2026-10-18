"""Audio files: WAV read here, the rest decoded by ffmpeg, as 16 kHz mono."""

import shutil
import subprocess

import numpy as np
import pytest
from scipy.io import wavfile

import mmp_sense
from mmp_sense.io.audio import AudioError, read_audio

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="no ffmpeg")


def two_tones(seconds=2.0, rate=48000):
    t = np.arange(int(seconds * rate)) / rate
    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    right = 0.5 * np.sin(2 * np.pi * 1000 * t)
    return np.stack([left, right], axis=1)


def write_wav(path, data, rate=48000, kind="int16"):
    if kind == "int16":
        wavfile.write(path, rate, np.round(data * 32767).astype(np.int16))
    elif kind == "float32":
        wavfile.write(path, rate, data.astype(np.float32))
    elif kind == "int24":
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(rate),
                        "-ac", str(data.shape[1]), "-i", "-", "-c:a", "pcm_s24le", str(path)],
                       input=data.astype("<f4").tobytes(), check=True)
    return path


def peaks(series, top=2):
    spectrum = mmp_sense.dsp.amplitude_spectrum(series.values[:, 0], series.sample_rate)
    frequency, amplitude = spectrum
    strongest = np.argsort(amplitude[:, 0])[::-1][:top]
    return sorted(frequency[strongest])


def test_a_stereo_48k_wav_becomes_16k_mono_audio(tmp_path):
    series = read_audio(write_wav(tmp_path / "tones.wav", two_tones()))
    assert series.channels == ["audio"] and series.sample_rate == 16000
    assert series.n_samples == 32000
    assert series.time[0] == 0.0
    assert series.metadata["source_rate"] == 48000 and series.metadata["source_channels"] == 2
    assert peaks(series) == pytest.approx([440, 1000], abs=2)
    assert np.abs(series.values).max() <= 1.0


def test_the_files_own_rate_can_be_kept(tmp_path):
    path = write_wav(tmp_path / "tones.wav", two_tones())
    series = read_audio(path, rate=None)
    assert series.sample_rate == 48000 and series.n_samples == 96000


@pytest.mark.parametrize("kind", ["float32", pytest.param("int24", marks=needs_ffmpeg)])
def test_sample_formats_read_alike(tmp_path, kind):
    data = two_tones(0.5)
    reference = read_audio(write_wav(tmp_path / "a.wav", data), rate=None).values
    other = read_audio(write_wav(tmp_path / f"b_{kind}.wav", data, kind=kind), rate=None).values
    assert np.allclose(reference, other, atol=2 / 32768)


@needs_ffmpeg
def test_a_phone_style_m4a_opens_the_same(tmp_path):
    wav = write_wav(tmp_path / "tones.wav", two_tones())
    m4a = tmp_path / "tones.m4a"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-c:a", "aac",
                    "-b:a", "128k", str(m4a)], check=True)
    series = read_audio(m4a)
    assert series.sample_rate == 16000
    assert series.n_samples == pytest.approx(32000, abs=2048)   # encoder padding
    assert series.metadata["decoder"] == "ffmpeg"
    assert peaks(series) == pytest.approx([440, 1000], abs=2)


def test_without_ffmpeg_it_says_what_is_missing(tmp_path, monkeypatch):
    clip = tmp_path / "clip.m4a"
    clip.write_bytes(b"\x00" * 100)
    monkeypatch.setenv("PATH", "")
    with pytest.raises(AudioError, match="needs ffprobe.*ffmpeg"):
        read_audio(clip)


@pytest.mark.parametrize("name, content", [("empty.wav", b""), ("junk.wav", b"not a wav"),
                                           pytest.param("junk.mp3", b"not audio",
                                                        marks=needs_ffmpeg)])
def test_a_broken_file_is_said_plainly(tmp_path, name, content):
    path = tmp_path / name
    path.write_bytes(content)
    with pytest.raises(AudioError, match=name):
        read_audio(path)


def test_load_opens_audio_as_a_session(tmp_path):
    session = mmp_sense.load(write_wav(tmp_path / "tones.wav", two_tones()))
    assert session.buffer.channels == ["audio"]
    assert session.buffer.rate == pytest.approx(16000)
    assert len(session.buffer) == 32000
    assert mmp_sense.load(tmp_path / "tones.wav", rate=0).buffer.rate == pytest.approx(48000)


def test_graph_ops_work_on_audio(tmp_path):
    rate = 16000
    t = np.arange(3 * rate) / rate
    noise = 0.01 * np.random.default_rng(0).normal(size=len(t))
    voice = np.where((t > 1.0) & (t < 1.4), 0.4 * np.sin(2 * np.pi * 700 * t), 0.0)
    wavfile.write(tmp_path / "burst.wav", rate, (noise + voice).astype(np.float32))
    session = mmp_sense.load(tmp_path / "burst.wav")
    speech = session.values(mmp_sense.signal("audio").frames("30ms").band_power(300, 3400))
    inside = (speech.index > 1.1) & (speech.index < 1.35)
    outside = (speech.index > 2.0)
    assert np.nanmin(speech.values[inside]) > 20 * np.nanmax(speech.values[outside])


def test_the_window_holds_all_of_a_long_file(tmp_path):
    from mmp_sense.ui.app import build_session, parse_args

    rate = 8000
    wavfile.write(tmp_path / "long.wav", rate, np.zeros(90 * rate, dtype=np.int16) + 1)
    session = build_session(parse_args(["--file", str(tmp_path / "long.wav")]))
    session.poll()
    assert session.buffer.duration == pytest.approx(90, abs=0.01)
    assert session.buffer.rate == pytest.approx(16000)


def test_a_long_csv_is_held_whole_too(tmp_path):
    from mmp_sense.ui.app import build_session, parse_args

    t = np.arange(0, 90, 0.01)
    (tmp_path / "long.csv").write_text(
        "time,v\n" + "\n".join(f"{x:.2f},0" for x in t) + "\n")
    session = build_session(parse_args(["--file", str(tmp_path / "long.csv")]))
    session.poll()
    assert session.buffer.duration == pytest.approx(89.99, abs=0.02)


def test_the_window_looks_at_the_sound_itself(tmp_path, qapp):
    pytest.importorskip("PySide6")
    from mmp_sense.ui.app import build_session, parse_args
    from mmp_sense.ui.main_window import MainWindow

    session = build_session(parse_args(["--file", str(write_wav(tmp_path / "t.wav", two_tones()))]))
    window = MainWindow(session, window_s=10.0, console=False)
    try:
        assert window.focus_signal == "audio"
        kinds = {p.kind: p.signal for p in window.panes[1:]}
        assert kinds == {"spectrum": "audio", "spectrogram": "audio"}
    finally:
        window.close()


def _dominant(values, rate):
    frequency, amplitude = mmp_sense.dsp.amplitude_spectrum(values, rate)
    return frequency[np.argmax(amplitude[:, 0])]


def test_a_stereo_wav_can_keep_its_channels(tmp_path):
    path = write_wav(tmp_path / "stereo.wav", two_tones())
    series = read_audio(path, keep_channels=True)
    assert series.channels == ["ch0", "ch1"]
    assert series.sample_rate == 16000
    assert _dominant(series.values[:, 0], 16000) == pytest.approx(440, abs=2)
    assert _dominant(series.values[:, 1], 16000) == pytest.approx(1000, abs=2)
    session = mmp_sense.load(path, keep_channels=True)
    assert session.channels == ["ch0", "ch1"]


@needs_ffmpeg
def test_a_compressed_stereo_clip_can_keep_its_channels(tmp_path):
    wav = write_wav(tmp_path / "stereo.wav", two_tones())
    clip = tmp_path / "stereo.flac"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), str(clip)], check=True)
    series = read_audio(clip, keep_channels=True)
    assert series.channels == ["ch0", "ch1"]
    assert _dominant(series.values[:, 1], 16000) == pytest.approx(1000, abs=2)
