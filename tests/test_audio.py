"""Audio files: WAV read here, the rest decoded by ffmpeg, at their own rate and channels."""

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


def _dominant(values, rate):
    frequency, amplitude = mmp_sense.dsp.amplitude_spectrum(values, rate)
    return frequency[np.argmax(amplitude[:, 0])]


def test_a_stereo_48k_wav_keeps_its_rate_and_channels(tmp_path):
    series = read_audio(write_wav(tmp_path / "tones.wav", two_tones()))
    assert series.channels == ["ch0", "ch1"] and series.sample_rate == 48000
    assert series.n_samples == 96000
    assert series.time[0] == 0.0
    assert series.metadata["source_rate"] == 48000 and series.metadata["source_channels"] == 2
    assert _dominant(series.values[:, 0], 48000) == pytest.approx(440, abs=2)
    assert _dominant(series.values[:, 1], 48000) == pytest.approx(1000, abs=2)
    assert np.abs(series.values).max() <= 1.0


def test_a_mono_wav_is_one_channel_called_audio(tmp_path):
    t = np.arange(4410) / 44100
    wavfile.write(tmp_path / "mono.wav", 44100, np.sin(2 * np.pi * 440 * t).astype(np.float32))
    series = read_audio(tmp_path / "mono.wav")
    assert series.channels == ["audio"] and series.sample_rate == 44100
    assert series.n_samples == 4410


@pytest.mark.parametrize("kind", ["float32", pytest.param("int24", marks=needs_ffmpeg)])
def test_sample_formats_read_alike(tmp_path, kind):
    data = two_tones(0.5)
    reference = read_audio(write_wav(tmp_path / "a.wav", data)).values
    other = read_audio(write_wav(tmp_path / f"b_{kind}.wav", data, kind=kind)).values
    assert np.allclose(reference, other, atol=2 / 32768)


@needs_ffmpeg
def test_a_phone_style_m4a_opens_the_same(tmp_path):
    wav = write_wav(tmp_path / "tones.wav", two_tones())
    m4a = tmp_path / "tones.m4a"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-c:a", "aac",
                    "-b:a", "128k", str(m4a)], check=True)
    series = read_audio(m4a)
    assert series.channels == ["ch0", "ch1"] and series.sample_rate == 48000
    assert series.n_samples == pytest.approx(96000, abs=2048)   # encoder padding
    assert series.metadata["decoder"] == "ffmpeg"
    assert _dominant(series.values[:, 0], 48000) == pytest.approx(440, abs=2)
    assert _dominant(series.values[:, 1], 48000) == pytest.approx(1000, abs=2)


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
    assert session.buffer.channels == ["ch0", "ch1"]
    assert session.buffer.rate == pytest.approx(48000)
    assert len(session.buffer) == 96000


@pytest.mark.parametrize("gone", ["rate", "keep_channels"])
def test_load_takes_no_audio_options(tmp_path, gone):
    with pytest.raises(TypeError, match=gone):
        mmp_sense.load(write_wav(tmp_path / "tones.wav", two_tones()), **{gone: 0})


@pytest.mark.parametrize("flag", ["--audio-rate=8000", "--audio-channels"])
def test_the_command_lines_take_no_audio_options(flag, capsys):
    from mmp_sense.mcp.server import main as mcp_main
    from mmp_sense.ui.app import parse_args

    for parse in (parse_args, mcp_main):
        with pytest.raises(SystemExit) as stopped:
            parse([flag])
        assert stopped.value.code == 2
        assert "unrecognized arguments" in capsys.readouterr().err


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
    assert session.buffer.rate == pytest.approx(8000)


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
        assert window.focus_signal == "ch0"
        kinds = {p.kind: p.signal for p in window.panes[1:]}
        assert kinds == {"spectrum": "ch0", "spectrogram": "ch0"}
    finally:
        window.close()


@needs_ffmpeg
def test_a_compressed_stereo_clip_keeps_its_channels(tmp_path):
    wav = write_wav(tmp_path / "stereo.wav", two_tones())
    clip = tmp_path / "stereo.flac"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), str(clip)], check=True)
    series = read_audio(clip)
    assert series.channels == ["ch0", "ch1"] and series.sample_rate == 48000
    assert _dominant(series.values[:, 1], 48000) == pytest.approx(1000, abs=2)
