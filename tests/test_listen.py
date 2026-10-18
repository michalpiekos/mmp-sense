"""Listening to a signal: the sound made of it, and the button that plays it."""

import numpy as np
import pandas as pd
import pytest

import mmp_sense
from mmp_sense.core.query import as_audio
from mmp_sense.core.session import Session
from mmp_sense.core.data import TimeSeries
from mmp_sense.sources.recording import SeriesSource


def _dominant(sound, rate):
    spectrum = np.abs(np.fft.rfft(sound))
    return np.fft.rfftfreq(len(sound), 1.0 / rate)[np.argmax(spectrum)]


@pytest.fixture
def tone():
    time = np.arange(16000) / 16000.0
    return mmp_sense.from_frame(pd.DataFrame({"a": 0.5 * np.sin(2 * np.pi * 1000 * time)},
                                       index=time))


# -- the sound -----------------------------------------------------------


def test_a_tone_keeps_its_pitch_and_length(tone):
    sound = tone.audio("a", rate=48000)
    assert sound.dtype == np.float32
    assert len(sound) == 48000
    assert _dominant(sound, 48000) == pytest.approx(1000, abs=1)
    assert np.abs(sound).max() == pytest.approx(0.5, abs=0.01)    # kept as it was


def test_faster_is_shorter_and_higher(tone):
    sound = tone.audio("a", rate=48000, speed=2.0)
    assert len(sound) == 24000
    assert _dominant(sound, 48000) == pytest.approx(2000, abs=2)


def test_the_stretch_rule_applies(tone):
    assert len(tone.audio("a", 0.25, 0.5, rate=16000)) == pytest.approx(4000, abs=2)
    tone.set_selection((0.0, 0.1))
    assert len(tone.audio("a", rate=16000)) == pytest.approx(1600, abs=2)


def test_sixteen_bit_counts_keep_their_level():
    values = np.array([0, 16384, -32768, 32767], dtype=np.int16)
    sound = as_audio(values, 4.0, 4)
    assert sound[1] == pytest.approx(0.5)
    assert sound.min() >= -1.0 and sound.max() <= 1.0


def test_large_values_lose_their_mean_and_fit():
    time = np.arange(1000) / 1000.0
    values = 9.81 + 3.0 * np.sin(2 * np.pi * 50 * time)       # m/s^2, with gravity
    sound = as_audio(values, 1000.0, 1000)
    assert np.abs(sound).max() == pytest.approx(0.9, abs=0.01)
    assert abs(sound.mean()) < 0.01


def test_a_gap_is_silence_and_the_ends_fade():
    values = np.full(1000, 0.5)
    values[400:600] = np.nan
    sound = as_audio(values, 1000.0, 1000)
    assert np.all(sound[400:600] == 0.0)
    assert sound[0] == 0.0 and sound[-1] < 0.5 / 4
    assert sound[500 // 2] == pytest.approx(0.5)


def test_a_recipe_and_a_second_stream_can_be_heard(tone):
    filtered = tone.audio(mmp_sense.signal("a").highpass(100.0) * 20, rate=16000)
    assert 0.5 < np.abs(filtered).max() <= 0.91          # scaled to fit, not clipped
    time = np.arange(8000) / 8000.0
    other = TimeSeries(name="/p/mic", time=time,
                       values=0.25 * np.sin(2 * np.pi * 300 * time)[:, None],
                       channels=["amplitude"], sample_rate=8000.0)
    tone.add(SeriesSource(other))
    while tone.poll():
        pass
    sound = tone.audio("mic:amplitude", rate=48000)
    assert len(sound) == 48000                               # at the mic's own rate
    assert _dominant(sound, 48000) == pytest.approx(300, abs=1)


# -- the button ---------------------------------------------------------------

pytest.importorskip("PySide6")
from PySide6.QtCore import QObject, Signal       # noqa: E402


class Speaker(QObject):
    """Stands in for the sound device the test machines do not have."""

    finished = Signal()

    def __init__(self, available=True):
        super().__init__()
        self.available = available
        self.problem = "" if available else "no sound output device"
        self.rate = 8000
        self.played = []
        self.playing = False

    def play(self, samples):
        self.played.append(np.asarray(samples))
        self.playing = True

    def stop(self, quietly=False):
        if not self.playing:
            return
        self.playing = False
        if not quietly:
            self.finished.emit()


@pytest.fixture
def ui(qapp, session):
    from mmp_sense.ui.main_window import MainWindow

    speaker = Speaker()
    window = MainWindow(session, window_s=10.0, console=False, player=speaker)
    window.show()
    window.tick()
    try:
        yield window, speaker
    finally:
        window.close()


def test_listen_plays_the_selection_and_the_cursor_runs_over_it(ui, session):
    window, speaker = ui
    session.set_selection((2.0, 3.0))
    window.toggle_sound()
    assert len(speaker.played[-1]) == pytest.approx(8000, abs=20)  # one second at 8 kHz
    assert window.hearing == session.channels[0]
    assert session.playing and session.cursor == pytest.approx(2.0)
    assert "listening to" in window.statusBar().currentMessage()
    assert window.sound_action.text() == "Stop"


def test_without_a_selection_it_plays_everything_from_the_start(ui, session):
    window, speaker = ui
    start, end = session.span
    session.set_cursor((start + end) / 2)
    window.toggle_sound()
    assert len(speaker.played[-1]) == pytest.approx((end - start) * 8000, abs=20)
    assert session.cursor == pytest.approx(start)


def test_listen_again_stops_the_sound_and_the_replay(ui, session):
    window, speaker = ui
    window.toggle_sound()
    window.toggle_sound()
    assert not speaker.playing and not session.playing
    assert window.hearing is None and window.sound_action.text() == "Listen"


def test_pausing_the_replay_stops_the_sound(ui, session):
    window, speaker = ui
    window.toggle_sound()
    window.toggle_play()                                         # Pause
    assert not speaker.playing and window.hearing is None


def test_the_sound_ending_ends_the_replay(ui, session):
    window, speaker = ui
    window.toggle_sound()
    speaker.stop()                                               # played to its end
    assert not session.playing and window.hearing is None


def test_the_replay_reaching_its_end_lets_the_sound_play_out(ui, session):
    window, speaker = ui
    session.set_selection((2.0, 2.5))
    window.toggle_sound()
    session.tick(now=session.clock() + 10.0)                     # the cursor is there first
    assert not session.playing
    assert speaker.playing and window.hearing is not None
    speaker.stop()
    assert window.hearing is None and window.sound_action.text() == "Listen"


def test_the_speed_box_applies(ui, session):
    window, speaker = ui
    session.set_selection((2.0, 4.0))
    window.speed.setCurrentText("2x")
    window.toggle_sound()
    assert len(speaker.played[-1]) == pytest.approx(8000, abs=20)
    assert session.speed == 2.0


def test_a_group_is_heard_by_its_first_member(ui, session):
    window, speaker = ui
    window.focus_signal = "channels"
    assert window.heard_signal() == session.channels[0]


def test_without_a_sound_device_listen_is_disabled(qapp, session):
    from mmp_sense.ui.main_window import MainWindow

    window = MainWindow(session, console=False, player=Speaker(available=False))
    try:
        assert not window.sound_action.isEnabled()
        assert "no sound output device" in window.sound_action.toolTip()
        window.toggle_sound()                                    # the key does nothing
        assert window.hearing is None
    finally:
        window.close()


def test_the_real_player_says_why_it_cannot_play(qapp):
    from mmp_sense.ui.audio import Player

    player = Player()
    assert player.available == (player.problem == "")
    assert player.rate > 0
    if not player.available:
        with pytest.raises(RuntimeError, match="cannot play sound"):
            player.play(np.zeros(10, np.float32))
