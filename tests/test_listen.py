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


# -- the amplifier -------------------------------------------------------


def _db(sound):
    return 20 * np.log10(np.sqrt(np.mean(np.asarray(sound, dtype=float) ** 2)))


def _sine(amplitude, rate=8000, seconds=2.0, hz=440.0):
    time = np.arange(int(rate * seconds)) / rate
    return amplitude * np.sin(2 * np.pi * hz * time)


def test_auto_brings_a_faint_tone_to_a_comfortable_level():
    sound = as_audio(_sine(1.41e-3), 8000.0, 8000, amp="auto")
    assert _db(sound) == pytest.approx(-20.0, abs=1.0)
    assert len(sound) == 16000
    assert _dominant(sound, 8000) == pytest.approx(440, abs=1)


def test_a_fixed_gain_is_that_gain():
    plain, raised = as_audio(_sine(1e-3), 8000.0, 8000), as_audio(_sine(1e-3), 8000.0, 8000, amp=24)
    assert _db(raised) - _db(plain) == pytest.approx(24.0, abs=0.1)


def test_a_knock_does_not_keep_the_noise_down():
    noise = 1e-3 * np.random.default_rng(1).standard_normal(80000)
    knock = noise.copy()
    knock[40000:40040] = 1.0
    sound = as_audio(knock, 8000.0, 8000, amp="auto")
    assert _db(sound[:30000]) == pytest.approx(-20.0, abs=2.0)
    assert np.abs(sound).max() <= 0.9 + 1e-6
    assert np.abs(sound[39000:41000]).max() > 0.5                # the knock is still there


def test_too_much_gain_is_limited_not_clipped():
    sound = as_audio(_sine(0.5), 8000.0, 8000, amp=40)
    assert np.abs(sound).max() <= 0.9 + 1e-6
    middle = sound[4000:12000]
    spectrum = np.abs(np.fft.rfft(middle * np.hanning(len(middle))))
    frequency = np.fft.rfftfreq(len(middle), 1 / 8000)
    fundamental = spectrum[np.abs(frequency - 440) < 10].max()
    harmonics = spectrum[frequency > 600].max()
    assert 20 * np.log10(harmonics / fundamental) < -40


def test_silence_stays_silent():
    assert not as_audio(np.zeros(8000), 8000.0, 8000, amp="auto").any()


def test_a_gap_stays_silent():
    values = _sine(1e-3)
    values[4000:6000] = np.nan
    sound = as_audio(values, 8000.0, 8000, amp="auto")
    assert not sound[4400:5600].any()
    assert _db(sound[:3000]) == pytest.approx(-20.0, abs=1.5)


def test_an_offset_does_not_hold_the_gain_down():
    offset, plain = as_audio(_sine(1e-3) + 0.5, 8000.0, 8000, amp="auto"), as_audio(_sine(1e-3), 8000.0, 8000, amp="auto")
    assert _db(offset) == pytest.approx(_db(plain), abs=0.5)


def test_a_loud_recording_is_not_turned_down_by_auto():
    plain, auto = as_audio(_sine(0.8), 8000.0, 8000), as_audio(_sine(0.8), 8000.0, 8000, amp="auto")
    assert _db(auto) == pytest.approx(_db(plain), abs=0.5)


def test_auto_follows_speed_and_reaches_bands():
    sound = as_audio(_sine(1.41e-3), 8000.0, 16000, speed=2.0, bands=[(300, 600)], amp="auto")
    assert _db(sound) == pytest.approx(-20.0, abs=1.0)
    assert _dominant(sound, 16000) == pytest.approx(880, abs=2)


@pytest.mark.parametrize("amp", [-1, 61, "loud", True])
def test_a_bad_amp_is_refused(amp):
    with pytest.raises(ValueError, match="amp is"):
        as_audio(_sine(0.1), 8000.0, 8000, amp=amp)


def test_session_audio_takes_amp(tone):
    assert _db(tone.audio("a", rate=8000, amp=12)) > _db(tone.audio("a", rate=8000))


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


def test_play_sounds_the_selection_and_the_cursor_runs_over_it(ui, session):
    window, speaker = ui
    session.set_selection((2.0, 3.0))
    session.set_cursor(9.0)
    window.toggle_play()
    assert len(speaker.played[-1]) == pytest.approx(8000, abs=20)  # one second at 8 kHz
    assert window.hearing == session.channels[0]
    assert session.playing and session.cursor == pytest.approx(2.0)
    assert "playing" in window.statusBar().currentMessage()
    assert "amp auto" in window.statusBar().currentMessage()
    assert window.play_action.text() == "Pause"


def test_inside_the_selection_it_starts_at_the_cursor(ui, session):
    window, speaker = ui
    session.set_selection((2.0, 4.0))
    session.set_cursor(3.5)
    window.toggle_play()
    assert len(speaker.played[-1]) == pytest.approx(4000, abs=20)
    assert session.cursor == pytest.approx(3.5)


def test_without_a_selection_it_starts_at_the_cursor(ui, session):
    window, speaker = ui
    start, end = session.span
    session.set_cursor(4.0)
    window.toggle_play()
    assert len(speaker.played[-1]) == pytest.approx((end - 4.0) * 8000, abs=20)
    assert session.cursor == pytest.approx(4.0)


def test_from_the_end_it_starts_again(ui, session):
    window, speaker = ui
    start, end = session.span
    session.set_cursor(end)
    window.toggle_play()
    assert len(speaker.played[-1]) == pytest.approx((end - start) * 8000, abs=20)
    assert session.cursor == pytest.approx(start)


@pytest.mark.parametrize("amp", ["amp off", "amp auto"])
def test_the_sound_at_the_cursor_is_the_sound_of_that_moment(ui, session, amp):
    window, speaker = ui
    window.amp.setCurrentText(amp)
    session.set_selection((1.0, 3.0))
    window.toggle_play()
    whole = speaker.played[-1]
    window.toggle_play()
    session.set_cursor(2.0)
    window.toggle_play()
    part = speaker.played[-1]
    fade = 100
    np.testing.assert_allclose(part[fade:-fade], whole[len(whole) - len(part) + fade:-fade],
                               atol=1e-6)


def test_pause_then_play_resumes(ui, session):
    window, speaker = ui
    session.set_cursor(4.0)
    window.toggle_play()
    window.toggle_play()
    assert not speaker.playing and not session.playing
    assert window.hearing is None and window.play_action.text() == "Play"
    session.set_cursor(6.0)
    window.toggle_play()
    assert session.cursor == pytest.approx(6.0)
    assert window.play_action.text() == "Pause"


def test_pausing_from_the_session_stops_the_sound(ui, session):
    window, speaker = ui
    window.toggle_play()
    session.pause()
    assert not speaker.playing and window.hearing is None


def test_the_sound_ending_ends_the_replay(ui, session):
    window, speaker = ui
    window.toggle_play()
    speaker.stop()                                               # played to its end
    assert not session.playing and window.hearing is None
    assert window.play_action.text() == "Play"


def test_the_replay_reaching_its_end_lets_the_sound_play_out(ui, session):
    window, speaker = ui
    session.set_selection((2.0, 2.5))
    window.toggle_play()
    session.tick(now=session.clock() + 10.0)                     # the cursor is there first
    assert not session.playing
    assert speaker.playing and window.hearing is not None
    assert window.play_action.text() == "Pause"
    window.toggle_play()
    assert window.hearing is None and window.play_action.text() == "Play"


def test_the_speed_box_applies(ui, session):
    window, speaker = ui
    session.set_selection((2.0, 4.0))
    session.set_cursor(2.0)
    window.speed.setCurrentText("2x")
    window.toggle_play()
    assert len(speaker.played[-1]) == pytest.approx(8000, abs=20)
    assert session.speed == 2.0


def test_a_group_is_heard_by_its_first_member(ui, session):
    window, speaker = ui
    window.focus_signal = "channels"
    assert window.heard_signal() == session.channels[0]


def test_listen_and_its_key_are_gone(ui):
    window, _speaker = ui
    assert not hasattr(window, "sound_action")
    assert not hasattr(window, "toggle_sound")


def test_without_a_sound_device_play_runs_silently(qapp, session):
    from mmp_sense.ui.main_window import MainWindow

    window = MainWindow(session, console=False, player=Speaker(available=False))
    try:
        session.set_cursor(4.0)
        window.toggle_play()
        assert session.playing and session.cursor == pytest.approx(4.0)
        assert window.hearing is None
        assert window.silent_because == "no sound output device"
        assert "silent: no sound output device" in window.statusBar().currentMessage()
        window.toggle_play()
        assert not session.playing
    finally:
        window.close()


def test_a_signal_that_cannot_sound_still_plays(ui, session, monkeypatch):
    window, speaker = ui
    monkeypatch.setattr(session, "audio", lambda *a, **k: (_ for _ in ()).throw(ValueError("no")))
    window.toggle_play()
    assert session.playing and window.hearing is None
    assert "cannot play" in window.silent_because


def test_the_amp_box_starts_on_auto_and_reaches_the_sound(ui, session, monkeypatch):
    window, speaker = ui
    asked = []
    real = session.audio
    monkeypatch.setattr(session, "audio", lambda *a, **k: asked.append(k["amp"]) or real(*a, **k))
    window.toggle_play()
    window.toggle_play()
    window.amp.setCurrentText("amp +24 dB")
    window.toggle_play()
    assert asked == ["auto", 24.0]


def test_changing_the_amp_restarts_the_sound_at_the_cursor(ui, session, clock):
    window, speaker = ui
    window.amp.setCurrentText("amp off")
    session.set_cursor(4.0)
    window.toggle_play()
    plays = len(speaker.played)
    clock.advance(1.0)
    session.tick()
    window.amp.setCurrentText("amp +12 dB")
    assert len(speaker.played) == plays + 1
    assert speaker.playing and session.playing
    assert session.cursor == pytest.approx(5.0)
    assert len(speaker.played[-1]) == pytest.approx((session.span[1] - 5.0) * 8000, abs=20)


def test_changing_the_amp_while_paused_waits_for_play(ui, session):
    window, speaker = ui
    window.toggle_play()
    window.toggle_play()
    plays = len(speaker.played)
    window.amp.setCurrentText("amp +12 dB")
    assert len(speaker.played) == plays and not session.playing


def test_sound_starts_at_the_cursor_on_a_later_stream(qapp):
    from mmp_sense.ui.main_window import MainWindow

    rate = 1000.0
    early, late = np.arange(4000) / rate, 2.0 + np.arange(2000) / rate
    session = Session(SeriesSource(TimeSeries(time=early, values=np.zeros(4000),
                                              channels=["v"], sample_rate=rate), name="a")).start()
    session.add(SeriesSource(TimeSeries(time=late, values=np.ones(2000) * 0.5, channels=["w"],
                                        sample_rate=rate), name="b"), "b")
    session.poll()
    session.stop_intake()
    window = MainWindow(session, console=False, player=Speaker())
    try:
        window.focus_signal = "b:w"
        window.amp.setCurrentText("amp off")
        session.set_cursor(3.0)
        window.toggle_play()
        assert len(window.player.played[-1]) == pytest.approx(8000, abs=40)
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
