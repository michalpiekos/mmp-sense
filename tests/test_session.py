"""The shared model: what it holds, what it tells, and what it refuses."""

import numpy as np
import pandas as pd
import pytest

import mmp_sense
from mmp_sense import Session, SyntheticAccelerometer


def twice():
    return (mmp_sense.signal("accel_x") * 2).named("twice")


def test_polling_fills_the_buffer_and_follows_the_newest(session, clock, source):
    assert len(session.buffer) == 5000
    assert session.cursor == pytest.approx(session.buffer.t_end)
    clock.advance(1.0)
    assert session.poll() == 500
    assert session.cursor == pytest.approx(session.buffer.t_end)


def test_moving_the_cursor_stops_it_following(session, clock):
    session.set_cursor(3.0)
    assert session.cursor == 3.0 and not session.following
    clock.advance(1.0)
    session.poll()
    assert session.cursor == 3.0
    session.follow_live()
    assert session.following and session.cursor == pytest.approx(session.buffer.t_end)


def test_the_cursor_stays_inside_what_is_held(session):
    session.set_cursor(-100.0)
    assert session.cursor == pytest.approx(session.buffer.t_start)
    session.set_cursor(1e6)
    assert session.cursor == pytest.approx(session.buffer.t_end)


def test_stepping_by_sample_window_and_second(session):
    session.set_cursor(5.0)
    session.step_cursor(1, unit="sample")
    assert session.cursor == pytest.approx(5.002)
    session.step_cursor(-1, unit="window")
    assert session.cursor == pytest.approx(5.002 - session.analysis_window)
    session.step_cursor(2, unit="second")
    assert session.cursor == pytest.approx(6.502, abs=1e-6)
    with pytest.raises(ValueError, match="unknown unit"):
        session.step_cursor(1, unit="furlong")


def test_a_selection_is_ordered_and_clearable(session):
    session.set_selection((6.0, 4.0))
    assert session.selection == (4.0, 6.0)
    session.clear_selection()
    assert session.selection is None


def test_the_analysis_window_must_be_positive(session):
    session.analysis_window = 0.25
    assert session.analysis_window == 0.25
    with pytest.raises(ValueError, match="must be positive"):
        session.analysis_window = 0.0


def test_listeners_hear_what_changed(session):
    heard = []
    stop = session.listen(lambda kind, payload: heard.append((kind, payload)))
    session.set_cursor(2.0)
    session.set_selection((1.0, 2.0))
    session.analysis_window = 0.1
    session.show(twice(), plot=False)
    session.plot("twice")
    kinds = [kind for kind, _ in heard]
    assert kinds == ["cursor", "selection", "analysis", "signals", "plot"]
    stop()
    session.set_cursor(3.0)
    assert len(heard) == 5


def test_a_broken_listener_does_not_stop_the_session(session):
    def explode(kind, payload):
        raise RuntimeError("no")

    session.listen(explode)
    session.set_cursor(1.0)          # must not raise
    assert session.cursor == 1.0


def test_playback_moves_the_cursor_and_stops_at_the_end(session, clock):
    session.play(start=1.0, speed=2.0, until=2.0)
    assert session.playing and not session.following
    clock.advance(0.25)
    session.tick()
    assert session.cursor == pytest.approx(1.5)
    clock.advance(0.5)
    session.tick()
    assert session.cursor == pytest.approx(2.0)
    assert not session.playing


def test_playback_defaults_to_the_selection(session, clock):
    session.set_selection((3.0, 3.5))
    session.play()
    assert session.cursor == pytest.approx(3.0)
    clock.advance(1.0)
    session.tick()
    assert session.cursor == pytest.approx(3.5) and not session.playing


def test_pausing_and_speed_are_refused_when_silly(session):
    with pytest.raises(ValueError, match="speed must be positive"):
        session.play(speed=0)
    session.pause()                  # pausing when stopped is not an error


def test_plot_marks_a_signal_active_and_refuses_an_unknown_one(session):
    session.show(twice(), plot=False)
    session.plot("twice")
    assert "twice" in session.active_signals
    with pytest.raises(Exception, match="no signal"):
        session.plot("nothing")


def test_the_context_says_what_is_being_looked_at(session):
    session.set_cursor(5.0)
    session.set_selection((4.9, 5.1))
    session.analysis_window = 0.2
    context = session.context
    assert context.cursor == 5.0
    assert context.selection == (4.9, 5.1)
    assert context.analysis_window == 0.2
    assert context.dataset == "synthetic"
    assert context.rate == 500.0
    as_dict = context.to_dict()
    assert as_dict["selection"] == [4.9, 5.1]
    assert set(as_dict) >= {
        "cursor", "selection", "active_signals", "analysis_window", "dataset"
    }


def test_attaching_another_source_keeps_the_recipes(session, clock):
    session.show(twice())
    other = SyntheticAccelerometer(rate=100.0, knock_interval=0.0, clock=clock)
    session.attach(other)
    clock.advance(1.0)
    session.poll()
    assert "twice" in session.signals
    assert session.rate == 100.0
    assert len(session.buffer) == 100


def test_the_session_can_be_used_without_a_source():
    session = Session(channels=["a", "b"], rate=50.0)
    assert session.poll() == 0
    assert session.window().empty
    assert session.span == (0.0, 0.0)


def test_stopping_the_intake_keeps_everything_it_took(session, clock, source):
    session.show(twice())
    session.set_cursor(5.0)
    session.set_selection((4.9, 5.1))
    held = len(session.buffer)

    session.stop_intake()
    assert not session.running
    clock.advance(5.0)
    assert session.poll() == 0

    assert len(session.buffer) == held
    assert session.cursor == 5.0
    assert session.selection == (4.9, 5.1)
    assert "twice" in session.signals
    assert len(session.values("twice", *session.span)) == held
    assert session.stats("accel_x", 4.9, 5.1)["n"] > 0


def test_a_stopped_session_can_still_replay_what_it_holds(session, clock):
    session.stop_intake()
    session.play(start=1.0, until=2.0, speed=1.0)
    clock.advance(0.5)
    session.tick()
    assert session.cursor == pytest.approx(1.5)


def test_starting_the_intake_begins_from_an_empty_buffer(session, clock):
    session.show(twice())
    session.set_selection((4.9, 5.1))
    session.stop_intake()

    session.start_intake()
    assert session.running
    assert len(session.buffer) == 0
    assert session.cursor == 0.0
    assert session.selection is None
    assert session.following
    # The recipe survives and computes over the new samples.
    assert "twice" in session.signals

    clock.advance(2.0)
    assert session.poll() == 1000
    assert session.buffer.t_start == 0.0
    assert len(session.values("twice")) == 1000


def test_starting_while_running_restarts_cleanly(session, clock):
    assert session.running and len(session.buffer) == 5000
    session.start_intake()
    assert len(session.buffer) == 0
    clock.advance(1.0)
    assert session.poll() == 500


def test_starting_the_intake_stops_playback(session, clock):
    session.play(start=1.0)
    assert session.playing
    session.start_intake()
    assert not session.playing


def test_the_intake_state_is_told_and_is_in_the_context(session):
    heard = []
    session.listen(lambda kind, payload: heard.append((kind, payload)))

    session.stop_intake()
    assert ("intake", {"running": False}) in heard
    assert session.context.running is False

    session.start_intake()
    assert ("intake", {"running": True}) in heard
    assert session.context.running is True


def test_stopping_the_intake_twice_says_nothing_the_second_time(session):
    heard = []
    session.listen(lambda kind, payload: heard.append(kind))
    session.stop_intake()
    session.stop_intake()
    assert heard.count("intake") == 1


def test_a_session_with_no_source_is_not_running():
    session = Session(channels=["a"], rate=10.0)
    assert not session.running
    session.buffer.append(np.arange(10) / 10.0, np.arange(10.0)[:, None])
    assert len(session.buffer) == 10


def test_a_source_that_cannot_restart_leaves_the_last_capture(session, clock):
    class Stubborn(SyntheticAccelerometer):
        def restart(self):
            raise ConnectionError("the device went away")

    held = len(session.buffer)
    session.source = Stubborn(rate=500.0, clock=clock)
    with pytest.raises(ConnectionError):
        session.start_intake()
    assert len(session.buffer) == held


# -- asking for numbers ----------------------------------------------------


def test_the_time_rule_is_the_same_for_every_question(session):
    start, end = session.span
    everything = session.values("accel_x")
    assert everything.index[0] == pytest.approx(start)
    assert everything.index[-1] == pytest.approx(end)
    session.set_selection((4.0, 4.5))
    assert session.values("accel_x").index[[0, -1]].tolist() == pytest.approx([4.0, 4.5])
    assert session.values("accel_x", 1.0, 2.0).index[[0, -1]].tolist() == pytest.approx([1.0, 2.0])
    assert session.values("accel_x", 9.0).index[-1] == pytest.approx(end)
    assert session.stats("accel_x")["n"] == len(session.values("accel_x"))
    assert session.spectrum_summary("accel_x")["samples"] == len(session.values("accel_x"))
    knock = (mmp_sense.signal().highpass(10).magnitude().peak("20ms") > 1).events(name="knock")
    assert len(session.events(knock)) == 0              # the knock is at 5 s
    session.clear_selection()
    assert len(session.events(knock)) == 1


def test_values_are_pandas_indexed_by_time(session):
    one = session.values("accel_z", 1.0, 1.1)
    assert isinstance(one, pd.Series) and one.index.name == "time"
    several = session.values(mmp_sense.signal(), 1.0, 1.1)
    assert isinstance(several, pd.DataFrame)
    assert list(several.columns) == ["accel_x", "accel_y", "accel_z"]
    frames = session.values(mmp_sense.signal("accel_z").frames("200ms", "50ms"), 1.0, 2.0)
    assert frames.columns.name == "frequency_hz"
    assert frames.columns[1] == pytest.approx(5.0)
    assert np.diff(frames.index).mean() == pytest.approx(0.05)
    with pytest.raises(TypeError, match="events"):
        session.values((mmp_sense.signal("accel_z") > 0).events())


def test_a_recipe_is_computed_over_everything_then_cut(session):
    """A filter has its history at the start of a short stretch."""
    hp = mmp_sense.signal("accel_z").highpass(1.0)
    cut = session.values(hp, 4.0, 4.1)
    assert abs(cut.iloc[0]) < 0.5                  # settled, not starting from 9.8
    alone = mmp_sense.from_frame(session.values("accel_z", 4.0, 4.1).to_frame())
    assert alone.values(hp).iloc[0] == 0.0         # a filter started there starts at rest


def test_events_are_a_table_of_start_end_and_duration(session):
    knock = (mmp_sense.signal().highpass(10).magnitude().peak("20ms") > 1).events(name="knock")
    table = session.events(knock)
    assert list(table.columns[:3]) == ["start", "end", "duration"]
    assert table["start"].iloc[0] == pytest.approx(5.0, abs=0.1)
    with pytest.raises(TypeError, match="values"):
        session.events("accel_x")


def test_stats_spectrum_and_summary(session):
    stats = session.stats("accel_z", 1.0, 2.0)
    assert set(stats.index) == {"n", "min", "max", "mean", "rms", "std", "peak_to_peak"}
    assert stats["mean"] == pytest.approx(9.81, abs=0.1)
    assert list(session.stats(mmp_sense.signal(), 1.0, 2.0).columns) == ["accel_x", "accel_y", "accel_z"]
    spectrum = session.spectrum("accel_z", 1.0, 1.999)
    assert spectrum.index.name == "frequency_hz"
    assert spectrum.index[1] == pytest.approx(1.0)
    padded = session.spectrum("accel_z", 1.0, 1.999, pad_to=2000)
    assert padded.index[1] == pytest.approx(0.25)
    summary = session.spectrum_summary("accel_z", 1.0, 1.999, bands=[(0, 50), (50, 250)])
    assert summary["resolution_hz"] == pytest.approx(1.0)
    assert summary["share 0-50 Hz"] + summary["share 50-250 Hz"] == pytest.approx(1.0, abs=0.02)
    with pytest.raises(ValueError, match="one signal"):
        session.spectrum(mmp_sense.signal(), 1.0, 2.0)


def test_a_step_of_a_shown_recipe_can_be_asked_for_by_name(session):
    strength = mmp_sense.signal().highpass(10).magnitude().peak("20ms", name="strength")
    knock = (strength > 1).events(name="knock")
    session.show(knock.repeated(2, gap=(0, "1s"), name="pair"))
    assert len(session.values("strength")) == len(session.buffer)
    assert len(session.events("knock")) == 1
    with pytest.raises(mmp_sense.SignalError, match="nothing"):
        session.values("nothing")
