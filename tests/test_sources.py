"""Where samples come from: simulated and recorded."""

import numpy as np
import pytest

import mmp_sense
from mmp_sense import SeriesSource, Session, SyntheticAccelerometer
from mmp_sense.core.data import TimeSeries


def test_the_simulator_produces_what_the_clock_allows(clock):
    source = SyntheticAccelerometer(rate=200.0, knock_interval=0.0, clock=clock).start()
    assert source.read()[0].size == 0
    clock.advance(0.5)
    time, values = source.read()
    assert len(time) == 100
    assert values.shape == (100, 3)
    assert time[0] == 0.0 and time[-1] == pytest.approx(0.495)
    # Samples are produced once, not again.
    assert source.read()[0].size == 0


def test_a_knock_lands_where_it_was_asked_for(clock):
    source = SyntheticAccelerometer(
        rate=500.0, knock_interval=0.0, noise=0.0, clock=clock
    ).start()
    source.knock(at=1.0, amplitude=5.0)
    clock.advance(2.0)
    time, values = source.read()
    magnitude = np.sqrt((values**2).sum(axis=1))
    peak = time[int(np.argmax(np.abs(magnitude - 9.81)))]
    assert peak == pytest.approx(1.0, abs=0.02)
    assert source.event_spans() == [(pytest.approx(0.92), pytest.approx(1.08))]


def test_knocks_arrive_on_their_own_when_an_interval_is_set(clock):
    source = SyntheticAccelerometer(rate=100.0, knock_interval=1.0, clock=clock).start()
    clock.advance(5.0)
    source.read()
    assert len(source.knocks) >= 3


def test_a_series_source_hands_over_everything_at_once():
    time = np.arange(100) / 100.0
    series = TimeSeries(name="run", time=time, values=time[:, None], channels=["v"])
    source = SeriesSource(series).start()
    read_time, values = source.read()
    assert len(read_time) == 100 and source.finished
    assert source.read()[0].size == 0


def test_a_series_source_replays_against_the_clock(clock):
    time = np.arange(100) / 100.0
    series = TimeSeries(name="run", time=time, values=time[:, None], channels=["v"])
    source = SeriesSource(series, replay=True, clock=clock).start()
    clock.advance(0.5)
    # Everything up to and including half a second in has arrived.
    assert len(source.read()[0]) == 51
    clock.advance(0.5)
    assert len(source.read()[0]) == 49
    assert source.finished
    source.rewind()
    clock.advance(1.0)
    assert len(source.read()[0]) == 100


def test_a_recording_reaches_the_session_like_a_stream(clock):
    time = np.arange(500) / 500.0
    values = np.stack([np.sin(2 * np.pi * 10 * time)] * 3, axis=1)
    series = TimeSeries(
        name="bench", time=time, values=values,
        channels=["accel_x", "accel_y", "accel_z"], units="m/s^2",
    )
    session = Session(SeriesSource(series), buffer_s=10.0)
    session.poll()
    assert session.dataset == "bench"
    assert session.rate == pytest.approx(500.0, rel=1e-3)
    magnitude = mmp_sense.signal("accel_x", "accel_y", "accel_z").magnitude()
    assert session.stats(magnitude, 0.0, 1.0)["max"] == pytest.approx(
        np.sqrt(3.0), rel=1e-2
    )


def test_a_restarted_simulator_produces_from_zero_again(clock):
    """The sample count must not survive: it would outrun the new clock."""
    source = SyntheticAccelerometer(rate=200.0, knock_interval=0.0, clock=clock).start()
    clock.advance(2.0)
    assert len(source.read()[0]) == 400
    source.stop()
    clock.advance(5.0)

    source.restart()
    clock.advance(0.5)
    time, values = source.read()
    assert len(time) == 100
    assert time[0] == 0.0 and values.shape == (100, 3)


def test_a_restarted_simulator_forgets_the_knocks_it_had(clock):
    source = SyntheticAccelerometer(
        rate=500.0, knock_interval=0.0, noise=0.0, clock=clock
    ).start()
    source.knock(at=0.2, amplitude=5.0)
    clock.advance(1.0)
    source.read()

    source.restart()
    clock.advance(1.0)
    _time, values = source.read()
    # The old knock was at 0.2, which the new run covers; nothing of it
    # may reappear there.
    assert source.knocks == []
    assert np.abs(np.sqrt((values**2).sum(axis=1)) - 9.81).max() < 0.1


def test_a_restarted_recording_plays_from_its_first_sample(clock):
    time = np.arange(10) / 10.0
    series = TimeSeries(name="ramp", time=time, values=time[:, None], channels=["v"])
    source = SeriesSource(series, replay=True, clock=clock).start()
    clock.advance(1.0)
    assert len(source.read()[0]) == 10
    assert source.finished

    source.restart()
    clock.advance(1.0)
    read_time, values = source.read()
    assert len(read_time) == 10 and values[0, 0] == 0.0


def test_restarting_a_plain_source_is_a_stop_and_a_start():
    from mmp_sense.sources.base import Source

    class Counting(Source):
        def __init__(self):
            self.log = []

        def start(self):
            self.log.append("start")
            return self

        def stop(self):
            self.log.append("stop")

    source = Counting()
    assert source.restart() is source
    assert source.log == ["stop", "start"]
