"""Microphone arrays: direction, position, the power map and a beam."""

import numpy as np
import pandas as pd
import pytest

import mmp_sense
from mmp_sense import Array
from mmp_sense.examples.array import RATE, far, near, square
from mmp_sense.sources.recording import SeriesSource
from mmp_sense.core.data import TimeSeries
from mmp_sense.core.session import Session

RING = square()


@pytest.mark.parametrize("azimuth", [30.0, 135.0, 250.0])
def test_a_far_source_is_found_within_three_degrees(azimuth):
    session = mmp_sense.from_frame(far(RING, azimuth))
    found = session.direction(RING)
    error = (found["azimuth_deg"] - azimuth + 180) % 360 - 180
    assert abs(error) < 3
    assert found["residual_s"] < 1 / RATE


def test_the_sign_is_right_by_an_independent_construction():
    # Noise heard 5 samples earlier by the microphone at +x than by the one
    # at the origin, built by shifting, not by the solver's own steering.
    rng = np.random.default_rng(2)
    sound = rng.standard_normal(16005)
    pair = Array({"near": (0.0, 0.0), "far": (-5 / RATE * 343.0, 0.0)})
    time = np.arange(16000) / RATE
    table = pd.DataFrame({"near": sound[5:], "far": sound[:-5]}, index=time)
    session = mmp_sense.from_frame(table)
    assert session.delay("near", "far")["delay"] == pytest.approx(5 / RATE, abs=0.1 / RATE)
    ring = Array({"a": (0.0, 0.0), "b": (-5 / RATE * 343.0, 0.0), "c": (0.0, 0.3)})
    table["c"] = sound[5:]                          # level with a along y: no delay
    found = mmp_sense.from_frame(table.rename(columns={"near": "a", "far": "b"})).direction(ring)
    assert abs((found["azimuth_deg"] + 180) % 360 - 180) < 3     # toward +x


def test_a_flat_array_given_in_three_dimensions_tells_the_elevation():
    flat = Array({name: (*xy, 0.0) for name, xy in zip(RING.names, RING.points.tolist())})
    session = mmp_sense.from_frame(far(flat, 60.0, 40.0))
    found = session.direction(flat)
    assert found["azimuth_deg"] == pytest.approx(60.0, abs=3)
    assert found["elevation_deg"] == pytest.approx(40.0, abs=5)


def test_three_dimensions_give_the_elevation():
    cube = Array({"a": (0, 0, 0), "b": (0.3, 0, 0), "c": (0, 0.3, 0), "d": (0, 0, 0.3),
                  "e": (0.3, 0.3, 0.3)})
    session = mmp_sense.from_frame(far(cube, 70.0, 25.0))
    found = session.direction(cube)
    assert found["azimuth_deg"] == pytest.approx(70.0, abs=3)
    assert found["elevation_deg"] == pytest.approx(25.0, abs=5)


def test_two_microphones_give_the_angle_to_their_axis():
    pair = Array({"a": (0, 0), "b": (0.3, 0)})
    session = mmp_sense.from_frame(far(pair, 60.0))
    found = session.direction(pair)
    assert np.isnan(found["azimuth_deg"])
    assert found["axis_angle_deg"] == pytest.approx(60.0, abs=3)


def test_a_near_source_is_located_within_five_centimetres():
    wide = Array({"m0": (0, 0), "m1": (2, 0), "m2": (2, 2), "m3": (0, 2)})
    session = mmp_sense.from_frame(near(wide, (1.2, 0.7)))
    point = session.locate(wide)
    assert np.hypot(point["x"] - 1.2, point["y"] - 0.7) < 0.05


def test_too_few_microphones_to_locate_say_so():
    pair = Array({"a": (0, 0), "b": (1, 0)})
    session = mmp_sense.from_frame(near(pair, (0.3, 0.4)))
    assert np.isnan(session.locate(pair)["x"])


def test_the_map_peaks_at_the_source():
    session = mmp_sense.from_frame(far(RING, 200.0))
    power = session.direction_map(RING, step_deg=1)
    assert power.idxmax() == pytest.approx(200.0, abs=3)
    assert power.max() > 2 * power.median()


def _snr_db(source_only, noise_only, recipe):
    signal = mmp_sense.from_frame(source_only).values(recipe).iloc[2000:]
    noise = mmp_sense.from_frame(noise_only).values(recipe).iloc[2000:]
    return 10 * np.log10(signal.var() / noise.var())


def test_a_beam_toward_the_source_hears_it_better_than_one_microphone():
    source = far(RING, 135.0, hiss=0.0)
    hiss = far(RING, 135.0, hiss=0.3) - source
    one = _snr_db(source, hiss, mmp_sense.signal("m0"))
    toward = _snr_db(source, hiss, mmp_sense.beam(RING, 135.0))
    away = mmp_sense.from_frame(source).values(mmp_sense.beam(RING, 315.0)).iloc[2000:].var()
    assert toward - one > 4.0
    assert mmp_sense.from_frame(source).values(mmp_sense.beam(RING, 135.0)).iloc[2000:].var() > away


@pytest.mark.parametrize("azimuth", [0.0, 210.0])
def test_a_beam_is_late_by_what_it_declares(azimuth):
    recipe = mmp_sense.beam(RING, azimuth)
    table = far(RING, azimuth, hiss=0.0)
    session = mmp_sense.from_frame(table)
    earliest = RING.names[int(np.argmin(mmp_sense.dsp.steering(RING.points, RING.speed, azimuth)))]
    lag = session.delay(earliest, recipe, 0.2, 1.8, max_lag=0.005)["delay"]
    assert lag == pytest.approx(recipe.latency, abs=1 / RATE)


def test_a_beam_wants_eight_kilohertz():
    time = np.arange(2000) / 4000.0
    slow = mmp_sense.from_frame(pd.DataFrame({n: np.zeros(2000) for n in RING.names}, index=time))
    with pytest.raises(Exception, match="8000 samples a second"):
        slow.values(mmp_sense.beam(RING, 0.0))


def test_a_beam_over_a_gap_carried_on_is_the_whole():
    _beam_live_equals_whole(gap=True)


def test_a_beam_reads_one_stream():
    a = TimeSeries(name="/p/m", time=np.arange(100) / RATE, values=np.zeros((100, 1)),
                   channels=["m0"], sample_rate=RATE)
    b = TimeSeries(name="/q/m", time=np.arange(100) / RATE, values=np.zeros((100, 1)),
                   channels=["m1"], sample_rate=RATE)
    session = Session()
    session.add(SeriesSource(a))
    session.add(SeriesSource(b))
    while session.poll():
        pass
    pair = Array({"m:m0": (0, 0), "q_m:m1": (0.3, 0)})
    with pytest.raises(Exception, match="several streams"):
        session.values(mmp_sense.beam(pair, 0.0))


def test_the_direction_holds_across_two_synchronized_streams():
    table = far(RING, 120.0, seconds=6.0)
    # Two phones, two microphones each, the second phone's clock 20 ms late.
    first = TimeSeries(name="/a/mic", time=table.index.to_numpy(), channels=["m0", "m1"],
                       values=table[["m0", "m1"]].to_numpy(), sample_rate=RATE)
    second = TimeSeries(name="/b/mic", time=table.index.to_numpy() + 0.020,
                        channels=["m2", "m3"], values=table[["m2", "m3"]].to_numpy(),
                        sample_rate=RATE)
    session = Session()
    session.add(SeriesSource(first))
    session.add(SeriesSource(second))
    while session.poll():
        pass
    session.set_clock("b_mic", 0.020)
    split = Array({"mic:m0": RING.points[0], "mic:m1": RING.points[1],
                   "b_mic:m2": RING.points[2], "b_mic:m3": RING.points[3]})
    found = session.direction(split, 1.0, 5.0)
    assert found["azimuth_deg"] == pytest.approx(120.0, abs=3)


def test_the_beam_carried_on_live_is_the_whole():
    _beam_live_equals_whole(gap=False)


def _beam_live_equals_whole(gap):
    from mmp_sense.core.buffer import ChannelBuffer
    from mmp_sense.core.graph import Evaluator
    from mmp_sense.core.signals import SignalTable

    table = far(RING, 45.0, seconds=3.0)
    time, values = table.index.to_numpy(), table.to_numpy()
    if gap:
        values = values.copy()
        values[20000:20100, 2] = np.nan
    live = SignalTable(ChannelBuffer(list(table.columns), span_s=1.0, rate=RATE,
                                     max_capacity=int(RATE)))
    whole = SignalTable(ChannelBuffer(list(table.columns), span_s=1e6, rate=RATE))
    recipe = mmp_sense.beam(RING, 45.0).envelope(200)
    evaluator = Evaluator()
    rng = np.random.default_rng(0)
    at = 0
    while at < len(time):
        n = int(rng.integers(1, 2000))
        for t in (live, whole):
            t.buffer.append(time[at:at + n], values[at:at + n])
        at += n
        window = live.window()
        got = evaluator.evaluate(recipe, window)
        reference = whole.window()
        reference.live = None
        expected = Evaluator().evaluate(recipe, reference)
        keep = expected.time >= window.t_start
        np.testing.assert_allclose(got.values, expected.values[keep], atol=1e-9,
                                   equal_nan=True)


def test_the_direction_pane_points_at_the_source(qapp):
    pytest.importorskip("pyqtgraph")
    from mmp_sense.ui.main_window import MainWindow

    session = mmp_sense.from_frame(far(RING, 300.0))
    session.add_array("ring", RING)
    window = MainWindow(session, console=False)
    try:
        pane = window.add_pane("direction", "ring")
        session.set_selection((0.2, 1.8))
        pane.refresh({"selection"})
        assert pane.azimuth == pytest.approx(300.0, abs=3)
        assert "azimuth" in pane.status.text()
    finally:
        window.close()


def test_an_array_needs_its_signals_and_one_kind_of_position():
    with pytest.raises(ValueError):
        Array({"a": (0, 0)})
    with pytest.raises(ValueError, match="one place"):
        Array({"a": (0, 0), "b": (0, 0)})
    with pytest.raises(ValueError, match="metres"):
        Array({"a": (0, 0), "b": (np.nan, 0)})
    with pytest.raises(ValueError):
        Array({"a": (0, 0), "b": (1, 0, 0)})
    session = mmp_sense.from_frame(far(RING, 0.0, seconds=0.5))
    with pytest.raises(Exception, match="no signal"):
        session.add_array("bad", Array({"m0": (0, 0), "zz": (1, 0)}))
