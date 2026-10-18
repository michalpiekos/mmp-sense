"""A live result carried on from the last window is the one a single
computation over the whole stream gives, over the samples still held."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmp_sense.core.buffer import ChannelBuffer
from mmp_sense.core.graph import Evaluator, GraphError, Lookback, hold, register_op, signal
from mmp_sense.core.graph.ops import REGISTRY, _OPS
from mmp_sense.core.signals import SignalTable
from mmp_sense.examples.double_knock import detector, synthesize

RATE = 1000.0
CHANNELS = ["accel_x", "accel_y", "accel_z"]


@pytest.fixture(scope="module")
def stream():
    table, _ = synthesize(duration=16.0, rate=RATE, seed=3)
    return table.index.to_numpy(float), table.to_numpy(float)


class Pair:
    """A live buffer of `span_s` that scrolls, and one that keeps everything."""

    def __init__(self, span_s: float = 3.0, channels=CHANNELS) -> None:
        self.live = SignalTable(ChannelBuffer(channels, span_s=span_s, rate=RATE,
                                              max_capacity=int(span_s * RATE)))
        self.whole = SignalTable(ChannelBuffer(channels, span_s=1e6, rate=RATE))
        self.evaluator = Evaluator()

    def append(self, time, values) -> None:
        self.live.buffer.append(time, values)
        self.whole.buffer.append(time, values)

    def carried(self, node):
        window = self.live.window()
        assert window.live is not None
        return window, self.evaluator.evaluate(node, window)

    def reference(self, node, t_start: float):
        window = self.whole.window()
        window.live = None                      # computed in full, as before
        result = Evaluator().evaluate(node, window)
        if node.output == "events":
            table = result.table
            return table[table["timestamp"] >= t_start].reset_index(drop=True)
        keep = result.time >= t_start
        return result.time[keep], np.asarray(result.values)[keep]


def _chunks(n: int, seed: int = 0, most: int = 240):
    rng = np.random.default_rng(seed)
    at = 0
    while at < n:
        size = int(rng.integers(1, most))
        yield at, min(n, at + size)
        at += size


def _same(pair, node, t_start, result, context=""):
    expected = pair.reference(node, t_start)
    if node.output == "events":
        got = result.table.reset_index(drop=True)
        if not len(got) and not len(expected):
            return
        columns = [c for c in expected.columns if c not in ("label", "source")]
        assert list(got["timestamp"]) == list(expected["timestamp"]), context
        pd.testing.assert_frame_equal(got[columns], expected[columns], check_dtype=False,
                                      obj=context)
        return
    time, values = expected
    np.testing.assert_array_equal(result.time, time, err_msg=context)
    got = np.asarray(result.values)
    if got.dtype == bool:
        np.testing.assert_array_equal(got, values, err_msg=context)
    else:
        np.testing.assert_allclose(got, np.asarray(values, float), rtol=1e-9, atol=1e-9,
                                   equal_nan=True, err_msg=context)


def _check(stream, nodes, *, span_s=3.0, every=5, seed=0, most=240):
    time, values = stream
    pair = Pair(span_s)
    for i, (a, b) in enumerate(_chunks(len(time), seed, most)):
        pair.append(time[a:b], values[a:b])
        if i % every and b < len(time):
            continue
        for node in nodes:
            window, result = pair.carried(node)
            _same(pair, node, window.t_start, result, f"{node.title} at {b}")
    return pair


accel = signal(*CHANNELS, name="accel")
hp = accel.magnitude().highpass(cutoff_hz=10, name="hp")
frames = hp.frames("80ms", "10ms", name="frames")

SERIES = {
    "signal": accel,
    "one channel": signal("accel_z"),
    "select": accel["accel_x"],
    "magnitude": accel.magnitude(),
    "highpass": hp,
    "highpass of a vector": accel.highpass(cutoff_hz=5),
    "lowpass": hp.lowpass(cutoff_hz=30),
    "peak": hp.peak("40ms"),
    "energy": hp.energy("80ms"),
    "rms": hp.rms("25ms"),
    "zero_crossing_rate": hp.zero_crossing_rate("50ms"),
    "arith by a value": hp * 2.0 + 1.0,
    "arith of two": hp.energy("80ms") / hp.peak("40ms"),
    "arith against frames": hp.energy("80ms") - frames.dominant_hz(),
    "compare": hp.energy("80ms") > 5.0,
    "compare of two": hp.energy("80ms") > hp.peak("40ms"),
    "between": hp.between(-0.5, 0.5),
    "and": (hp.energy("80ms") > 5.0) & frames.dominant_hz().between(120, 350),
    "or": (hp.energy("80ms") > 5.0) | (hp.peak("40ms") > 2.0),
    "not": ~(hp.energy("80ms") > 5.0),
    "frames": frames,
    "frames with no hop": hp.frames("64ms"),
    "band_power": frames.band_power(120, 350),
    "band_share": frames.band_share(120, 350),
    "dominant_hz": frames.dominant_hz(min_hz=20),
    "centroid_hz": frames.centroid_hz(),
    "spread_hz": frames.spread_hz(),
    "flatness": frames.flatness(),
    "rolloff_hz": frames.rolloff_hz(0.9),
    "flux": frames.flux(),
    "flux, normalised": frames.flux(normalize=True),
    "steady": frames.steady("50ms"),
    "line_power": frames.line_power(100, 400, above=4.0),
    "line_share of steady": frames.steady("40ms").line_share(100, 400, above=4.0),
    "lowpass, order 4": hp.lowpass(cutoff_hz=30, order=4),
    "highpass, order 3 of a vector": accel.highpass(cutoff_hz=5, order=3),
    "bandpass": accel["accel_z"].bandpass(80, 300),
    "bandstop": hp.bandstop(45, 55, order=2),
    "notch": hp.notch(90, q=10),
    "envelope": hp.envelope(30),
    "envelope of a vector": accel.highpass(cutoff_hz=5).envelope(),
    "abs": accel["accel_x"].abs(),
    "delay": accel["accel_x"].delay(accel["accel_y"], "80ms", "20ms", max_lag="10ms"),
    "delay, plain": hp.delay(accel["accel_z"], "64ms", method="plain"),
    "repetition_hz": hp.frames("200ms", "50ms").repetition_hz(20, 120, low_hz=40,
                                                             high_hz=450),
    "peak_hz": frames.peak_hz(100, 400),
    "peak_hz near a number": frames.peak_hz(50, 450, near=200.0, within=40),
    "peak_hz near a recipe": frames.peak_hz(50, 450, near=frames.dominant_hz(min_hz=20),
                                            within=30),
    "fill": hp.energy("80ms").fill(),
    "fill, at most": frames.repetition_hz(20, 120, low_hz=40, high_hz=450).fill("150ms"),
    "align": frames.align(frames.dominant_hz(min_hz=20), 200.0),
    "steady of aligned": frames.align(frames.dominant_hz(min_hz=20), 200.0).steady("50ms"),
}


@pytest.mark.parametrize("name", list(SERIES))
def test_a_series_carried_on_is_the_whole_streams(stream, name):
    _check(stream, [SERIES[name]])


@pytest.mark.parametrize("name", ["lowpass, order 4", "bandpass", "notch", "envelope"])
def test_a_filter_over_a_gap_carried_on_is_the_whole_streams(stream, name):
    time, values = stream
    values = values.copy()
    values[5000:5003] = np.nan
    _check((time, values), [SERIES[name]], every=1)


def test_a_delay_over_a_gap_carried_on_is_the_whole_streams(stream):
    time, values = stream
    values = values.copy()
    values[3000:3005, 1] = np.nan                 # a gap in the second signal
    values[7000:7400, 0] = 0.0                    # and silence in the first
    _check((time, values), [accel["accel_x"].delay(accel["accel_y"], "80ms", "20ms")],
           every=1)


loud = (hp.energy("80ms") > 5.0).events(refractory="100ms", name="loud")
EVENTS = {
    "events": loud,
    "events, min_duration": (hp.energy("80ms") > 2.0).events(min_duration="30ms"),
    "events of frames": frames.dominant_hz().between(120, 350).events(refractory="50ms"),
    "lasting": (hp.energy("80ms") > 2.0).events().lasting(min="20ms", max="200ms"),
    "repeated": loud.repeated(2, gap=("80ms", "500ms")),
    "repeated thrice": loud.repeated(3, gap=("80ms", "700ms")),
    "followed_by itself": loud.followed_by(loud, gap=("80ms", "600ms")),
    "followed_by another": loud.followed_by(
        (hp.peak("40ms") > 1.0).events(refractory="50ms"), gap=("0ms", "400ms")),
}


@pytest.mark.parametrize("name", list(EVENTS))
def test_events_carried_on_are_the_whole_streams(stream, name):
    _check(stream, [EVENTS[name]], every=3)


def test_the_double_knock_detector_carried_on_is_the_whole_streams(stream):
    nodes = detector()
    _check(stream, list(nodes.values()), every=2, seed=7)


def test_single_samples_and_whole_seconds_alike(stream):
    _check(stream, [SERIES["and"], EVENTS["repeated"]], most=3, every=97)
    _check(stream, [SERIES["and"], EVENTS["repeated"]], most=2500, every=1, seed=2)


def test_carrying_on_computes_only_the_new_samples(stream):
    time, values = stream
    pair = Pair()
    node = hp.energy("80ms")
    pair.append(time[:2000], values[:2000])
    pair.carried(node)
    pair.append(time[2000:2010], values[2000:2010])
    calls = []
    original = pair.evaluator.apply
    pair.evaluator.apply = lambda ctx, inputs, params: (
        calls.append(len(inputs[0].time) if inputs else 0) or original(ctx, inputs, params))
    pair.carried(node)
    assert max(calls) < 200                        # the new ten, and the energy's window


def test_an_op_with_no_rule_is_computed_in_full(stream):
    @register_op("test_centred", causal=False)
    def centred(ctx, series):
        values = np.asarray(series.values, float)
        return series.with_values(values - np.nanmean(values))

    from mmp_sense import dsp

    time, values = stream
    pair = Pair()
    node = hp.apply("test_centred")
    energy = node.energy("50ms")
    try:
        for a, b in _chunks(8000, seed=5):
            pair.append(time[a:b], values[a:b])
            window, below = pair.carried(hp)
            _, got = pair.carried(node)
            expected = below.values - np.nanmean(below.values)
            np.testing.assert_allclose(got.values, expected, equal_nan=True)
            _, got = pair.carried(energy)
            np.testing.assert_allclose(got.values, dsp.trailing_sum(expected ** 2, 50),
                                       rtol=1e-9, atol=1e-9, equal_nan=True)
    finally:
        _OPS.pop("test_centred")
        REGISTRY["version"] += 1


def test_an_op_can_say_how_it_carries_on(stream):
    @register_op("test_doubled", incremental=Lookback(0))
    def doubled(ctx, series):
        return series.with_values(np.asarray(series.values, float) * 2)

    try:
        _check(stream, [hp.apply("test_doubled")])
    finally:
        _OPS.pop("test_doubled")
        REGISTRY["version"] += 1


def test_an_unbounded_pattern_is_computed_in_full(stream):
    unbounded = loud.repeated(2, gap=("80ms", None))
    _check(stream, [unbounded], span_s=6.0)


def _fresh(pair, node):
    window = pair.live.window()
    window.live = None
    return Evaluator().evaluate(node, window)


def test_a_restart_starts_again(stream):
    time, values = stream
    pair = Pair()
    node = EVENTS["repeated"]
    energy = hp.energy("80ms")
    pair.append(time[:4000], values[:4000])
    pair.carried(node), pair.carried(energy)
    pair.live.buffer.append(time[:2000], values[:2000])     # a new timeline
    assert pair.live.buffer.restarts == 1
    _, result = pair.carried(energy)
    expected = _fresh(pair, energy)
    np.testing.assert_allclose(result.values, expected.values, equal_nan=True)
    _, events = pair.carried(node)
    pd.testing.assert_frame_equal(events.table, _fresh(pair, node).table)


def test_a_cleared_buffer_starts_again(stream):
    time, values = stream
    pair = Pair()
    energy = hp.energy("80ms")
    pair.append(time[:4000], values[:4000])
    pair.carried(energy)
    pair.live.buffer.clear()
    pair.live.buffer.append(time[5000:6000], values[5000:6000])
    _, result = pair.carried(energy)
    np.testing.assert_allclose(result.values, _fresh(pair, energy).values, equal_nan=True)


def test_another_buffer_starts_again(stream):
    time, values = stream
    pair = Pair()
    energy = hp.energy("80ms")
    pair.append(time[:4000], values[:4000])
    pair.carried(energy)
    buffer = ChannelBuffer(CHANNELS, span_s=3.0, rate=RATE)
    buffer.append(time[4000:6000], values[4000:6000])
    pair.live.rebind(buffer)
    _, result = pair.carried(energy)
    np.testing.assert_allclose(result.values, _fresh(pair, energy).values, equal_nan=True)


def test_changed_params_start_again(stream):
    time, values = stream
    pair = Pair()
    node = signal("accel_z").highpass(cutoff_hz=10).energy("80ms")
    pair.append(time[:2000], values[:2000])
    pair.carried(node)
    pair.append(time[2000:4100], values[2000:4100])
    pair.carried(node)
    changed = node.with_params(length=0.05)
    window, result = pair.carried(changed)
    # The energy starts from the window; the filter beneath it carried on.
    _, expected = pair.reference(changed, window.t_start)
    assert np.isnan(result.values[:49]).all()
    np.testing.assert_allclose(result.values[49:], expected[49:], rtol=1e-9)


def test_a_redefined_signal_starts_again(stream):
    time, values = stream
    pair = Pair()
    pair.live.define("twice", lambda w: w["accel_z"] * 2)
    node = signal("twice").highpass(cutoff_hz=10).energy("80ms")
    pair.append(time[:4000], values[:4000])
    pair.carried(node)
    pair.live.define("twice", lambda w: w["accel_z"] * 3)
    _, result = pair.carried(node)
    np.testing.assert_allclose(result.values, _fresh(pair, node).values, equal_nan=True)


def _churn(pair, stream, nodes, *, held=()):
    time, values = stream
    for a, b in _chunks(8000, seed=6):
        pair.append(time[a:b], values[a:b])
        window = pair.live.window()
        fresh = pair.live.window()
        fresh.live = None
        for node in nodes:
            got = pair.evaluator.evaluate(node, window)
            expected = Evaluator().evaluate(node, fresh)
            np.testing.assert_allclose(got.values, expected.values, rtol=1e-9, atol=1e-9,
                                       equal_nan=True, err_msg=f"{node.title} at {b}")
        for node in held:
            expected = hold(Evaluator().evaluate(node, fresh), window.time)
            np.testing.assert_allclose(pair.evaluator.held(node, window), expected,
                                       equal_nan=True, err_msg=f"held {node.title} at {b}")


def test_what_leans_on_an_op_with_no_rule_is_computed_in_full(stream):
    @register_op("test_centred2", causal=False)
    def centred(ctx, series):
        values = np.asarray(series.values, float)
        return series.with_values(values - np.nanmean(values))

    time, values = stream
    try:
        centred_hp = hp.apply("test_centred2")
        pair = Pair()
        for a, b in _chunks(8000, seed=6):
            pair.append(time[a:b], values[a:b])
            window = pair.live.window()
            below = pair.evaluator.evaluate(hp, window).values
            expected = below - np.nanmean(below)
            np.testing.assert_allclose(pair.evaluator.held(centred_hp, window), expected,
                                       equal_nan=True, err_msg=f"held at {b}")
            got = pair.evaluator.evaluate(hp - centred_hp, window).values
            np.testing.assert_allclose(got, below - expected, equal_nan=True,
                                       err_msg=f"difference at {b}")
    finally:
        _OPS.pop("test_centred2")
        REGISTRY["version"] += 1


def test_a_defined_signal_is_computed_in_full(stream):
    pair = Pair()
    pair.live.define("dz", lambda w: w["accel_z"] - w["accel_z"].mean())
    pair.append(*[a[:1500] for a in stream])
    _churn(pair, stream, [signal("dz").energy("50ms")])


def test_equal_timestamps_align_as_in_full(stream):
    time, values = stream
    floored = np.floor(time * 500) / 500                    # pairs of equal times
    pair = Pair()
    node = hp.energy("80ms") - hp.peak("40ms")
    start = 0
    for a, b in _chunks(8000, seed=8):
        pair.append(floored[a:b], values[a:b])
        if start == 0:
            pair.carried(node)
            start = 1
            continue
        window, got = pair.carried(node)
        whole = pair.whole.window()
        whole.live = None
        full = Evaluator().evaluate(node, whole)
        n = len(got.values)
        np.testing.assert_allclose(got.values, np.asarray(full.values)[-n:],
                                   rtol=1e-9, atol=1e-9, equal_nan=True, err_msg=f"at {b}")


def test_a_live_window_cannot_be_written(stream):
    pair = Pair()
    pair.append(*[a[:500] for a in stream])
    window = pair.live.window()
    with pytest.raises(ValueError):
        window.values[0, 0] = 1.0
    with pytest.raises(ValueError):
        window.time[0] = 1.0


def test_what_is_handed_out_is_read_only_and_stays_put(stream):
    time, values = stream
    pair = Pair()
    node = hp.energy("80ms")
    pair.append(time[:2000], values[:2000])
    _, first = pair.carried(node)
    kept = np.array(first.values, copy=True)
    with pytest.raises(ValueError):
        first.values[0] = 1.0
    for a in range(2000, 9000, 700):
        pair.append(time[a:a + 700], values[a:a + 700])
        pair.carried(node)
    np.testing.assert_array_equal(first.values, kept)


def test_a_failing_op_is_named_and_forgotten(stream):
    time, values = stream
    pair = Pair()
    pair.append(time[:2000], values[:2000])
    with pytest.raises(GraphError, match="frames"):
        pair.carried(signal("accel_z").frames("0.5ms"))


def test_held_values_follow_the_window(stream):
    time, values = stream
    pair = Pair()
    node = frames.dominant_hz()
    for a, b in _chunks(9000, seed=4):
        pair.append(time[a:b], values[a:b])
        window = pair.live.window()
        got = pair.evaluator.held(node, window)
        whole = pair.whole.window()
        whole.live = None
        full = Evaluator().evaluate(node, whole)
        expected = np.asarray(hold(full, window.time), float)
        np.testing.assert_allclose(got, expected, equal_nan=True, err_msg=f"at {b}")
