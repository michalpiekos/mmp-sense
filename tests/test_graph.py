"""Detector graphs: the transforms, the time rules, the composition, the run."""

import subprocess
import sys

import numpy as np
import pytest

import mmp_sense
from mmp_sense import dsp
from mmp_sense.core.data import TimeSeries
from mmp_sense.analysis import session_of
from mmp_sense.core.graph import GraphError, Node, Series, hold, register_op, run, seconds


def series_of(values, rate=1000.0, channels=None, t0=0.0):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    channels = channels or [f"c{i}" for i in range(values.shape[1])]
    time = t0 + np.arange(len(values)) / rate
    return TimeSeries(name="test", time=time, values=values, channels=channels,
                      sample_rate=rate)


def pulses(times, duration=3.0, rate=1000.0, width=0.01):
    """A channel "v" that is 1 for `width` seconds from each of `times`."""
    t = np.arange(int(duration * rate)) / rate
    v = np.zeros_like(t)
    for start in times:
        v[(t >= start) & (t < start + width)] = 1.0
    return series_of(v, rate, ["v"])


def run_on(node, data):
    """Every step of `node` over some data: a TimeSeries, a path or a session."""
    if isinstance(data, TimeSeries):
        data = session_of(data)
    elif isinstance(data, (str, bytes)) or hasattr(data, "suffix"):
        data = mmp_sense.load(data)
    return run(node, data)


def evaluate(node, series):
    return run_on(node, series).output


# -- the transforms ------------------------------------------------------


def test_energy_is_the_sum_of_squares_over_the_window():
    out = evaluate(mmp_sense.signal("v").energy(3), series_of([1, 2, 3], rate=1.0, channels=["v"]))
    assert out.values[-1] == pytest.approx(14.0)
    # Fewer samples than the window is no value at all, not a small one.
    assert np.isnan(out.values[:2]).all()


def test_magnitude_of_three_four_zero_is_five():
    data = series_of([[3.0, 4.0, 0.0]], channels=["x", "y", "z"])
    assert evaluate(mmp_sense.signal("x", "y", "z").magnitude(), data).values[0] == pytest.approx(5.0)


def test_the_highpass_follows_its_equation_and_resumes_from_its_state():
    rate, cutoff = 100.0, 5.0
    x = np.random.default_rng(0).normal(size=200)
    whole, _ = dsp.highpass1(x, rate, cutoff)
    first, state = dsp.highpass1(x[:70], rate, cutoff)
    second, _ = dsp.highpass1(x[70:], rate, cutoff, state)
    assert np.allclose(whole, np.concatenate([first, second]))

    alpha = dsp.highpass1_alpha(rate, cutoff)
    rc = 1 / (2 * np.pi * cutoff)
    assert alpha == pytest.approx(rc / (rc + 1 / rate))
    expected = [0.0]
    for n in range(1, len(x)):
        expected.append(alpha * (expected[-1] + x[n] - x[n - 1]))
    assert np.allclose(whole, expected)


def test_the_highpass_starts_at_rest_and_removes_an_offset():
    rate = 1000.0
    t = np.arange(2000) / rate
    out = evaluate(mmp_sense.signal("v").highpass(cutoff_hz=10),
                   series_of(9.81 + np.sin(2 * np.pi * 100 * t), rate, ["v"]))
    assert abs(out.values[0]) < 1e-12
    assert abs(out.values[1000:].mean()) < 0.01
    assert out.values[1000:].max() == pytest.approx(1.0, abs=0.1)


def test_the_highpass_rejects_a_cutoff_beyond_nyquist():
    with pytest.raises(GraphError, match="Nyquist"):
        evaluate(mmp_sense.signal("v").highpass(cutoff_hz=600), series_of(np.zeros(10), 1000.0, ["v"]))


def test_the_lowpass_follows_its_equation_and_resumes_from_its_state():
    rate, cutoff = 100.0, 5.0
    x = np.random.default_rng(0).normal(size=200)
    whole, _ = dsp.lowpass1(x, rate, cutoff)
    first, state = dsp.lowpass1(x[:70], rate, cutoff)
    second, _ = dsp.lowpass1(x[70:], rate, cutoff, state)
    assert np.allclose(whole, np.concatenate([first, second]))

    b = dsp.lowpass1_alpha(rate, cutoff)
    rc = 1 / (2 * np.pi * cutoff)
    assert b == pytest.approx((1 / rate) / (rc + 1 / rate))
    expected, y = [], x[0]
    for value in x:
        y = y + b * (value - y)
        expected.append(y)
    assert np.allclose(whole, expected)


def test_the_lowpass_passes_a_constant_from_the_first_sample():
    out = evaluate(mmp_sense.signal("v").lowpass(cutoff_hz=10),
                   series_of(np.full(500, 9.81), 1000.0, ["v"]))
    assert np.allclose(out.values, 9.81)


@pytest.mark.parametrize("ratio, low, high", [(10.0, 0.08, 0.12), (0.1, 0.95, 1.0)])
def test_the_lowpass_attenuates_above_its_cutoff(ratio, low, high):
    rate, cutoff = 1000.0, 10.0
    t = np.arange(20000) / rate
    x = np.sin(2 * np.pi * cutoff * ratio * t)
    out = evaluate(mmp_sense.signal("v").lowpass(cutoff_hz=cutoff), series_of(x, rate, ["v"]))
    gain = np.abs(out.values[10000:]).max()
    assert low < gain < high


def test_the_lowpass_restarts_at_rest_after_a_gap():
    x = np.array([1.0, 1.0, np.nan, 5.0, 5.0])
    out, state = dsp.lowpass1(x, 100.0, 5.0)
    assert np.isnan(out[2])
    assert out[3] == pytest.approx(5.0)
    assert state == (5.0, pytest.approx(5.0))


def test_the_lowpass_rejects_a_cutoff_beyond_nyquist():
    with pytest.raises(GraphError, match="Nyquist"):
        evaluate(mmp_sense.signal("v").lowpass(cutoff_hz=500), series_of(np.zeros(10), 1000.0, ["v"]))


def test_the_lowpass_counts_its_time_constant_as_latency():
    node = mmp_sense.signal("v").lowpass(cutoff_hz=10, name="lp")
    assert node.latency == pytest.approx(1 / (2 * np.pi * 10))
    assert "lp: lowpass(10Hz)" in node.graph()
    assert node.highpass(cutoff_hz=5).latency == pytest.approx(node.latency)


def test_a_highpass_then_a_lowpass_is_a_band_pass():
    rate = 1000.0
    t = np.arange(20000) / rate
    band = mmp_sense.signal("v").highpass(cutoff_hz=5).lowpass(cutoff_hz=50)

    def gain(frequency):
        x = 9.81 + np.sin(2 * np.pi * frequency * t)
        out = evaluate(band, series_of(x, rate, ["v"]))
        return np.abs(out.values[10000:]).max()

    from scipy import signal as sig

    a = dsp.highpass1_alpha(rate, 5)
    b = dsp.lowpass1_alpha(rate, 50)

    def response(frequency):
        _, hp = sig.freqz([a, -a], [1, -a], worN=[frequency], fs=rate)
        _, lp = sig.freqz([b], [1, -(1 - b)], worN=[frequency], fs=rate)
        return abs(hp[0] * lp[0])

    for frequency in (0.5, 16, 400):
        assert gain(frequency) == pytest.approx(response(frequency), rel=0.02)
    assert gain(16) > 0.85
    assert gain(0.5) < 0.15 and gain(400) < 0.15


METRICS = [
    ("band_power", dict(low_hz=80, high_hz=100),
     lambda f, a, bw: mmp_sense.dsp.band_power(f, a, 80, 100, bw)),
    ("band_share", dict(low_hz=80, high_hz=100),
     lambda f, a, bw: mmp_sense.dsp.band_power(f, a, 80, 100, bw, relative=True)),
    ("centroid_hz", {}, lambda f, a, bw: mmp_sense.dsp.spectral_centroid(f, a, bw)),
    ("spread_hz", {}, lambda f, a, bw: mmp_sense.dsp.spectral_spread(f, a, bw)),
    ("flatness", {}, lambda f, a, bw: mmp_sense.dsp.spectral_flatness(f, a, bw)),
    ("rolloff_hz", dict(fraction=0.9), lambda f, a, bw: mmp_sense.dsp.spectral_rolloff(f, a, bw, 0.9)),
    ("flux", {}, lambda f, a, bw: mmp_sense.dsp.spectral_flux(f, a, bw)),
    ("flux", dict(normalize=True), lambda f, a, bw: mmp_sense.dsp.spectral_flux(f, a, bw, True)),
    ("line_power", dict(low_hz=80, high_hz=300),
     lambda f, a, bw: mmp_sense.dsp.line_power(f, a, 80, 300, bw)),
    ("line_share", dict(low_hz=80, high_hz=300, above=4.0, floor_hz=30.0),
     lambda f, a, bw: mmp_sense.dsp.line_power(f, a, 80, 300, bw, 4.0, 30.0, relative=True)),
]


@pytest.mark.parametrize("name, params, expected", METRICS)
def test_a_measure_of_frames_is_its_dsp_function_per_frame(name, params, expected):
    x = np.random.default_rng(5).normal(size=3000)
    data = series_of(x, 1000.0, ["v"])
    frames = mmp_sense.signal("v").frames("100ms")
    spectrum = evaluate(frames, data)
    measured = evaluate(getattr(frames, name)(**params), data)
    want = expected(spectrum.coords["frequency"], spectrum.values, spectrum.coords["enbw"])
    assert np.allclose(measured.values, want, equal_nan=True)
    assert np.allclose(measured.time, spectrum.time)
    node = getattr(frames, name)(**params)
    assert node.latency == pytest.approx(frames.latency)
    assert node.label.startswith(name)


def test_a_measure_of_frames_refuses_a_plain_signal():
    data = series_of(np.zeros(500), 1000.0, ["v"])
    with pytest.raises(GraphError, match="frames"):
        evaluate(mmp_sense.signal("v").band_power(1, 100), data)


def test_band_power_units_are_the_signal_units_squared():
    data = series_of(np.random.default_rng(1).normal(size=500), 1000.0, ["v"])
    data.units = "m/s^2"
    frames = mmp_sense.signal("v").frames("100ms")
    assert evaluate(frames.band_power(1, 100), data).units == "(m/s^2)^2"
    assert evaluate(frames.band_share(1, 100), data).units == ""
    assert evaluate(frames.centroid_hz(), data).units == "Hz"


def test_measures_refuse_a_misspelt_parameter():
    frames = mmp_sense.signal("v").frames("1s")
    with pytest.raises(Exception, match="widow"):
        frames.apply("band_power", widow="1s", low_hz=1, high_hz=2)
    with pytest.raises(Exception, match="fraction"):
        frames.apply("centroid_hz", fraction=0.5)


def test_the_knock_shows_in_band_share_and_flux(session):
    frames = mmp_sense.signal().magnitude().highpass(cutoff_hz=10).frames("80ms", "20ms")
    share = session.values(frames.band_share(70, 110))
    flux = session.values(frames.flux())
    near = (share.index > 5.0) & (share.index < 5.1)
    quiet = (share.index > 2.0) & (share.index < 4.0)
    assert np.nanmax(share.values[near]) > 0.5
    assert np.nanmedian(share.values[quiet]) < 0.2
    rise = (flux.index > 4.98) & (flux.index < 5.1)
    assert np.nanmax(flux.values[rise]) > 2 * np.nanmedian(flux.values[quiet])


def test_steady_keeps_each_bin_at_its_smallest_over_its_length():
    rng = np.random.default_rng(8)
    t = np.arange(6000) / 1000.0
    x = np.sin(2 * np.pi * 120 * t) + 0.01 * rng.normal(size=len(t))
    x[3000:3600] += 2 * np.sin(2 * np.pi * 300 * t[3000:3600])     # a 0.6 s burst
    data = series_of(x, 1000.0, ["v"])
    frames = mmp_sense.signal("v").frames("200ms", "50ms")
    held = frames.steady("1s")
    spectrum, out = evaluate(frames, data), evaluate(held, data)
    want = dsp.trailing_min(spectrum.values, 20)
    assert np.allclose(out.values, want, equal_nan=True)
    assert np.allclose(out.time, spectrum.time)
    assert held.latency == pytest.approx(frames.latency + 1.0)
    burst = evaluate(frames.band_power(250, 350), data).values
    kept = evaluate(held.band_power(250, 350), data).values
    assert np.nanmax(burst) > 1.0 and np.nanmax(kept) < 1e-3       # it came and went
    tone = evaluate(held.line_share(80, 400), data).values
    assert np.nanmin(tone) > 0.9                                   # the tone stayed
    with pytest.raises(GraphError, match="frames"):
        evaluate(mmp_sense.signal("v").steady("1s"), data)


def test_trailing_max_looks_only_backwards():
    x = np.random.default_rng(1).normal(size=100)
    for width in (1, 2, 5, 8):
        expected = [x[i - width + 1:i + 1].max() if i >= width - 1 else np.nan
                    for i in range(len(x))]
        assert np.allclose(dsp.trailing_max(x, width), expected, equal_nan=True)


def test_peak_is_the_largest_absolute_value_in_the_window():
    out = evaluate(mmp_sense.signal("v").peak(3), series_of([1, -5, 2, 0, 0, 0], 1.0, ["v"]))
    assert list(out.values[2:]) == [5.0, 5.0, 2.0, 0.0]


def test_zero_crossing_rate_of_a_sine_is_twice_its_frequency():
    rate = 1000.0
    t = np.arange(4000) / rate
    out = evaluate(mmp_sense.signal("v").zero_crossing_rate("0.5s"),
                   series_of(np.sin(2 * np.pi * 50 * t + 0.1), rate, ["v"]))
    assert np.nanmedian(out.values) == pytest.approx(100.0, rel=0.03)
    assert out.units == "Hz"


def test_dominant_frequency_finds_the_tone_and_is_stamped_at_frame_ends():
    rate = 1000.0
    t = np.arange(2000) / rate
    node = mmp_sense.signal("v").frames("100ms", "50ms").dominant_hz()
    out = evaluate(node, series_of(np.sin(2 * np.pi * 120 * t), rate, ["v"]))
    assert np.allclose(out.values, 120.0)
    # Frames end on sample numbers that are multiples of the hop.
    assert out.time[0] == pytest.approx(0.1)
    assert np.allclose(np.diff(out.time), 0.05)
    assert out.rate == pytest.approx(20.0)


def test_fft_is_a_spectrum_per_frame():
    rate = 1000.0
    t = np.arange(1000) / rate
    out = evaluate(mmp_sense.signal("v").frames("200ms"), series_of(2 * np.sin(2 * np.pi * 50 * t), rate, ["v"]))
    assert out.shape == (101,)
    frequency = out.coords["frequency"]
    assert frequency[np.argmax(out.values[0])] == pytest.approx(50.0)
    assert out.values[0].max() == pytest.approx(2.0, rel=0.02)


def test_durations_are_seconds_or_text():
    assert seconds("80ms") == pytest.approx(0.08)
    assert seconds("1.5s") == 1.5
    assert seconds("250us") == pytest.approx(0.00025)
    assert seconds(0.2) == 0.2
    with pytest.raises(ValueError):
        seconds("eighty")
    assert mmp_sense.signal().energy("80ms").params["length"] == pytest.approx(0.08)


# -- time alignment and predicates ------------------------------------


def test_hold_takes_the_latest_value_at_or_before_each_instant():
    slow = Series(time=[1.0, 2.0], values=[10.0, 20.0])
    assert list(hold(slow, np.array([0.5, 1.0, 1.5, 2.5]))[1:]) == [10.0, 10.0, 20.0]
    assert np.isnan(hold(slow, np.array([0.5]))[0])
    flags = Series(time=[1.0], values=np.array([True]))
    assert list(hold(flags, np.array([0.5, 1.0]))) == [False, True]


def test_predicates_compose_and_nan_is_never_true():
    data = series_of([0, 1, 2, 3, np.nan], 1.0, ["v"])
    v = mmp_sense.signal("v")
    assert list(evaluate(v > 1, data).values) == [False, False, True, True, False]
    assert list(evaluate(v >= 1, data).values) == [False, True, True, True, False]
    assert list(evaluate(v < 1, data).values) == [True, False, False, False, False]
    assert list(evaluate(v <= 1, data).values) == [True, True, False, False, False]
    assert list(evaluate(v.between(1, 2), data).values) == [False, True, True, False, False]
    assert list(evaluate((v > 0) & (v < 3), data).values) == [False, True, True, False, False]
    assert list(evaluate((v < 1) | (v > 2), data).values) == [True, False, False, True, False]
    assert list(evaluate(~(v > 1), data).values) == [True, True, False, False, True]


def test_two_time_bases_meet_on_the_left_ones():
    rate = 1000.0
    t = np.arange(1000) / rate
    data = series_of(np.sin(2 * np.pi * 200 * t), rate, ["v"])
    v = mmp_sense.signal("v")
    both = (v.energy("10ms") > 0) & v.frames("100ms").dominant_hz().between(150, 250)
    out = evaluate(both, data)
    assert len(out.time) == 1000
    # Nothing is true before the first spectrum exists, 99 ms in.
    assert not out.values[:99].any()
    assert out.values[200:].all()


def test_a_node_is_not_a_truth_value():
    with pytest.raises(TypeError, match="between"):
        bool(mmp_sense.signal("v") > 1)


def test_arithmetic_on_nodes():
    data = series_of([1.0, 2.0], 1.0, ["v"])
    v = mmp_sense.signal("v")
    assert list(evaluate(v * 2 + 1, data).values) == [3.0, 5.0]
    assert list(evaluate(v / v, data).values) == [1.0, 1.0]


# -- events and patterns --------------------------------------------------


def test_a_condition_becomes_one_event_per_stretch():
    events = evaluate((mmp_sense.signal("v") > 0.5).events(name="hit"), pulses([1.0, 1.3, 2.5]))
    table = events.table
    assert list(table["timestamp"]) == pytest.approx([1.0, 1.3, 2.5])
    assert list(table["end_time"]) == pytest.approx([1.009, 1.309, 2.509])
    assert list(table["duration"]) == pytest.approx([0.009] * 3)
    assert set(table["label"]) == {"hit"}
    assert set(table["source"]) == {"hit"}


def test_refractory_and_min_duration():
    data = pulses([1.0, 1.05, 2.0], width=0.02)
    hits = mmp_sense.signal("v") > 0.5
    assert list(run_on(hits.events(refractory="100ms"), data).timestamps) == pytest.approx([1.0, 2.0])
    assert run_on(hits.events(min_duration="50ms"), data).count == 0


def test_followed_by_pairs_an_event_with_the_next_in_range():
    a = (mmp_sense.signal("v") > 0.5).events(name="A")
    result = run_on(a.followed_by(a, gap=(0, "500ms"), name="AA"), pulses([1.0, 1.3, 2.5]))
    assert result.count == 1
    row = result.events.table.iloc[0]
    assert (row["first"], row["second"]) == pytest.approx((1.0, 1.3))
    assert row["start_time"] == pytest.approx(1.0)
    assert row["end_time"] == pytest.approx(1.309)
    assert row["label"] == "AA"


def test_followed_by_uses_each_event_once():
    a = (mmp_sense.signal("v") > 0.5).events()
    data = pulses([1.0, 1.3, 1.6])
    assert run_on(a.followed_by(a, gap=(0, "500ms")), data).count == 1
    assert run_on(a.followed_by(a, gap=("400ms", "1s")), data).count == 1


@pytest.mark.parametrize("times, expected", [
    ([1.0, 1.3, 1.6], [[1.0, 1.3, 1.6]]),
    ([1.0, 1.3, 1.6, 1.9], [[1.0, 1.3, 1.6], [1.3, 1.6, 1.9]]),
    ([1.0, 1.3], []),
    ([1.0, 1.3, 2.5], []),
    ([1.0, 1.1, 1.4, 1.7], [[1.0, 1.4, 1.7], [1.1, 1.4, 1.7]]),
    ([1.0, 1.3, 1.6, 3.0, 3.3, 3.6], [[1.0, 1.3, 1.6], [3.0, 3.3, 3.6]]),
])
def test_repeated_finds_each_run_of_n_in_rhythm(times, expected):
    a = (mmp_sense.signal("v") > 0.5).events()
    table = run_on(a.repeated(3, gap=("200ms", "450ms")), pulses(times, duration=4.0)).events.table
    assert len(table) == len(expected)
    if expected:
        assert [list(t) for t in table["times"]] == [pytest.approx(e) for e in expected]
        assert list(table["start_time"]) == pytest.approx([e[0] for e in expected])
        assert list(table["end_time"]) == pytest.approx([e[-1] + 0.009 for e in expected])


def test_repeated_needs_two_or_more_and_a_gap():
    a = (mmp_sense.signal("v") > 0.5).events()
    with pytest.raises(ValueError):
        a.repeated(1, gap=(0, 1))
    with pytest.raises(TypeError, match="gap"):
        a.repeated(3, gap="1s")


def test_followed_by_between_two_kinds_of_event():
    data = series_of(np.stack([pulses([1.0, 2.0]).values[:, 0],
                               pulses([1.2, 3.0]).values[:, 0]], axis=1),
                     1000.0, ["a", "b"])
    a = (mmp_sense.signal("a") > 0.5).events()
    b = (mmp_sense.signal("b") > 0.5).events()
    result = run_on(a.followed_by(b, gap=(0, "500ms")), data)
    assert list(result.timestamps) == pytest.approx([1.0])
    assert run_on(b.followed_by(a, gap=(0, "1s")), data).count == 1


def test_duration_keeps_the_events_of_the_right_length():
    data = pulses([1.0, 2.0], width=0.05)
    data.values[2000:2200] = 1.0
    events = (mmp_sense.signal("v") > 0.5).events()
    assert run_on(events.lasting(min="100ms"), data).count == 1
    assert run_on(events.lasting(max="100ms"), data).count == 1


# -- the graph as a thing to look at -----------------------------------


def knock_graph():
    accel = mmp_sense.signal("accel_x", "accel_y", "accel_z")
    hp = accel.magnitude().highpass(cutoff_hz=10, name="hp")
    energy = hp.energy("80ms", name="energy")
    freq = hp.frames("80ms").dominant_hz(name="freq")
    strong = (energy > 0.25).named("strong")
    knock = (strong & freq.between(60, 150)).events(name="knock", refractory="100ms")
    return knock.followed_by(knock, gap=("80ms", "500ms"), name="double_knock")


def test_the_graph_prints_as_a_tree_down_to_the_source():
    text = knock_graph().graph()
    lines = text.splitlines()
    assert lines[0] == "double_knock: followed_by(gap 80ms..500ms)"
    assert "├── knock: events(refractory=100ms)" in text
    assert "energy: energy(80ms)" in text
    assert "hp: highpass(10Hz)" in text
    assert "signal[accel_x,accel_y,accel_z]" in text
    assert "between 60..150" in text
    assert lines[-1] == "└── knock: events(refractory=100ms) (above)"


def test_latency_counts_the_slowest_path():
    graph = knock_graph()
    assert graph.find("energy").latency == pytest.approx(0.08)
    assert graph.latency == pytest.approx(0.6)
    assert graph.causal and graph.stateful and graph.windowed
    timing = graph.find("hp").timing
    assert timing.stateful and not timing.windowed and timing.delay_s == 0


def test_a_graph_survives_plain_data():
    graph = knock_graph()
    data = graph.to_dict()
    assert data["output"] == "double_knock"
    assert data["nodes"]["energy"] == {"op": "energy", "inputs": ["hp"],
                                       "params": {"length": 0.08}, "name": "energy"}
    import json

    again = Node.from_dict(json.loads(json.dumps(data)))
    assert again.key == graph.key


def test_with_params_changes_every_use_of_a_named_node():
    graph = knock_graph()
    changed = graph.with_params("knock", refractory="200ms")
    assert changed.find("knock").params["refractory"] == pytest.approx(0.2)
    assert all(n.params["refractory"] == pytest.approx(0.2)
               for n in changed.walk() if n.op == "events")
    assert graph.find("knock").params["refractory"] == pytest.approx(0.1)
    with pytest.raises(KeyError):
        graph.with_params("nothing", length=1)
    with pytest.raises(TypeError):
        graph.with_params("energy", widow=1)


def test_a_changed_threshold_recomputes_only_what_depends_on_it(session):
    session.stop_intake()
    graph = knock_graph()
    run(graph, session, session.evaluator)
    energy = graph.find("energy")
    assert session.evaluator.count(energy) == 1

    tighter = graph.with_params("strong", value=0.5)
    result = run(tighter, session, session.evaluator)
    assert session.evaluator.count(energy) == 1
    assert session.evaluator.count(tighter.find("strong")) == 1
    assert result.execution_metadata["nodes"]["energy"]["computed"] is False
    assert result.execution_metadata["nodes"]["strong"]["computed"] is True


def test_new_data_is_computed_afresh(session, clock):
    graph = knock_graph()
    run(graph, session, session.evaluator)
    clock.advance(1.0)
    session.poll()
    run(graph, session, session.evaluator)
    assert session.evaluator.count(graph.find("energy")) == 2


def test_a_custom_op_needs_no_change_to_mmp_sense():
    @register_op("test_clip", description="Limit to +-limit")
    def clip(ctx, series, limit=1.0):
        return series.with_values(np.clip(series.values, -limit, limit))

    data = series_of([-3.0, 0.5, 3.0], 1.0, ["v"])
    clipped = mmp_sense.signal("v").apply("test_clip", limit=2.0, name="clipped")
    assert list(evaluate(clipped, data).values) == [-2.0, 0.5, 2.0]
    assert list(evaluate(clipped > 1.5, data).values) == [False, False, True]
    assert "clipped: test_clip(limit=2)" in (clipped > 1.5).graph()
    assert clipped.with_params("clipped", limit=0.1).params["limit"] == 0.1


def test_an_op_that_returns_the_wrong_thing_is_named():
    @register_op("test_wrong")
    def wrong(ctx, series):
        return series.values

    with pytest.raises(GraphError, match="test_wrong"):
        evaluate(mmp_sense.signal("v").apply("test_wrong"), series_of([1.0], 1.0, ["v"]))


def test_vectors_and_components():
    data = series_of([[1.0, 2.0], [3.0, 4.0]], 1.0, ["x", "y"])
    accel = mmp_sense.signal("x", "y")
    assert evaluate(accel, data).channels == ["x", "y"]
    assert list(evaluate(accel["y"], data).values) == [2.0, 4.0]
    assert list(evaluate(accel.energy(1)[0], data).values) == [1.0, 9.0]


# -- running it ----------------------------------------------------------


def test_run_keeps_every_step_and_what_it_was_run_with(session):
    graph = knock_graph()
    result = run_on(graph, session)
    assert result.count == len(result.timestamps) == len(result.events)
    assert set(result.intermediates) >= {"hp", "energy", "freq", "strong", "knock", "double_knock"}
    assert isinstance(result["energy"], Series)
    assert result[graph.find("knock")] is result.intermediates["knock"]
    assert result.parameters == graph.to_dict()
    meta = result.execution_metadata
    assert meta["samples"] == len(session.buffer)
    assert meta["latency_s"] == pytest.approx(0.6)
    assert meta["nodes"]["energy"]["length_s"] == pytest.approx(0.08)


def test_the_simulated_knock_is_found(session):
    knock = knock_graph().find("knock")
    found = run_on(knock, session).timestamps
    assert len(found) == 1
    assert found[0] == pytest.approx(5.0, abs=0.06)


def test_load_and_run_a_csv_without_a_window(tmp_path):
    from mmp_sense.io.csv import write_timeseries

    path = write_timeseries(pulses([1.0, 1.3, 2.5]), tmp_path / "bench.csv")
    recording = mmp_sense.load(path)
    assert not recording.running
    a = (mmp_sense.signal("v") > 0.5).events()
    assert run_on(a, recording).count == 3
    assert run_on(a, str(path)).count == 3


def test_the_core_runs_without_qt():
    code = (
        "import sys, numpy as np, pandas as pd, mmp_sense\n"
        "t = np.arange(1000) / 1000.0\n"
        "s = mmp_sense.from_frame(pd.DataFrame({'v': np.sin(2*np.pi*200*t)}, index=t))\n"
        "r = s.events((mmp_sense.signal('v').energy('10ms') > 1).events())\n"
        "assert len(r) == 1, r\n"
        "assert not any(m.startswith(('PySide6', 'pyqtgraph')) for m in sys.modules)\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# -- in the session ------------------------------------------------------


def test_show_publishes_values_as_signals_and_events_as_detections(session):
    heard = []
    session.listen(lambda kind, payload: heard.append(kind))
    graph = knock_graph()
    names = session.show(graph.find("energy"), graph.find("knock"), graph)
    assert names == ["energy", "knock", "double_knock"]
    assert session.signals.get("energy").source == "graph:energy"
    assert session.detection_names == ["knock", "double_knock"]
    assert "detections" in heard and "chart" in heard

    window = session.window()
    energy = session.values(graph.find("energy"))
    assert np.allclose(window["energy"], energy.values, equal_nan=True)
    assert np.allclose(session.values("energy").values, energy.values, equal_nan=True)
    assert len(session.events("knock")) == 1


def test_showing_a_name_again_replaces_it_and_hide_removes_it(session):
    graph = knock_graph()
    session.show(graph.find("knock"))
    stricter = graph.find("knock").with_params("strong", value=1e9)
    session.show(stricter)
    assert session.shown["knock"] is stricter
    assert len(session.events("knock")) == 0
    session.hide("knock")
    assert session.detection_names == []
    session.show(graph.find("hp"))
    session.hide("hp")
    assert "hp" not in session.signals


def test_an_undrawn_detector_is_still_shown_and_computed(session):
    heard = []
    graph = knock_graph()
    session.show(graph.find("knock"), graph, graph.find("energy"))
    session.listen(lambda kind, payload: heard.append(kind))

    session.set_drawn("knock", False)
    assert not session.drawn("knock") and session.drawn("double_knock")
    assert "knock" in session.shown and session.detection_names == ["knock", "double_knock"]
    assert len(session.events("knock")) == 1
    assert heard == ["detections"]
    session.set_drawn("knock", False)
    assert heard == ["detections"]          # no change, no news

    session.set_drawn("knock", True)
    assert session.drawn("knock")
    with pytest.raises(KeyError):
        session.set_drawn("energy", False)   # a signal, not events
    with pytest.raises(KeyError):
        session.set_drawn("nothing", False)


def test_showing_again_draws_and_hiding_forgets(session):
    graph = knock_graph()
    session.show(graph.find("knock"))
    session.set_drawn("knock", False)
    session.show(graph.find("knock").with_params("strong", value=1e9))
    assert session.drawn("knock")
    session.set_drawn("knock", False)
    session.hide("knock")
    assert not session.drawn("knock")
    session.show(graph.find("knock"))
    assert session.drawn("knock")


def test_unnamed_nodes_are_given_names_and_vectors_components(session):
    names = session.show(mmp_sense.signal().highpass(cutoff_hz=5), plot=False)
    assert names == ["highpass"]
    assert [n for n in session.signals.names if n.startswith("highpass.")] == [
        "highpass.accel_x", "highpass.accel_y", "highpass.accel_z"]
    assert len(session.window()["highpass.accel_z"]) == len(session.buffer)


def test_show_takes_only_nodes(session):
    with pytest.raises(TypeError):
        session.show("energy")


# -- the demo ------------------------------------------------------------


def test_the_demo_needs_both_features():
    from mmp_sense.examples import double_knock as demo

    table, truth = demo.synthesize(duration=30.0)
    recording = mmp_sense.from_frame(table)
    nodes = demo.detector()

    def scored(name, expected):
        return demo.score(recording.events(nodes[name])["start"], expected)

    assert scored("knock", truth["knock"]) == {
        "expected": len(truth["knock"]), "found": len(truth["knock"]),
        "hits": len(truth["knock"]), "missed": 0, "false": 0}
    double = scored("double_knock", truth["double_knock"])
    assert double["missed"] == double["false"] == 0
    assert scored("loud", truth["knock"])["false"] > 0
    assert scored("pitched", truth["knock"])["false"] > 0


def test_redefining_an_op_is_not_served_from_the_cache(session):
    session.stop_intake()

    @register_op("test_scaled")
    def scaled(ctx, series):
        return series.with_values(series.values * 2)

    node = mmp_sense.signal("accel_z").apply("test_scaled")
    first = session.values(node).values[0]

    @register_op("test_scaled")
    def scaled_again(ctx, series):
        return series.with_values(series.values * 3)

    assert session.values(node).values[0] == pytest.approx(first * 1.5)


def test_a_misspelt_parameter_is_caught_when_the_node_is_made():
    with pytest.raises(TypeError, match="widow"):
        mmp_sense.signal().apply("energy", widow=0.1)


def test_every_parameter_that_was_set_is_printed():
    text = (mmp_sense.signal("v") > 1).events(refractory=1, min_duration="10ms").graph()
    assert text.splitlines()[0] == "events(refractory=1s, min_duration=10ms)"


def test_a_node_named_like_a_channel_is_refused_whole(session):
    with pytest.raises(ValueError, match="channel"):
        session.show(mmp_sense.signal().magnitude(name="accel_x"))
    assert session.shown == {}


def test_another_dataset_is_never_served_this_ones_results():
    from mmp_sense import Session, SeriesSource

    loud = series_of(np.r_[np.zeros(500), np.ones(500)], 1000.0, ["v"])
    quiet = series_of(np.zeros(1000), 1000.0, ["v"])
    session = Session(SeriesSource(loud))
    session.poll()
    hits = (mmp_sense.signal("v") > 0.5).events(name="hits")
    session.show(hits)
    for i in range(50):
        data, expected = (quiet, 0) if i % 2 else (loud, 1)
        session.attach(SeriesSource(data), keep_signals=False)
        session.poll()
        session.show(hits)
        assert len(session.events("hits")) == expected


def test_windowed_ops_stack_and_a_gap_stays_local():
    rate = 1000.0
    t = np.arange(2000) / rate
    v = np.sin(2 * np.pi * 50 * t)
    v[1000] = np.nan
    data = series_of(v, rate, ["v"])
    x = mmp_sense.signal("v")
    for node in (x.energy("10ms").energy("10ms"), x.peak("10ms").energy("10ms"),
                 x.energy("10ms").highpass(cutoff_hz=5), x.highpass(cutoff_hz=5).peak("10ms"),
                 x.zero_crossing_rate("50ms")):
        values = evaluate(node, data).values
        assert np.isfinite(values[1100:]).all(), node.graph()
        assert np.isfinite(values[100:990]).all(), node.graph()
    assert np.isnan(dsp.trailing_max(np.array([1.0, np.nan, 0.0, 0.0]), 2)[1:3]).all()


def test_the_highpass_restarts_at_rest_after_a_gap():
    out, state = dsp.highpass1(np.array([1.0, 2.0, np.nan, 5.0, 6.0]), 100.0, 5.0)
    assert np.isnan(out[2]) and out[3] == 0.0 and np.isfinite(out[4])
    x = np.random.default_rng(2).normal(size=(50, 2))
    x[10, 0] = x[29, 1] = np.nan
    whole, _ = dsp.highpass1(x, 100.0, 5.0)
    for split in (20, 30):
        first, state = dsp.highpass1(x[:split], 100.0, 5.0)
        second, _ = dsp.highpass1(x[split:], 100.0, 5.0, state)
        assert np.allclose(whole, np.concatenate([first, second]), equal_nan=True)


def test_showing_steps_does_not_recompute_what_they_share(session):
    session.stop_intake()
    energy = mmp_sense.signal().magnitude().energy("50ms", name="energy")
    strong = (energy > 0.5).named("strong")
    knock = strong.events(name="knock")
    session.show(energy, strong, knock)
    session.window()["energy"]
    assert session.evaluator.count(energy) == 1
    session.show(knock.with_params("strong", value=1.0))
    session.events("knock")
    assert session.evaluator.count(energy) == 1


def test_a_redefined_signal_is_read_afresh(session):
    session.stop_intake()
    session.show((mmp_sense.signal("accel_z") * 0).named("level"))
    node = mmp_sense.signal("level").energy("10ms")
    assert np.nanmax(session.values(node).values) == 0
    session.show((mmp_sense.signal("accel_z") * 0 + 1).named("level"))
    assert np.nanmax(session.values(node).values) == pytest.approx(5.0)


def test_ids_never_collide_with_names():
    graph = mmp_sense.signal("x", name="n1").magnitude().highpass(cutoff_hz=1, name="n0")
    data = graph.to_dict()
    assert len(data["nodes"]) == 3
    assert Node.from_dict(data).key == graph.key


def test_latency_is_honest_about_hops_and_open_ends():
    x = mmp_sense.signal("x")
    assert x.frames("80ms").latency == pytest.approx(0.1)
    assert x.frames("80ms", "10ms").dominant_hz().latency == pytest.approx(0.09)
    events = (x > 0).events()
    assert events.followed_by(events).latency == float("inf")


def test_the_frame_length_is_tunable_by_name():
    freq = mmp_sense.signal("v").frames("80ms", name="frames").dominant_hz(name="freq")
    tuned = (freq > 100).with_params("frames", length="40ms")
    assert tuned.find("frames").params["length"] == pytest.approx(0.04)
    rate = 1000.0
    t = np.arange(1000) / rate
    out = evaluate(tuned.find("freq"), series_of(np.sin(2 * np.pi * 100 * t), rate, ["v"]))
    assert np.allclose(out.values, 100.0)
    assert "frames: frames(length=40ms)" in tuned.graph()
