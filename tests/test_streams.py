"""Several streams in one session: names, time, and recipes across them."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import mmp_sense
from mmp_sense import SignalError
from mmp_sense.core.buffer import ChannelBuffer
from mmp_sense.core.data import TimeSeries
from mmp_sense.core.graph import Evaluator, hold, signal
from mmp_sense.core.graph.evaluate import home_of
from mmp_sense.core.session import Session, stream_names
from mmp_sense.core.signals import SignalTable, Stream
from mmp_sense.sources.recording import SeriesSource

A_RATE, B_RATE = 1000.0, 400.0


def _series(name, rate, duration, channels, offset=0.0, seed=0):
    rng = np.random.default_rng(seed)
    time = offset + np.arange(int(duration * rate)) / rate
    values = np.stack([np.sin(2 * np.pi * (3 + i) * time) + 0.3 * rng.standard_normal(len(time))
                       for i in range(len(channels))], axis=1)
    return TimeSeries(name=name, time=time, values=values, channels=channels,
                      sample_rate=rate)


@pytest.fixture
def two():
    session = Session(SeriesSource(_series("/phone/a", A_RATE, 6.0, ["x", "y"])))
    session.add(SeriesSource(_series("/phone/b", B_RATE, 5.0, ["x", "z"], offset=0.5, seed=1)))
    while session.poll():
        pass
    return session


# -- names ---------------------------------------------------------------


def test_streams_are_named_from_their_sources():
    class Named:
        def __init__(self, name):
            self.name = name

    assert stream_names([Named("/a/mic"), Named("/b/accel")]) == ["mic", "accel"]
    assert stream_names([Named("/a/mic"), Named("/b/mic")]) == ["a_mic", "b_mic"]
    assert stream_names([Named("/a/mic")], taken=["mic"]) == ["a_mic"]
    assert stream_names([Named("mic")], taken=["mic"]) == ["mic2"]
    assert stream_names([Named("x:y")]) == ["x_y"]


def test_with_several_streams_channels_are_listed_by_full_name(two):
    assert list(two.streams) == ["a", "b"]
    assert two.channels == ["a:x", "a:y", "b:x", "b:z"]
    assert two.groups["a"] == ["a:x", "a:y"] and two.groups["b"] == ["b:x", "b:z"]


def test_a_bare_name_finds_the_one_stream_that_has_it(two):
    assert two.signals.canonical("y") == "a:y"
    assert two.signals.stream_of("z") == "b"
    assert two.values("z").equals(two.values("b:z"))


def test_an_ambiguous_bare_name_says_which_streams_have_it(two):
    with pytest.raises(SignalError, match="a:x, b:x"):
        two.values("x")
    with pytest.raises(SignalError, match="several streams"):
        two.window()["x"]


def test_one_stream_keeps_its_bare_names_and_answers_to_full_ones():
    session = Session(SeriesSource(_series("/phone/a", A_RATE, 1.0, ["x", "y"])))
    session.poll()
    assert session.channels == ["x", "y"]
    assert list(session.streams) == ["a"]
    assert session.values("a:x").equals(session.values("x"))


def test_removing_a_stream_brings_back_bare_names(two):
    two.remove("b")
    assert two.channels == ["x", "y"]
    with pytest.raises(ValueError, match="at least one"):
        two.remove("a")


def test_what_is_plotted_keeps_meaning_the_same_when_a_stream_is_added():
    session = Session(SeriesSource(_series("/phone/a", A_RATE, 1.0, ["x", "y"])))
    assert session.active_signals == ["x", "y"]
    session.add(SeriesSource(_series("/phone/b", B_RATE, 1.0, ["x", "z"])))
    assert session.active_signals == ["a:x", "a:y", "b:x", "b:z"]
    session.remove("a")
    assert session.active_signals == ["x", "z"]


def test_a_recipe_may_not_take_a_channels_bare_name(two):
    with pytest.raises(ValueError, match="is a channel"):
        two.show((signal("a:x") * 2).named("x"))
    with pytest.raises(SignalError, match="is a channel"):
        two.signals.define("z", lambda window: window["a:y"])


def test_a_stream_whose_channel_is_named_like_a_shown_signal_is_refused():
    session = Session(SeriesSource(_series("/phone/a", A_RATE, 1.0, ["x", "y"])))
    session.show((signal("x") * 2).named("z"))
    source = SeriesSource(_series("/phone/b", B_RATE, 1.0, ["x", "z"]))
    with pytest.raises(ValueError, match="z: a shown signal"):
        session.add(source)
    assert list(session.streams) == ["a"]
    with pytest.raises(ValueError, match="taken"):
        session.add(source, name="z")


def test_a_stream_cannot_be_named_with_the_separator():
    buffer = ChannelBuffer(["x"])
    with pytest.raises(SignalError, match="':'"):
        SignalTable(streams=[Stream("a:b", buffer)])


# -- time ----------------------------------------------------------------


def test_the_span_and_the_cursor_cover_every_stream(two):
    t0, t1 = two.span
    assert t0 == pytest.approx(0.0)
    assert t1 == pytest.approx(max(6.0 - 1 / A_RATE, 5.5 - 1 / B_RATE))
    two.set_cursor(100.0)
    assert two.cursor == pytest.approx(t1)


def test_each_stream_keeps_its_own_rate_and_times(two):
    window = two.window()
    other = window.of("b")
    assert window.rate == A_RATE and other.rate == B_RATE
    assert other.t_start == pytest.approx(0.5)
    assert window.home("b:z") is other
    # Asked of the other stream's window, a channel is held onto its times.
    held = other["a:x"]
    index = np.searchsorted(window.time, other.time, side="right") - 1
    np.testing.assert_allclose(held, window["a:x"][index])


def test_playback_runs_over_every_stream(two):
    two.play(0.0, speed=2.0)
    two.tick(now=two.clock() + 1.0)
    assert two.cursor == pytest.approx(2.0, abs=0.1)
    two.tick(now=two.clock() + 100.0)
    assert not two.playing
    assert two.cursor == pytest.approx(two.span[1])


def test_restarting_intake_empties_every_stream(two):
    two.start_intake()
    assert all(len(s.buffer) == 0 for s in two.streams.values())
    while two.poll():
        pass
    assert all(len(s.buffer) > 0 for s in two.streams.values())


# -- recipes across streams --------------------------------------------------


def test_a_recipe_across_streams_stops_where_the_earlier_stream_does(two):
    both = (signal("a:x") - signal("b:x")).named("both")
    values = two.values(both)
    b_end = two.streams["b"].buffer.t_end
    assert values.index[-1] <= b_end
    assert values.index[-1] > b_end - 2 / A_RATE
    # On a's times, b's value is the latest at or before each.
    a = two.values("a:x")
    b = two.values("b:x")
    t = values.index[len(values) // 2]
    expected = a[t] - b[b.index <= t].iloc[-1]
    assert values[t] == pytest.approx(expected)


def test_a_shown_recipe_is_drawn_on_its_first_streams_times(two):
    two.show((signal("b:z").rms("50ms") > signal("a:y").peak("20ms")).named("louder"))
    assert two.signals.stream_of("louder") == "b"
    window = two.window()
    assert len(window.home("louder")["louder"]) == len(window.of("b"))


def test_a_signal_of_two_streams_at_once_is_refused(two):
    with pytest.raises(Exception, match="several streams"):
        two.values(signal("a:x", "b:z"))


class Pair:
    """Two live buffers that scroll, and the same two keeping everything."""

    def __init__(self, span_s: float = 2.0, rates=(A_RATE, B_RATE)) -> None:
        self.rates = rates

        def streams(span, cap):
            return [Stream("a", ChannelBuffer(["x", "y"], span_s=span, rate=rates[0],
                                              max_capacity=cap(rates[0]))),
                    Stream("b", ChannelBuffer(["x", "z"], span_s=span, rate=rates[1],
                                              max_capacity=cap(rates[1])))]
        self.live = SignalTable(streams=streams(span_s, lambda r: int(span_s * r)))
        self.whole = SignalTable(streams=streams(1e6, lambda r: 1 << 23))
        self.evaluator = Evaluator()

    def append(self, stream, time, values) -> None:
        self.live.streams[stream].buffer.append(time, values)
        self.whole.streams[stream].buffer.append(time, values)

    def check(self, node, context):
        window = self.live.window()
        result = self.evaluator.evaluate(node, window)
        whole = self.whole.window()
        whole.live = None
        for other in ("a", "b"):
            whole.of(other).live = None
        expected = Evaluator().evaluate(node, whole)
        start = window.of(home_of(node, self.live)).t_start
        if node.output == "events":
            got = result.table
            table = expected.table
            table = table[table["timestamp"] >= start].reset_index(drop=True)
            assert list(got["timestamp"]) == list(table["timestamp"]), context
            self.events = len(got)
            return
        keep = expected.time >= start
        np.testing.assert_array_equal(result.time, expected.time[keep], err_msg=context)
        np.testing.assert_allclose(np.asarray(result.values, float),
                                   np.asarray(expected.values, float)[keep],
                                   rtol=1e-9, atol=1e-9, equal_nan=True, err_msg=context)


def _feed(pair, nodes, seed=0, every=1, lag=0.5, offset=0.0013):
    """Both streams in uneven pieces, either ahead of the other by up to
    `lag` seconds -- less than the live buffers hold, or what one has not
    yet been matched with would scroll out before it could be."""
    a = _series("a", pair.rates[0], 8.0, ["x", "y"], seed=seed)
    b = _series("b", pair.rates[1], 8.0, ["x", "z"], offset=offset, seed=seed + 1)
    rng = np.random.default_rng(seed)
    ia = ib = 0
    tick = 0
    while ia < a.n_samples or ib < b.n_samples:
        ta = a.time[ia] if ia < a.n_samples else np.inf
        tb = b.time[ib] if ib < b.n_samples else np.inf
        if ta < tb - lag:
            take_a = True
        elif tb < ta - lag:
            take_a = False
        else:
            take_a = ia < a.n_samples and (ib >= b.n_samples or rng.random() < 0.5)
        if take_a:
            n = int(rng.integers(1, 0.3 * pair.rates[0]))
            pair.append("a", a.time[ia:ia + n], a.values[ia:ia + n])
            ia += n
        else:
            n = int(rng.integers(1, 0.3 * pair.rates[1]))
            pair.append("b", b.time[ib:ib + n], b.values[ib:ib + n])
            ib += n
        tick += 1
        if tick % every:
            continue
        for node in nodes:
            pair.check(node, f"{node.title} at tick {tick}")


hp = signal("a:x").highpass(5.0)
ACROSS = {
    "difference": hp - signal("b:x"),
    "other stream first": signal("b:z").lowpass(20.0) * signal("a:y").rms("30ms"),
    "compare": signal("b:z").rms("50ms") > signal("a:x").peak("20ms"),
    "against frames": hp.energy("80ms") - signal("b:x").frames("100ms", "20ms").dominant_hz(),
    "and": (hp.energy("80ms") > 5.0) & (signal("b:z").peak("40ms") > 0.9),
    "events": ((hp.energy("80ms") > 8.0) & (signal("b:z") > 0.5)).events(refractory="50ms"),
    "delay": signal("a:x").delay(signal("b:x"), "100ms", "25ms", max_lag="20ms"),
    "delay, other stream first": signal("b:z").delay(signal("a:y"), "200ms", max_lag="30ms"),
    "followed_by": (hp.peak("20ms") > 1.2).events(refractory="300ms").followed_by(
        (signal("b:z").peak("20ms") > 1.0).events(refractory="300ms"), gap=(0.0, 0.5)),
}


@pytest.mark.parametrize("name", list(ACROSS))
def test_a_recipe_across_streams_carried_on_is_the_whole_streams(name):
    pair = Pair()
    _feed(pair, [ACROSS[name]])
    if ACROSS[name].output == "events":
        assert pair.events > 0


def test_a_recipe_across_streams_starts_again_with_the_intake():
    pair = Pair()
    nodes = [ACROSS["difference"], ACROSS["against frames"], ACROSS["events"]]
    _feed(pair, nodes, every=9)
    for stream in ("a", "b"):
        pair.live.streams[stream].buffer.clear()
    pair.whole = Pair().whole
    _feed(pair, nodes, seed=2, every=9)


def test_a_recipe_across_streams_starts_again_when_the_session_does():
    a = _series("/p/a", A_RATE, 3.0, ["x", "y"])
    b = _series("/p/b", B_RATE, 3.0, ["x", "z"], offset=0.0013, seed=1)
    clock = [0.0]
    sources = [SeriesSource(x, replay=True, clock=lambda: clock[0]) for x in (a, b)]
    session = Session(buffer_s=10.0)
    for source in sources:
        session.add(source)
    node = (signal("a:x").highpass(5.0) - signal("b:x")).named("d")
    session.show(node)

    def run():
        for _ in range(30):
            clock[0] += 0.1
            session.poll()
            session.window().home("d")["d"]          # a tick draws it
        whole = Session(buffer_s=10.0)
        for x in (a, b):
            whole.add(SeriesSource(x))
        while whole.poll():
            pass
        np.testing.assert_allclose(session.values(node).to_numpy(),
                                   whole.values(node).to_numpy(), equal_nan=True)

    run()
    session.start_intake()
    clock[0] = 100.0
    for source in sources:
        source.rewind()
    run()


def test_a_delay_across_a_gap_in_the_other_stream_is_the_whole_streams():
    pair = Pair()
    original = pair.append

    def gapped(stream, time, values):
        if stream == "b":
            values = values.copy()
            values[(time > 3.0) & (time < 3.1)] = np.nan
        original(stream, time, values)

    pair.append = gapped
    _feed(pair, [ACROSS["delay"]])


def test_several_recipes_share_their_steps_and_stay_right():
    _feed(Pair(), list(ACROSS.values()), seed=4, every=7)


@pytest.mark.parametrize("name", ["difference", "other stream first", "against frames"])
def test_streams_of_one_rate_whose_samples_line_up_are_not_confused(name):
    # Equal rates, a third of a sample apart: the two streams' sample
    # numbers coincide, and must not be taken for the same samples.
    _feed(Pair(rates=(1000.0, 1000.0)), [ACROSS[name]], offset=0.0003, every=2)


def test_a_one_stream_recipe_held_onto_another_streams_times_waits_for_it():
    pair = Pair(rates=(1000.0, 100.0))
    node = signal("b:z").lowpass(10.0)
    a = _series("a", 1000.0, 3.0, ["x", "y"])
    b = _series("b", 100.0, 3.0, ["x", "z"])
    for i in range(0, 3000, 50):
        pair.append("a", a.time[i:i + 50], a.values[i:i + 50])
        held = len(pair.whole.streams["b"].buffer)
        j = max(held, i // 10 - 5)                  # b is 50 ms behind a
        pair.append("b", b.time[held:j], b.values[held:j])
        window = pair.live.window()
        got = pair.evaluator.held(node, window)
        expected = np.asarray(hold(Evaluator().evaluate(node, _whole(pair)), window.time),
                              dtype=float)
        final = window.time <= pair.live.streams["b"].buffer.t_end
        np.testing.assert_allclose(got[final], expected[final], equal_nan=True,
                                   err_msg=f"at {i}")
        assert np.isnan(got[~final]).all()


def test_a_one_stream_recipe_beside_one_across_streams_is_not_recomputed():
    from mmp_sense.core.graph import evaluate

    pair = Pair()
    alone = hp.rms("30ms")
    across = alone - signal("b:x")
    a = _series("a", A_RATE, 4.0, ["x", "y"])
    b = _series("b", B_RATE, 4.0, ["x", "z"])
    first = next(evaluate._GENERATIONS)
    for i in range(0, 4000, 100):
        pair.append("a", a.time[i:i + 100], a.values[i:i + 100])
        pair.append("b", b.time[i * 2 // 5:(i + 100) * 2 // 5], b.values[i * 2 // 5:(i + 100) * 2 // 5])
        window = pair.live.window()
        pair.evaluator.evaluate(alone, window)
        pair.evaluator.evaluate(across, window)
    # Eight steps between the two (three alone, five across): each was
    # computed afresh once, and carried on for the other 39 ticks.
    assert next(evaluate._GENERATIONS) - first - 1 == 8


def test_a_recipe_across_streams_is_held_as_nan_past_what_both_hold():
    pair = Pair(span_s=1e3)
    a = _series("a", A_RATE, 2.0, ["x", "y"])
    b = _series("b", B_RATE, 1.0, ["x", "z"])
    pair.append("a", a.time, a.values)
    pair.append("b", b.time, b.values)
    node = (signal("a:x") - signal("b:x")).named("d")
    window = pair.live.window()
    held = pair.evaluator.held(node, window)
    assert np.isnan(held[window.time > b.time[-1]]).all()
    assert np.isfinite(held[(window.time <= b.time[-1]) & (window.time >= b.time[0])]).all()
    # Carried on as b catches up, the values once NaN are given.
    more = _series("b", B_RATE, 2.0, ["x", "z"])
    pair.append("b", more.time[len(b.time):], more.values[len(b.time):])
    again = pair.evaluator.held(node, pair.live.window())
    expected = pair.evaluator.held(node, _whole(pair))
    np.testing.assert_allclose(again, expected, equal_nan=True)


def _whole(pair):
    window = pair.whole.window()
    window.live = None
    return window


# -- the sample type ---------------------------------------------------------


class Pcm:
    """A source of 16-bit samples, eight channels."""

    name = "/array/mic"
    kind = "live"
    channels = [f"m{i}" for i in range(8)]
    units = ""
    rate = 16000.0
    dtype = np.dtype(np.int16)

    def __init__(self):
        self.sent = False

    def start(self):
        return self

    def stop(self):
        pass

    def restart(self):
        self.sent = False
        return self

    def read(self):
        if self.sent:
            return np.zeros(0), np.zeros((0, 8), np.int16)
        self.sent = True
        time = np.arange(16000) / self.rate
        values = (np.sin(2 * np.pi * 440 * time)[:, None] * 12000
                  * np.arange(1, 9) / 8).astype(np.int16)
        return time, values


def test_a_16_bit_stream_is_held_as_16_bit_and_computed_as_floats():
    session = Session(Pcm())
    session.poll()
    assert session.buffer.values.dtype == np.int16
    assert session.window().values.dtype == np.int16
    rms = session.values(signal("m7").rms("10ms"))
    assert rms.dtype == float
    assert rms.iloc[-1] == pytest.approx(12000 / np.sqrt(2), rel=0.01)
    assert session.values("m3").dtype == float
    energy = session.values(signal("m7").energy("10ms"))
    assert energy.iloc[-1] > 1e8                       # no 16-bit overflow


def test_a_16_bit_stream_gives_what_the_same_samples_as_floats_give():
    pcm = Pcm()
    time, values = pcm.read()

    class Floats(Pcm):
        dtype = np.dtype(np.float64)

        def read(self):
            if self.sent:
                return np.zeros(0), np.zeros((0, 8))
            self.sent = True
            return time, values.astype(float)

    as_int, as_float = Session(Pcm()), Session(Floats())
    as_int.poll()
    as_float.poll()
    recipes = [signal("m7").highpass(50.0).rms("10ms"),
               signal("m0", "m7").magnitude().peak("5ms"),
               (signal("m3") * signal("m5")).named("product"),
               signal("m7").frames("32ms", "8ms").dominant_hz(),
               (signal("m7").energy("5ms") > 1e9).events()]
    for recipe in recipes:
        if recipe.output == "events":
            assert as_int.events(recipe).equals(as_float.events(recipe))
            continue
        np.testing.assert_allclose(as_int.values(recipe).to_numpy(),
                                   as_float.values(recipe).to_numpy(), equal_nan=True,
                                   err_msg=recipe.title)


# -- the window --------------------------------------------------------------


def test_a_pane_draws_each_stream_on_its_own_times(qapp, two):
    pytest.importorskip("pyqtgraph")
    from mmp_sense.ui.main_window import MainWindow

    window = MainWindow(two, window_s=10.0, console=False)
    window.show()
    window.tick()
    try:
        pane = window.panes[0]
        assert set(pane.signal_names()) == {"a:x", "a:y", "b:x", "b:z"}
        a_x = pane._curves["a:x"].xData
        b_z = pane._curves["b:z"].xData
        np.testing.assert_allclose(a_x, two.streams["a"].buffer.snapshot()[0])
        np.testing.assert_allclose(b_z, two.streams["b"].buffer.snapshot()[0])
        assert "b: " in window.statusBar().currentMessage()
        # The spectrum of b is at b's rate.
        spectrum = [p for p in window.panes if p.kind == "spectrum"][0]
        spectrum.chooser.setCurrentText("b:z")
        two.set_cursor(3.0)
        window.tick()
        assert "Hz/bin" in spectrum.status.text()
        resolution = float(spectrum.status.text().split(", ")[1].split(" Hz")[0])
        assert resolution == pytest.approx(B_RATE / round(0.5 * B_RATE), rel=0.05)
    finally:
        window.close()


def test_the_signal_list_groups_signals_by_stream(qapp, two):
    pytest.importorskip("pyqtgraph")
    from mmp_sense.ui.widgets.signal_dock import SignalDock

    two.show((signal("b:z").rms("50ms")).named("loud"))
    dock = SignalDock(two)
    tops = [dock.tree.topLevelItem(i) for i in range(dock.tree.topLevelItemCount())]
    assert [t.text(0) for t in tops] == ["a", "b"]
    assert [tops[1].child(i).text(0) for i in range(tops[1].childCount())] == \
        ["b:x", "b:z", "loud"]


def test_the_assistant_sees_the_streams_and_asks_by_full_name(two):
    from mmp_sense.mcp.tools import call

    context = call(two, "get_current_context")
    assert [s["name"] for s in context["streams"]] == ["a", "b"]
    assert context["streams"][1]["rate"] == B_RATE
    listed = call(two, "list_signals")
    assert {"name": "b:z", "kind": "channel", "units": "", "stream": "b",
            "description": "from b"} in listed["signals"]
    stats = call(two, "get_statistics", {"signal": "b:z"})
    assert stats["stats"]["n"] == two.streams["b"].buffer.size


def test_a_stream_added_to_a_stopped_session_brings_what_it_has():
    session = Session(SeriesSource(_series("/p/a", A_RATE, 1.0, ["x", "y"])))
    session.poll()
    session.stop_intake()
    session.add(SeriesSource(_series("/p/b", B_RATE, 1.0, ["x", "z"])))
    assert len(session.streams["b"].buffer) == int(B_RATE)
    assert not session.running
