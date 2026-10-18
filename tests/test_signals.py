"""The signal table: channels and shown signals, computed per window."""

import numpy as np
import pytest

import mmp_sense
from mmp_sense import SignalError
from mmp_sense.core.buffer import ChannelBuffer
from mmp_sense.core.signals import SignalTable


@pytest.fixture
def table():
    buffer = ChannelBuffer(["x", "y"], span_s=10.0, rate=100.0, units="m/s^2")
    time = np.arange(1000) / 100.0
    buffer.append(time, np.stack([np.sin(time), np.cos(time)], axis=1))
    return SignalTable(buffer)


def test_channels_come_from_the_source(table):
    assert table.channels == ["x", "y"]
    assert table.names == ["x", "y"]
    assert table.window()["x"][:3] == pytest.approx(np.sin(np.arange(3) / 100.0))


def test_a_derived_signal_sees_the_window_and_follows_the_data(table):
    table.define("energy", lambda window: window["x"] ** 2)
    window = table.window(0.0, 0.1)
    assert window["energy"] == pytest.approx(window["x"] ** 2)
    table.buffer.append([10.0], [[0.0, 1.0]])
    assert len(table.window()["energy"]) == 1001


def test_a_window_computes_a_derived_signal_once(table):
    calls = []

    def count(window):
        calls.append(1)
        return window["x"] * 2

    table.define("twice", count)
    window = table.window()
    window["twice"], window["twice"], window["twice"]
    assert len(calls) == 1


def test_a_signal_that_depends_on_itself_is_refused(table):
    table.define("loop", lambda window: window["loop"])
    with pytest.raises(SignalError, match="depends on itself"):
        table.window()["loop"]


def test_unknown_names_are_reported_with_what_is_available(table):
    with pytest.raises(SignalError, match="no signal"):
        table.window()["missing"]


def test_a_channel_may_not_be_shadowed_or_dropped(table):
    with pytest.raises(SignalError, match="is a channel"):
        table.define("x", lambda window: window["y"] * 2)
    with pytest.raises(SignalError, match="is a channel"):
        table.drop("x")


def test_a_wrong_length_result_is_refused(table):
    table.define("short", lambda window: np.zeros(3))
    with pytest.raises(SignalError, match="produced 3 values"):
        table.window()["short"]


def test_shown_signals_survive_a_new_source(session, source, clock):
    session.show((mmp_sense.signal("accel_x") * 2).named("twice"))
    session.attach(source)
    assert "twice" in session.signals
    assert "accel_x" in session.signals


def r(window):
    return np.sqrt(window["x"] ** 2 + window["y"] ** 2)


def test_a_window_becomes_a_series_and_a_frame(table):
    table.define("r", r)
    window = table.window(0.0, 0.5)
    series = window.series(["x", "r"])
    assert series.channels == ["x", "r"]
    assert series.values.shape == (len(window), 2)
    frame = window.to_frame(["r"])
    assert list(frame.columns) == ["r"]
    assert frame.index.name == "time"


def test_slicing_a_window_keeps_the_signals(table):
    table.define("r", r)
    piece = table.window().slice(1.0, 2.0)
    assert piece.t_start == pytest.approx(1.0)
    assert piece["r"] == pytest.approx(np.ones(len(piece)))
