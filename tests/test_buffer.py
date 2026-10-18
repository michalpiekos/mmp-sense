"""The live store: what it keeps, what it drops, and what it promises."""

import numpy as np
import pytest

from mmp_sense.core.buffer import MIN_CAPACITY, ChannelBuffer


def fill(buffer, start, count, rate=100.0):
    time = start + np.arange(count) / rate
    values = np.stack([time, -time], axis=1)
    return buffer.append(time, values)


def test_keeps_the_newest_span_and_drops_the_rest():
    buffer = ChannelBuffer(["x", "y"], span_s=1.0, rate=100.0)
    fill(buffer, 0.0, 1000)
    assert buffer.t_end == pytest.approx(9.99)
    # It holds at least the span it was asked for, and not the whole run.
    assert buffer.duration >= 1.0
    assert len(buffer) < 1000 + MIN_CAPACITY


def test_snapshot_returns_the_interval_asked_for():
    buffer = ChannelBuffer(["x", "y"], span_s=10.0, rate=100.0)
    fill(buffer, 0.0, 500)
    time, values = buffer.snapshot(1.0, 1.1)
    assert time[0] == pytest.approx(1.0)
    assert time[-1] == pytest.approx(1.1)
    assert values.shape == (len(time), 2)
    # A copy: the caller may hold it while the source keeps writing.
    values[0, 0] = 999.0
    assert buffer.snapshot(1.0, 1.1)[1][0, 0] != 999.0


def test_a_late_sample_is_dropped_not_inserted():
    buffer = ChannelBuffer(["x"], span_s=10.0, rate=10.0)
    buffer.append([0.0, 0.1, 0.2], [[0.0], [1.0], [2.0]])
    kept = buffer.append([0.15], [[9.0]])
    assert kept == 0 and buffer.late == 1
    time, _ = buffer.snapshot()
    assert np.all(np.diff(time) >= 0)


def test_a_batch_out_of_order_is_sorted_rather_than_dropped():
    buffer = ChannelBuffer(["x"], span_s=10.0)
    buffer.append([0.3, 0.1, 0.2], [[3.0], [1.0], [2.0]])
    time, values = buffer.snapshot()
    assert list(time) == [0.1, 0.2, 0.3]
    assert list(values[:, 0]) == [1.0, 2.0, 3.0]


def test_a_step_back_in_time_starts_a_new_timeline():
    buffer = ChannelBuffer(["x"], span_s=10.0)
    buffer.append(np.arange(10) + 100.0, np.zeros((10, 1)))
    buffer.append([0.0, 0.1], [[1.0], [2.0]])
    assert buffer.restarts == 1
    assert len(buffer) == 2
    assert buffer.t_start == 0.0


def test_clear_empties_it_and_bumps_the_version():
    buffer = ChannelBuffer(["x"], span_s=10.0)
    fill(ChannelBuffer(["x", "y"], span_s=1.0), 0.0, 10)
    buffer.append([0.0], [[1.0]])
    version = buffer.version
    buffer.clear()
    assert len(buffer) == 0 and buffer.version > version


def test_rate_is_declared_or_measured():
    declared = ChannelBuffer(["x"], rate=250.0)
    assert declared.rate == 250.0
    measured = ChannelBuffer(["x"])
    measured.append(np.arange(100) / 50.0, np.zeros((100, 1)))
    assert measured.rate == pytest.approx(50.0, rel=1e-6)


def test_shape_and_channel_errors_are_named():
    buffer = ChannelBuffer(["x", "y"])
    with pytest.raises(ValueError, match="values for"):
        buffer.append([0.0, 0.1], [[1.0], [2.0]])
    with pytest.raises(KeyError, match="no channel"):
        buffer.index("z")
    with pytest.raises(ValueError, match="at least one channel"):
        ChannelBuffer([])


def test_an_interval_that_spans_the_wrap_comes_back_in_order():
    """The ring overwrites in place; a snapshot must not show the seam."""
    buffer = ChannelBuffer(["x"], span_s=0.0, rate=0.0, max_capacity=1024)
    time = np.arange(3000) / 100.0
    buffer.append(time, time[:, None])
    held, values = buffer.snapshot()
    assert len(held) == buffer.capacity
    assert np.all(np.diff(held) > 0)
    assert values[:, 0] == pytest.approx(held)

    # An interval crossing the point where the ring wrapped.
    middle = held[len(held) // 2]
    piece, piece_values = buffer.snapshot(middle - 0.5, middle + 0.5)
    assert np.all(np.diff(piece) > 0)
    assert piece_values[:, 0] == pytest.approx(piece)
    assert piece[0] >= middle - 0.5 and piece[-1] <= middle + 0.5


def test_only_the_interval_asked_for_is_copied():
    buffer = ChannelBuffer(["x"], span_s=60.0, rate=1000.0)
    time = np.arange(60000) / 1000.0
    buffer.append(time, time[:, None])
    piece, values = buffer.snapshot(30.0, 30.5)
    assert len(piece) == 501
    assert values.base is None       # a copy, not a view into the ring
