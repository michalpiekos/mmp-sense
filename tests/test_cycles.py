"""Cycles averaged: an engine of six firings, alike or with one weak,
whose rate glides -- on sounds whose answers are known."""

import numpy as np
import pandas as pd
import pytest

import mmp_sense
from mmp_sense import dsp, signal
from mmp_sense.analysis import CycleAverage

RATE = 16000
SECONDS = 6.0
PER_CYCLE = 720


def _firing(u):
    """One firing, by position in the cycle after it: a ring that dies away."""
    return np.exp(-u / 0.025) * np.sin(2 * np.pi * 12 * u)


def _cycle(theta, strengths):
    """The engine's sound at cycle position `theta`, firing j at j / 6."""
    out = np.zeros_like(theta)
    for j, a in enumerate(strengths):
        out += a * _firing((theta - j / 6) % 1.0)
    return out


def _engine(strengths=(1.0,) * 6, noise=0.3, glide=0.05, seed=0):
    """A session: `mic`, the engine plus noise, and `rate`, its cycle rate,
    falling `glide` over the recording and warped as if passing by."""
    time = np.arange(int(SECONDS * RATE)) / RATE
    rate = 20.0 * (1 - glide * time / SECONDS) * (1 + 0.02 * np.sin(2 * np.pi * time / SECONDS))
    theta = np.cumsum(rate) / RATE
    rng = np.random.default_rng(seed)
    mic = _cycle(theta, strengths) + noise * rng.standard_normal(len(time))
    return mmp_sense.from_frame(pd.DataFrame({"mic": mic, "rate": rate}, index=time))


def _true_orders(strengths):
    theta = np.arange(PER_CYCLE) / PER_CYCLE
    return np.abs(np.fft.rfft(_cycle(theta, strengths))) * 2 / PER_CYCLE


def test_alike_firings_read_alike():
    average = _engine().cycle_average("mic", "rate", lock=6)
    assert isinstance(average, CycleAverage)
    assert 110 <= average.cycles <= 120
    parts = average.parts(6)
    assert np.allclose(parts["rms"], 1.0, atol=0.03)
    assert np.allclose(parts["lag"], 0.0, atol=0.02)
    uneven = average.uneven(6)
    assert uneven["uneven"] < 3 * uneven["noise"]


def test_a_weak_firing_is_found():
    strengths = (1.0, 1.0, 1.0, 0.6, 1.0, 1.0)
    average = _engine(strengths).cycle_average("mic", "rate", lock=6)
    rms = average.parts(6)["rms"].sort_values()
    assert rms.iloc[0] == pytest.approx(0.6 / (5.6 / 6), rel=0.1)
    assert (rms.iloc[1:] > 0.95).all()
    uneven = average.uneven(6)
    assert uneven["uneven"] >= 5 * uneven["noise"]
    orders = average.orders(12)
    assert orders.loc[[1, 2, 3, 4, 5], "ratio"].mean() > 3


def test_the_average_matches_the_engine_and_lock_keeps_high_orders():
    strengths = (1.0, 1.0, 1.0, 0.6, 1.0, 1.0)
    session = _engine(strengths, noise=0.2)
    truth = _true_orders(strengths)
    off = signal("rate") * 1.002                         # a rate measured 0.2 % high
    locked = session.cycle_average("mic", off, lock=6).orders(36)["amplitude"]
    loose = session.cycle_average("mic", off).orders(36)["amplitude"]
    assert locked[36] == pytest.approx(truth[36], rel=0.1)
    assert locked[6] == pytest.approx(truth[6], rel=0.05)
    assert loose[36] < 0.5 * truth[36]


def test_noise_falls_as_the_root_of_the_cycles():
    time = np.arange(int(SECONDS * RATE)) / RATE
    rng = np.random.default_rng(1)
    session = mmp_sense.from_frame(pd.DataFrame({"mic": rng.standard_normal(len(time))}, index=time))
    average = session.cycle_average("mic", 20.0, per_cycle=360)
    spread = average.mean.std()
    assert 0.7 < spread / average.noise.std() < 1.4
    assert spread < 2.0 / np.sqrt(average.cycles)


def test_a_fixed_rate_averages_like_its_signal():
    time = np.arange(int(SECONDS * RATE)) / RATE
    theta = 20.0 * time
    mic = _cycle(theta, (1.0, 1.0, 1.0, 0.6, 1.0, 1.0))
    session = mmp_sense.from_frame(pd.DataFrame({"mic": mic, "rate": np.full(len(time), 20.0)},
                                          index=time))
    by_number = session.cycle_average("mic", 20.0, 1.0, 5.0)
    by_signal = session.cycle_average("mic", "rate", 1.0, 5.0)
    assert by_number.rate_hz == by_signal.rate_hz == 20.0
    assert by_number.cycles == by_signal.cycles == 79
    assert np.allclose(by_number.mean, by_signal.mean, atol=1e-9)


def test_a_gap_does_not_stop_it():
    session = _engine()
    time = session.values("mic").index
    mic = session.values("mic").to_numpy().copy()
    mic[(time > 2.0) & (time < 2.05)] = np.nan
    gapped = mmp_sense.from_frame(pd.DataFrame({"mic": mic, "rate": session.values("rate").to_numpy()},
                                         index=time))
    average = gapped.cycle_average("mic", "rate")
    assert np.isfinite(average.mean).all()
    assert np.allclose(average.parts(6)["rms"], 1.0, atol=0.05)


def test_a_lost_line_is_said_so():
    session = _engine(noise=2.0)
    with pytest.raises(ValueError, match="was lost"):
        session.cycle_average("mic", "rate", lock=5, lock_hz=9.0)


def test_too_short_and_uneven_cuts_are_refused():
    session = _engine()
    with pytest.raises(ValueError, match="2 or more"):
        session.cycle_average("mic", "rate", 1.0, 1.08)
    average = session.cycle_average("mic", "rate", 1.0, 2.0)
    with pytest.raises(ValueError, match="equal parts"):
        average.parts(7)


def test_cycle_count_locks_onto_a_line():
    time = np.arange(4 * RATE) / RATE
    true = 2 * np.pi * 120.0 * time + 0.8 * np.sin(2 * np.pi * 0.5 * time)
    values = np.cos(true)
    cycles = dsp.cycle_count(values, RATE, np.full(len(time), 20.0), lock=6)
    phase_error = np.angle(np.exp(1j * (2 * np.pi * 6 * cycles - true)))
    middle = slice(RATE // 2, -RATE // 2)
    assert np.ptp(np.unwrap(phase_error[middle])) < 0.1
    rows = dsp.cut_cycles(values, cycles, 120)
    assert rows.shape[1] == 120 and len(rows) >= 78

