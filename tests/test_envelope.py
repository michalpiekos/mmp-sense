"""The envelope: how loud, instant by instant, without looking ahead."""

import numpy as np
import pandas as pd
import pytest

import mmp_sense
from mmp_sense import dsp

RATE = 8000.0


def _am(seconds=4.0, carrier=440.0, depth=0.5, rate_hz=3.0):
    time = np.arange(int(seconds * RATE)) / RATE
    level = 1 + depth * np.sin(2 * np.pi * rate_hz * time)
    session = mmp_sense.from_frame(pd.DataFrame({"x": level * np.sin(2 * np.pi * carrier * time)},
                                          index=time))
    return session, time, level


@pytest.mark.parametrize("min_hz", [20.0, 100.0])
def test_the_envelope_follows_the_modulation_after_its_lag(min_hz):
    session, time, level = _am()
    recipe = mmp_sense.signal("x").envelope(min_hz)
    envelope = session.values(recipe).to_numpy()
    lag = int(round(recipe.latency * RATE))
    settled = slice(lag + int(0.1 * RATE), None)
    expected = level[:len(level) - lag][int(0.1 * RATE):]
    assert np.max(np.abs(envelope[settled] - expected) / expected) < 0.05


def test_the_declared_lag_is_the_lag():
    assert mmp_sense.signal("x").envelope(50).latency == pytest.approx(
        dsp.envelope_delay(RATE, 50), abs=1 / RATE)


def test_a_constant_has_itself_for_envelope_and_no_start_up_step():
    time = np.arange(4000) / RATE
    session = mmp_sense.from_frame(pd.DataFrame({"x": np.full(4000, -2.0)}, index=time))
    np.testing.assert_allclose(session.values(mmp_sense.signal("x").envelope()), 2.0, atol=1e-9)


def test_too_low_a_rate_for_the_band_is_refused():
    with pytest.raises(ValueError, match="quarter of the rate"):
        dsp.envelope(np.zeros(10), 100.0, min_hz=30.0)


def test_abs_rectifies():
    time = np.arange(4) / RATE
    session = mmp_sense.from_frame(pd.DataFrame({"x": [-1.0, 2.0, -3.0, 0.0]}, index=time))
    assert session.values(mmp_sense.signal("x").abs()).tolist() == [1.0, 2.0, 3.0, 0.0]
