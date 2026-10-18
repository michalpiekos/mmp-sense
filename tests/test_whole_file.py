"""A file is held whole, however long; a live stream stays capped."""

import numpy as np
import pytest

import mmp_sense
from mmp_sense.analysis import session_of
from mmp_sense.core.data import TimeSeries
from mmp_sense.core import buffer as buffer_module
from mmp_sense.core import session as session_module
from mmp_sense.core.session import Session


@pytest.fixture
def small_cap(monkeypatch):
    """The live cap made small, so a long file is cheap to test."""
    monkeypatch.setattr(buffer_module, "MAX_CAPACITY", 4096)
    monkeypatch.setattr(session_module, "MAX_CAPACITY", 4096)
    return 4096


def long_series(n=20_000, rate=1000.0):
    return TimeSeries(time=np.arange(n) / rate, values=np.arange(n, dtype=float)[:, None],
                      channels=["v"], sample_rate=rate)


def test_a_loaded_series_is_held_whole_past_the_live_cap(small_cap):
    series = long_series()
    session = session_of(series)
    assert len(session.buffer) == series.n_samples
    assert session.buffer.t_start == 0.0            # the start is not dropped
    assert len(session.values(mmp_sense.signal("v").highpass(cutoff_hz=1))) == series.n_samples


def test_a_file_opened_in_the_window_is_held_whole(small_cap, tmp_path):
    from scipy.io import wavfile

    from mmp_sense.ui.app import build_session, parse_args

    wavfile.write(tmp_path / "long.wav", 8000, np.ones(8000 * 3, dtype=np.int16))
    session = build_session(parse_args(["--file", str(tmp_path / "long.wav"), "--audio-rate", "8000"]))
    session.poll()
    assert len(session.buffer) == 8000 * 3 > small_cap


def test_a_live_buffer_keeps_the_cap(small_cap):
    session = Session(buffer_s=3600.0, rate=1000.0)
    assert session.buffer.capacity == small_cap
