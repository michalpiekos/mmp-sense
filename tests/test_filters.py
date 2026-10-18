"""Butterworth filters, a notch, and their responses."""

import numpy as np
import pandas as pd
import pytest
from scipy import signal as sp

import mmp_sense
from mmp_sense import dsp

RATE = 1000.0


def _tones(*frequencies, seconds=8.0):
    time = np.arange(int(seconds * RATE)) / RATE
    columns = {f"f{f:g}": np.sin(2 * np.pi * f * time) for f in frequencies}
    return mmp_sense.from_frame(pd.DataFrame(columns, index=time))


def _gain_db(session, recipe, settle=2.0):
    """The output's rms over the input's, once the filter has settled."""
    out = session.values(recipe).loc[settle:].to_numpy()
    return 20 * np.log10(np.sqrt(np.mean(out ** 2)) / np.sqrt(0.5))


@pytest.mark.parametrize("kind, cutoff", [("lowpass", 100.0), ("highpass", 100.0)])
def test_a_butterworth_is_3_db_down_at_its_cutoff(kind, cutoff):
    session = _tones(cutoff)
    recipe = getattr(mmp_sense.signal("f100"), kind)(cutoff, order=4)
    assert _gain_db(session, recipe) == pytest.approx(-3.01, abs=0.1)


def test_order_four_falls_24_db_an_octave_and_more():
    session = _tones(400.0, 25.0)
    assert _gain_db(session, mmp_sense.signal("f400").lowpass(100, order=4)) < -45
    assert _gain_db(session, mmp_sense.signal("f25").highpass(100, order=4)) < -45


def test_a_bandpass_keeps_its_band_and_its_edges_are_3_db_down():
    session = _tones(30.0, 100.0, 300.0, 50.0, 200.0)
    band = lambda name: mmp_sense.signal(name).bandpass(50, 200, order=4)
    assert _gain_db(session, band("f100")) == pytest.approx(0.0, abs=0.2)
    assert _gain_db(session, band("f50")) == pytest.approx(-3.01, abs=0.1)
    assert _gain_db(session, band("f200")) == pytest.approx(-3.01, abs=0.1)
    assert _gain_db(session, band("f300")) < -20 and _gain_db(session, band("f30")) < -20


def test_a_bandstop_loses_its_band():
    session = _tones(50.0, 150.0)
    assert _gain_db(session, mmp_sense.signal("f50").bandstop(40, 60)) < -30
    assert _gain_db(session, mmp_sense.signal("f150").bandstop(40, 60)) == pytest.approx(0, abs=0.2)


def test_a_notch_removes_its_tone_and_leaves_the_rest():
    session = _tones(50.0, 200.0)
    assert _gain_db(session, mmp_sense.signal("f50").notch(50)) < -30
    assert _gain_db(session, mmp_sense.signal("f200").notch(50)) == pytest.approx(0.0, abs=0.5)


def test_a_band_filter_named_without_an_order_is_of_order_four():
    session = _tones(30.0)
    bare = mmp_sense.signal("f30").apply("bandpass", low_hz=50.0, high_hz=200.0)
    four = mmp_sense.signal("f30").bandpass(50, 200, order=4)
    np.testing.assert_allclose(session.values(bare), session.values(four))
    assert bare.latency == pytest.approx(four.latency)
    pd.testing.assert_frame_equal(bare.response(RATE, [30.0]), four.response(RATE, [30.0]))
    as_typed = mmp_sense.signal("f30").apply("lowpass", cutoff_hz=10.0, order=4.0)
    assert as_typed.title == "lowpass(10Hz, order 4)"
    with pytest.raises(Exception, match="whole number"):
        session.values(mmp_sense.signal("f30").apply("lowpass", cutoff_hz=10.0, order=2.5))
    with pytest.raises(Exception, match="whole number"):
        session.values(mmp_sense.signal("f30").lowpass(10, order=2.5))
    assert bare.title == four.title == "bandpass(50Hz, 200Hz)"
    assert mmp_sense.signal("f30").bandpass(50, 200, order=1).title == "bandpass(50Hz, 200Hz, order 1)"


def test_a_constant_passes_a_lowpass_without_a_step():
    time = np.arange(2000) / RATE
    session = mmp_sense.from_frame(pd.DataFrame({"g": np.full(2000, 9.81)}, index=time))
    np.testing.assert_allclose(session.values(mmp_sense.signal("g").lowpass(20, order=4)), 9.81)
    np.testing.assert_allclose(session.values(mmp_sense.signal("g").highpass(20, order=4)), 0.0,
                               atol=1e-9)


def test_order_one_is_the_filter_it_always_was():
    session = _tones(30.0)
    old = session.values(mmp_sense.signal("f30").lowpass(10))
    values, _ = dsp.lowpass1(session.values("f30").to_numpy(), RATE, 10.0)
    np.testing.assert_array_equal(old.to_numpy(), values)
    assert mmp_sense.signal("f30").lowpass(10).title == "lowpass(10Hz)"
    assert mmp_sense.signal("f30").lowpass(10, order=1).key == mmp_sense.signal("f30").lowpass(10).key


def test_the_response_is_scipys():
    recipe = mmp_sense.signal("x").highpass(2.0).bandpass(80, 300, order=3).notch(50)
    frequency = np.linspace(1.0, 499.0, 300)
    response = recipe.response(RATE, frequency)
    h = np.ones(len(frequency), complex)
    for sos in (sp.tf2sos([dsp.highpass1_alpha(RATE, 2.0), -dsp.highpass1_alpha(RATE, 2.0)],
                          [1.0, -dsp.highpass1_alpha(RATE, 2.0)]),
                sp.butter(3, (80, 300), "bandpass", fs=RATE, output="sos"),
                sp.tf2sos(*sp.iirnotch(50, 30, fs=RATE))):
        h *= sp.sosfreqz(sos, worN=frequency, fs=RATE)[1]
    np.testing.assert_allclose(response["gain_db"], 20 * np.log10(np.abs(h)), atol=1e-6)
    assert response.loc[response.index[np.argmin(abs(frequency - 160))], "group_delay_s"] > 0


def test_a_filter_says_how_late_it_is():
    assert mmp_sense.signal("x").lowpass(10).latency == pytest.approx(1 / (2 * np.pi * 10))
    late = mmp_sense.signal("x").lowpass(10, order=4).latency
    measured = mmp_sense.signal("x").lowpass(10, order=4).response(1000.0, [0.1])["group_delay_s"]
    assert late == pytest.approx(float(measured.iloc[0]), rel=0.02)
    assert mmp_sense.signal("x").bandpass(100, 400).latency > 0


def test_a_mistaken_filter_is_refused():
    session = _tones(30.0)
    with pytest.raises(Exception, match="Nyquist"):
        session.values(mmp_sense.signal("f30").lowpass(600, order=2))
    with pytest.raises(Exception, match="low < high"):
        session.values(mmp_sense.signal("f30").bandpass(200, 100))
