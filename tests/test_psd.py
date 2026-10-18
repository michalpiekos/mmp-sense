"""The averaged spectrum: Welch's power spectral density."""

import numpy as np
import pandas as pd
import pytest

import mmp_sense

RATE = 1000.0


def _session(values):
    time = np.arange(len(values)) / RATE
    return mmp_sense.from_frame(pd.DataFrame({"x": values}, index=time))


def test_a_tones_power_is_the_area_under_its_peak():
    time = np.arange(20000) / RATE
    density = _session(3.0 * np.sin(2 * np.pi * 125.0 * time)).psd("x", segment=1024)
    width = density.index[1] - density.index[0]
    near = (density.index > 115) & (density.index < 135)
    assert density[near].sum() * width == pytest.approx(3.0 ** 2 / 2, rel=0.02)
    assert density.idxmax() == pytest.approx(125.0, abs=1.0)


@pytest.mark.parametrize("segment", [256, 2048])
def test_white_noise_reads_its_variance_over_half_the_rate(segment):
    rng = np.random.default_rng(0)
    density = _session(2.0 * rng.standard_normal(60000)).psd("x", segment=segment)
    inside = density.iloc[1:-1]
    assert inside.mean() == pytest.approx(4.0 / (RATE / 2), rel=0.1)


def test_gaps_do_not_lower_the_density():
    rng = np.random.default_rng(3)
    values = rng.standard_normal(40000)
    values[:20000] = np.nan
    values[30000:30010] = np.nan
    density = _session(values).psd("x", segment=1024)
    assert density.iloc[1:-1].mean() == pytest.approx(1.0 / (RATE / 2), rel=0.1)


def test_the_stretch_rule_and_the_units():
    rng = np.random.default_rng(1)
    session = _session(rng.standard_normal(10000))
    session.set_selection((2.0, 4.0))
    # The selection is 2001 samples: one segment, no longer than that.
    assert len(session.psd("x", segment=4096)) == 2001 // 2 + 1
    assert session.psd("x").index.name == "frequency_hz"


def test_the_psd_pane_follows_its_controls(qapp):
    pytest.importorskip("pyqtgraph")
    from mmp_sense.ui.main_window import MainWindow

    time = np.arange(20000) / RATE
    session = _session(np.sin(2 * np.pi * 50 * time))
    window = MainWindow(session, console=False)
    try:
        pane = window.add_pane("psd", "x")
        assert len(pane.frequency) == 513
        assert pane.frequency[np.argmax(pane.density)] == pytest.approx(50, abs=1)
        x, y = pane.curve.getData()
        assert y.max() == pytest.approx(10 * np.log10(pane.density.max()), abs=1e-6)
        pane.db.setChecked(False)
        assert pane.curve.getData()[1].max() == pytest.approx(pane.density.max())
        pane.segment.setCurrentText("4096")
        assert len(pane.frequency) == 2049
        pane.log_frequency.setChecked(True)
        assert pane.curve.xData.min() > 0                  # no 0 Hz on a log axis
        session.set_selection((1.0, 2.0))
        pane.refresh({"selection"})
        assert "1.000-2.000 s" in pane.status.text()
        assert pane.describe()["segment"] == 4096
    finally:
        window.close()


def test_add_pane_lists_the_panes_by_name(qapp):
    pytest.importorskip("pyqtgraph")
    from PySide6.QtWidgets import QToolButton
    from mmp_sense.ui.main_window import MainWindow

    window = MainWindow(_session(np.zeros(1000)), console=False)
    try:
        add = next(b for b in window.findChildren(QToolButton) if b.text() == "Add pane")
        items = {a.text(): a.toolTip() for a in add.menu().actions()}
        assert {"psd", "spectrogram", "spectrum", "signal"} <= set(items)
        assert items["psd"] == "The averaged spectrum: power per Hz, in dB"
    finally:
        window.close()
