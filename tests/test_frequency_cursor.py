"""The frequency cursor: in the session, on the frequency charts, from
clicks and keys, and the peaks it lands on."""

import math
import re

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import QPointF, Qt

import mmp_sense
from mmp_sense.dsp.peaks import next_peak, refine_peak, snap_to_peak, step_bin

RATE = 1000.0
TONES = (50.37, 120.0)


def _tones(seconds=20.0):
    time = np.arange(int(seconds * RATE)) / RATE
    values = np.sin(2 * np.pi * TONES[0] * time) + 0.3 * np.sin(2 * np.pi * TONES[1] * time)
    values += 0.03 * np.random.default_rng(0).standard_normal(len(time))
    return mmp_sense.from_frame(pd.DataFrame({"x": values}, index=time))


@pytest.fixture
def ui(qapp):
    from mmp_sense.ui.main_window import MainWindow

    session = _tones()
    session.set_selection((0.0, 10.0))                 # 0.1 Hz bins in the spectrum
    window = MainWindow(session, console=False)
    window.show()
    window.tick()
    try:
        yield window, session
    finally:
        window.close()


def _pane(window, kind):
    return next(p for p in window.panes if p.kind == kind)


class _Click:
    """What pyqtgraph's scene hands a click handler."""

    def __init__(self, viewbox, x, y, modifiers=Qt.NoModifier, button=Qt.LeftButton):
        self._pos = viewbox.mapViewToScene(QPointF(x, y))
        self._modifiers, self._button = modifiers, button

    def button(self):
        return self._button

    def modifiers(self):
        return self._modifiers

    def scenePos(self):
        return self._pos


def _click(pane, x, y=None, modifiers=Qt.NoModifier, view=((0, 500), (0, 500))):
    viewbox = pane.plot.getViewBox()
    # Offscreen, nothing is painted, so the view has not ranged itself yet.
    viewbox.setRange(xRange=view[0], yRange=view[1], padding=0)
    if y is None:
        y = sum(viewbox.viewRange()[1]) / 2
    handler = pane.marks._clicked if hasattr(pane, "marks") else pane._clicked
    handler(_Click(viewbox, x, y, modifiers))


# -- the session ---------------------------------------------------------


def test_the_frequency_cursor_starts_unplaced_and_tells_who_listens():
    session = _tones(2.0)
    heard = []
    session.listen(lambda kind, payload: heard.append((kind, payload)))
    assert session.frequency_cursor is None
    session.frequency_cursor = 50.0
    assert session.frequency_cursor == 50.0
    assert heard[-1] == ("frequency", {"frequency_cursor": 50.0})
    session.set_frequency_cursor(-3.0)
    assert session.frequency_cursor == 0.0
    session.frequency_cursor = None
    assert session.frequency_cursor is None
    with pytest.raises(ValueError):
        session.set_frequency_cursor(math.nan)


def test_the_frequency_cursor_is_independent_of_the_time_cursor():
    session = _tones(2.0)
    session.set_cursor(1.0)
    session.frequency_cursor = 120.0
    assert session.cursor == 1.0
    session.set_cursor(0.5)
    assert session.frequency_cursor == 120.0


def test_the_context_reports_it():
    session = _tones(2.0)
    assert session.context.to_dict()["frequency_cursor"] is None
    session.frequency_cursor = 75.0
    assert session.context.to_dict()["frequency_cursor"] == 75.0


# -- the peaks it lands on -----------------------------------------------


def _db_spectrum(seconds=10.0):
    spectrum = _tones(seconds).spectrum("x", 0.0, seconds)
    return spectrum.index.to_numpy(), 20 * np.log10(spectrum.to_numpy())


def test_a_peak_is_refined_between_bins():
    frequency, db = _db_spectrum()
    assert frequency[1] - frequency[0] == pytest.approx(0.1, rel=0.01)
    i = int(np.argmax(db))
    assert abs(frequency[i] - TONES[0]) > 0.02            # no bin sits on the tone
    assert refine_peak(frequency, db, i) == pytest.approx(TONES[0], abs=0.01)


def test_snapping_finds_the_peak_within_reach_only():
    frequency, db = _db_spectrum()
    assert snap_to_peak(frequency, db, 50.2) == pytest.approx(TONES[0], abs=0.01)
    far = snap_to_peak(frequency, db, 80.0)
    assert abs(far - 80.0) <= 0.35


def test_stepping_goes_bin_by_bin_and_peak_by_peak():
    frequency, db = _db_spectrum()
    assert step_bin(frequency, None, +1) == frequency[0]
    assert step_bin(frequency, 50.0, +1) == pytest.approx(50.1, abs=1e-6)
    assert step_bin(frequency, 50.0, -1) == pytest.approx(49.9, abs=1e-6)
    assert next_peak(frequency, db, None, +1) == pytest.approx(TONES[0], abs=0.01)
    assert next_peak(frequency, db, TONES[0], +1) == pytest.approx(TONES[1], abs=0.01)
    assert next_peak(frequency, db, TONES[1], -1) == pytest.approx(TONES[0], abs=0.01)
    assert next_peak(frequency, db, TONES[1], +1) is None


def test_a_peak_is_judged_against_the_floor_around_it_not_the_whole_spectrum():
    # Faint lines in a quiet stretch, beside a far noisier one that would set
    # the bar for the whole spectrum.
    rng = np.random.default_rng(1)
    frequency = np.arange(24000.0)
    db = np.where(frequency < 1000, 0.5, 5.0) * rng.standard_normal(len(frequency))
    for hz in (300, 400):
        db[hz] = 8.0
    stops, hz = [], 0.0
    while (hz := next_peak(frequency, db, hz, +1)) is not None and hz < 1000:
        stops.append(round(hz))
    assert {300, 400} <= set(stops)


# -- the charts ----------------------------------------------------------


def test_every_frequency_chart_draws_it_and_reads_it_out(ui):
    window, session = ui
    psd = window.add_pane("psd", "x")
    window.tick()
    spectrum, spectrogram = _pane(window, "spectrum"), _pane(window, "spectrogram")
    assert not spectrum.marks.line.isVisible() and spectrum.marks.readout.text() == ""
    session.frequency_cursor = 120.0
    window.tick()
    assert spectrum.marks.line.isVisible() and spectrum.marks.line.value() == 120.0
    assert psd.marks.line.value() == 120.0
    assert spectrogram.frequency_line.isVisible()
    assert spectrogram.frequency_line.value() == 120.0
    assert re.fullmatch(r"120 Hz, bin 1[12]\d\.?\d* Hz: 0\.29\d*", spectrum.marks.readout.text())
    assert psd.marks.readout.text().startswith("120 Hz, bin ")
    assert psd.marks.readout.text().endswith(" dB")
    assert " s, 120 Hz: " in spectrogram.readout.text()
    session.frequency_cursor = None
    window.tick()
    assert not spectrum.marks.line.isVisible() and not psd.marks.line.isVisible()
    assert not spectrogram.frequency_line.isVisible()
    assert spectrum.marks.readout.text() == ""


def test_the_psd_reads_out_in_its_own_units(ui):
    window, session = ui
    psd = window.add_pane("psd", "x")
    session.frequency_cursor = 120.0
    psd.db.setChecked(False)
    assert not psd.marks.readout.text().endswith(" dB")


def test_a_click_lands_on_the_nearest_peak_and_alt_where_clicked(ui):
    window, session = ui
    spectrum = _pane(window, "spectrum")
    _click(spectrum, 50.2)
    assert session.frequency_cursor == pytest.approx(TONES[0], abs=0.01)
    _click(spectrum, 50.2, modifiers=Qt.AltModifier)
    assert session.frequency_cursor == pytest.approx(50.2, abs=1e-6)
    _click(spectrum, 80.0, modifiers=Qt.ShiftModifier)               # a band's, not the cursor's
    assert session.frequency_cursor == pytest.approx(50.2, abs=1e-6)


def test_a_click_on_a_log_axis_is_at_its_frequency(ui):
    window, session = ui
    psd = window.add_pane("psd", "x")
    psd.log_frequency.setChecked(True)
    window.tick()
    _click(psd, math.log10(120.0), modifiers=Qt.AltModifier, view=((0, 3), (-100, 0)))
    assert session.frequency_cursor == pytest.approx(120.0, rel=1e-6)
    window.tick()
    assert psd.marks.line.value() == pytest.approx(math.log10(120.0))


def test_a_click_on_the_spectrogram_sets_both_cursors(ui):
    window, session = ui
    spectrogram = _pane(window, "spectrogram")
    _click(spectrogram, 7.5, 200.0, view=((0, 20), (0, 500)))
    assert session.cursor == pytest.approx(7.5, abs=1e-6)
    assert session.frequency_cursor == pytest.approx(200.0, abs=1e-6)


def test_harmonics_are_marked_while_asked_for(ui):
    window, session = ui
    spectrum = _pane(window, "spectrum")
    session.frequency_cursor = 120.0
    window.tick()
    assert spectrum.marks.harmonic_frequencies() == []
    window.harmonics_action.setChecked(True)
    assert spectrum.marks.harmonic_frequencies() == pytest.approx([240.0, 360.0, 480.0])
    psd = window.add_pane("psd", "x")
    window.tick()
    assert psd.marks.harmonic_frequencies() == pytest.approx([240.0, 360.0, 480.0])
    window.harmonics_action.setChecked(False)
    assert spectrum.marks.harmonic_frequencies() == []


def test_the_keys_step_by_bin_and_by_peak(ui):
    window, session = ui
    window.step_frequency(+1, peak=True)
    assert session.frequency_cursor == pytest.approx(TONES[0], abs=0.01)
    window.step_frequency(+1, peak=True)
    assert session.frequency_cursor == pytest.approx(TONES[1], abs=0.01)
    frequency, _ = window.frequency_values()
    at = int(np.argmin(np.abs(frequency - 120.0)))
    window.step_frequency(+1)
    assert session.frequency_cursor == frequency[at + 1]
    window.step_frequency(-1)
    window.step_frequency(-1)
    assert session.frequency_cursor == frequency[at - 1]
    window.step_frequency(-1, peak=True)
    assert session.frequency_cursor == pytest.approx(TONES[0], abs=0.01)
