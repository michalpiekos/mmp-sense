"""Frequency bands: picked out of a spectrum and heard alone."""

import numpy as np
import pandas as pd
import pytest

import mmp_sense
from mmp_sense.core.query import as_audio, band_only


def _level_db(sound, rate, hz):
    spectrum = np.abs(np.fft.rfft(sound * np.hanning(len(sound))))
    frequency = np.fft.rfftfreq(len(sound), 1.0 / rate)
    near = np.abs(frequency - hz) < 20
    return 20 * np.log10(spectrum[near].max() + 1e-12)


@pytest.fixture
def two_tones():
    time = np.arange(16000) / 16000.0
    values = 0.4 * np.sin(2 * np.pi * 300 * time) + 0.1 * np.sin(2 * np.pi * 3000 * time)
    return mmp_sense.from_frame(pd.DataFrame({"a": values}, index=time))


# -- the mask ----------------------------------------------------------


def test_a_band_keeps_its_tone_and_drops_the_other(two_tones):
    sound = two_tones.audio("a", rate=16000, bands=[(2500, 3500)])
    assert _level_db(sound, 16000, 3000) - _level_db(sound, 16000, 300) > 40
    assert np.abs(sound).max() == pytest.approx(0.9, abs=0.02)     # raised to be heard


def test_without_bands_the_sound_is_as_before(two_tones):
    plain = two_tones.audio("a", rate=16000)
    assert _level_db(plain, 16000, 300) > _level_db(plain, 16000, 3000)
    assert np.abs(plain).max() <= 0.5


def test_several_bands_are_a_union():
    rate = 8000.0
    time = np.arange(8000) / rate
    values = sum(np.sin(2 * np.pi * f * time) for f in (200, 1000, 3000))
    out = band_only(values, rate, [(150, 250), (2900, 3100)])
    for hz, kept in ((200, True), (1000, False), (3000, True)):
        level = _level_db(out, rate, hz) - _level_db(values, rate, hz)
        assert (level > -1) if kept else (level < -60)


def test_a_band_from_zero_is_a_lowpass():
    rate = 1000.0
    values = 2.0 + np.sin(2 * np.pi * 300 * np.arange(1000) / rate)
    out = band_only(values, rate, [(0, 50)])
    assert np.allclose(out, 2.0, atol=0.02)


def test_a_gap_stays_a_gap_and_the_rest_is_finite():
    values = np.sin(np.arange(1000) / 5.0)
    values[100:200] = np.nan
    out = band_only(values, 1000.0, [(10, 100)])
    assert np.isnan(out[100:200]).all() and np.isfinite(out[200:]).all()
    sound = as_audio(values, 1000.0, 1000, bands=[(10, 100)])
    assert np.all(sound[100:200] == 0.0)


def test_sixteen_bit_counts_can_be_banded():
    time = np.arange(8000) / 8000.0
    values = (8000 * np.sin(2 * np.pi * 440 * time)).astype(np.int16)
    sound = as_audio(values, 8000.0, 8000, bands=[(400, 480)])
    assert np.abs(sound).max() == pytest.approx(0.9, abs=0.02)


# -- the session's bands -------------------------------------------------


def test_bands_are_named_ticked_and_listed(session):
    seen = []
    session.listen(lambda kind, payload: seen.append(kind))
    first = session.add_band(100, 200)
    second = session.add_band(1000, 2000, "whine", on=False)
    assert (first, second) == ("band 1", "whine")
    assert [b.range for b in session.bands] == [(100.0, 200.0), (1000.0, 2000.0)]
    assert session.heard_bands == [(100.0, 200.0)]
    assert seen.count("bands") == 2


def test_a_band_is_edited_ticked_renamed_and_removed(session):
    name = session.add_band(100, 200)
    session.edit_band(name, high_hz=300)
    session.set_band_on(name, False)
    assert session.band(name).range == (100.0, 300.0) and not session.band(name).on
    session.edit_band(name, rename="hum")
    assert [b.name for b in session.bands] == ["hum"]
    session.add_band(5, 10)
    assert session.bands[-1].name == "band 1"                 # the name came free
    session.remove_band("hum")
    session.clear_bands()
    assert session.bands == []


@pytest.mark.parametrize("low, high", [(200, 100), (100, 100), (-1, 10), (0, np.inf)])
def test_a_band_that_makes_no_sense_is_refused(session, low, high):
    with pytest.raises(ValueError):
        session.add_band(low, high)
    assert session.bands == []


def test_names_are_unique_and_unknown_ones_say_so(session):
    session.add_band(1, 2, "x")
    with pytest.raises(ValueError):
        session.add_band(3, 4, "x")
    session.add_band(3, 4, "y")
    with pytest.raises(ValueError):
        session.edit_band("y", rename="x")
    with pytest.raises(KeyError):
        session.remove_band("z")


def test_audio_takes_the_bands_themselves(two_tones):
    two_tones.add_band(2500, 3500)
    sound = two_tones.audio("a", rate=16000, bands=two_tones.bands)
    assert _level_db(sound, 16000, 3000) - _level_db(sound, 16000, 300) > 40


# -- the window ----------------------------------------------------------------

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt                                    # noqa: E402

from test_listen import Speaker                                  # noqa: E402


@pytest.fixture
def ui(qapp, session):
    from mmp_sense.ui.main_window import MainWindow

    speaker = Speaker()
    window = MainWindow(session, window_s=10.0, console=False, player=speaker)
    window.show()
    window.tick()
    try:
        yield window, speaker
    finally:
        window.close()


def _strips(pane):
    return {name: tuple(s.getRegion()) for name, s in pane.bands.strips.items()}


def test_the_form_adds_a_band_with_typed_edges(ui, session):
    window, _ = ui
    dialog = window.band_dock.add()
    assert "holds up to" in dialog.limit.text()
    dialog.low.setValue(12.345)
    dialog.high.setValue(40.5)
    dialog.accept()
    assert [b.range for b in session.bands] == [(12.345, 40.5)]
    window.tick()
    assert window.band_dock.tree.topLevelItem(0).text(1) == "12.345"


def test_the_form_refuses_a_backwards_band_and_a_taken_name(ui, session):
    window, _ = ui
    session.add_band(1, 2, "hum")
    dialog = window.band_dock.add()
    dialog.low.setValue(50)
    dialog.high.setValue(10)
    assert not dialog.buttons.button(dialog.buttons.StandardButton.Ok).isEnabled()
    dialog.accept()
    dialog.high.setValue(60)
    dialog.name.setText("hum")
    dialog.accept()
    assert "already" in dialog.problem.text() and len(session.bands) == 1


def test_the_form_edits_a_band(ui, session):
    window, _ = ui
    session.add_band(10, 20)
    window.tick()
    window.band_dock.tree.setCurrentItem(window.band_dock.tree.topLevelItem(0))
    dialog = window.band_dock.edit()
    assert dialog.low.value() == 10 and dialog.high.value() == 20
    dialog.high.setValue(30)
    dialog.name.setText("thump")
    dialog.accept()
    assert session.band("thump").range == (10.0, 30.0)


def test_unticking_in_the_dock_turns_a_band_off(ui, session):
    window, _ = ui
    session.add_band(10, 20)
    window.tick()
    window.band_dock.tree.topLevelItem(0).setCheckState(0, Qt.Unchecked)
    assert session.heard_bands == []


def test_shift_drag_on_a_spectrum_adds_a_band_every_chart_shows(ui, session):
    window, _ = ui
    spectrum = next(p for p in window.panes if p.kind == "spectrum")
    psd = window.add_pane("psd")
    spectrum._viewbox.region_dragged.emit(30.0, 10.0, False)
    assert spectrum.bands.draft.isVisible() and session.bands == []
    spectrum._viewbox.region_dragged.emit(30.0, 10.0, True)
    assert [b.range for b in session.bands] == [(10.0, 30.0)]
    window.tick()
    assert _strips(spectrum) == {"band 1": (10.0, 30.0)}
    assert _strips(psd) == {"band 1": (10.0, 30.0)}


def test_dragging_a_strip_edge_changes_the_band(ui, session):
    window, _ = ui
    spectrum = next(p for p in window.panes if p.kind == "spectrum")
    session.add_band(10, 20)
    window.tick()
    strip = spectrum.bands.strips["band 1"]
    strip.setRegion((10.0, 25.0))
    strip.sigRegionChangeFinished.emit(strip)
    assert session.band("band 1").range == (10.0, 25.0)


def test_a_log_axis_puts_strips_at_log_edges_and_keeps_zero(ui, session):
    window, _ = ui
    psd = window.add_pane("psd")
    session.add_band(0, 10)
    psd.log_frequency.setChecked(True)
    window.tick()
    low, high = _strips(psd)["band 1"]
    assert high == pytest.approx(1.0)
    strip = psd.bands.strips["band 1"]
    strip.setRegion((low, 1.5))                                  # only the top edge moved
    strip.sigRegionChangeFinished.emit(strip)
    assert session.band("band 1").range == pytest.approx((0.0, 10 ** 1.5))


def test_listen_plays_only_the_ticked_bands(ui, session):
    window, speaker = ui
    window.toggle_sound()
    whole = speaker.played[-1]
    window.toggle_sound()
    session.add_band(10, 20)
    session.add_band(30, 40, on=False)
    window.toggle_sound()
    banded = speaker.played[-1]
    assert len(banded) == len(whole) and not np.allclose(banded, whole)
    assert np.abs(banded).max() == pytest.approx(0.9, abs=0.02)
    assert "1 band" in window.statusBar().currentMessage()
    window.toggle_sound()
    session.set_band_on("band 1", False)
    window.toggle_sound()
    assert np.allclose(speaker.played[-1], whole)


def test_the_dock_starts_collapsed_and_opens_on_the_first_band(ui, session):
    window, _ = ui
    assert window.bands_dock.collapsed
    session.add_band(10, 20)
    window.tick()
    assert not window.bands_dock.collapsed
    window.set_dock_collapsed(window.bands_dock, True)
    session.add_band(30, 40)
    window.tick()
    assert window.bands_dock.collapsed                           # closed by hand stays so


def test_bands_past_what_the_signal_holds_are_said_not_played(ui, session):
    window, speaker = ui
    name, top = window.nyquist()
    session.add_band(top + 1, top + 10)
    window.toggle_sound()
    assert speaker.played == [] and window.hearing is None
    assert "holds nothing above" in window.statusBar().currentMessage()
