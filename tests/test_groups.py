"""Signal groups: several signals drawn together on one chart."""

import pytest

import mmp_sense
from conftest import show_basics
from mmp_sense.core.signals import SignalError


def shown_vector(session, name="hp"):
    session.show(mmp_sense.signal("accel_x", "accel_y", "accel_z").highpass(cutoff_hz=5, name=name),
                 plot=False)
    return [f"{name}.accel_x", f"{name}.accel_y", f"{name}.accel_z"]


def test_a_shown_vector_is_a_group_by_its_name(session):
    members = shown_vector(session)
    assert session.groups["hp"] == members
    assert session.members("hp") == members
    assert session.groups["channels"] == list(session.buffer.channels)
    session.show(mmp_sense.signal().magnitude().energy("80ms", name="energy2"), plot=False)
    assert "energy2" not in session.groups           # one value: a signal, not a group
    session.hide("hp")
    assert "hp" not in session.groups


def test_a_group_of_any_signals(session):
    show_basics(session)
    members = shown_vector(session)
    heard = []
    session.listen(lambda kind, payload: heard.append((kind, payload.get("signal"))))
    session.group("motion", ["magnitude", members[2]])
    assert session.groups["motion"] == ["magnitude", "hp.accel_z"]
    session.chart("motion")
    assert ("chart", "motion") in heard

    session.hide("hp")                                # a member goes: skipped
    assert session.groups["motion"] == ["magnitude"]
    session.group("motion", None)
    assert "motion" not in session.groups


def test_a_group_name_must_be_free_and_its_members_real(session):
    show_basics(session)
    shown_vector(session)
    with pytest.raises(ValueError):
        session.group("magnitude", ["accel_x"])       # a signal's name
    with pytest.raises(ValueError):
        session.group("hp", ["accel_x"])              # a shown node's name
    with pytest.raises(ValueError):
        session.group("channels", ["accel_x"])
    with pytest.raises(SignalError):
        session.group("g", ["accel_x", "nothing"])
    with pytest.raises(SignalError):
        session.chart("nothing")
    session.group("g", ["accel_x"])
    with pytest.raises(ValueError):
        session.show(mmp_sense.signal("accel_x").highpass(cutoff_hz=5, name="g"))


pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from mmp_sense.ui.main_window import MainWindow          # noqa: E402


@pytest.fixture
def window(qapp, session):
    window = MainWindow(show_basics(session), window_s=10.0, console=False)
    window.show()
    window.tick()
    try:
        yield window
    finally:
        window.close()


def curves(pane):
    return sorted(pane._curves)


def test_showing_a_vector_opens_one_chart_with_every_component(window, session):
    before = len(window.panes)
    session.show(mmp_sense.signal("accel_x", "accel_y", "accel_z").highpass(cutoff_hz=5, name="hp"))
    window.tick()
    assert len(window.panes) == before + 1
    pane = window.panes[-1]
    assert pane.signal == "hp" and pane.chooser.currentText() == "hp"
    assert curves(pane) == ["hp.accel_x", "hp.accel_y", "hp.accel_z"]
    assert pane.plot.getPlotItem().getAxis("left").labelText == "hp"
    legend = pane.plot.getPlotItem().legend
    assert sorted(label.text for _sample, label in legend.items) == curves(pane)


def test_plotting_a_group_opens_one_chart(window, session):
    session.group("motion", ["magnitude", "accel_z"])
    before = len(window.panes)
    session.chart("motion")
    window.tick()
    assert len(window.panes) == before + 1
    assert curves(window.panes[-1]) == ["accel_z", "magnitude"]


def test_a_group_can_be_picked_in_a_signal_charts_dropdown(window, session, qapp):
    session.group("motion", ["magnitude", "accel_z"])
    qapp.processEvents()                      # the news crosses to the Qt thread
    window.tick()
    pane = window.add_pane("signal", "magnitude")
    assert "motion" in pane.choices()
    pane.chooser.setCurrentText("motion")
    window.tick()
    assert curves(pane) == ["accel_z", "magnitude"]
    spectrum = [p for p in window.panes if p.kind == "spectrum"][0]
    assert "motion" not in spectrum.choices() and "channels" not in spectrum.choices()


def test_a_chart_whose_group_goes_falls_back(window, session, qapp):
    session.show(mmp_sense.signal("accel_x", "accel_y").highpass(cutoff_hz=5, name="hp"))
    window.tick()
    pane = window.panes[-1]
    session.hide("hp")
    qapp.processEvents()
    window.tick()
    assert pane.signal in pane.choices()
    assert "hp" not in pane.choices()


def test_a_chart_switched_on_stopped_data_draws_at_once(window, session):
    session.stop_intake()
    pane = window.add_pane("signal", "magnitude")
    pane.chooser.setCurrentText("accel_y")
    x, y = pane._curves["accel_y"].getData()
    assert x is not None and len(x) == len(session.buffer)
