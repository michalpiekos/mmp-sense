"""Scripts: a file anywhere, given the session, drawing named results."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import mmp_sense
import mmp_sense_send
from mmp_sense import Session, from_frame, run_script
from mmp_sense.core.plots import Plot


def write(path: Path, text: str) -> Path:
    path.write_text(textwrap.dedent(text))
    return path


# -- samples -----------------------------------------------------------


def test_samples_are_the_selection_as_numpy(session):
    session.set_selection((4.0, 6.0))
    channel = session.channels[2]
    samples = session.samples(channel)
    expected = session.values(channel, 4.0, 6.0)
    np.testing.assert_array_equal(samples.values, expected.to_numpy(float))
    np.testing.assert_array_equal(samples.time, expected.index.to_numpy(float))
    assert samples.rate == 500.0
    assert samples.name == channel
    assert samples.units == session.signals.get(channel).units


def test_samples_without_a_selection_are_everything_held(session):
    channel = session.channels[0]
    assert len(session.samples(channel).values) == len(session.values(channel))


def test_samples_of_a_recipe(session):
    recipe = mmp_sense.signal(session.channels[0]).highpass(2.0)
    assert len(session.samples(recipe, 1.0, 2.0).values) == len(session.values(recipe, 1.0, 2.0))


def test_samples_refuse_several_components(session):
    with pytest.raises(ValueError, match="takes one signal"):
        session.samples(mmp_sense.signal().highpass(2.0, name="hp"))


def test_samples_give_each_streams_own_rate():
    from mmp_sense import SeriesSource
    from mmp_sense.core.data import TimeSeries

    def series(name, rate):
        time = np.arange(int(rate)) / rate
        return SeriesSource(TimeSeries(time=time, values=np.sin(time)[:, None],
                                       channels=["v"], sample_rate=rate), name=name)

    session = Session(series("slow", 100.0)).start()
    session.add(series("fast", 1000.0), "fast")
    session.poll()
    assert session.samples("fast:v").rate == 1000.0
    assert session.samples(f"{session.signals.first}:v").rate == 100.0
    assert len(session.samples("fast:v").values) == 1000


# -- plots -------------------------------------------------------------


def test_replotting_replaces_and_registers_nothing_else(session):
    names, streams, groups = list(session.signals.names), list(session.streams), session.groups
    heard = []
    session.listen(lambda kind, payload: heard.append((kind, payload.get("name"))))
    for n in range(5):
        session.plot("result", [0, 1, 2], [n, n, n])
    assert list(session.plots) == ["result"]
    assert session.plots["result"].y[0] == 4
    assert heard.count(("plots", "result")) == 5
    assert session.signals.names == names
    assert list(session.streams) == streams
    assert session.groups == groups


def test_a_plot_holds_a_read_only_copy(session):
    x = np.arange(3.0)
    plot = session.plot("p", x, x)
    x[0] = 9
    assert plot.x[0] == 0
    with pytest.raises(ValueError):
        plot.x[0] = 1


def test_a_plot_takes_pandas_and_keeps_labels(session):
    series = pd.Series([1.0, 2.0])
    plot = session.plot("p", series.index, series, x_label="Lag", x_units="s",
                        y_label="Level", y_units="dB")
    assert isinstance(plot, Plot)
    assert (plot.x_label, plot.x_units, plot.y_label, plot.y_units) == ("Lag", "s", "Level", "dB")


@pytest.mark.parametrize("x, y", [
    ([1, 2], [1]),
    ([[1, 2]], [[1, 2]]),
    ([1j], [1]),
    (["a"], [1]),
])
def test_a_bad_plot_is_refused_and_the_old_one_stays(session, x, y):
    old = session.plot("p", [1], [2])
    with pytest.raises(ValueError):
        session.plot("p", x, y)
    assert session.plots["p"] is old


def test_a_plot_needs_a_name(session):
    with pytest.raises(ValueError, match="needs a name"):
        session.plot("", [1], [1])


def test_removing_a_plot(session):
    session.plot("p", [1], [2])
    session.remove_plot("p")
    assert session.plots == {}
    with pytest.raises(KeyError, match="no plot 'p'"):
        session.remove_plot("p")


# -- run_script --------------------------------------------------------


def test_main_is_given_the_session(session, tmp_path):
    script = write(tmp_path / "s.py", """\
        def main(session):
            return session
    """)
    assert run_script(script, session) is session


def test_a_script_is_read_afresh(session, tmp_path):
    script = tmp_path / "s.py"
    for n in (1, 2):
        script.write_text(f"def main(session):\n    return {n}\n")
        assert run_script(script, session) == n


def test_a_sibling_module_is_found_and_its_edits_count(session, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path.parent)
    folder = tmp_path / "scripts"
    folder.mkdir()
    write(folder / "s.py", """\
        import helper_for_mmp

        def main(session):
            return helper_for_mmp.VALUE
    """)
    for value in ("1", "22"):
        (folder / "helper_for_mmp.py").write_text(f"VALUE = {value}\n")
        assert run_script(folder / "s.py", session) == int(value)
    assert "helper_for_mmp" not in sys.modules
    assert str(folder) not in sys.path


def test_what_was_imported_from_elsewhere_stays(session, tmp_path):
    sys.modules.pop("wave", None)
    script = write(tmp_path / "s.py", "import wave\n\ndef main(session):\n    pass\n")
    run_script(script, session)
    assert "wave" in sys.modules


def test_a_dataclass_in_a_script_works(session, tmp_path):
    script = write(tmp_path / "s.py", """\
        from __future__ import annotations
        from dataclasses import dataclass

        @dataclass
        class Result:
            value: int

        def main(session):
            return Result(1).value
    """)
    assert run_script(script, session) == 1


def test_a_script_can_run_another(session, tmp_path):
    inner = write(tmp_path / "inner.py", "def main(session):\n    return 1\n")
    outer = write(tmp_path / "outer.py", f"""\
        import mmp_sense

        def main(session):
            return mmp_sense.run_script({str(inner)!r}, session) + 1
    """)
    assert run_script(outer, session) == 2


def test_a_script_without_main_says_so(session, tmp_path):
    with pytest.raises(ValueError, match="defines no main"):
        run_script(write(tmp_path / "s.py", "x = 1\n"), session)


def test_a_failing_script_points_at_itself_and_cleans_up(session, tmp_path):
    script = write(tmp_path / "s.py", "def main(session):\n    1 / 0\n")
    with pytest.raises(ZeroDivisionError) as caught:
        run_script(script, session)
    assert str(caught.traceback[-1].path) == str(script)
    assert str(tmp_path) not in sys.path


def test_a_missing_script(session, tmp_path):
    with pytest.raises(FileNotFoundError):
        run_script(tmp_path / "gone.py", session)


def test_the_example_runs_headless():
    time = np.arange(1000) / 1000.0
    session = from_frame(pd.DataFrame({"v": np.sin(2 * np.pi * 50 * time)}, index=time))
    from mmp_sense.examples import spectrum_script

    run_script(spectrum_script.__file__, session)
    plot = session.plots["v spectrum"]
    assert plot.x[np.argmax(plot.y)] == pytest.approx(50.0)
    assert (plot.x_label, plot.x_units) == ("Frequency", "Hz")


def test_a_script_runs_from_another_directory_in_a_new_process(tmp_path):
    other = tmp_path / "elsewhere"
    other.mkdir()
    write(tmp_path / "helper_for_mmp.py", "SCALE = 2.0\n")
    write(tmp_path / "s.py", """\
        import helper_for_mmp

        def main(session):
            samples = session.samples("v")
            session.plot("doubled", samples.time, samples.values * helper_for_mmp.SCALE)
            print(session.plots["doubled"].y.sum())
    """)
    code = ("import sys, numpy as np, pandas as pd, mmp_sense\n"
            "session = mmp_sense.from_frame(pd.DataFrame({'v': [1.0, 2.0]}, index=[0.0, 1.0]))\n"
            "mmp_sense.run_script(sys.argv[1], session)\n")
    done = subprocess.run([sys.executable, "-c", code, "../s.py"], cwd=other,
                          env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "6.0"


# -- sending -----------------------------------------------------------


def test_script_call_resolves_in_the_senders_directory(tmp_path, monkeypatch):
    script = tmp_path / "it's.py"
    script.write_text("")
    monkeypatch.chdir(tmp_path)
    code = mmp_sense_send.script_call("it's.py")
    assert code == f"run({str(script.resolve())!r})\n"


def test_sending_a_missing_script_fails_at_the_sender(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert mmp_sense_send.main(["--script", "gone.py"]) == 1
    assert "no such script" in capsys.readouterr().err


# -- the window --------------------------------------------------------


pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from mmp_sense.ui.main_window import MainWindow          # noqa: E402
from mmp_sense.ui.panes import XYPane                    # noqa: E402


def wait_for(qapp, condition, timeout=8.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        qapp.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def window(qapp, session):
    window = MainWindow(session, window_s=10.0, console=False)
    window.timer.stop()
    window.show()
    try:
        yield window
    finally:
        window.close()


def xy_panes(window):
    return [p for p in window.panes if isinstance(p, XYPane)]


def test_a_plot_opens_a_pane_with_its_axes(window, session):
    session.plot("result", [0, 1, 2], [3, 4, 5], x_label="Frequency", x_units="Hz",
                 y_label="Level", y_units="dB")
    [pane] = xy_panes(window)
    assert pane.signal == "result"
    x, y = pane.curve.getData()
    assert list(x) == [0, 1, 2] and list(y) == [3, 4, 5]
    bottom, left = pane.plot.getAxis("bottom"), pane.plot.getAxis("left")
    assert (bottom.labelText, bottom.labelUnits) == ("Frequency", "Hz")
    assert (left.labelText, left.labelUnits) == ("Level", "dB")
    assert not bottom.autoSIPrefix and not left.autoSIPrefix


def test_rerunning_updates_the_one_pane(window, session):
    count = len(window.panes)
    names = list(session.signals.names)
    for n in range(4):
        session.plot("result", [0, 1], [n, n + 1], y_units=f"u{n}")
    [pane] = xy_panes(window)
    assert len(window.panes) == count + 1
    assert list(pane.curve.getData()[1]) == [3, 4]
    assert pane.plot.getAxis("left").labelUnits == "u3"
    assert session.signals.names == names


def test_a_second_name_gets_a_second_pane(window, session):
    session.plot("a", [0], [0])
    session.plot("b", [0], [0])
    assert sorted(p.signal for p in xy_panes(window)) == ["a", "b"]


def test_a_closed_pane_comes_back_on_the_next_plot(window, session):
    session.plot("result", [0], [0])
    window.remove_pane(xy_panes(window)[0])
    assert xy_panes(window) == []
    session.plot("result", [0], [1])
    assert len(xy_panes(window)) == 1


def test_remove_plot_closes_its_pane(window, session):
    session.plot("a", [0], [0])
    session.plot("b", [0], [0])
    session.remove_plot("a")
    assert [p.signal for p in xy_panes(window)] == ["b"]
    assert xy_panes(window)[0].choices() == ["b"]


def test_an_empty_xy_pane_picks_up_a_plot(window, session):
    pane = window.add_pane("xy")
    assert pane.signal == ""
    assert pane.status.text().startswith("no plot")
    session.plot("result", [0, 1], [1, 2])
    assert xy_panes(window) == [pane]
    assert pane.chooser.currentText() == "result"
    assert len(pane.curve.getData()[0]) == 2


def test_gaps_in_a_plot_draw(window, session):
    session.plot("gaps", [0.0, 1.0, 2.0, np.inf], [1.0, np.nan, 3.0, 4.0])
    window.grab()


def test_an_xy_pane_named_like_a_signal_does_not_hide_it(window, session):
    channel = session.channels[1]
    session.plot(channel, [0], [0])
    before = [p for p in window.panes if p.kind == "signal"]
    window.plot_signal(channel)
    assert len([p for p in window.panes if p.kind == "signal"]) == len(before) + 1


def test_a_plot_from_another_thread_reaches_the_window(qapp, window, session):
    import threading

    thread = threading.Thread(target=session.plot, args=("far", [0], [1]))
    thread.start()
    thread.join()
    assert wait_for(qapp, lambda: len(xy_panes(window)) == 1)


@pytest.fixture
def listening(qapp, session):
    window = MainWindow(session, window_s=10.0, console=True, listen=True)
    if window.console is None:
        window.close()
        pytest.skip("no in-process kernel available")
    window.show()
    try:
        yield window
    finally:
        window.close()


def test_a_sent_script_runs_in_the_window_from_anywhere(listening, session, qapp, tmp_path,
                                                        monkeypatch):
    script = write(tmp_path / "exp.py", """\
        from mmp_sense import Session

        def main(session: Session) -> None:
            samples = session.samples(session.channels[0])
            session.plot("sent", samples.time, samples.values, x_label="Time", x_units="s",
                         y_label=samples.name, y_units=samples.units)
    """)
    session.set_selection((1.0, 2.0))
    monkeypatch.chdir(tmp_path)
    assert mmp_sense_send.send(mmp_sense_send.script_call(script.name)) == "ok"
    assert wait_for(qapp, lambda: len(xy_panes(listening)) == 1)
    [pane] = xy_panes(listening)
    assert len(pane.curve.getData()[0]) == len(session.samples(session.channels[0]).values)
    assert pane.plot.getAxis("bottom").labelUnits == "s"
    session.clear_selection()
    mmp_sense_send.send(mmp_sense_send.script_call(script.name))
    held = len(session.samples(session.channels[0]).values)
    assert wait_for(qapp, lambda: len(pane.curve.getData()[0]) == held)
    assert xy_panes(listening) == [pane]


def test_a_failing_sent_script_turns_the_title_red(listening, qapp, tmp_path):
    script = write(tmp_path / "bad.py", "def main(session):\n    1 / 0\n")
    assert mmp_sense_send.send(mmp_sense_send.script_call(str(script))) == "ok"
    title = listening.console_dock.titleBarWidget()
    assert wait_for(qapp, lambda: title.error is not None)
    assert title.error.startswith("ZeroDivisionError")
    assert listening.isVisible()


def test_live_serves_run(listening):
    import mmp_sense.live

    assert callable(mmp_sense.live.run)
