"""`mmp_sense.live`: the console's names, typed for an editor, live in the console."""

import subprocess
import sys

import pytest


def test_outside_a_console_the_import_says_why():
    code = ("import sys\n"
            "try:\n"
            "    from mmp_sense.live import session\n"
            "except ImportError as exc:\n"
            "    print('ERR', exc)\n"
            "print('QT', 'PySide6' in sys.modules)\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout
    assert "ERR mmp_sense.live.session: no mmp-sense console in this process" in out
    assert "mmp-sense-send" in out
    assert "QT False" in out


def test_an_unknown_name_is_not_offered():
    import mmp_sense.live

    with pytest.raises(AttributeError):
        mmp_sense.live.nothing                                    # noqa: B018


pytest.importorskip("PySide6")
pytest.importorskip("qtconsole")
pytest.importorskip("ipykernel")

from mmp_sense.ui.main_window import MainWindow          # noqa: E402


def test_in_the_console_the_names_are_the_windows_own(qapp, session):
    window = MainWindow(session, window_s=10.0, console=True)
    try:
        if window.console is None:
            pytest.skip("no in-process kernel available")
        shell = window.console.kernel_manager.kernel.shell
        shell.run_cell("from mmp_sense.live import session as s, ui as u, show, signal\n"
                       "from mmp_sense.live import *")
        ns = shell.user_ns
        assert ns["s"] is session and ns["session"] is session
        assert ns["u"] is window
        assert ns["show"] == session.show
        shell.run_cell("knock = (signal().magnitude().energy('80ms') > 5).events(name='k')\n"
                       "show(knock)")
        assert "k" in session.shown
    finally:
        window.close()
