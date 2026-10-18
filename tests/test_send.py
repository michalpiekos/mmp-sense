"""Code sent from an editor runs in the window's console: `--listen`, `mmp-sense-send`."""

import os
import socket
import stat
import subprocess
import sys
import threading
import time

import pytest

import mmp_sense_send


@pytest.fixture(autouse=True)
def runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.delenv("MMP_SENSE_SOCKET", raising=False)
    return tmp_path


# -- mmp-sense-send, no Qt ---------------------------------------------------------


def test_an_indented_selection_is_dedented():
    selected = "        for i in range(3):\n            total += i\n"
    assert mmp_sense_send.prepare(selected) == "for i in range(3):\n    total += i\n"


def test_with_no_window_listening_it_says_so(runtime, capsys):
    stale = mmp_sense_send.directory() / "999999.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(stale))
    server.close()                          # a window that died
    assert mmp_sense_send.listeners() == []
    assert not stale.exists()

    command = os.path.join(os.path.dirname(sys.executable), "mmp-sense-send")
    result = subprocess.run([command], input="x = 1\n",
                            capture_output=True, text=True, env=os.environ)
    assert result.returncode == 1
    assert "no mmp-sense console listening" in result.stderr


def fake_window(path, answer=b"ok\n"):
    received = []
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen(1)

    def serve():
        connection, _ = server.accept()
        with connection:
            data = b""
            while chunk := connection.recv(4096):
                data += chunk
            received.append(data.decode())
            connection.sendall(answer)
        server.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    return received, thread


def test_the_code_goes_to_the_named_socket_and_ok_comes_back(runtime):
    path = runtime / "window.sock"
    received, thread = fake_window(path)
    assert mmp_sense_send.send("x = 1\n", path) == "ok"
    thread.join(2)
    assert received == ["x = 1\n"]


def test_an_error_answer_fails_the_command(runtime, monkeypatch, capsys):
    path = runtime / "window.sock"
    fake_window(path, b"error: not UTF-8\n")
    monkeypatch.setattr(sys, "stdin", type("In", (), {"read": lambda self: "x\n"})())
    assert mmp_sense_send.main(["--socket", str(path)]) == 1
    assert "not UTF-8" in capsys.readouterr().err


def test_mmp_sense_send_does_not_import_mmp_sense():
    code = "import sys, mmp_sense_send; print('mmp_sense' in sys.modules or 'numpy' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "False"


# -- the window ------------------------------------------------------------------

pytest.importorskip("PySide6")
pytest.importorskip("qtconsole")
pytest.importorskip("ipykernel")

from mmp_sense.ui.main_window import MainWindow          # noqa: E402


def wait_for(qapp, condition, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        qapp.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def listening(qapp, session):
    window = MainWindow(session, window_s=10.0, console=True, listen=True)
    if window.console is None:
        window.close()
        pytest.skip("no in-process kernel available")
    window.show()
    shell = window.console.kernel_manager.kernel.shell
    try:
        yield window, shell
    finally:
        window.close()


def test_sent_code_runs_in_the_console_and_is_shown(listening, qapp):
    window, shell = listening
    path = window.listener.path
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert mmp_sense_send.listeners() == [path]
    assert mmp_sense_send.send("x = 41 + 1\n") == "ok"
    assert wait_for(qapp, lambda: shell.user_ns.get("x") == 42)
    assert "x = 41 + 1" in window.console._control.toPlainText()


def test_an_indented_block_runs(listening, qapp):
    _window, shell = listening
    mmp_sense_send.send(mmp_sense_send.prepare("    total = 0\n    for i in range(4):\n        total += i\n"))
    assert wait_for(qapp, lambda: shell.user_ns.get("total") == 6)


def test_sends_wait_for_a_busy_console_and_keep_their_order(listening, qapp, monkeypatch):
    window, shell = listening
    runner = window.listener.runner
    busy = [True]                                   # e.g. waiting on input()
    real_busy = runner._busy
    monkeypatch.setattr(runner, "_busy", lambda: busy[0] or real_busy())
    mmp_sense_send.send("order = ['a']\n")
    mmp_sense_send.send("order.append('b')\n")
    assert wait_for(qapp, lambda: runner.pending == 2)
    assert "order" not in shell.user_ns
    busy[0] = False
    mmp_sense_send.send("order.append('c')\n")
    assert wait_for(qapp, lambda: shell.user_ns.get("order") == ["a", "b", "c"])
    assert wait_for(qapp, lambda: runner.pending == 0)


def test_back_to_back_sends_all_run_in_order(listening, qapp):
    _window, shell = listening
    for i in range(5):
        mmp_sense_send.send(f"seen = globals().get('seen', []) + [{i}]\n")
    assert wait_for(qapp, lambda: shell.user_ns.get("seen") == [0, 1, 2, 3, 4])


def test_what_was_half_typed_is_put_back(listening, qapp):
    window, shell = listening
    assert wait_for(qapp, lambda: not window.console._executing)
    window.console.input_buffer = "session.cur"
    mmp_sense_send.send("y = 7\n")
    assert wait_for(qapp, lambda: shell.user_ns.get("y") == 7)
    assert wait_for(qapp, lambda: window.console.input_buffer == "session.cur")


def test_closing_the_window_removes_the_socket(qapp, session):
    window = MainWindow(session, window_s=10.0, console=True, listen=True)
    if window.console is None:
        window.close()
        pytest.skip("no in-process kernel available")
    path = window.listener.path
    assert path.exists()
    window.close()
    assert not path.exists()


def test_without_listen_there_is_no_socket(qapp, session):
    window = MainWindow(session, window_s=10.0, console=True)
    try:
        assert window.listener is None
        assert mmp_sense_send.listeners() == []
    finally:
        window.close()


def test_the_mcp_endpoints_do_not_list_the_console_sockets(listening):
    from mmp_sense.mcp.endpoint import endpoints

    assert endpoints() == []


def test_listen_without_the_console_does_not_listen(qapp, session):
    window = MainWindow(session, window_s=10.0, console=False, listen=True)
    try:
        assert window.listener is None
        assert mmp_sense_send.listeners() == []
    finally:
        window.close()


def test_listen_is_off_unless_asked_for():
    from mmp_sense.ui.app import parse_args

    assert parse_args([]).listen is False
    assert parse_args(["--listen"]).listen is True
    assert parse_args(["--listen", "/tmp/x.sock"]).listen == "/tmp/x.sock"


# -- a failed send shows on the console's title ----------------------------------


def title(window):
    return window.console_dock.titleBarWidget()


def settle(qapp, window):
    assert wait_for(qapp, lambda: window.listener.runner.pending == 0
                    and window.listener.runner._saved is None)


def test_a_failed_send_turns_the_title_red_until_the_console_is_opened(listening, qapp):
    window, _shell = listening
    assert window.console_dock.collapsed
    assert title(window).error is None
    mmp_sense_send.send("1/0\n")
    assert wait_for(qapp, lambda: title(window).error is not None)
    assert "ZeroDivisionError" in title(window).error
    assert "ZeroDivisionError" in title(window).label.toolTip()
    assert "bold" in title(window).label.styleSheet()

    mmp_sense_send.send("fine = 1\n")                       # collapsed: still red
    settle(qapp, window)
    assert window.console.kernel_manager.kernel.shell.user_ns["fine"] == 1
    assert title(window).error is not None

    window.set_dock_collapsed(window.console_dock, False)
    assert title(window).error is None
    assert title(window).label.styleSheet() == ""


def test_with_the_console_open_a_good_send_clears_the_red(listening, qapp):
    window, _shell = listening
    window.set_dock_collapsed(window.console_dock, False)
    mmp_sense_send.send("sesion.cursor\n")
    assert wait_for(qapp, lambda: title(window).error is not None)
    assert "NameError" in title(window).error
    settle(qapp, window)
    mmp_sense_send.send("ok_now = 1\n")
    settle(qapp, window)
    assert wait_for(qapp, lambda: title(window).error is None)


def test_a_syntax_error_in_sent_code_turns_it_red(listening, qapp):
    window, _shell = listening
    mmp_sense_send.send("def broken(:\n")
    assert wait_for(qapp, lambda: title(window).error is not None)
    assert "SyntaxError" in title(window).error


def test_an_error_typed_at_the_prompt_does_not(listening, qapp):
    window, _shell = listening
    window.console.execute("1/0")
    for _ in range(30):
        qapp.processEvents()
        time.sleep(0.01)
    assert title(window).error is None
