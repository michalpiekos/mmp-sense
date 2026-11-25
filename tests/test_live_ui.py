"""File -> Open stream, Receive streams, the Live streams dock and Save selection, offscreen."""

from __future__ import annotations

import time

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

import mmp_stream                                               # noqa: E402
from PySide6.QtWidgets import QApplication                      # noqa: E402
from fake_relay import FakeRelay                                # noqa: E402
from senselab import Senselab                                   # noqa: E402
from test_streaming import RATE, two_nodes                      # noqa: E402
from mmp_sense.settings import Settings                         # noqa: E402
from mmp_sense.ui.main_window import MainWindow                 # noqa: E402
from mmp_sense.ui.panes import SignalPane                       # noqa: E402
from mmp_sense.ui.panes.base import ALL_CHANNELS                # noqa: E402


@pytest.fixture
def window(qapp, session, monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    window = MainWindow(session, window_s=10.0, console=False)
    told = []
    window.files.tell = lambda caption, text, problem=False: told.append((caption, text, problem))
    window.files.told = told
    window.show()
    window.tick()
    try:
        yield window
    finally:
        window.live.leave()
        window.close()


def until(window, condition, seconds=5.0):
    end = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < end, "timed out"
        window.tick()
        window.live.update()
        QApplication.processEvents()          # what other threads sent the window
        time.sleep(0.01)


def watching(window, relay):
    names = window.live.open_streams(two_nodes(relay), relay=relay.address)
    session = window.session
    until(window, lambda: all(len(session.streams[n].buffer) == 2 * RATE for n in names)
          and window.live.live.epoch is not None)
    window.live.update()
    return names


def test_open_stream_shows_a_chart_per_stream_and_the_dock(window):
    with FakeRelay() as relay:
        names = watching(window, relay)
        charts = [p for p in window.panes if isinstance(p, SignalPane)]
        assert [p.signal for p in charts] == names, "one amplitude chart per stream"
        assert window.focus_signal == f"{names[0]}:S"
        dock = window.live.dock
        assert not window.live.live_dock.collapsed
        assert [dock.tree.topLevelItem(i).text(0) for i in range(2)] == names
        assert dock.tree.topLevelItem(0).text(3) == "node ntp ±20 ms"
        assert dock.tree.topLevelItem(1).text(3) == "receiver arrival ±100 ms"
        assert dock.epoch.text() == "Time 0 is 2026-11-15 08:00:00 UTC"
        assert window.files.shown is None

        dock.tree.setCurrentItem(dock.tree.topLevelItem(1))
        dock.trim.setValue(-30.0)
        assert window.live.live.streams[names[1]].trim == pytest.approx(-0.03)

        window.live.close_stream(names[1])
        assert [p.signal for p in window.panes if isinstance(p, SignalPane)] == [ALL_CHANNELS]


def test_the_menu_offers_streams_and_cancelling_does_nothing(window):
    texts = [a.text() for a in window.files.menu.actions()]
    assert texts.index("Open s&tream...") == texts.index("&Open...") + 1
    assert texts.index("Save s&election to workspace") == texts.index("Save a&ll") + 1
    window.live.ask_streams = lambda relay: None
    assert window.live.open_streams() == []
    assert not window.live.live.streams


def test_the_relay_comes_from_the_settings(window):
    Settings(server="http://192.168.1.2:8080").save()
    asked = []
    window.live.ask_streams = lambda relay: asked.append(relay)
    window.live.open_streams()
    assert asked == ["192.168.1.2:8091"], "the server's host by default"


def test_receive_streams_watches_and_keeps_each_stream_whole(window, tmp_path):
    files = window.files
    files.create_workspace(tmp_path / "ws", "Field day")
    live = window.live
    assert live.start_receiving("127.0.0.1:0", "127.0.0.1:0")
    try:
        assert live.receive_action.isChecked()
        relay = live.receiving_relay()
        assert "Receiving streams on 127.0.0.1:" in live.receiving_label.text()
        asked = []
        live.ask_streams = lambda r: asked.append(r)
        live.open_streams()
        assert asked == [relay], "the window's own relay while it receives"

        node = f"127.0.0.1:{live.receiver.address[1]}"
        sid = mmp_stream.send(node, "tetrahedral", "tetra1", seconds=0.5, title="Bench", log=lambda _l: None)
        until(window, lambda: any(r.sync.session_id == sid for r in files.workspace.recordings))
        (recording,) = files.workspace.recordings
        assert recording.saved and recording.origin == "session" and recording.title == "Bench"
        assert not (tmp_path / "ws" / "received" / sid).exists(), "moved in, not copied"
        assert (files.workspace.path_of(recording) / "mics.wav").is_file()

        with Senselab(tmp_path / "srv") as server:
            files.synchronize(server.url, wait=True)
            assert [r.outcome for r in files.last_results] == ["sent"]
            assert server.manifest(sid)["title"] == "Bench"
            assert server.store.session(sid)["missing"] == []
    finally:
        live.stop_receiving()
    assert not live.receive_action.isChecked() and live.receiving_label.text() == ""


def test_receiving_on_a_port_in_use_is_told(window):
    import socket

    busy = socket.socket()
    busy.bind(("127.0.0.1", 0))
    busy.listen()
    try:
        assert not window.live.start_receiving(f"127.0.0.1:{busy.getsockname()[1]}", "127.0.0.1:0")
        assert window.files.told[-1][2] and "Cannot receive" in window.files.told[-1][1]
        assert not window.live.receive_action.isChecked()
    finally:
        busy.close()


def test_a_selection_is_kept_in_the_workspace_and_synchronized(window, tmp_path):
    with FakeRelay() as relay:
        names = watching(window, relay)
        assert window.live.save_selection() == []
        assert window.files.told[-1][2], "nothing selected"
        window.session.set_selection((0.5, 1.5))
        added = window.live.save_selection()
        assert [r.title for r in added] == ["tetrahedral mics 08:00:00", "tetrahedral mics 08:00:00"]
        assert all(not r.saved for r in added)
        assert added[0].started_at_utc.startswith("2026-11-15T08:00:00.50")
        assert added[0].meta["channel_names"] == ["S", "N", "W", "E"]

        files = window.files
        files.create_workspace(tmp_path / "ws", "Field day")
        files.save_all()
        with Senselab(tmp_path / "srv") as server:
            files.synchronize(server.url, wait=True)
            assert [r.outcome for r in files.last_results] == ["sent", "sent"]
            first = added[0]
            m = server.manifest(first.sync.session_id)
            assert m["meta"]["channel_names"] == ["S", "N", "W", "E"]
            assert m["meta"]["sample_format"] == "s16"
            assert m["meta"]["clock"]["source"] == "ntp" and m["meta"]["clock"]["uncertainty_ns"] == 20_000_000
            assert m["meta"]["stream"] == "mics"
            assert m["started_utc_ns"] // 10_000_000 == 179_472_960_050
            clock = server.manifest(added[1].sync.session_id)["meta"]["clock"]
            assert (clock["from"], clock["uncertainty_ns"], clock["domain"]) == ("receiver", 100_000_000, "ntp:mmp-stream-receiver")

        # Showing a recording leaves the live view, and the fragment has its channels' names.
        files.dock.list.setCurrentRow(0)
        window.tick()
        assert files.shown is first and not window.live.live.streams
        assert window.live.live_dock.collapsed
        assert window.session.channels == ["S", "N", "W", "E"]
        assert [p.signal for p in window.panes if isinstance(p, SignalPane)] == [ALL_CHANNELS]
        assert names[0] not in window.session.streams
