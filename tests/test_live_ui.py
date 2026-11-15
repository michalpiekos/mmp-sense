"""File -> Open stream, the Live streams dock and Save selection, offscreen."""

from __future__ import annotations

import time

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from fake_relay import FakeRelay                                # noqa: E402
from fake_senselab import TOKEN, FakeSenselab                   # noqa: E402
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
        assert dock.tree.topLevelItem(1).text(3) == "gateway arrival ±100 ms"
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
    window.live.ask_streams = lambda relay, token_for: None
    assert window.live.open_streams() == []
    assert not window.live.live.streams


def test_the_relay_and_token_come_from_the_settings(window, tmp_path):
    token = tmp_path / "relay.token"
    token.write_text("mmps_relay\n")
    Settings(server="http://192.168.1.2:8080", token_file=str(token)).save()
    asked = []
    window.live.ask_streams = lambda relay, token_for: asked.append(
        (relay, token_for(relay), token_for("192.168.1.23:8091"), token_for("localhost:18091")))
    window.live.open_streams()
    assert asked == [("192.168.1.2:8091", "mmps_relay", None, "mmps_relay")], \
        "the token goes to the server's host, or through a tunnel on this machine, only"


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
        with FakeSenselab() as fake:
            files.synchronize(fake.url, TOKEN, wait=True)
            assert [r.outcome for r in files.last_results] == ["sent", "sent"]
            first = added[0]
            declared = fake.samples[first.sync.sample_id]["declaration"]
            assert declared["encoding"]["channel_names"] == ["S", "N", "W", "E"]
            assert declared["encoding"]["sample_format"] == "s16"
            session = fake.sessions[first.sync.capture_session_id]
            assert session["clock"] == {"source": "ntp", "uncertainty_ns": 20_000_000}
            assert session["started_at_utc"] == first.started_at_utc
            assert session["metadata"]["live"]["stream"] == "mics"
            assert fake.sessions[added[1].sync.capture_session_id]["clock"] == {
                "source": "ntp", "uncertainty_ns": 100_000_000, "domain": "ntp:stream-gateway"}

        # Showing a recording leaves the live view, and the fragment has its channels' names.
        files.dock.list.setCurrentRow(0)
        window.tick()
        assert files.shown is first and not window.live.live.streams
        assert window.live.live_dock.collapsed
        assert window.session.channels == ["S", "N", "W", "E"]
        assert [p.signal for p in window.panes if isinstance(p, SignalPane)] == [ALL_CHANNELS]
        assert names[0] not in window.session.streams
