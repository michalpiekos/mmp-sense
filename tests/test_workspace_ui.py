"""The File menu and the Workspace dock, offscreen."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from fake_senselab import TOKEN, FakeSenselab                 # noqa: E402
from test_workspace import files                              # noqa: E402,F401
from mmp_sense.ui.app import build, parse_args                # noqa: E402
from mmp_sense.ui.main_window import MainWindow               # noqa: E402
from mmp_sense.workspace import Workspace                     # noqa: E402
from mmp_sense.workspace import sync                          # noqa: E402

MENU = ["&New", "&Open...", "Open s&tream...", "Open &workspace...", "&Create workspace...", "",
        "&Save", "Save wor&kspace", "Save workspace &as...", "Save a&ll", "Save s&election to workspace",
        "", "S&ynchronize"]


@pytest.fixture
def window(qapp, session, monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    window = MainWindow(session, window_s=10.0, console=False)
    told = []
    window.files.tell = lambda caption, text, problem=False: told.append((caption, text, problem))
    window.files.told = told
    window.show()
    window.tick()
    try:
        yield window
    finally:
        window.close()


def test_the_file_menu_holds_what_was_asked_for(window):
    files = window.files
    assert [a.text() for a in files.menu.actions()] == MENU
    assert not files.new_action.isEnabled()
    assert window.menuBar().actions()[0].text() == "&File"
    assert files.workspace_dock.collapsed            # nothing in it yet


def test_open_adds_the_file_and_shows_it(window, files):
    window.files.open_file(files[0])
    window.tick()
    workspace = window.files.workspace
    assert [r.name for r in workspace.recordings] == ["knock.wav"]
    assert window.files.shown is workspace.recordings[0]
    assert window.session.span[1] == pytest.approx(1.0, abs=0.01)
    assert not window.files.workspace_dock.collapsed
    assert "untitled" in window.windowTitle()
    assert window.files.dock.list.count() == 1


def test_save_in_an_untitled_workspace_creates_it_first(window, files, tmp_path):
    files_ = window.files
    files_.open_file(files[0])
    files_.ask_folder = lambda _caption: str(tmp_path / "ws")
    files_.ask_text = lambda *_a, **_k: "Field day"
    files_.save()
    workspace = files_.workspace
    assert workspace.folder == tmp_path / "ws" and workspace.title == "Field day"
    assert workspace.path_of(workspace.recordings[0]).read_bytes() == files[0].read_bytes()
    assert "Field day" in window.windowTitle()


def test_save_all_then_open_workspace_shows_the_same(window, files, tmp_path):
    f = window.files
    f.create_workspace(tmp_path / "ws", "Field day")
    f.open_file(files[0])
    f.open_file(files[1])
    f.dock.list.setCurrentRow(0)
    f.dock.title.setText("Knock")
    f.dock.title.textEdited.emit("Knock")
    assert f.shown.name == "knock.wav" and f.shown.title == "Knock"
    f.save_all()

    f.open_workspace(tmp_path / "ws")
    assert [r.title for r in f.workspace.recordings] == ["Knock", "walk"]
    assert f.shown is f.workspace.recordings[0]
    f.dock.list.setCurrentRow(1)
    window.tick()
    assert f.shown.name == "walk.mcap"
    assert "x" in window.session.channels


def test_save_workspace_as_works_on_the_copy(window, files, tmp_path):
    f = window.files
    f.create_workspace(tmp_path / "one", "One")
    f.open_file(files[0])
    f.save_all()
    before = (f.workspace.id, f.workspace.recordings[0].sync.capture_session_id)
    f.save_workspace_as(tmp_path / "two")
    assert f.workspace.folder == tmp_path / "two"
    assert (f.workspace.id, f.workspace.recordings[0].sync.capture_session_id) != before
    assert Workspace.open(tmp_path / "one").id == before[0]


def test_synchronize_sends_and_keeps_the_token_out_of_the_workspace(window, files, tmp_path):
    f = window.files
    f.create_workspace(tmp_path / "ws", "Field day")
    f.open_file(files[0])
    f.save_all()
    with FakeSenselab() as fake:
        f.synchronize(fake.url, TOKEN, wait=True)
        assert [r.outcome for r in f.last_results] == ["sent"]
        assert len(fake.samples) == 1
        assert f.told[-1][0] == "Synchronize" and not f.told[-1][2]
        assert sync.load_token(fake.url) == TOKEN
        assert TOKEN not in (tmp_path / "ws" / "workspace.json").read_text()
        assert Workspace.open(tmp_path / "ws").server == fake.url
        assert "on the server" in f.dock.list.item(0).text()
        assert f.dock.started.isReadOnly()

        f.ask_text = lambda *_a, **_k: pytest.fail("nothing should be asked")
        set_server(f, fake.url)
        f.synchronize(wait=True)             # the address from the settings, the token remembered
        assert [r.outcome for r in f.last_results] == ["up to date"]


def set_server(files, url):
    def fill(settings):
        settings.server = url
        return True

    files.ask_settings = fill
    files.edit_settings()


def test_edit_settings_keeps_the_server_for_every_window(window, qapp, session):
    assert window.menuBar().actions()[1].text() == "&Edit"
    assert [a.text() for a in window.files.edit_menu.actions()] == ["&Settings..."]
    set_server(window.files, "http://srv:8080/")
    other = MainWindow(session, console=False)
    try:
        seen = []
        other.files.ask_settings = lambda settings: seen.append(settings.server) or False
        other.files.edit_settings()
        assert seen == ["http://srv:8080"]
    finally:
        other.close()


def test_synchronize_without_a_server_opens_the_settings(window, files, tmp_path):
    f = window.files
    f.create_workspace(tmp_path / "ws", "Field day")
    f.open_file(files[0])
    f.save_all()
    with FakeSenselab() as fake:
        opened = []

        def fill(settings):
            opened.append(settings.server)
            settings.server = fake.url
            return True

        f.ask_settings = fill
        f.ask_text = lambda *_a, **_k: TOKEN
        f.synchronize(wait=True)
        assert opened == [""]
        assert [r.outcome for r in f.last_results] == ["sent"]


def test_a_workspace_is_not_sent_to_a_second_server(window, files, tmp_path):
    f = window.files
    f.create_workspace(tmp_path / "ws", "Field day")
    f.open_file(files[0])
    f.save_all()
    with FakeSenselab() as fake:
        f.synchronize(fake.url, TOKEN, wait=True)
        calls = len(fake.calls)
        set_server(f, "http://elsewhere:8080")
        f.synchronize(token=TOKEN, wait=True)
        assert f.told[-1][2] and fake.url in f.told[-1][1]
        assert len(fake.calls) == calls


def test_a_refused_token_is_forgotten(window, files, tmp_path):
    f = window.files
    f.create_workspace(tmp_path / "ws", "Field day")
    f.open_file(files[0])
    f.open_file(files[1])
    f.save_all()
    with FakeSenselab() as fake:
        f.synchronize(fake.url, "mmps_wrong", wait=True)
        assert [r.outcome for r in f.last_results] == ["failed", "failed"]
        assert "not tried" in f.last_results[1].message
        assert [path for _m, path in fake.calls] == [f"/capture-sessions/{f.workspace.recordings[0].sync.capture_session_id}"]
        assert sync.load_token(fake.url) is None
        assert f.told[-1][2]


def test_synchronize_needs_a_workspace_on_disk(window, files):
    window.files.open_file(files[0])
    window.files.synchronize("http://127.0.0.1:9", TOKEN, wait=True)
    assert window.files.told[-1][2]


def test_the_command_line_opens_a_workspace_folder(qapp, files, tmp_path):
    workspace = Workspace.create(tmp_path / "ws", "Field day")
    workspace.add(files[0])
    workspace.save_all()
    session, opened, shown = build(parse_args([str(tmp_path / "ws")]))
    assert opened.id == workspace.id and shown.name == "knock.wav"
    window = MainWindow(session, console=False, workspace=opened, shown=shown)
    try:
        assert "Field day" in window.windowTitle()
        assert window.files.dock.list.currentRow() == 0
    finally:
        window.close()


def test_the_command_line_puts_a_file_in_an_untitled_workspace(qapp, files):
    session, workspace, shown = build(parse_args([str(files[0])]))
    assert workspace.folder is None and workspace.recordings == [shown]
    assert not shown.saved


def test_the_token_is_read_from_the_token_file_in_the_settings(window, files, tmp_path):
    f = window.files
    f.create_workspace(tmp_path / "ws", "Field day")
    f.open_file(files[0])
    f.save_all()
    token_file = tmp_path / "mmp-sense.token"
    token_file.write_text(TOKEN + "\n")
    with FakeSenselab() as fake:
        def fill(settings):
            settings.server, settings.token_file = fake.url, f" {token_file} "
            return True

        f.ask_settings = fill
        f.edit_settings()
        f.ask_text = lambda *_a, **_k: pytest.fail("nothing should be asked")
        f.synchronize(wait=True)
        assert [r.outcome for r in f.last_results] == ["sent"]
        assert sync.load_token(fake.url) is None            # not copied anywhere

        token_file.write_text("mmps_wrong")
        f.workspace.recordings[0].notes = "changed"
        f.synchronize(wait=True)
        assert f.told[-1][2] and "token file" in f.told[-1][1]
        assert token_file.read_text() == "mmps_wrong"


def test_a_missing_token_file_is_told(window, files, tmp_path):
    f = window.files
    f.create_workspace(tmp_path / "ws", "Field day")
    f.open_file(files[0])
    f.save_all()

    def fill(settings):
        settings.server, settings.token_file = "http://127.0.0.1:9", str(tmp_path / "none")
        return True

    f.ask_settings = fill
    f.edit_settings()
    f.synchronize(wait=True)
    assert f.told[-1][2] and "token file" in f.told[-1][1]
    assert f.syncing is None


def test_the_settings_dialog_edits_the_address_and_the_token_file(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from mmp_sense.settings import Settings
    from mmp_sense.ui.settings_dialog import SettingsDialog

    settings = Settings(server="http://a:8080", token_file="/old")
    dialog = SettingsDialog(settings)
    assert (dialog.server.text(), dialog.token_file.text()) == ("http://a:8080", "/old")
    dialog.server.setText("http://b:8080/ ")
    dialog.token_file.setText(" ~/mmp-sense.token ")
    dialog.accept()
    settings.save()
    assert Settings.load() == Settings("http://b:8080", "~/mmp-sense.token")
