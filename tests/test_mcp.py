"""The tools, the socket a session serves them on, and the MCP server."""

import asyncio
import json
import socket

import pytest

from mmp_sense.mcp import tools
from mmp_sense.mcp.endpoint import Endpoint, EndpointClient, endpoints


def test_the_tools_are_the_seven_questions():
    assert set(tools.TOOLS) == {
        "get_current_context", "list_signals", "get_statistics", "get_values",
        "get_events", "get_spectrum_summary", "get_delay",
    }
    for described in tools.describe_tools():
        assert described["description"]


def test_a_served_session_answers_over_its_socket(session):
    session.set_cursor(5.0)
    session.set_selection((4.9, 5.1))
    path = session.serve()
    try:
        assert path.exists()
        assert oct(path.stat().st_mode)[-3:] == "600"
        with EndpointClient(path) as client:
            context = client.call("get_current_context")
            assert context["cursor"] == 5.0
            assert context["selection"] == [4.9, 5.1]
            # No interval given: the answer is about the selection.
            stats = client.call("get_statistics", {"signal": "accel_z"})
            assert stats["stretch"] == [4.9, 5.1]
            assert {t["name"] for t in client.tools()} == set(tools.TOOLS)
    finally:
        session.close_endpoint()
    assert not path.exists()


def test_serving_twice_keeps_one_socket(session):
    first = session.serve()
    assert session.serve() == first
    session.close_endpoint()


def test_a_refused_call_is_an_error_not_a_crash(session):
    endpoint = Endpoint(session, None)
    reply = json.loads(endpoint.handle(b'{"id": 3, "tool": "nope"}'))
    assert reply == {"id": 3, "ok": False, "error": reply["error"], "type": "KeyError"}
    assert "no tool" in reply["error"]

    bad = json.loads(endpoint.handle(b"not json"))
    assert bad["ok"] is False and "not JSON" in bad["error"]

    missing = json.loads(
        endpoint.handle(b'{"tool": "get_statistics", "args": {"signal": "nope"}}')
    )
    assert missing["ok"] is False and "no channel or shown name" in missing["error"]


def test_a_dead_session_leaves_no_socket_behind(session, tmp_path):
    from mmp_sense.mcp import endpoint as endpoint_module

    directory = endpoint_module.default_directory()
    stale = directory / "99999999.sock"
    stale.touch()
    try:
        assert stale not in endpoints()
        assert not stale.exists()
    finally:
        stale.unlink(missing_ok=True)


def test_the_client_refuses_a_socket_that_is_not_there(tmp_path):
    with pytest.raises(OSError):
        EndpointClient(tmp_path / "nothing.sock")


def test_the_server_exposes_every_tool(session):
    pytest.importorskip("mcp")
    from mmp_sense.mcp.server import build_server

    class Backend:
        calls = []

        def call(self, name, args=None):
            self.calls.append((name, args))
            return tools.call(session, name, args or {})

    backend = Backend()
    server = build_server(backend)
    listed = asyncio.run(server.list_tools())
    names = {tool.name for tool in listed}
    assert names == set(tools.TOOLS)

    result = asyncio.run(server.call_tool("get_current_context", {}))
    assert backend.calls[-1][0] == "get_current_context"
    assert result


def test_the_server_proxies_a_running_session(session):
    pytest.importorskip("mcp")
    from mmp_sense.mcp.server import ProxyBackend

    session.set_cursor(4.0)
    path = session.serve()
    try:
        backend = ProxyBackend(path)
        assert backend.call("get_current_context")["cursor"] == 4.0
        assert "the mmp-sense session at" in backend.description
        backend.close()
    finally:
        session.close_endpoint()
