"""A local MCP server over an mmp-sense session.

    mmp-sense-mcp                      attach to a running window, else simulate
    mmp-sense-mcp --endpoint PATH      a particular running session
    mmp-sense-mcp --no-attach          a simulated session of its own
    mmp-sense-mcp walk.mcap            a session of its own on a file

Attached, the tools answer about the window the user is looking at:
`get_current_context` is their cursor, and `get_statistics` with no
stretch is their selection. That is the point of the thing -- nobody
should have to dictate a timestamp to an assistant that could ask.

The tools are the ones in `tools.py`, all read-only.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

from . import tools
from .endpoint import EndpointClient, endpoints

__all__ = ["build_server", "LocalBackend", "ProxyBackend", "main"]


class LocalBackend:
    """A session in this process."""

    def __init__(self, session, poll_interval: float = 0.05) -> None:
        self.session = session
        self.poll_interval = poll_interval
        self._stopping = threading.Event()
        self._thread = threading.Thread(
            target=self._poll, name="mmp-sense-mcp-poll", daemon=True
        )
        self._thread.start()

    def _poll(self) -> None:
        while not self._stopping.wait(self.poll_interval):
            try:
                self.session.poll()
            except Exception as exc:                          # noqa: BLE001
                print(f"mmp-sense-mcp: polling failed: {exc}", file=sys.stderr)

    def call(self, name: str, args: dict | None = None) -> dict:
        if name == "list_tools":
            return {"tools": tools.describe_tools()}
        return tools.call(self.session, name, args)

    def close(self) -> None:
        self._stopping.set()
        self.session.stop()

    @property
    def description(self) -> str:
        return f"its own session on {self.session.dataset}"


class ProxyBackend:
    """A session running elsewhere, reached through its socket."""

    def __init__(self, path) -> None:
        self.client = EndpointClient(path)

    def call(self, name: str, args: dict | None = None) -> dict:
        return self.client.call(name, args)

    def close(self) -> None:
        self.client.close()

    @property
    def description(self) -> str:
        return f"the mmp-sense session at {self.client.path}"


def build_server(backend, name: str = "mmp-sense"):
    """An MCP server whose tools are answered by `backend`."""
    from mcp.server.mcpserver import MCPServer

    server = MCPServer(
        name=name,
        instructions=(
            "mmp-sense is a signal exploration environment. Start with "
            "get_current_context to find out what the user is looking at, "
            "then get_statistics and get_values for a signal, "
            "get_spectrum_summary for its frequencies, and get_events for "
            "what a detector found. Without t0/t1 every tool answers about "
            "the user's selection, else everything held."
        ),
    )

    def get_current_context() -> dict:
        """Cursor, frequency cursor, selection, the time held, the streams,
        the shown signals and detectors, and whether samples are still arriving."""
        return backend.call("get_current_context")

    def list_signals() -> dict:
        """Every signal: the streams' channels (stream:channel when there
        are several) and what the user has shown."""
        return backend.call("list_signals")

    def get_statistics(signal: str, t0: float | None = None,
                       t1: float | None = None) -> dict:
        """n, min, max, mean, rms, std and peak_to_peak of one signal.
        Without t0/t1: the user's selection, else everything held."""
        return backend.call("get_statistics", {"signal": signal, "t0": t0, "t1": t1})

    def get_values(signal: str, t0: float | None = None, t1: float | None = None,
                   points: int = 200) -> dict:
        """The shape of a signal: a min/max envelope of a few hundred points."""
        return backend.call("get_values", {"signal": signal, "t0": t0, "t1": t1,
                                           "points": points})

    def get_events(detector: str, t0: float | None = None, t1: float | None = None,
                   limit: int = 200) -> dict:
        """The events of a shown detector: start, end, duration."""
        return backend.call("get_events", {"detector": detector, "t0": t0, "t1": t1,
                                           "limit": limit})

    def get_spectrum_summary(signal: str, t0: float | None = None,
                             t1: float | None = None,
                             bands: list[list[float]] | None = None) -> dict:
        """The spectrum of a stretch in a few numbers, and the power and
        share of each band [[low_hz, high_hz], ...]."""
        return backend.call("get_spectrum_summary", {"signal": signal, "t0": t0,
                                                     "t1": t1, "bands": bands})

    def get_delay(x: str, y: str, t0: float | None = None, t1: float | None = None,
                  max_lag: float | None = None, method: str = "phat") -> dict:
        """How much later signal y hears what signal x does (s, positive when
        y is later), with the correlation peak and its snr."""
        return backend.call("get_delay", {"x": x, "y": y, "t0": t0, "t1": t1,
                                          "max_lag": max_lag, "method": method})

    for function in (get_current_context, list_signals, get_statistics, get_values,
                     get_events, get_spectrum_summary, get_delay):
        server.add_tool(function)
    return server


def _own_session(args) -> LocalBackend:
    from ..core.session import Session, stream_names

    topics = [t for t in (args.topic or []) if t]
    path = args.path or args.file
    buffer_s, max_samples = args.buffer, None
    if path:
        from ..analysis import file_buffer
        from ..sources.recording import FileSources

        sources = FileSources(path, topics=topics, audio_rate=args.audio_rate,
                              keep_channels=args.audio_channels, rate=args.rate,
                              scale=args.scale, units=args.units)
        buffer_s, max_samples = file_buffer(sources, buffer_s)
    else:
        from ..sources.synthetic import SyntheticAccelerometer

        sources = [SyntheticAccelerometer(rate=args.rate or 1000.0)]
    session = Session(buffer_s=buffer_s, max_samples=max_samples)
    names = stream_names(sources) if len(sources) > 1 else [None]
    for source, name in zip(sources, names):
        session.add(source, name)
    return LocalBackend(session.start())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mmp-sense-mcp", description="Expose an mmp-sense session over MCP (stdio)."
    )
    parser.add_argument("--endpoint", help="socket of a running mmp-sense session")
    parser.add_argument("--list", action="store_true",
                        help="list the sessions serving on this machine")
    parser.add_argument("--no-attach", action="store_true",
                        help="do not attach to a running session")
    parser.add_argument("path", nargs="?", metavar="FILE",
                        help="a data file to read instead, in any format mmp-sense "
                             "opens")
    parser.add_argument("--file", help="the same, as an option")
    parser.add_argument("--topic", action="append", default=[],
                        help="an MCAP topic; again for another (default: every "
                             "plottable topic)")
    parser.add_argument("--audio-rate", type=int, default=16000, metavar="HZ",
                        help="rate audio is resampled to (default 16000); 0 keeps "
                             "the file's own")
    parser.add_argument("--audio-channels", action="store_true",
                        help="keep an audio file's channels rather than mixing them")
    parser.add_argument("--rate", type=float, default=None, metavar="HZ",
                        help="sample rate of a file that carries no time; of the "
                             "simulated source otherwise (default 1000)")
    parser.add_argument("--scale", type=float, default=None,
                        help="units per count of a sensor's WAV")
    parser.add_argument("--units", default=None, help="the units of the values opened")
    parser.add_argument("--buffer", type=float, default=60.0,
                        help="seconds of history to keep")
    args = parser.parse_args(argv)
    if args.path and args.file:
        parser.error("name the file once: as FILE or as --file")

    if args.list:
        found = endpoints()
        for path in found:
            print(path)
        if not found:
            print("no mmp-sense session is serving", file=sys.stderr)
        return 0 if found else 1

    backend = None
    if args.endpoint:
        backend = ProxyBackend(args.endpoint)
    elif not args.no_attach and not (args.path or args.file):
        found = endpoints()
        if found:
            backend = ProxyBackend(found[0])
    if backend is None:
        try:
            backend = _own_session(args)
        except (OSError, ValueError, KeyError) as exc:
            print(f"mmp-sense-mcp: {exc.args[0] if exc.args else exc}", file=sys.stderr)
            return 1

    print(f"mmp-sense-mcp: serving {backend.description}", file=sys.stderr)
    server = build_server(backend)
    try:
        server.run(transport="stdio")
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
