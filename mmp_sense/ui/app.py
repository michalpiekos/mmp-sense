"""Starting the GUI.

    mmp-sense walk.mcap                     a recording: every plottable topic
    mmp-sense walk.mcap --topic /sensor/accelerometer --topic /sensor/gyroscope
    mmp-sense clip.m4a                      a voice clip, as 16 kHz audio
    mmp-sense accel.wav --scale 0.0024 --units m/s^2   a sensor's WAV
    mmp-sense table.tsv                     delimited text, a column per channel
    mmp-sense capture.npy --rate 400        an array, with its sample rate
    mmp-sense                               a simulated accelerometer
    mmp-sense FILE --replay                 a file, played back in time
    mmp-sense --serve                       answerable by an MCP server
    mmp-sense --listen                      running code sent by mmp-sense-send
"""

from __future__ import annotations

import argparse
import sys

__all__ = ["main", "build_session"]

#: The simulated accelerometer's rate unless --rate gives one.
SIMULATED_RATE = 1000.0


def build_session(args):
    """The session the window will show."""
    from ..core.session import Session, stream_names

    topics = [t for t in (args.topic or []) if t]
    buffer_s = args.buffer
    max_samples = None
    path = args.path or args.file
    if path:
        from ..sources.recording import FileSources

        sources = FileSources(path, replay=args.replay, speed=args.speed, topics=topics,
                              audio_rate=args.audio_rate,
                              keep_channels=args.audio_channels, rate=args.rate,
                              scale=args.scale, units=args.units)
        from ..analysis import file_buffer

        # All of a file, not the last --buffer seconds of it.
        buffer_s, max_samples = file_buffer(sources, buffer_s)
    else:
        from ..sources.synthetic import SyntheticAccelerometer

        sources = [SyntheticAccelerometer(rate=args.rate or SIMULATED_RATE)]

    session = Session(buffer_s=buffer_s, max_samples=max_samples)
    names = stream_names(sources) if len(sources) > 1 else [None]
    for source, name in zip(sources, names):
        session.add(source, name)
    session.start()
    session.poll()
    return session


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mmp-sense", description="Explore sensor signals: look at them, "
        "measure them, turn what you see into a detector."
    )
    parser.add_argument("path", nargs="?", metavar="FILE",
                        help="a data file to open: MCAP, WAV or other audio (m4a, mp3, "
                             "ogg... with ffmpeg), CSV/TSV, or NumPy .npy/.npz; "
                             "without one, simulated data")
    parser.add_argument("--file", help="the same, as an option")
    parser.add_argument("--topic", action="append", default=[],
                        help="an MCAP topic to open; again for another beside it, on "
                             "the same time axis (default: every plottable topic)")
    parser.add_argument("--audio-rate", type=int, default=16000, metavar="HZ",
                        help="rate audio files are resampled to (default 16000, "
                             "enough for speech); 0 keeps the file's own")
    parser.add_argument("--audio-channels", action="store_true",
                        help="keep an audio file's channels (ch0, ch1, ..) "
                             "rather than mixing them to one")
    parser.add_argument("--replay", action="store_true",
                        help="play a file back in time rather than loading it at once")
    parser.add_argument("--speed", type=float, default=1.0, help="replay speed")
    parser.add_argument("--rate", type=float, default=None, metavar="HZ",
                        help="sample rate of a file that carries no time (an array, "
                             "a table without a time column); of the simulated "
                             "source otherwise (default 1000)")
    parser.add_argument("--scale", type=float, default=None,
                        help="units per count of a sensor's WAV (srv-senselab's "
                             "encoding.scale); it opens in counts without it")
    parser.add_argument("--units", default=None,
                        help="the units of the values opened, e.g. m/s^2")
    parser.add_argument("--buffer", type=float, default=60.0,
                        help="seconds of history to keep (default 60)")
    parser.add_argument("--window", type=float, default=30.0,
                        help="seconds on screen (default 30)")
    parser.add_argument("--analysis-window", type=float, default=0.5,
                        help="seconds the FFT is taken over (default 0.5)")
    parser.add_argument("--peak-hold", type=float, default=2.0, metavar="SECONDS",
                        help="how long the spectrum's peak markers stay; 0 "
                             "hides them (default 2)")
    parser.add_argument("--listen", nargs="?", const=True, default=False,
                        metavar="SOCKET",
                        help="run code sent by mmp-sense-send (e.g. from an editor) "
                             "in the console; off by default")
    parser.add_argument("--serve", nargs="?", const="", default=None,
                        metavar="SOCKET",
                        help="answer the MCP tools on a local socket, so "
                             "mmp-sense-mcp can attach to this window")
    args = parser.parse_args(argv)
    if args.path and args.file:
        parser.error("name the file once: as FILE or as --file")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print(
            "mmp-sense: the GUI needs PySide6 and pyqtgraph; install them with\n"
            "    pip install 'mmp-sense[gui]'",
            file=sys.stderr,
        )
        return 1

    application = QApplication.instance() or QApplication(sys.argv[:1])
    # The app_id a Wayland compositor sees, and matches window rules on.
    # Without it the window arrives named after the interpreter, "python3".
    application.setApplicationName("mmp-sense")
    application.setDesktopFileName("mmp-sense")
    try:
        session = build_session(args)
    except Exception as exc:                                  # noqa: BLE001
        # A KeyError's message would print inside quotes.
        message = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        print(f"mmp-sense: {message}", file=sys.stderr)
        return 1
    session.analysis_window = args.analysis_window

    if args.serve is not None:
        path = session.serve(args.serve or None)
        print(f"mmp-sense: serving the MCP tools at {path}", file=sys.stderr)

    from .main_window import MainWindow

    window = MainWindow(
        session,
        window_s=args.window,
        listen=args.listen,
        peak_hold_s=args.peak_hold,
    )
    if window.listener is not None:
        print(f"mmp-sense: running code sent to {window.listener.path}", file=sys.stderr)
    elif args.listen:
        print("mmp-sense: --listen needs the console (qtconsole, ipykernel); not listening",
              file=sys.stderr)
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
