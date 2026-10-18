#!/usr/bin/env python3
"""How long one tick of the window takes.

A tick is what the application does four to twenty times a second: take
what the source produced, advance playback, recompute the signals the
panes draw, and redraw them. It has to stay far inside the refresh
interval or the window stutters under the user's hands.

    python scripts/bench-tick.py                # a real window
    python scripts/bench-tick.py --offscreen    # no display; faster than real
    python scripts/bench-tick.py --detector     # and the double-knock detector

Measure on a real window when the number matters: the offscreen platform
does not rasterise the same way, and reports a cost that is not the
user's.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate", type=float, default=1000.0)
    parser.add_argument("--buffer", type=float, default=60.0)
    parser.add_argument("--span", type=float, default=30.0,
                        help="seconds on screen")
    parser.add_argument("--ticks", type=int, default=200)
    parser.add_argument("--interval", type=float, default=0.05,
                        help="seconds of data per tick")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--detector", action="store_true",
                        help="show the double-knock detector: its energy and "
                             "frequency drawn, its knocks marked")
    args = parser.parse_args(argv)

    if args.offscreen:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"

    from PySide6.QtWidgets import QApplication

    from mmp_sense import Session, SyntheticAccelerometer
    from mmp_sense.ui.main_window import MainWindow

    application = QApplication.instance() or QApplication([])
    clock = Clock()
    source = SyntheticAccelerometer(rate=args.rate, clock=clock, knock_interval=2.0)
    session = Session(source, buffer_s=args.buffer, clock=clock).start()

    # Start full: a tick over a full buffer is the expensive one.
    clock.advance(args.buffer)
    session.poll()
    if args.detector:
        from mmp_sense.examples.double_knock import detector

        nodes = detector()
        session.show(nodes["energy"], nodes["freq"], nodes["knock"], nodes["double_knock"],
                     plot=False)

    window = MainWindow(session, window_s=args.span, console=False)
    if args.detector:
        window.add_pane("signal", "energy")
        window.add_pane("signal", "freq")
    window.show()
    window.timer.stop()
    application.processEvents()

    for _ in range(args.ticks):
        clock.advance(args.interval)
        window.tick()
        application.processEvents()

    times = np.array(window.tick_times) * 1000.0
    print(
        f"{len(session.buffer)} samples at {session.rate:g} Hz, "
        f"{args.span:g} s on screen, {len(window.panes)} panes"
        f"{', the detector' if args.detector else ''}, "
        f"{'offscreen' if args.offscreen else 'a real window'}"
    )
    print(
        f"tick: median {np.median(times):.1f} ms, "
        f"p95 {np.percentile(times, 95):.1f} ms, max {times.max():.1f} ms "
        f"({len(times)} ticks)"
    )
    window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
