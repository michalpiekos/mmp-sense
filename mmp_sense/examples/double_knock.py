"""Find knocks and double knocks in a synthetic accelerometer recording.

    python -m mmp_sense.examples.double_knock          the backtest, printed
    python -m mmp_sense.examples.double_knock --gui    the same, in the window

A minute of three-axis data, a sensor on a door: gravity on z, noise,
knocks along z (a short 200 Hz ring), and pairs of knocks a few hundred
milliseconds apart. Two disturbances are planted to fool a detector that
looks at one feature only: a heavy slow bump (plenty of energy, the wrong
frequency) and a faint high-pitched buzz (the right frequency, no
energy). Energy and frequency together reject both.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

import mmp_sense

__all__ = ["synthesize", "detector", "score", "main"]

RATE = 1000.0
KNOCK_HZ = 200.0


def _knock(t, onset, amplitude=3.0, tau=0.015):
    after = t - onset
    ring = amplitude * np.exp(-np.clip(after, 0, None) / tau) * np.sin(2 * np.pi * KNOCK_HZ * after)
    return np.where(after >= 0, ring, 0.0)


def synthesize(duration: float = 60.0, rate: float = RATE, seed: int = 1):
    """(table, truth): the recording, time as its index, and when each
    thing happened."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * rate)) / rate
    noise = 0.05
    x = noise * rng.standard_normal(len(t)) + 0.05 * np.sin(2 * np.pi * 0.3 * t)
    y = noise * rng.standard_normal(len(t))
    z = 9.81 + noise * rng.standard_normal(len(t))

    truth: dict[str, list[float]] = {"knock": [], "double_knock": [], "bump": [], "buzz": []}
    slots = np.arange(2.0, duration - 2.0, 2.5)
    kinds = ["knock", "double_knock", "bump", "buzz"]
    for i, start in enumerate(slots):
        onset = float(start + rng.uniform(0, 0.5))
        kind = kinds[i % len(kinds)]
        if kind == "knock":
            z += _knock(t, onset, amplitude=rng.uniform(2.5, 4.0))
            truth["knock"].append(onset)
        elif kind == "double_knock":
            second = onset + rng.uniform(0.18, 0.35)
            z += _knock(t, onset, amplitude=rng.uniform(2.5, 4.0))
            z += _knock(t, second, amplitude=rng.uniform(2.5, 4.0))
            truth["knock"] += [onset, second]
            truth["double_knock"].append(onset)
        elif kind == "bump":
            length = 0.15
            inside = (t >= onset) & (t < onset + length)
            z[inside] += 4.0 * np.sin(np.pi * (t[inside] - onset) / length)
            truth["bump"].append(onset)
        else:
            inside = (t >= onset) & (t < onset + 1.0)
            z[inside] += 0.15 * np.sin(2 * np.pi * 250.0 * t[inside])
            truth["buzz"].append(onset)

    table = pd.DataFrame({"accel_x": x, "accel_y": y, "accel_z": z},
                         index=pd.Index(t, name="time"))
    return table, truth


def detector(threshold: float = 5.0, band=(120.0, 350.0)) -> dict[str, mmp_sense.Node]:
    """The nodes of the detector, by name, from the source to the pattern."""
    accel = mmp_sense.signal("accel_x", "accel_y", "accel_z", name="accel")
    hp = accel.magnitude().highpass(cutoff_hz=10, name="hp")
    energy = hp.energy("80ms", name="energy")
    freq = hp.frames("80ms", "10ms").dominant_hz(name="freq")
    knock = ((energy > threshold) & freq.between(*band)).events(
        name="knock", refractory="100ms")
    double_knock = knock.repeated(2, gap=("80ms", "500ms"), name="double_knock")
    return {"accel": accel, "hp": hp, "energy": energy, "freq": freq,
            "knock": knock, "double_knock": double_knock,
            "loud": (energy > threshold).events(name="loud", refractory="100ms"),
            "pitched": freq.between(*band).events(name="pitched", refractory="100ms")}


def score(found, expected, tolerance: float = 0.06) -> dict[str, int]:
    """Hits, misses and false alarms, an event matching within `tolerance`."""
    found, expected = list(found), list(expected)
    hits = sum(any(abs(f - e) <= tolerance for f in found) for e in expected)
    false = sum(not any(abs(f - e) <= tolerance for e in expected) for f in found)
    return {"expected": len(expected), "found": len(found),
            "hits": hits, "missed": len(expected) - hits, "false": false}


def report(session, nodes, truth) -> str:
    lines = [f"{'detector':<14}{'expected':>9}{'found':>7}{'hits':>6}{'missed':>8}{'false':>7}"]
    wanted = {"knock": truth["knock"], "double_knock": truth["double_knock"],
              "loud": truth["knock"], "pitched": truth["knock"]}
    for name, expected in wanted.items():
        s = score(session.events(nodes[name])["start"], expected)
        lines.append(f"{name:<14}{s['expected']:>9}{s['found']:>7}{s['hits']:>6}"
                     f"{s['missed']:>8}{s['false']:>7}")
    lines += ["", nodes["double_knock"].graph(), "",
              f"latency of double_knock: {nodes['double_knock'].latency * 1000:.0f} ms"]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--gui", action="store_true", help="open the window as well")
    parser.add_argument("--threshold", type=float, default=5.0)
    args = parser.parse_args(argv)

    table, truth = synthesize()
    session = mmp_sense.from_frame(table, name="door")
    nodes = detector(threshold=args.threshold)
    print(report(session, nodes, truth))
    if not args.gui:
        return 0

    from PySide6.QtWidgets import QApplication

    from ..ui.main_window import MainWindow

    application = QApplication.instance() or QApplication(sys.argv[:1])
    window = MainWindow(session, window_s=15.0)
    for pane in window.panes[1:]:
        window.remove_pane(pane)
    session.show(nodes["hp"], nodes["energy"], nodes["freq"],
                 nodes["knock"], nodes["double_knock"])
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
