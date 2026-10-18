"""Hear a light plane through wind and shouting.

    python -m mmp_sense.examples.plane                     a synthetic field, scored
    python -m mmp_sense.examples.plane RECORDING.mcap      the detector on a recording
    python -m mmp_sense.examples.plane --gui [RECORDING]   the same, in the window

The recorded plane sounds as a comb of steady spectral lines -- harmonics
of about 19 Hz, a few hundred hertz up -- while wind is a broadband
rumble low down and a shout a stack of harmonics that glides and stops. The detector keeps, bin by bin, only what has sounded for the
last two seconds (`steady`), and asks how much of 150-1000 Hz is in lines
(`line_share`): wind is floor, a shout has come and gone, the engine
stays.

The synthetic field has wind, shouts and room noise throughout and a
plane only for a stretch in the middle, so the detector is scored on
when it says there is a plane and when there is none.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from scipy import signal as sig

import mmp_sense

__all__ = ["synthesize", "detector", "score", "main"]

RATE = 16000.0
#: The line spacing of the recorded plane, at the start. The comb's
#: strongest lines are these multiples of it.
ENGINE_HZ = 19.0
STRONG = (8, 12, 16, 24, 28, 32, 38)
BAND = (150.0, 1000.0)


def _plane(t, rng, start, end, level):
    """A comb of engine lines, drifting down 4 % over the recording,
    fading in over two seconds and out over two."""
    f0 = ENGINE_HZ * (1 - 0.04 * t / max(t[-1], 1e-9))
    phase = 2 * np.pi * np.cumsum(f0) / RATE
    out = np.zeros(len(t))
    for k in range(4, 60):
        if k * ENGINE_HZ > 2000:
            break
        size = (1.0 if k in STRONG else rng.uniform(0.05, 0.3)) / np.sqrt(k / 8)
        out += size * np.sin(k * phase + rng.uniform(0, 2 * np.pi))
    out *= 1 + 0.15 * np.sin(2 * np.pi * 0.23 * t + rng.uniform(0, 6))
    fade = np.clip((t - start) / 2.0, 0, 1) * np.clip((end - t) / 2.0, 0, 1)
    return level * out / np.std(out) * fade


def _wind(t, rng, level):
    """Air buffeting the microphone: a rumble below a few hundred hertz,
    in gusts."""
    w = np.cumsum(rng.standard_normal(len(t)))
    w = sig.sosfilt(sig.butter(2, 3.0, "hp", fs=RATE, output="sos"), w)
    w = sig.sosfilt(sig.butter(4, 300.0, "lp", fs=RATE, output="sos"), w)
    gust = 0.3 + np.clip(np.sin(2 * np.pi * 0.11 * t + rng.uniform(0, 6))
                         + 0.7 * np.sin(2 * np.pi * 0.29 * t + rng.uniform(0, 6)), 0, None)
    w *= gust
    return level * w / np.std(w)


def _shouts(t, rng, level, every=3.0):
    """Shouts: a voice's harmonics, gliding and wavering, for half a
    second to a second and a half, every few seconds."""
    out, onsets = np.zeros(len(t)), []
    at = rng.uniform(0.5, every)
    while at < t[-1] - 1.5:
        length = rng.uniform(0.4, 1.4)
        inside = (t >= at) & (t < at + length)
        tt = t[inside] - at
        pitch = rng.uniform(250, 500) * (1 + 0.15 * np.sin(np.pi * tt / length)) \
            + 8 * np.sin(2 * np.pi * 5.5 * tt)
        phase = 2 * np.pi * np.cumsum(pitch) / RATE
        voice = sum(np.sin(k * phase) / k for k in range(1, 15))
        out[inside] += voice * np.hanning(inside.sum())
        onsets.append(float(at))
        at += length + rng.uniform(0.5, every)
    return level * out / max(np.abs(out).max(), 1e-12), onsets


def synthesize(duration: float = 30.0, *, plane=(8.0, 22.0), wind=True, shouts=True,
               room=True, plane_level=0.002, seed: int = 1):
    """(table, truth): the field's sound, `audio` from -1 to 1 indexed by
    time, and when the plane was there (`plane`: (start, end) or None)
    and each shout began. `plane=None` leaves the plane out."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * RATE)) / RATE
    x = np.zeros(len(t))
    truth = {"plane": [] if plane is None else [tuple(plane)], "shouts": []}
    if plane is not None:
        x += _plane(t, rng, *plane, plane_level)
    if wind:
        x += _wind(t, rng, 0.05)
    if shouts:
        voice, truth["shouts"] = _shouts(t, rng, 0.3)
        x += voice
    if room:
        hiss = sig.lfilter([1.0], [1.0, -0.95], rng.standard_normal(len(t)))
        x += 0.004 * hiss / np.std(hiss)
    table = pd.DataFrame({"audio": x}, index=pd.Index(t, name="time"))
    return table, truth


def detector(band=BAND, *, share: float = 0.22, steady="2s", min_duration="3s",
             frames=("500ms", "100ms")) -> dict[str, mmp_sense.Node]:
    """The nodes of the detector, by name, from the microphone to the plane."""
    mic = mmp_sense.signal(name="mic")
    spectra = mic.frames(*frames, name="spectra")
    held = spectra.steady(steady, name="held")
    plane_share = held.line_share(*band, name="plane_share")
    return {
        "mic": mic,
        "band_power": spectra.band_power(*band, name="band_power"),
        "line_share": spectra.line_share(*band, name="line_share"),
        "plane_power": held.line_power(*band, name="plane_power"),
        "plane_share": plane_share,
        "plane": (plane_share > share).events(min_duration=min_duration, name="plane"),
    }


def score(session, node, truth, margin: float = 3.0) -> dict[str, float]:
    """How much of the time the detector was right: of the seconds with a
    plane, the share it said so (`recall`); of those without, the share
    it said there was one (`false`). `margin` seconds after each change
    are not judged: the detector needs them to hear it."""
    t0, t1 = session.span
    step = 0.1
    time = np.arange(t0 + margin, t1, step)
    said = np.zeros(len(time), bool)
    for _, row in session.events(node).iterrows():
        said |= (time >= row["start"]) & (time <= row["end"])
    there = np.zeros(len(time), bool)
    judged = np.ones(len(time), bool)
    for start, end in truth["plane"]:
        there |= (time >= start) & (time <= end)
        judged &= ~(((time >= start) & (time < start + margin))
                    | ((time >= end) & (time < end + margin)))
    plane, none = there & judged, ~there & judged
    return {"recall": float(said[plane].mean()) if plane.any() else float("nan"),
            "false": float(said[none].mean()) if none.any() else float("nan"),
            "events": int(len(session.events(node)))}


def report(session, nodes, truth=None) -> str:
    events = session.events(nodes["plane"])
    lines = ["plane heard:"]
    lines += [f"  {row.start:7.1f} s .. {row.end:7.1f} s" for row in events.itertuples()] \
        or ["  never"]
    if truth is not None:
        s = score(session, nodes["plane"], truth)
        lines += ["", f"plane there:   {truth['plane'] or 'never'}",
                  f"recall {s['recall']:.2f}   false alarms {s['false']:.2f} of the time"]
    share = session.values(nodes["plane_share"]).dropna()
    lines += ["", f"plane_share: min {share.min():.2f}  median {share.median():.2f}"
                  f"  max {share.max():.2f}",
              "", nodes["plane"].graph(),
              f"latency of plane: {nodes['plane'].latency:.1f} s"]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("recording", nargs="?", help="an MCAP, WAV or CSV to run it on")
    parser.add_argument("--topic", help="the stream of the recording, if it has several")
    parser.add_argument("--share", type=float, default=0.22, help="the plane_share threshold")
    parser.add_argument("--gui", action="store_true", help="open the window as well")
    args = parser.parse_args(argv)

    nodes = detector(share=args.share)
    if args.recording:
        session = mmp_sense.load(args.recording, topic=args.topic)
        print(report(session, nodes))
    else:
        for label, plane in (("wind, shouts and a plane from 8 s to 22 s", (8.0, 22.0)),
                             ("wind and shouts, no plane", None)):
            table, truth = synthesize(plane=plane)
            session = mmp_sense.from_frame(table)
            print(f"== {label}\n{report(session, nodes, truth)}\n")
    if not args.gui:
        return 0

    from PySide6.QtWidgets import QApplication

    from ..ui.main_window import MainWindow

    application = QApplication.instance() or QApplication(sys.argv[:1])
    window = MainWindow(session, window_s=36.0)
    session.show(nodes["band_power"], nodes["plane_share"], nodes["plane_power"], nodes["plane"])
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
