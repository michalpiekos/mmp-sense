"""A square of four microphones, and a sound to find.

    python -m mmp_sense.examples.array

A source of noise, far off or near, heard by four microphones 30 cm
apart, each with its own hiss. From the delays between them the
session finds the direction (far) or the point (near), maps the power
over directions, and a beam pointed at the source hears it better than
any one microphone.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import signal as _signal

from ..core.array import SPEED_OF_SOUND, Array

__all__ = ["square", "far", "near", "RATE"]

RATE = 16000.0
FINE = 8


def square(side: float = 0.3, prefix: str = "m") -> Array:
    """Four microphones on the corners of a square centred on the origin."""
    h = side / 2
    return Array({f"{prefix}0": (-h, -h), f"{prefix}1": (h, -h),
                  f"{prefix}2": (h, h), f"{prefix}3": (-h, h)})


def _source(seconds: float, rng) -> tuple[np.ndarray, float]:
    fine = RATE * FINE
    sound = rng.standard_normal(int((seconds + 1.0) * fine))
    sos = _signal.butter(8, (200.0, 0.4 * RATE), btype="bandpass", fs=fine, output="sos")
    return _signal.sosfiltfilt(sos, sound), fine


def _heard(array: Array, delays, gains, seconds, hiss, seed) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sound, fine = _source(seconds, rng)
    time = np.arange(int(seconds * RATE)) / RATE
    start = 0.5
    columns = {}
    for name, delay, gain in zip(array.names, delays, gains):
        at = (time - delay + start) * fine
        columns[name] = gain * np.interp(at, np.arange(len(sound)), sound) \
            + hiss * rng.standard_normal(len(time))
    return pd.DataFrame(columns, index=pd.Index(time, name="time"))


def far(array: Array, azimuth_deg: float, elevation_deg: float = 0.0, *,
        seconds: float = 2.0, hiss: float = 0.3, seed: int = 0) -> pd.DataFrame:
    """The array's signals for a far source in that direction."""
    from ..dsp import steering

    arrival = steering(array.points, array.speed, azimuth_deg, elevation_deg)
    return _heard(array, arrival, np.ones(len(array.names)), seconds, hiss, seed)


def near(array: Array, position, *, seconds: float = 2.0, hiss: float = 0.1,
         seed: int = 0) -> pd.DataFrame:
    """The array's signals for a source at `position` (metres)."""
    distance = np.linalg.norm(array.points - np.asarray(position, float), axis=1)
    return _heard(array, distance / array.speed, 1.0 / np.maximum(distance, 0.1),
                  seconds, hiss, seed)


def main(argv=None) -> int:
    import mmp_sense

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--azimuth", type=float, default=135.0)
    args = parser.parse_args(argv)

    ring = square()
    session = mmp_sense.from_frame(far(ring, args.azimuth))
    found = session.direction(ring)
    print(f"far source at {args.azimuth:g} deg: found {found['azimuth_deg']:.1f} deg "
          f"(residual {found['residual_s'] * 1e6:.1f} us)")
    power = session.direction_map(ring)
    print(f"the map peaks at {power.idxmax():g} deg")
    source = far(ring, args.azimuth, hiss=0.0)
    hiss = far(ring, args.azimuth) - source

    def snr(recipe):
        heard = mmp_sense.from_frame(source).values(recipe).iloc[2000:].var()
        return 10 * np.log10(heard / mmp_sense.from_frame(hiss).values(recipe).iloc[2000:].var())

    gain = snr(mmp_sense.beam(ring, args.azimuth)) - snr(mmp_sense.signal(ring.names[0]))
    print(f"a beam toward it hears it {gain:+.1f} dB clearer than one microphone")

    wide = Array({"m0": (0, 0), "m1": (2, 0), "m2": (2, 2), "m3": (0, 2)})
    session = mmp_sense.from_frame(near(wide, (1.2, 0.7)))
    point = session.locate(wide)
    print(f"near source at (1.2, 0.7) m: found ({point['x']:.3f}, {point['y']:.3f}) m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
