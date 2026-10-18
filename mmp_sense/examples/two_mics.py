"""Two microphones on two clocks, lined up by the claps both heard.

    python -m mmp_sense.examples.two_mics

Two phones record one room. The second one's clock is 37 ms late and
runs 80 ppm fast, so what both heard reads later on it, and later still
as the minute goes on. Claps are what the two have in common: their
delay, window by window, gives the offset and the drift, and correcting
the second phone's clock by them puts the two on one time line.
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy import signal as _signal

from ..core.data import TimeSeries

__all__ = ["synthesize", "OFFSET_S", "DRIFT"]

OFFSET_S = 0.037
DRIFT = 80e-6
#: The room's own sound is drawn at this many times 16 kHz, finer than
#: any microphone samples it.
FINE = 8


def _room(duration: float, rate: float, claps_every: float, rng) -> tuple[np.ndarray, float]:
    """The room's sound as microphones at `rate` hear it: quiet noise, and
    a clap -- a burst decaying in 30 ms -- every `claps_every` s."""
    fine = 16000.0 * FINE
    n = int((duration + 1.0) * fine)
    sound = 0.01 * rng.standard_normal(n)
    clap = int(0.12 * fine)
    envelope = np.exp(-np.arange(clap) / (0.03 * fine / 4))
    for start in np.arange(0.5, duration, claps_every):
        at = int((start + rng.uniform(-0.1, 0.1)) * fine)
        sound[at:at + clap] += 0.8 * envelope * rng.standard_normal(clap)
    # A microphone hears nothing above its Nyquist frequency: sampled
    # unfiltered, each would pick up an alias of its own.
    sos = _signal.butter(8, 0.4 * rate, fs=fine, output="sos")
    return _signal.sosfiltfilt(sos, sound), fine


def _heard(sound, fine, stamps, true_times, gain, noise, rng):
    values = gain * np.interp(true_times * fine, np.arange(len(sound)), sound)
    return values + noise * rng.standard_normal(len(stamps))


def synthesize(duration: float = 60.0, rate: float = 16000.0, *, rate_b: float | None = None,
               offset: float = OFFSET_S, drift: float = DRIFT, claps_every: float = 2.0,
               seed: int = 0) -> tuple[TimeSeries, TimeSeries]:
    """(a, b): one room heard by two microphones. `a`'s clock is right;
    `b`'s stamps a sound that happened at `t` as `t + offset + drift * t`,
    at `rate_b` samples a second (`rate` by default)."""
    rng = np.random.default_rng(seed)
    rate_b = rate if rate_b is None else rate_b
    sound, fine = _room(duration, min(rate, rate_b), claps_every, rng)
    stamps_a = np.arange(int(duration * rate)) / rate
    stamps_b = np.arange(int(duration * rate_b)) / rate_b
    true_b = (stamps_b - offset) / (1.0 + drift)
    keep = true_b >= 0
    stamps_b, true_b = stamps_b[keep], true_b[keep]
    a = TimeSeries(name="/phoneA/mic", time=stamps_a, channels=["amplitude"],
                   values=_heard(sound, fine, stamps_a, stamps_a, 1.0, 0.005, rng)[:, None],
                   sample_rate=rate)
    b = TimeSeries(name="/phoneB/mic", time=stamps_b, channels=["amplitude"],
                   values=_heard(sound, fine, stamps_b, true_b, 0.6, 0.008, rng)[:, None],
                   sample_rate=rate_b)
    return a, b


def main(argv=None) -> int:
    from .. import Session
    from ..sources.recording import SeriesSource

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--duration", type=float, default=60.0)
    args = parser.parse_args(argv)

    a, b = synthesize(args.duration)
    session = Session()
    session.add(SeriesSource(a))
    session.add(SeriesSource(b))
    while session.poll():
        pass
    late = session.delay("mic:amplitude", "phoneB_mic:amplitude", 40.0, 42.0, max_lag="100ms")
    print(f"before: phone B hears a clap at 41 s {late['delay'] * 1e3:.3f} ms late")
    estimate = session.synchronize("mic:amplitude", "phoneB_mic:amplitude", max_lag="100ms")
    print(f"estimated: {estimate!r}")
    planted = OFFSET_S + DRIFT * estimate.at
    print(f"planted:   offset {planted * 1e3:.4f} ms at {estimate.at:.6g} s, "
          f"drift {DRIFT * 1e6:+.2f} ppm")
    late = session.delay("mic:amplitude", "phoneB_mic:amplitude", 40.0, 42.0, max_lag="100ms")
    print(f"after:     {late['delay'] * 1e6:+.1f} us")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
