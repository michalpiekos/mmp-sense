# The mmp-sense API

Six ideas, and a call or two for each:

| idea | what it is | where |
|---|---|---|
| **session** | the samples held, and where you are in them | `mmp_sense.load(...)`, the viewer's `session` |
| **stream** | one source's samples, on its own clock | `session.streams` |
| **channel** | a named column a stream produces | `session.channels` |
| **recipe** | a calculation, written down, run later | `mmp_sense.signal(...)` and its methods |
| **stretch** | the time a question is about, `t0` to `t1` seconds | an argument of every question |
| **result** | the answer, as pandas | `session.values(...)`, `session.events(...)`, ... |

```python
import mmp_sense

session = mmp_sense.load("knocks.mcap", topic="/phone/accel")      # a recording
strength = mmp_sense.signal("ax", "ay", "az").highpass(2.0).magnitude().peak("60ms")
session.values(strength, 4.0, 6.0)          # a pandas Series, indexed by time
knock = (strength > 0.7).events(refractory="145ms", name="knock")
session.events(knock)                       # a DataFrame: start, end, duration
```

In the viewer's console `session` is the window's own, `signal` is
`mmp_sense.signal`, `show` draws, and `ui` is the window:
`ui.show_distribution(on=True)` is its **PDF** button.

## Sessions

| call | gives |
|---|---|
| `mmp_sense.load(path, topic=None, *, topics=None, sample_rate=None, scale=None, units=None)` | a session holding all of a data file, in any format the window opens ([the formats](formats.md)): MCAP (every plottable topic, a stream each, on one clock; `topic` or `topics` picks them), sound (at its own rate; channel `audio`, or `ch0`, `ch1`, ... for several), a sensor's WAV (in counts, or times `scale` in `units`), delimited text or a NumPy array (`sample_rate`: for a file that carries no time) |
| `mmp_sense.from_frame(table)` | a session holding a pandas DataFrame: its index is time in seconds, its columns the channels |
| `mmp_sense.Session(source)` | a session on any source: `SyntheticAccelerometer`, `FileSources(path, replay=True)[0]`, one of your own |
| `mmp_sense.io.read_file(path, **options)` | the series a file holds, as `TimeSeries`, without a session; `mmp_sense.io.kind_of(path)` says which format it is |

What a session knows: `session.channels`, `session.streams`,
`session.rate` (the first stream's samples per second), `session.span`
(the first and last time held, in any stream), `session.cursor`,
`session.selection` (`(t0, t1)` or `None`; `set_selection`,
`clear_selection`), `session.frequency_cursor` (Hz, or `None` when not
placed; independent of the time cursor, shared by the frequency charts),
`session.running`, `session.context` (all of it at once).
`session.samples(signal, t0=None, t1=None)` gives one signal as NumPy:
see [Scripts](#scripts).

A live session takes samples when polled (`session.poll()`; the window
polls on a timer). `session.stop_intake()` stops taking them and keeps
everything held, to be explored; `session.start_intake()` starts again
from an empty buffer. `session.listen(callback)` hears every change --
`cursor`, `frequency`, `selection`, `data`, `signals`, `detections`, `bands`, `intake`, ... --
and returns a function that stops it.

## Streams

A session holds one stream per source -- two phones' microphones, an
accelerometer beside a gyroscope -- each at its own rate, under one
cursor, one selection and one set of recipes:

```python
a, b = mmp_sense.FileSources("claps.mcap", topics=["/phoneA/mic", "/phoneB/mic"])
session = mmp_sense.Session(a)       # stream "mic"
session.add(b)                       # "phoneB_mic"
session.streams                     # {"mic": Stream, "phoneB_mic": Stream}
session.remove("phoneB_mic")
```

A stream is named after its source's topic -- the last segment, with
the ones before it when that alone is taken -- or by `add(source,
name)`. With several streams a channel's name is `stream:channel`
(`mic:amplitude`), and each stream is a group of its channels. A
bare channel name still works while one stream has it, and is an error
naming the candidates when several do. With one stream, channels go by
their bare names, and `stream:channel` works too.

All streams share the session's clock: the recording's in a file, the
source's own live. Nothing corrects one device's clock against
another's.

`session.window(t0, t1, stream=None)` is one stream's window (the first
by default); `window.of("other")` is another stream's, taken with it,
and `window.home(name)` the window of the stream `name` is of. Asked of
a window, a signal of another stream is given at that window's times,
each value the latest at or before.

Samples are held as the source sends them -- 16-bit audio as 16-bit --
and computed as floats.

## Two signals, two clocks

How two signals relate in time -- two microphones, or a microphone and an
accelerometer. `y` is read at `x`'s times, by straight lines between its
samples, so the two may be of different streams and rates; the stretch
rule is the questions'.

| call | returns |
|---|---|
| `session.correlation(x, y, t0, t1, max_lag=None, method="phat")` | `pd.Series` indexed by lag (s): how alike `y` is to `x` that much later. `"phat"` (GCC-PHAT) peaks sharply whatever the sound and through echoes; `"plain"` is the normalised correlation, -1..1, better for slow sensors |
| `session.delay(x, y, t0, t1, max_lag=None, method="phat")` | `pd.Series`: `delay` (s, between samples; positive when `y` hears later), `peak`, `snr` (the peak over the rest). NaN when nothing stands out, as in silence |
| `session.coherence(x, y, t0, t1, segment=256)` | `pd.Series` indexed by frequency: how much of `y` follows from `x` there, 0..1 |

Two devices do not agree on the time. With a sound both heard -- a few
claps over a minute -- the second's clock can be put right:

```python
estimate = session.synchronize("phoneA_mic:amplitude", "phoneB_mic:amplitude",
                               t0, t1, window="1s", max_lag="100ms")
estimate.offset, estimate.drift_ppm     # how late B's clock was, and how fast it ran
session.clock_of("phoneB_mic")          # the correction now applied
session.set_clock("phoneB_mic")         # back to the times as B sent them
```

`estimate_sync` measures the delay over half-overlapping windows of the
stretch and fits `delay = offset + drift * (t - at)` through them,
leaving out weak windows (`min_snr`) and outliers; `synchronize` also
corrects `y`'s stream by it -- every timestamp held and every one to
come -- unless fewer than 3 windows agree or the fit is off by more than
a sample of the slower stream, when it raises and changes nothing.
Synchronizing again refines the correction. `set_clock(stream, offset,
drift_ppm, at)` sets one by hand: a sample stamped `t` happened at `t -
offset - drift * (t - at)`. `python -m mmp_sense.examples.two_mics` shows it
end to end.

## Recorded clocks

A session opened from a raw export of srv-senselab (`mmp_sense.load("DIR/<session-id>")`,
see [the formats](formats.md)) carries the device's clocks as it recorded them:

```python
session.clocks()               # id, name, kind, hz, bits, timeline, source, domain
session.clock_observations()   # a row per reading: time (session s), kind, clock,
                               # value (that clock's ticks), uncertainty_ns, pps,
                               # gateway_utc_ns, gateway_send_utc_ns, gateway_recv_utc_ns
```

`kind` is `device` (readings the device took at one instant: UTC against its
monotonic clock, PPS edges), `pong` (the device's reading when a gateway ping
arrived, between the gateway's send and receive UTC) or `arrival` (the
gateway's UTC when a data message arrived, against its timestamp). Values are
raw 64-bit integers, kept exactly; nothing is fitted or corrected. Other files
have no recorded clocks: both are empty.

## Arrays: where a sound came from

```python
ring = mmp_sense.Array({"mic:m0": (-0.15, -0.15), "mic:m1": (0.15, -0.15),
                  "mic:m2": (0.15, 0.15), "mic:m3": (-0.15, 0.15)})   # metres
session.direction(ring, t0, t1)      # azimuth_deg, elevation_deg, residual_s
session.locate(ring, t0, t1)         # x, y (z) of a near source
session.direction_map(ring, t0, t1)  # steered power by azimuth
session.add_array("ring", ring)      # for the direction pane
```

An array places microphones -- signals by name, of one stream or of
synchronized ones -- in two or three dimensions; `speed` is the speed
of sound, 343 m/s by default. Azimuth is degrees counter-clockwise from
+x, elevation from the x-y plane; microphones all in one plane tell how
far off it a source is, not to which side. Each answer comes from the
GCC-PHAT delay between every pair over the stretch: `direction` fits a
far source's direction to them, `locate` a near source's point (it
needs a microphone more than the dimensions), and `direction_map` sums
the pairs' correlation at the delays each azimuth would give (SRP-PHAT),
1 where all agree. `python -m mmp_sense.examples.array` shows all three, and
a beam.

## Asking: the seven questions

Each takes `x` -- a channel's name, a shown recipe's name, or a recipe
-- and a stretch. **The stretch rule is the same for all seven:** `t0`
and `t1` if given; with neither, the selection if there is one, else
everything held; with one, from it to the end (or the start to it).

| call | returns |
|---|---|
| `session.values(x, t0, t1)` | `pd.Series` of values indexed by time (s); a `pd.DataFrame` with a column per component for several (`signal("ax", "ay", "az")`) and a column per frequency for frames |
| `session.events(x, t0, t1)` | `pd.DataFrame` of the events starting in the stretch: `start`, `end`, `duration` (s), and what the recipe adds -- `times` for `repeated`, `first`, `second`, `gap` for `followed_by` |
| `session.stats(x, t0, t1)` | `pd.Series`: `n, min, max, mean, rms, std, peak_to_peak`; a DataFrame, a column per component, for several |
| `session.distribution(x, t0, t1, bins=64, low=None, high=None)` | `pd.Series` of density indexed by value (each bin's centre): a histogram whose area is 1, of the values between `low` and `high` -- the values' own range if not given |
| `session.spectrum(x, t0, t1, taper="hann", pad_to=None, align=None, ref_hz=None)` | `pd.Series` of amplitude indexed by frequency (Hz): how strong each frequency is in the stretch. `align`, a rate's signal or recipe, divides out its drift first (see *Combs*) |
| `session.psd(x, t0, t1, segment=1024, overlap=0.5, taper="hann", align=None, ref_hz=None)` | `pd.Series` of power spectral density indexed by frequency (Hz), in units squared per Hz: Welch's average over `segment`-sample pieces of the stretch, gaps left out. Noise reads the same whatever `segment` is; a tone's power is the sum over its bins times their width |
| `session.lines(x, t0, t1, low_hz=0, high_hz=None, prominence=8, near=None, within=1.5)` | `pd.DataFrame` of lines, a row each: `hz` (between bins), `prominence_db`, `over_floor_db`, `level_db`. Listed from Welch's PSD at `resolution_hz=1.5`; or, with `near`, the line nearest each of those frequencies, measured from the whole stretch (see *Combs*) |
| `session.repetition_hz(x, t0, t1, min_hz=, max_hz=, low_hz=None, high_hz=None)` | `float`: how often the stretch repeats, from the autocorrelation of its lines, checked against half and double; NaN if no comb stands |
| `session.orders(x, rate, t0, t1, taper="hann", pad_to=None)` | `pd.Series` of amplitude indexed by `order`, multiples of `rate`: order tracking |
| `session.cycle_average(x, rate, t0, t1, lock=None, per_cycle=720, low_hz=None, high_hz=None, lock_hz=4)` | a `CycleAverage`: the stretch's cycles at `rate` averaged, the time-synchronous average (see *Combs*) |
| `session.spectrum_summary(x, t0, t1, bands=(), taper="hann")` | `pd.Series`: `samples, resolution_hz, dominant_hz, centroid_hz, spread_hz, flatness, rolloff_hz`, and `power lo-hi Hz`, `share lo-hi Hz` for each `(lo, hi)` in `bands` |

A recipe is computed over **everything held** and then cut to the
stretch, so a filter or a trailing measure starts the stretch with its
history, not from nothing.

A recipe's filters have a frequency response: `recipe.response(rate)`
is a DataFrame by frequency of `gain_db`, `phase_deg` and
`group_delay_s` of every filter from its signal along its first inputs.

`spectrum` removes the stretch's mean first (a constant, such as
gravity, would be a large 0 Hz bin). `taper` fades the stretch's ends
so the cut does not smear power into other frequencies: `"hann"`, or
`"boxcar"` for none. `pad_to` appends zeros to that many samples, which
draws the bins closer together without adding any information.

## Recipes

A recipe starts with `mmp_sense.signal(*channels)` -- every channel if none
are named -- and each method makes a new recipe; nothing is computed
until a question or `show` runs it. Every method takes `name=`, the
name it is shown and asked by. Durations are seconds or text: `0.06`,
`"60ms"`, `"1.5s"`. Frequencies are Hz.

| group | method | gives, per instant |
|---|---|---|
| **filters** | `.highpass(cutoff_hz, order=1)` | what changes faster than `cutoff_hz`: order 1 a first-order IIR, more a Butterworth that steep (6 dB an octave per order) |
| | `.lowpass(cutoff_hz, order=1)` | what changes slower than `cutoff_hz`, likewise |
| | `.bandpass(low_hz, high_hz, order=4)`, `.bandstop(low_hz, high_hz, order=4)` | what lies between two frequencies, or everything but it (Butterworth) |
| | `.notch(freq_hz, q=30)` | everything but one frequency, `freq_hz / q` wide -- mains hum |
| | `.envelope(min_hz=20)` | how loud, instant by instant: the size of the analytic signal (a causal Hilbert transformer), lagging about `2 / min_hz`; high-pass a constant offset away first |
| | `.abs()` | each value's size: the signal rectified |
| | `.fill(longest=None)` | each gap (NaN) filled with the last value before it, for at most `longest` -- a rate measured in most frames then holds through the rest |
| **combine** | `.magnitude()` | the size of a vector: `sqrt(x^2 + y^2 + ...)` |
| | `recipe["az"]` | one component |
| | `+ - * /`, `> >= < <=`, `.between(lo, hi)`, `& \| ~` | arithmetic; true or false; and, or, not |
| **trailing measures** | `.peak(length)` | the largest absolute value in the last `length` |
| | `.energy(length)` | the sum of squares over the last `length` |
| | `.rms(length)` | the root mean square over the last `length` |
| | `.zero_crossing_rate(length)` | sign changes per second over the last `length` |
| **frequency** | `.frames(length, hop=None)` | a spectrum of the last `length`, every `hop` (a quarter of `length` by default), stamped at its last sample |
| | `.band_power(lo, hi)` | of frames: the power between two frequencies (units squared) |
| | `.band_share(lo, hi)` | of frames: that power's share of all power, 0..1 |
| | `.dominant_hz(min_hz=0)` | of frames: the strongest bin's frequency |
| | `.centroid_hz()`, `.spread_hz()` | of frames: the power's centre of gravity, and its spread about it |
| | `.flatness()` | of frames: 1 for power spread evenly (noise), near 0 in a few bins (a tone) |
| | `.rolloff_hz(fraction=0.85)` | of frames: the frequency below which `fraction` of the power lies |
| | `.flux(normalize=False)` | of frames: how much the spectrum grew since the last frame (onsets); `normalize`: how much its shape changed |
| | `.line_power(lo, hi, above=10, floor_hz=60)` | of frames: the power in lines between two frequencies -- bins standing more than `above` times (10 dB) over the median of the `floor_hz` around them -- counted above that floor (units squared). An engine's harmonics are lines; wind and hiss are floor however loud |
| | `.line_share(lo, hi, above=10, floor_hz=60)` | of frames: the share of the band that is lines, 0..1, each bin first measured against its own floor so a sloping floor counts alike throughout; noise near 0, tones near 1 |
| | `.steady(length)` | of frames: frames again, each bin its smallest over the last `length` -- what sounded throughout; a shout or a gust that came and went is gone |
| | `.repetition_hz(min_hz, max_hz, low_hz=None, high_hz=None)` | of frames: how often the sound repeats, from the autocorrelation of the lines between `low_hz` (`min_hz`) and `high_hz` (40 × `max_hz`), checked against half and double; NaN where no comb stands |
| | `.peak_hz(lo, hi, near=None, within=None)` | of frames: the precise frequency of the strongest bin between two frequencies -- or within `within` Hz (two bins) of `near`, a number or a recipe such as `rate * 28` |
| | `.align(rate, ref_hz)` | of frames: frames again, each spectrum's frequencies scaled by `ref_hz / rate` at its time, so lines that move with `rate` stand still |
| **arrays** | `mmp_sense.beam(array, azimuth_deg, elevation_deg=0)` | what an array hears from one direction: delay and sum of its microphones (channels of one stream), lagging its widest delay and 0.5 ms more; 8 kHz or more |
| **two signals** | `.delay(other, length, hop=None, max_lag=None, method="phat")` | how much later `other` hears what this does over the last `length`, every `hop`, in seconds -- positive when `other` is later; NaN in silence. Another stream is read at this one's times, by straight lines between its samples |
| **events** | `.events(refractory=0, min_duration=0)` | of a condition: an event per stretch where it is true; one starting within `refractory` of the last, or shorter than `min_duration`, is dropped |
| | `.followed_by(other, gap=(min, max))` | of events: one, then one of `other` from `min` to `max` later (`max` None: any time); each event used once |
| | `.repeated(n, gap=(min, max))` | of events: `n` in a row, each `min` to `max` after the last; runs may share events |
| | `.lasting(min=None, max=None)` | of events: only those lasting so long |

An operator such as `>` or `*` takes no `name=`; name its result with
`.named("name")`: `(strength > 0.7).named("loud")`.

A frequency measure takes frames, never a plain signal, so the frame
length and hop are written once and shared:

```python
frames = mmp_sense.signal("az").highpass(2.0).frames("200ms", "25ms")
ringing = frames.band_power(40, 150, name="ringing")
share = frames.band_share(40, 150, name="share")
```

`steady` takes frames and gives frames, so any measure follows it: the
lines that have sounded for two seconds, and how much of a band they are,
is `frames.steady("2s").line_share(150, 1000)` -- an engine through wind
and shouting (`python -m mmp_sense.examples.plane`).

### Time

Nothing looks ahead: a value at `t` is of samples up to `t`, so what a
recipe says of a recording is what it would have said live. A trailing
measure not yet full reads NaN; so does a frame not yet complete. Two
inputs at different rates -- `strength` per sample, `ringing` per frame
-- meet on the left one's times, each instant taking the latest value
of the right at or before it; a comparison with NaN is false.

Live, a recipe is not computed again on every tick: each step carries
on from where it was with the samples that arrived, and drops what has
scrolled out of the buffer. So its values are those of one computation
over the whole stream since it was first shown, not since the oldest
sample still held: a filter does not start again from rest at the left
edge, a trailing measure is not NaN there, and an event is not cut off
there. A recording, or a window of `session.window(t0, t1)`, is
computed whole. A signal made with a function of the window
(`session.signals.define`), an op registered without a rule, and
whatever is computed from them, are computed over the whole window
every tick.
A recipe may read several streams: `mmp_sense.signal("left:x") -
mmp_sense.signal("right:x")`. It is computed on the times of its first
input's stream -- the left one's -- each other stream's value the
latest at or before, and only up to the newest sample of the stream
that is furthest behind, so that what it says is final: a stream that
stops holds the recipe where it stopped. A `signal(...)` reads one
stream. A stream lagging another by more than the other's buffer holds
loses what it was to be matched with, and the recipe starts again.

A source that declares no sample rate has its rate measured afresh on
every tick, so a trailing length or a filter's cutoff may differ very
slightly between the older and the newer values.

Every recipe says what that costs: `recipe.latency` is the seconds from
an instant to when the recipe can say anything about it (a trailing
length, a frame and its hop, a pattern's longest gap);
`recipe.timing` has the rest. `recipe.graph()` prints the recipe as a
tree; `recipe.with_params("knock", refractory="200ms")` makes a copy
with a named step changed; `recipe.to_dict()` and `Node.from_dict()`
turn it into plain data and back.

## Combs: a rate, its lines, and its drift

A source that repeats `r` times a second -- an engine, a propeller, a
voice -- makes lines at multiples of `r`, and they all move when `r`
does. Four steps: find the rate, follow it, divide its drift out, and
ask which rates explain the lines better than chance.

```python
from mmp_sense.analysis import comb_scan, fit_rates, heights, permutation_lift

frames = signal("amplitude").frames("1s", "250ms")
rate = frames.repetition_hz(12.5, 33, low_hz=150, high_hz=1000, name="rate")
line = frames.peak_hz(100, 1000, near=rate * 28, within=3)     # one line, while it glides

session.repetition_hz("amplitude", 2.2, 4.5, min_hz=12.5, max_hz=33, low_hz=150, high_hz=1000)
lines = session.lines("amplitude", 1, 12, low_hz=100, high_hz=1000, prominence=10,
                      align=rate, ref_hz=18.8)                  # where they are at 18.8 Hz
comb_scan(lines, np.arange(25, 120, 0.01), top=4)               # spacings against chance
fit = fit_rates(lines, 37.7, 93.2, band=(100, 1000))            # n * X + k * B
fit.x, fit.b, fit.fitted, fit.chance
```

**The rate.** `repetition_hz` is the peak of the autocorrelation of the
lines -- each bin's power over the median of the 60 Hz around it, so
wind's broad rumble barely counts -- between the periods `1 / max_hz`
and `1 / min_hz`. A comb is explained by half its rate too, and
sometimes by double, so the answer is checked on the spectrum: halved
if lines stand 3 dB over the bins between them at the odd multiples of
half of it (1.5 dB), doubled if they stand at its even multiples (3 dB)
and not its odd ones -- never out of `min_hz`..`max_hz`, so a range set
round one comb keeps to it. Of the autocorrelation's peaks within 90 %
of the highest, the shortest lag is the period: a period's multiples
peak nearly as high. A rate whose multiples stand no more than 1.5 dB
over the bins between them, on average, is NaN: no comb.

**A line.** `peak_hz` refines the strongest bin by a parabola through it
and its neighbours in dB, a small fraction of a bin. `near` may be a
recipe: `rate * 28` follows the 28th multiple wherever the rate goes.
`session.lines(..., near=[...])` measures given lines in one stretch
the same way, from its whole spectrum.

**Drift.** Every line of one source moves by the same factor. Dividing it
out lines them up again: `frames.align(rate, ref_hz)` for a recipe --
`frames.align(rate, 18.8).steady("4s")` keeps a comb that glides, which
`steady` alone would lose -- and `align=rate` on `spectrum`, `psd`,
`lines` and `repetition_hz` for a stretch, which resample the signal so
the rate reads `ref_hz` throughout (its mean over the stretch if not
given). A rate made of frames says what each frame's middle was, so a
stretch reads it there, half a frame before its stamp.

Alignment is only as good as the rate: a rate 0.5 % off moves a line at
1000 Hz by 5 Hz, and `steady` loses it as surely as a glide. Refine the
rate on a strong line before aligning by it, and hold it through the
frames it is missing:

```python
rough = frames.repetition_hz(90, 145, low_hz=150, high_hz=1500).fill("3s")
rate = (frames.peak_hz(300, 450, near=rough * 3, within=8) / 3).fill("3s")
share = frames.align(rate, 120).steady("2s").line_share(150, 1000)
``` `session.orders(x, rate)` is the same spectrum indexed by
multiples of the rate. A recipe's `align` needs `ref_hz`: the mean
would look ahead.

**Cycles.** `session.cycle_average` cuts a stretch into cycles of a
`rate` (a number, a signal or a recipe, in Hz), resamples each to
`per_cycle` points and averages them. What repeats once a cycle stays;
what does not shrinks as the square root of the number of cycles. Use it
to see what happens within one cycle, such as whether an engine's six
firings are alike:

```python
sixth = signal("amplitude").frames("500ms", "50ms").peak_hz(92, 145)
cycle = session.cycle_average("amplitude", sixth / 6, 15, 20, lock=6, low_hz=15, high_hz=1500)
cycle.mean, cycle.noise           # the averaged cycle, and the noise left in it
cycle.parts(6)                    # each of six parts: start, rms, lag
cycle.uneven(6)                   # power off the multiples of 6, over that on them
cycle.orders(12)                  # amplitude, noise and ratio of orders 1-12
```

`lock=6` follows the phase of the line at 6 times the rate, so a rate a
little off does not move the cuts over many cycles; lock on the
strongest line, and keep `lock_hz` under half the rate. If that line is
lost, its phase jumps back and the call says so. The signal is first
band-passed to `low_hz`..`high_hz`. Its top is held below half of
`per_cycle` times the slowest rate, so resampling folds nothing.

The `CycleAverage` has:
- `mean` and `noise`, both indexed by position in the cycle, 0 to 1.
  `noise` is half the difference between the average of the even cycles
  and the average of the odd ones: whatever repeats cancels there.
- `cycles`, how many were averaged, and `rate_hz`, the median rate.
- `orders(top=36)`: amplitude, noise and ratio of each order.
- `parts(n)`: the cycle in `n` equal parts, cut where the parts' average
  shape is quietest. A row each: `start` (position in the cycle), `rms`
  (over the parts' mean) and `lag` (in parts). Alike parts read 1 and 0.
- `uneven(n)`: the power of the orders up to `6 * n` that are not
  multiples of `n`, over the power of those that are, with `noise`, the
  same figure for the noise alone.

Position 0 in the cycle is wherever the average began. Which part is
which cylinder needs a reference from outside, such as a tachometer.

**Against chance** (`mmp_sense.analysis`, on `session.lines`' table or a list
of Hz):

| call | returns |
|---|---|
| `comb_scan(lines, spacings, within=0.5, apart=0.5, top=None)` | a DataFrame by spacing: `lines` within `within` Hz of a multiple, `chance` -- how many random frequencies would -- and `p` (binomial), best first; spacings within `apart` of a better one left out |
| `fit_rates(lines, x, b, within=0.5, most=4, nk=None, band=None, tries=500)` | a `RateFit`: `x` and `b` fitted to the lines at `n * x + k * b` (`abs(k) <= most`), `table` (`hz, n, k, off, fits`), `fitted`, and `chance` / `chance_max` over `tries` draws of random frequencies in `band`. `nk` keeps an earlier fit's placing: the same lines in another stretch |
| `heights(psd, hz, within=1.0, floor_hz=60)` | `pd.Series` by `hz`: how far the line nearest each stands over its floor, in dB |
| `permutation_lift(values, chosen, tries=5000)` | `lift` (how much higher the chosen values stand, on average) and `p` (how often shuffling does as well) |

## Showing

```python
session.show(strength, knock)        # a chart per value recipe; marks for events
session.hide("knock")
session.set_drawn("knock", False)    # marks off, still counted and askable
session.group("motion", ["strength", "az"])     # several on one chart; None removes
session.chart("motion", view=None)    # "spectrum", "spectrogram", "psd" for those panes
```

`show` makes a value recipe a signal of its name -- plotted, listed,
askable by name -- and one with components a group of `name.component`
signals. Showing again under a name replaces it. Once shown, the recipe
and each named step inside it can be asked for by name:
`session.events("knock")`.

## Scripts

A script is a Python file with `main(session)`. The session is an
argument, so one script runs on the window's session or on a loaded one:

```python
from mmp_sense import Session

def main(session: Session) -> None:
    s = session.samples("accel_x", 4.0, 6.0)
    session.plot("doubled", s.time, s.values * 2, x_label="Time", x_units="s",
                 y_label="Twice", y_units=s.units)
```

| call | gives |
|---|---|
| `mmp_sense.run_script(path, session)` | runs the file's `main(session)` and returns what it returns: read afresh each time, with modules beside it importable |
| `run(path)` | the same in the window's console, on its session; `mmp-sense-send --script PATH` sends it |
| `session.samples(x, t0=None, t1=None)` | `Samples` of one channel, shown name or recipe over the stretch (the selection, else everything held): `.values` and `.time` as float arrays, `.rate` in samples per second, `.units`, `.name` |
| `session.plot(name, x, y, *, x_label="", x_units="", y_label="", y_units="")` | draws `y` against `x` in an `xy` pane, 1-D and of equal length; returns the `Plot` |
| `session.plots` | every `Plot` by name: `.x`, `.y` (read-only copies) and the labels |
| `session.remove_plot(name)` | forgets a plot and closes its pane |

A plot is not a signal. Plotting under a name replaces what it held and
updates the same pane, so a rerun adds nothing to `session.signals`,
`session.streams` or `session.groups`; only a new name adds a plot. If the
pane was closed, the next plot of that name opens it. Units are shown as
given, without SI prefixes. With no window, a plot is only kept.

`mmp_sense` ships type information. `Session`, `Samples` and `Plot` are the
types to annotate with, and `SignalRef` is a channel, shown name or recipe.

## Adding an op

An op is a function of its inputs, registered by name; `recipe.apply`
uses it:

```python
@mmp_sense.register_op("clip", description="Limit to +-limit", menu={"limit": 1.0})
def clip(ctx, series, limit=1.0):
    return series.with_values(np.clip(series.values, -limit, limit))

clipped = strength.apply("clip", limit=2.0, name="clipped")
```

`series` has `time`, `values` and `rate`; return
`series.with_values(...)`. `register_op` also says what the op is:
`output="events"`, `windowed=True`, `length=` and `delay=` (seconds, or
the name of the parameter holding them), `on_frames=True` for a measure
of frames. `menu=` offers it in the viewer's **Add feature** box with
those starting values.

Live, an op without `incremental=` is computed over the whole buffer on
every tick. An op whose value at an instant depends only on the last
few values of its first input can say so, and it is then computed for
the new samples alone:

```python
@mmp_sense.register_op("clip", incremental=mmp_sense.pointwise)      # this instant only
@mmp_sense.register_op("smooth", incremental=mmp_sense.Lookback(lambda ctx, series, p: p["n"]))
```

`mmp_sense.Lookback(n)` takes the number of earlier values the op reads, or a
function of `(ctx, series, params)` giving it. Its result must be on
its first input's times; any other input is aligned onto them. Do not
declare it for an op that looks at the whole window, such as one
subtracting the mean.

## Playback

```python
session.play(start=8.0, speed=0.5, until=8.5)
session.tick()                       # advance to now; the window does this
session.pause()
```

Replay moves the cursor through history at `speed` times real time. The
samples are not moved or copied, and new ones keep arriving while it
plays. With no arguments it plays the selection, or from the cursor.
This is the cursor alone; the window's **Play** button also makes the
sound.

```python
sound = session.audio("mic:amplitude", t0, t1, rate=48000, speed=1.0)
sound = session.audio("mic:amplitude", amp="auto")     # a weak recording, raised
```

`session.audio(x, t0, t1, rate=48000, speed=1.0, bands=None, amp=None)` is a stretch of one
signal -- a channel or a recipe, by the stretch rule -- as sound:
float32 samples within -1..1 at `rate`, `speed` times as fast and as
high. 16-bit samples keep their level, values already within -1..1 are
kept, larger ones lose their mean and peak at 0.9, and a gap is silence.
`amp` takes a number of dB from 0 to 60, or `"auto"`, which brings the
louder moments to -20 dBFS: the mean goes, the sound is raised, and a
limiter holds it under 0.9. The window's **Play** button sounds it
through the speakers, with `amp="auto"` unless its **Amp** box says
otherwise.

```python
session.add_band(2800, 3200, "whine")         # -> "whine"; "band 1", ... unnamed
session.edit_band("whine", high_hz=3500, on=False, rename="hiss")
sound = session.audio("mic:amplitude", bands=session.heard_bands)
```

Frequency bands are the session's: `session.bands` lists them (`Band`:
`name`, `low_hz`, `high_hz`, `on`), `session.heard_bands` gives the
ticked ones as (low, high) pairs, and `set_band_on`, `remove_band`,
`clear_bands` do what they say. `audio(..., bands=[(low, high), ...])`
keeps only those frequencies of the stretch -- a zero-phase mask over
all of it, with soft edges -- and raises the result to peak at 0.9, or
to what `amp` says. Play sounds the ticked bands, or all of the sound
when none are ticked. `session.rate_of(x)` is a signal's or recipe's sample rate.

## Sources

A source produces samples and says what they are:

```python
class Source:
    name: str; kind: str; channels: list[str]; units: str; rate: float
    dtype: numpy.dtype        # what a sample is held as; float64 by default
    def start(self): ...      # idempotent; after it, the above are final
    def read(self): ...       # (time, values) for whatever is ready
    def stop(self): ...
    def restart(self): ...    # begin again, producing nothing made before
```

`restart()` is what `session.start_intake()` calls, and a source that
counts what it has produced must reset that count in it: the simulator
starts its clock and its samples from zero again, a recording plays from
its first sample, and a live stream steps over whatever its device
sent while the session was stopped. Stopping and starting is enough for
a source that keeps no count of its own, and is the default.

`SyntheticAccelerometer`, `FileSources(path, replay=False, speed=1.0,
**options)` (whatever a data file holds, a source per series, replayed
together), `RecordingSource(path, topic)` and `RecordingSources(path,
topics)` (an MCAP's topics), `CsvSource`, `AudioSource` and
`SeriesSource` (any `TimeSeries`) are the ones that come with
mmp-sense. A live source -- a device on a serial line or the network --
is one more class with these five methods. `session.attach(source)` swaps every stream for one and
keeps the shown recipes, which is how a detector worked out on a live
stream is then run over a recording; `session.add(source)` takes
another stream beside the ones there.

The simulated source knocks on demand, which is what makes it useful for
trying a detector:

```python
source.knock(amplitude=5.0)
source.event_spans()                 # the ground truth, for judging one
```

## MCP

`session.serve()` opens a Unix socket under `$XDG_RUNTIME_DIR/mmp-sense`,
readable only by you, and answers the tools on it; `mmp-sense-mcp` attaches to
it and speaks MCP over stdio. The tools are

```
get_current_context   list_signals   get_statistics
get_values            get_events     get_spectrum_summary
get_delay             get_clocks     get_clock_observations
```

`get_current_context` and `list_signals` list the streams, each with
its rate and channels. All of them only read, and none runs code it is
sent. Without `t0`/`t1`
they answer about the user's selection, else everything held -- the
stretch rule of the questions they are built on.

## Panes

A pane draws and computes nothing. Adding a kind of pane is subclassing
`Pane`, implementing `refresh`, and registering it:

```python
from mmp_sense.ui.views import register_view
from mmp_sense.ui.panes import Pane

class HistogramPane(Pane):
    kind = "histogram"
    time_axis = False

    def refresh(self, reasons, window=None):
        if window is None or self.signal not in self.session.signals:
            return
        counts, edges = numpy.histogram(window[self.signal], bins=64)
        ...

register_view("histogram", HistogramPane, description="Distribution")
```

It is then in the window's *Add pane* menu, by its name (the
description is the item's tooltip), and a module outside mmp-sense can
add a pane without editing mmp-sense.

## Layout

```
mmp_sense/
    dsp/        plain functions over ndarrays: filters, spectra, trailing measures
    core/
        buffer.py    the ring the samples land in
        signals.py   named columns over it, and windows
        session.py   the shared model
        query.py     the arithmetic behind the answers
        graph/       recipes: nodes, ops, evaluation
        data/        the typed datasets panes draw
    sources/    simulated, and read from a file
    io/         the file formats: MCAP, WAV and audio, delimited text, NumPy
    mcp/        the tools, the local socket, the stdio server
    ui/         Qt: the window, the panes, the console
    analysis.py load(), from_frame(): sessions without a window
    scripts.py  run_script(): a file's main(session)
    examples/   runnable worked examples
```

Dependencies point one way: `ui -> sources -> core -> dsp`, with `mcp`
alongside `ui`. Nothing under `core`, `dsp`, `io` or `mcp` imports Qt,
and no signal knows a pane exists.
