"""The operations a node can be, by name.

An op is a plain function `fn(ctx, *inputs, **params)` returning a
`Series` or an `EventSeries`, plus what it declares about itself:
whether it is causal, keeps state, looks through a window, and how much
delay that costs. Registering one is all it takes for it to be usable
in any graph -- the engine never names an op.

    @register_op("clip", description="Limit to +-limit")
    def clip(ctx, series, limit=1.0):
        return series.with_values(np.clip(series.values, -limit, limit))

    node.apply("clip", limit=2.0)
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np
import pandas as pd

from ... import dsp
from ..data import EventSeries
from .incremental import (Filter, Frontier, Lookback, Part, Rule, Stale, Tape,
                          pointwise)
from .series import Series, Timing, format_seconds, hold

__all__ = ["Op", "register_op", "op", "ops", "EVENT_COLUMNS"]

#: What every detection carries beyond EventSeries' own columns. The
#: timestamp is when the event happened; start and end bound it.
EVENT_COLUMNS = ["timestamp", "duration", "score", "source"]


@dataclass(frozen=True)
class Op:
    name: str
    fn: Callable[..., Any]
    output: str = "series"            # "series" or "events"
    causal: bool = True
    stateful: bool = False
    windowed: bool = False
    length: Any = None                # seconds, a param's name, or fn(params)
    delay: Any = None
    lookahead: Any = None
    durations: tuple[str, ...] = ()   # params given as "80ms" or seconds
    pattern: bool = False             # composes events into higher ones
    label: Callable[[dict], str] | None = None
    description: str = ""
    #: fn(ctx) -> what else, beyond its inputs, the op's result depends
    #: on -- for a leaf, which definitions of the signals it reads.
    depends: Callable[..., Any] | None = None
    #: Offered in the GUI's feature menu, with these starting values;
    #: None keeps it to the console.
    menu: dict[str, Any] | None = None
    #: Measures frames (a spectrum per frame) rather than a signal: the
    #: GUI's menu makes the frames first.
    on_frames: bool = False
    #: How a live result is carried on to new samples (`incremental.Rule`);
    #: None computes it over the whole window every time.
    incremental: Any = None

    def _amount(self, what, params: dict) -> float:
        if what is None:
            return 0.0
        if callable(what):
            return float(what(params))
        if isinstance(what, str):
            value = params.get(what)
            return float(value) if value is not None else 0.0
        return float(what)

    def timing(self, params: dict) -> Timing:
        """What this op costs by itself, before its inputs are counted."""
        return Timing(
            causal=self.causal,
            stateful=self.stateful,
            windowed=self.windowed,
            length_s=self._amount(self.length, params),
            delay_s=self._amount(self.delay, params),
            lookahead_s=self._amount(self.lookahead, params),
        )

    def text(self, params: dict) -> str:
        if self.label is not None:
            return self.label(params)
        if not params:
            return self.name
        return f"{self.name}({format_params(params, self.durations)})"

    def describe(self) -> dict:
        return {
            "name": self.name, "output": self.output, "causal": self.causal,
            "stateful": self.stateful, "windowed": self.windowed,
            "description": self.description,
        }


def format_params(params: dict, durations=()) -> str:
    parts = []
    for key, value in params.items():
        if value is None:
            continue
        if key in durations and isinstance(value, (int, float)):
            parts.append(f"{key}={format_seconds(value)}")
        elif isinstance(value, float):
            parts.append(f"{key}={value:g}")
        elif isinstance(value, (tuple, list)):
            parts.append(f"{key}={','.join(map(str, value))}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


_OPS: dict[str, Op] = {}
#: Bumped by every registration, so results of an op since redefined are
#: never served from a cache.
REGISTRY = {"version": 0}


def register_op(name: str | None = None, *, output: str = "series",
                causal: bool = True, stateful: bool = False,
                windowed: bool = False, length=None, delay=None,
                lookahead=None, durations=(), pattern: bool = False,
                label=None, description: str = "", depends=None, menu=None,
                on_frames: bool = False, incremental=None):
    """Register `fn(ctx, *inputs, **params)` as an op; a decorator.

    `menu` -- the starting values of its parameters -- offers an op of one
    signal to the GUI's feature menu, where a form asks for the rest.
    `incremental` -- a `Rule`, such as `pointwise` or `Lookback(n)` --
    lets a live result be carried on to new samples instead of computed
    over the whole buffer on every tick.
    """
    if output not in ("series", "events"):
        raise ValueError(f"an op outputs 'series' or 'events', not {output!r}")

    def wrap(fn):
        key = name or fn.__name__
        _OPS[key] = Op(
            name=key, fn=fn, output=output, causal=causal, stateful=stateful,
            windowed=windowed, length=length, delay=delay, lookahead=lookahead,
            durations=tuple(durations), pattern=pattern, label=label, depends=depends,
            description=description or (fn.__doc__ or "").strip().split("\n")[0],
            menu=None if menu is None else dict(menu), on_frames=bool(on_frames),
            incremental=incremental,
        )
        REGISTRY["version"] += 1
        return fn

    return wrap


def menu_ops() -> dict[str, Op]:
    """The ops offered in the GUI's feature menu, by name."""
    return {name: o for name, o in sorted(_OPS.items()) if o.menu is not None}


def op(name: str) -> Op:
    try:
        return _OPS[name]
    except KeyError:
        raise KeyError(f"no op {name!r}; have {', '.join(sorted(_OPS))}") from None


def ops() -> dict[str, Op]:
    return dict(_OPS)


# -- helpers for writing ops --------------------------------------------


def _samples(ctx, series: Series, seconds: float) -> int:
    rate = series.rate
    if rate <= 0:
        raise ValueError(f"{ctx.label}: a length in seconds needs a known sample rate")
    width = int(round(seconds * rate))
    if width < 1:
        raise ValueError(f"{ctx.label}: {seconds} s is less than one sample at {rate:g} Hz")
    return width


def _float(series: Series) -> np.ndarray:
    return np.asarray(series.values, dtype=float)


def _scalar(ctx, series: Series) -> np.ndarray:
    if series.shape:
        raise ValueError(
            f"{ctx.label}: takes one value per instant, got {series.shape}; "
            "take .magnitude() or pick a component with node['name']"
        )
    return _float(series)


# -- sources -----------------------------------------------------------


def _own_channels(window, names) -> bool:
    """Whether every name is a channel of the window's own stream."""
    table = window.table
    for name in names:
        signal = table.get(name)
        if signal is None or signal.kind != "channel" or signal.stream != window.stream:
            return False
    return True


def _definitions(window, names=()):
    table = window.table
    names = tuple(names or window.channels)
    if _own_channels(window, names):
        return (window.stream, *((n, table.get(n).serial) for n in names))
    # A derived signal reads others, which may have been defined again.
    return tuple((n, table.get(n).serial) for n in table.names)


class _Leaf(Rule):
    """The window's own samples: numbered by the window, taken whole.

    Only the source's channels: a signal defined with a function of the
    window may change its old values on every tick.
    """

    def fresh(self, ctx, inputs, params, live):
        out = ctx.evaluator.apply(ctx, [], params)
        names = params.get("names") or ctx.window.channels
        if not _own_channels(ctx.window, names):
            return Part(out), None
        numbers = Tape(np.arange(live.origin, live.end + 1))
        return Part(out, numbers.view()), numbers

    def extend(self, ctx, inputs, params, live, previous, numbers, end):
        if numbers is None:
            raise Stale("not the source's channels")
        numbers.drop(live.origin - (end + 1 - len(numbers)))
        numbers.append(np.arange(end + 1, live.end + 1))
        if len(numbers) != live.length:
            raise Stale("out of step with the window")
        return Part(ctx.evaluator.apply(ctx, [], params), numbers.view()), numbers


@register_op("signal", label=lambda p: f"signal[{','.join(p.get('names') or ()) or 'channels'}]",
             depends=_definitions, incremental=_Leaf(),
             description="Signals of the data, by name")
def _signal(ctx, names=()):
    window = ctx.window
    names = list(names) if names else window.channels
    if names == window.channels and len(names) > 1:
        values = window.values
    else:
        columns = [np.asarray(window[n]) for n in names]
        values = columns[0] if len(columns) == 1 else np.stack(columns, axis=1)
    return Series(
        time=window.time, values=values, rate=window.rate,
        units=window.table.units_for(names),
        coords={} if len(names) == 1 else {"channel": names},
    )


# -- per-sample --------------------------------------------------------


@register_op("magnitude", incremental=pointwise, description="Euclidean norm across the components")
def _magnitude(ctx, series):
    return series.with_values(dsp.magnitude(_float(series)), coords={})


def _first_order(kind):
    return dsp.highpass1 if kind == "highpass" else dsp.lowpass1


def _iir_order(kind, params) -> int:
    default = 1 if kind in ("lowpass", "highpass") else 4
    order = params.get("order", default)
    if order != int(order) or int(order) < 1:
        raise ValueError(f"order must be a whole number from 1, got {order}")
    return int(order)


def _iir_run(kind):
    """`fn(values, rate, state=None, **params)` for the Filter rule."""
    def run(values, rate, state=None, **params):
        order = _iir_order(kind, params) if kind != "notch" else None
        params = {k: v for k, v in params.items() if k != "order"}
        if kind in ("lowpass", "highpass") and order == 1:
            return _first_order(kind)(values, rate, params["cutoff_hz"], state)
        if kind == "notch":
            return dsp.sos_filter(values, dsp.design(kind, rate, **params), state)
        return dsp.sos_filter(values, dsp.design(kind, rate, order=order, **params), state)
    return run


def _iir_delay(kind):
    def delay(p):
        order = _iir_order(kind, p) if kind != "notch" else None
        if kind == "lowpass" and order == 1:
            return 1.0 / (2.0 * np.pi * p["cutoff_hz"])
        if kind == "highpass" and order == 1:
            return 0.0
        params = {k: v for k, v in p.items() if k != "order"}
        return dsp.group_delay(kind, **params, **({} if order is None else {"order": order}))
    return delay


def _iir_label(kind, keys):
    def label(p):
        inner = ", ".join(f"{p[k]:g}Hz" for k in keys)
        default = 1 if kind in ("lowpass", "highpass") else 4
        if kind != "notch" and p.get("order", default) != default:
            inner += f", order {p['order']:g}"
        if kind == "notch" and p.get("q", 30.0) != 30.0:
            inner += f", q {p['q']:g}"
        return f"{kind}({inner})"
    return label


_IIR_DESCRIPTIONS = {
    "highpass": "High-pass: first-order IIR, or Butterworth of `order`; from rest",
    "lowpass": "Low-pass: first-order IIR, or Butterworth of `order`; from rest",
    "bandpass": "Butterworth band-pass between two frequencies; from rest",
    "bandstop": "Butterworth band-stop between two frequencies; from rest",
    "notch": "A notch at one frequency, `freq_hz/q` wide; from rest",
}


def _register_iir(kind, keys, menu):
    run = _iir_run(kind)

    def fn(ctx, series, **params):
        values, _ = run(_float(series), series.rate, **params)
        return series.with_values(values)
    fn.__name__ = f"_{kind}"
    defaults = {"order": 1} if kind in ("lowpass", "highpass") else (
        {"q": 30.0} if kind == "notch" else {"order": 4})
    fn.__signature__ = inspect.Signature(
        [inspect.Parameter("ctx", inspect.Parameter.POSITIONAL_OR_KEYWORD),
         inspect.Parameter("series", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        + [inspect.Parameter(k, inspect.Parameter.KEYWORD_ONLY) for k in keys]
        + [inspect.Parameter(k, inspect.Parameter.KEYWORD_ONLY, default=v)
           for k, v in defaults.items()])
    register_op(kind, stateful=True, incremental=Filter(run), delay=_iir_delay(kind),
                label=_iir_label(kind, keys), menu=menu,
                description=_IIR_DESCRIPTIONS[kind])(fn)


_register_iir("highpass", ("cutoff_hz",), {"cutoff_hz": 10.0, "order": 1})
_register_iir("lowpass", ("cutoff_hz",), {"cutoff_hz": 10.0, "order": 1})
_register_iir("bandpass", ("low_hz", "high_hz"), {"low_hz": 20.0, "high_hz": 200.0, "order": 4})
_register_iir("bandstop", ("low_hz", "high_hz"), {"low_hz": 45.0, "high_hz": 55.0, "order": 4})
_register_iir("notch", ("freq_hz",), {"freq_hz": 50.0, "q": 30.0})


def filter_sections(node, rate: float) -> list[np.ndarray]:
    """The second-order sections of every filter along a recipe's first
    inputs, from its signal on: what `Node.response` multiplies."""
    from scipy import signal as _scipy

    chain = []
    while node is not None:
        if node.op in _IIR_DESCRIPTIONS:
            p = dict(node.params)
            if node.op in ("lowpass", "highpass") and p.get("order", 1) == 1:
                if node.op == "highpass":
                    a = dsp.highpass1_alpha(rate, p["cutoff_hz"])
                    b_, a_ = [a, -a], [1.0, -a]
                else:
                    beta = dsp.lowpass1_alpha(rate, p["cutoff_hz"])
                    b_, a_ = [beta, 0.0], [1.0, -(1.0 - beta)]
                chain.append(_scipy.tf2sos(b_, a_))
            else:
                if node.op != "notch":
                    p["order"] = _iir_order(node.op, p)
                chain.append(dsp.design(node.op, rate, **p))
        node = node.inputs[0] if node.inputs else None
    return chain[::-1]


@register_op("envelope", stateful=True, incremental=Filter(dsp.envelope),
             delay=lambda p: 2.0 / p.get("min_hz", 20.0),
             label=lambda p: f"envelope({p.get('min_hz', 20.0):g}Hz)", menu={"min_hz": 20.0},
             description="The size of the analytic signal (causal Hilbert); lags 2/min_hz")
def _envelope(ctx, series, min_hz=20.0):
    values, _ = dsp.envelope(_float(series), series.rate, min_hz)
    return series.with_values(values)


def _beam_delays(positions, speed, azimuth_deg, elevation_deg):
    """Per channel, how long to hold it back so a sound from that
    direction lines up in all: from when it arrives to when the last does."""
    arrival = dsp.steering(np.asarray(positions, float), speed, azimuth_deg, elevation_deg)
    return arrival.max() - arrival


def _beam_run(values, rate, state=None, *, positions, speed, azimuth_deg, elevation_deg=0.0):
    if rate * dsp.FRACTION_S < 4:
        raise ValueError(f"a beam needs {4 / dsp.FRACTION_S:g} samples a second or more "
                         f"to delay by fractions of one; this stream has {rate:g}")
    if np.ndim(values) != 2 or np.shape(values)[1] != len(positions):
        raise ValueError(f"a beam takes one signal per microphone: {len(positions)}")
    return dsp.delay_and_sum(values, rate, _beam_delays(positions, speed, azimuth_deg,
                                                        elevation_deg), state)


@register_op("beam", stateful=True, incremental=Filter(_beam_run),
             delay=lambda p: float(_beam_delays(p["positions"], p["speed"], p["azimuth_deg"],
                                                p.get("elevation_deg", 0.0)).max())
                             + dsp.FRACTION_S,
             label=lambda p: f"beam({p['azimuth_deg']:g}deg"
                             + (f", {p['elevation_deg']:g}deg" if p.get("elevation_deg") else "")
                             + ")",
             description="Delay and sum: what an array hears from one direction")
def _beam(ctx, series, positions, speed, azimuth_deg, elevation_deg=0.0):
    values, _ = _beam_run(_float(series), series.rate, positions=positions, speed=speed,
                          azimuth_deg=azimuth_deg, elevation_deg=elevation_deg)
    return series.with_values(values, coords={})


def _fill_run(values, rate, state=None, longest=None):
    """Each gap (NaN) filled with the last value before it, for at most
    `longest` seconds; the state is (last value, samples since it)."""
    values = np.asarray(values, dtype=float)
    last, age = state if state is not None else (np.nan, np.inf)
    n = len(values)
    if not n:
        return values.copy(), (last, age)
    at = np.arange(n)
    seen = np.maximum.accumulate(np.where(np.isfinite(values), at, -1))
    earlier = seen < 0
    filled = np.where(earlier, last, values[np.maximum(seen, 0)])
    since = np.where(earlier, age + at + 1, at - seen)
    limit = np.inf if longest is None else float(longest) * rate
    out = np.where(np.isfinite(values), values, np.where(since <= limit, filled, np.nan))
    return out, (float(filled[-1]), float(since[-1]))


@register_op("fill", stateful=True, incremental=Filter(_fill_run), durations=("longest",),
             label=lambda p: "fill" if p.get("longest") is None
             else f"fill({format_seconds(p['longest'])})",
             description="Each gap filled with the last value before it, for at most `longest`")
def _fill(ctx, series, longest=None):
    values, _state = _fill_run(_float(series), series.rate, None, longest)
    return series.with_values(values)


@register_op("abs", incremental=pointwise, description="The size of each value: rectified")
def _abs(ctx, series):
    return series.with_values(np.abs(_float(series)))


@register_op("select", incremental=pointwise, label=lambda p: f"[{p['key']}]", description="One component")
def _select(ctx, series, key):
    if isinstance(key, str):
        channels = series.channels
        if key not in channels:
            raise KeyError(f"{ctx.label}: no component {key!r}; have {', '.join(channels)}")
        key = channels.index(key)
    return series.with_values(series.values[:, key], coords={})


_ARITH = {"add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.divide}
_SYMBOL = {"add": "+", "sub": "-", "mul": "*", "div": "/",
           "gt": ">", "ge": ">=", "lt": "<", "le": "<="}


def _operands(first: Series, rest: tuple, value):
    a = first.values
    b = hold(rest[0], first.time) if rest else value
    return a, b


def _binary_label(p):
    other = "" if p.get("value") is None else f" {p['value']:g}"
    return f"{_SYMBOL[p['op']]}{other}" if other else _SYMBOL[p["op"]]


@register_op("arith", label=_binary_label, incremental=pointwise, description="Arithmetic, aligned on the left's time")
def _arith(ctx, first, *rest, op, value=None):
    a, b = _operands(first, rest, value)
    with np.errstate(divide="ignore", invalid="ignore"):
        return first.with_values(_ARITH[op](np.asarray(a, float), np.asarray(b, float)))


# -- trailing measures --------------------------------------------------
#
# Each value summarises the `length` seconds up to and including its
# instant: nothing looks ahead, so a recording gives what live would.


def _trailing(name, compute, description, units=lambda u: u):
    def fn(ctx, series, length):
        return series.with_values(compute(ctx, series, _samples(ctx, series, length)),
                                  units=units(series.units))
    fn.__name__ = f"_{name}"
    register_op(name, windowed=True, length="length", delay="length", durations=("length",),
                label=lambda p: f"{name}({format_seconds(p['length'])})",
                menu={"length": "80ms"}, description=description,
                incremental=Lookback(lambda ctx, s, p: _samples(ctx, s, p["length"]) + 1))(fn)


_trailing("peak", lambda ctx, s, w: dsp.trailing_max(np.abs(_float(s)), w),
          "Largest absolute value over the last `length`")
_trailing("energy", lambda ctx, s, w: dsp.trailing_sum(_float(s) ** 2, w),
          "Sum of squares over the last `length`", units=lambda u: f"({u})^2" if u else "")
_trailing("rms", lambda ctx, s, w: np.sqrt(dsp.trailing_sum(_float(s) ** 2, w) / w),
          "Root mean square over the last `length`")
_trailing("zero_crossing_rate",
          lambda ctx, s, w: dsp.zero_crossing_rate(_scalar(ctx, s), s.rate, max(2, w)),
          "Sign changes per second over the last `length`", units=lambda u: "Hz")


# -- frequency ---------------------------------------------------------------


def _first_end(series, step):
    # Frames end on sample numbers that are multiples of the hop, so a
    # live buffer that scrolls does not move the frames with it.
    return (-int(round(series.time[0] * series.rate))) % step if len(series.time) else 0


def _frame_plan(ctx, series, length, hop):
    values = _scalar(ctx, series)
    width = _samples(ctx, series, length)
    step = max(1, _samples(ctx, series, hop)) if hop else max(1, width // 4)
    if width < 2:
        raise ValueError(f"{ctx.label}: a frame needs at least 2 samples")
    return values, width, step


def _framed(series, values, starts, width, step, taper):
    frequency, amplitude = dsp.framed_spectrum(values, series.rate, starts, width, taper)
    return Series(
        time=series.time[starts + width - 1] if len(starts) else np.zeros(0),
        values=amplitude, rate=series.rate / step, units=series.units,
        coords={"frequency": frequency, "enbw": dsp.enbw(taper, width)},
    )


def _consecutive(numbers) -> bool:
    return len(numbers) == 0 or int(numbers[-1] - numbers[0]) == len(numbers) - 1


class _Frames(Rule):
    """Frames every `step` samples from the last one's end: the ones
    whose last sample is new are the new ones. Any other input is read
    at the frames' times, whole: its old values must stay put."""

    def compute(self, ctx, series, others, starts, width, step, params):
        return _framed(series, _scalar(ctx, series), starts, width, step,
                       params.get("taper", "hann"))

    def ready(self, series, others, starts, width):
        """The frames that can be computed yet: all of them."""
        return starts

    def fresh(self, ctx, inputs, params, live):
        series, numbers = inputs[0].result, inputs[0].pos
        values, width, step = _frame_plan(ctx, series, params["length"], params.get("hop"))
        first_end = _first_end(series, step)
        others = [p.result for p in inputs[1:]]
        starts = self.ready(series, others,
                            dsp.frame_starts(len(values), width, step, first_end), width)
        out = self.compute(ctx, series, others, starts, width, step, params)
        # With no sample yet, where the frames fall is not yet known.
        if numbers is None or not len(numbers) or not _consecutive(numbers) \
                or any(p.pos is None for p in inputs[1:]):
            return Part(out), None
        ends = numbers[starts + width - 1]
        if len(ends):
            following = int(ends[-1]) + step
        else:
            following = (int(numbers[0]) if len(numbers) else live.origin) + \
                width - 1 + ((first_end - (width - 1)) % step)
        tapes = (Tape(out.time), Tape(ends), Tape(out.values))
        out = replace(out, time=tapes[0].view(), values=tapes[2].view())
        return Part(out, tapes[1].view()), (width, step, following, tapes)

    def extend(self, ctx, inputs, params, live, previous, state, end):
        first = inputs[0]
        numbers = first.pos
        if state is None or numbers is None or not _consecutive(numbers):
            raise Stale("frames need their input's every sample")
        if any(p.pos is None for p in inputs[1:]):
            raise Stale("an input whose old values may change")
        width, step, following, (times, ends, values) = state
        series = first.result
        if _frame_plan(ctx, series, params["length"], params.get("hop"))[1:] != (width, step):
            raise Stale("the frames changed size")
        gone = int(np.searchsorted(ends.view(), live.origin, side="left"))
        for tape in (times, ends, values):
            tape.drop(gone)
        template = previous.result
        new_ends = np.arange(following, int(numbers[-1]) + 1, step) if len(numbers) else []
        if len(new_ends):
            starts = new_ends - int(numbers[0]) - width + 1
            ready = self.ready(series, [p.result for p in inputs[1:]], starts, width)
            new_ends = new_ends[np.isin(starts, ready)]
        if len(new_ends):
            at = new_ends - int(numbers[0])
            begin = int(at[0]) - width + 1
            if begin < 0:
                raise Stale("not enough history held")
            tail = replace(series, time=series.time[begin:], values=series.values[begin:])
            out = self.compute(ctx, tail, [p.result for p in inputs[1:]],
                               at - width + 1 - begin, width, step, params)
            times.append(out.time)
            ends.append(new_ends.astype(ends.view().dtype))
            values.append(out.values)
            following = int(new_ends[-1]) + step
            template = out
        result = replace(template, time=times.view(), values=values.view())
        return Part(result, ends.view()), (width, step, following, (times, ends, values))


def _frame_delay(p):
    """A frame is complete `length` after its first sample, and its
    result waits up to one more hop for the frame's end to come round."""
    length = p["length"]
    return length + (p.get("hop") or length / 4)


@register_op("frames", windowed=True, length="length", delay=_frame_delay,
             durations=("length", "hop"), incremental=_Frames(),
             label=lambda p: "frames(" + format_params(
                 {"length": p["length"], "hop": p.get("hop")}, ("length", "hop")) + ")",
             description="An amplitude spectrum of the last `length`, every `hop`")
def _frames(ctx, series, length, hop=None, taper="hann"):
    values, width, step = _frame_plan(ctx, series, length, hop)
    starts = dsp.frame_starts(len(values), width, step, _first_end(series, step))
    return _framed(series, values, starts, width, step, taper)


class _Delays(_Frames):
    """A delay per frame of the first input, the second read at its times."""

    def compute(self, ctx, series, others, starts, width, step, params):
        return _delays(ctx, series, others[0], starts, width, step, params)

    def ready(self, series, others, starts, width):
        return _readable(series, others[0], starts, width)


def _same_samples(other: Series, time: np.ndarray) -> bool:
    """Whether `time` is the end of `other`'s own times: the tail of one stream."""
    n = len(time)
    return n <= len(other.time) and np.array_equal(other.time[len(other.time) - n:], time)


def _readable(series, other, starts, width):
    """The frames whose every sample `other` can be read at: on other
    times, up to its last sample -- one past a frame's end must have
    come, or the line to it is not yet drawn."""
    if _same_samples(other, series.time):
        return starts
    last = other.time[-1] if len(other.time) else -np.inf
    ends = np.clip(starts + width - 1, 0, max(0, len(series.time) - 1))
    return starts[(series.time[ends] <= last) | (starts < 0)]


def _at_times(ctx, other: Series, time: np.ndarray) -> np.ndarray:
    """`other` at each of `time`: its own samples on its own times, else
    by straight lines between them -- none drawn across a gap (NaN), nor
    past its ends."""
    values = _scalar(ctx, other)
    if _same_samples(other, time):
        return values[len(values) - len(time):]
    if len(other.time) < 2:
        return np.full(len(time), np.nan)
    out = np.interp(time, other.time, np.nan_to_num(values), left=np.nan, right=np.nan)
    after = np.clip(np.searchsorted(other.time, time, side="left"), 0, len(values) - 1)
    before = np.clip(np.searchsorted(other.time, time, side="right") - 1, 0, len(values) - 1)
    out[~(np.isfinite(values[after]) & np.isfinite(values[before]))] = np.nan
    return out


def _delays(ctx, series, other, starts, width, step, params):
    values = _scalar(ctx, series)
    second = _at_times(ctx, other, series.time)
    max_lag = params.get("max_lag")
    max_lag = width / series.rate / 2 if max_lag is None else max_lag
    out = np.full(len(starts), np.nan)
    for i, start in enumerate(starts):
        a, b = values[start:start + width], second[start:start + width]
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            continue
        lags, correlation = dsp.cross_correlation(a, b, series.rate, max_lag=max_lag,
                                                  method=params.get("method", "phat"))
        out[i] = dsp.peak_lag(lags, correlation)[0]
    return Series(
        time=series.time[starts + width - 1] if len(starts) else np.zeros(0),
        values=out, rate=series.rate / step, units="s",
    )


@register_op("delay", windowed=True, length="length", delay=_frame_delay,
             durations=("length", "hop", "max_lag"), incremental=_Delays(),
             label=lambda p: "delay(" + format_params(
                 {"length": p["length"], "hop": p.get("hop"), "max_lag": p.get("max_lag"),
                  "method": None if p.get("method", "phat") == "phat" else p["method"]},
                 ("length", "hop", "max_lag")) + ")",
             description="How much later the second hears what the first does, per frame (s)")
def _delay(ctx, series, other, length, hop=None, max_lag=None, method="phat"):
    values, width, step = _frame_plan(ctx, series, length, hop)
    starts = dsp.frame_starts(len(values), width, step, _first_end(series, step))
    return _delays(ctx, series, other, _readable(series, other, starts, width), width, step,
                   {"max_lag": max_lag, "method": method})


def _metric(name, compute, *, units, first=(), description, menu=None):
    """An op measuring each frame of a spectrum: one number per frame."""
    def fn(ctx, spectrum, **params):
        if "frequency" not in spectrum.coords:
            raise ValueError(f"{ctx.label}: measures frames; make them with "
                             ".frames(length, hop) first")
        values = compute(spectrum.coords["frequency"], spectrum.values,
                         spectrum.coords.get("enbw", 1.5), **params)
        return spectrum.with_values(values, coords={}, units=units(spectrum.units))
    # Name the parameters, so a misspelt one is refused when the node is made.
    extra = list(inspect.signature(compute).parameters.values())[3:]
    fixed = list(inspect.signature(fn).parameters.values())[:2]
    fn.__signature__ = inspect.Signature(fixed + [
        x.replace(kind=inspect.Parameter.KEYWORD_ONLY) for x in extra])

    def label(p):
        inner = ", ".join(f"{p[k]:g}" if isinstance(p[k], float) else str(p[k])
                          for k in first if p.get(k) is not None)
        return f"{name}({inner})" if inner else name
    register_op(name, label=label, on_frames=True, description=description,
                menu=None if menu is None else dict(menu),
                incremental=Lookback(1) if name == "flux" else pointwise)(fn)


def _squared(units):
    return f"({units})^2" if units else ""


_metric("band_power",
        lambda f, a, bw, low_hz, high_hz: dsp.band_power(f, a, low_hz, high_hz, bw),
        units=_squared, first=("low_hz", "high_hz"), menu={"low_hz": 80.0, "high_hz": 100.0},
        description="Power between two frequencies, per frame (units squared)")
_metric("band_share",
        lambda f, a, bw, low_hz, high_hz: dsp.band_power(f, a, low_hz, high_hz, bw, True),
        units=lambda u: "", first=("low_hz", "high_hz"), menu={"low_hz": 80.0, "high_hz": 100.0},
        description="The share of all power between two frequencies, 0..1, per frame")
_metric("dominant_hz",
        lambda f, a, bw, min_hz=0.0: dsp.dominant_frequency(f, a, min_hz or 0.0),
        units=lambda u: "Hz", menu={},
        description="The frequency of the strongest bin, per frame")
_metric("centroid_hz", lambda f, a, bw: dsp.spectral_centroid(f, a, bw),
        units=lambda u: "Hz", menu={},
        description="The power's centre of gravity, per frame")
_metric("spread_hz", lambda f, a, bw: dsp.spectral_spread(f, a, bw),
        units=lambda u: "Hz", menu={},
        description="How far the power lies from the centroid, per frame")
_metric("flatness", lambda f, a, bw: dsp.spectral_flatness(f, a, bw),
        units=lambda u: "", menu={},
        description="1 when power is spread evenly (noise), near 0 in a few bins (a tone)")
_metric("rolloff_hz",
        lambda f, a, bw, fraction=0.85: dsp.spectral_rolloff(f, a, bw, fraction),
        units=lambda u: "Hz", first=("fraction",), menu={"fraction": 0.85},
        description="The frequency below which `fraction` of the power lies, per frame")
_metric("flux",
        lambda f, a, bw, normalize=False: dsp.spectral_flux(f, a, bw, normalize),
        units=lambda u: u, menu={},
        description="How much the spectrum grew since the last frame (onsets); "
                    "normalize: change of shape only")


_metric("line_power",
        lambda f, a, bw, low_hz, high_hz, above=10.0, floor_hz=60.0:
            dsp.line_power(f, a, low_hz, high_hz, bw, above, floor_hz),
        units=_squared, first=("low_hz", "high_hz"),
        menu={"low_hz": 150.0, "high_hz": 1000.0, "above": 10.0, "floor_hz": 60.0},
        description="Power in the lines between two frequencies -- bins standing `above` "
                    "times over the floor around them -- per frame (units squared)")
_metric("line_share",
        lambda f, a, bw, low_hz, high_hz, above=10.0, floor_hz=60.0:
            dsp.line_power(f, a, low_hz, high_hz, bw, above, floor_hz, relative=True),
        units=lambda u: "", first=("low_hz", "high_hz"),
        menu={"low_hz": 150.0, "high_hz": 1000.0, "above": 10.0, "floor_hz": 60.0},
        description="The share of a band that is lines, each bin against its own floor, "
                    "0..1, per frame")


_metric("repetition_hz",
        lambda f, a, bw, min_hz, max_hz, low_hz=None, high_hz=None, floor_hz=60.0,
        guard_db=dsp.GUARD_DB: dsp.repetition_rate(
            f, np.asarray(a, dtype=float) ** 2, min_hz, max_hz, low_hz=low_hz,
            high_hz=high_hz, floor_hz=floor_hz, guard_db=guard_db),
        units=lambda u: "Hz", first=("min_hz", "max_hz"),
        description="How often the sound repeats, from the autocorrelation of its lines, "
                    "checked against half and double, per frame (Hz)")


def _of_frames(ctx, spectrum, what):
    if "frequency" not in spectrum.coords:
        raise ValueError(f"{ctx.label}: {what}; make them with .frames(length, hop) first")


def _peak_label(p):
    near = "" if p.get("near") is None else f", near {p['near']:g}"
    return f"peak_hz({p['low_hz']:g}, {p['high_hz']:g}{near})"


@register_op("peak_hz", on_frames=True, incremental=pointwise, label=_peak_label,
             description="A line's precise frequency, per frame -- the strongest bin in a "
                         "band, or near a frequency or a recipe that follows it (Hz)")
def _peak_hz(ctx, spectrum, *rest, low_hz, high_hz, near=None, within=None):
    _of_frames(ctx, spectrum, "finds a line in frames")
    if rest:
        near = np.asarray(hold(rest[0], spectrum.time), dtype=float)
    values = dsp.line_peaks(spectrum.coords["frequency"], spectrum.values, low_hz, high_hz,
                            near=near, within=within)
    return spectrum.with_values(values, coords={}, units="Hz")


@register_op("align", on_frames=True, incremental=pointwise,
             label=lambda p: f"align({p['ref_hz']:g} Hz)",
             description="Frames with a drifting rate divided out: each spectrum's "
                         "frequencies scaled so the rate reads `ref_hz`")
def _align(ctx, spectrum, rate, ref_hz):
    _of_frames(ctx, spectrum, "aligns frames")
    ratio = np.asarray(hold(rate, spectrum.time), dtype=float) / float(ref_hz)
    return spectrum.with_values(
        dsp.align_frames(spectrum.coords["frequency"], spectrum.values, ratio))


@register_op("steady", windowed=True, length="length", delay="length", durations=("length",),
             label=lambda p: f"steady({format_seconds(p['length'])})",
             incremental=Lookback(lambda ctx, s, p: _samples(ctx, s, p["length"]) - 1),
             description="Of frames: each bin's smallest value over the last `length` -- "
                         "what has sounded all along")
def _steady(ctx, spectrum, length):
    if "frequency" not in spectrum.coords:
        raise ValueError(f"{ctx.label}: holds frames; make them with "
                         ".frames(length, hop) first")
    values = dsp.trailing_min(spectrum.values, _samples(ctx, spectrum, length))
    return spectrum.with_values(values)

# -- predicates --------------------------------------------------------

_COMPARE = {"gt": np.greater, "ge": np.greater_equal, "lt": np.less, "le": np.less_equal}


@register_op("compare", label=_binary_label, incremental=pointwise, description="A comparison, true or false per instant")
def _compare(ctx, first, *rest, op, value=None):
    a, b = _operands(first, rest, value)
    with np.errstate(invalid="ignore"):
        out = _COMPARE[op](np.asarray(a, float), np.asarray(b, float))
    return first.with_values(np.asarray(out, dtype=bool), units="")


@register_op("between", incremental=pointwise, label=lambda p: f"between {p['low']:g}..{p['high']:g}",
             description="Within [low, high], inclusive")
def _between(ctx, series, low, high):
    values = _float(series)
    with np.errstate(invalid="ignore"):
        out = (values >= low) & (values <= high)
    return series.with_values(out, units="")


def _truth(series: Series) -> np.ndarray:
    values = series.values
    if values.dtype == bool:
        return values
    with np.errstate(invalid="ignore"):
        return np.nan_to_num(np.asarray(values, float), nan=0.0) != 0


@register_op("and", label=lambda p: "AND", incremental=pointwise, description="Both true, aligned on the left's time")
def _and(ctx, first, second):
    held = hold(second.with_values(_truth(second)), first.time)
    return first.with_values(_truth(first) & held, units="")


@register_op("or", label=lambda p: "OR", incremental=pointwise, description="Either true, aligned on the left's time")
def _or(ctx, first, second):
    held = hold(second.with_values(_truth(second)), first.time)
    return first.with_values(_truth(first) | held, units="")


@register_op("not", label=lambda p: "NOT", incremental=pointwise, description="Negation")
def _not(ctx, series):
    return series.with_values(~_truth(series), units="")


# -- events ------------------------------------------------------------


def _events(ctx, records) -> EventSeries:
    for record in records:
        record.setdefault("label", ctx.name)
        record.setdefault("confidence", 1.0)
        record.setdefault("score", np.nan)
        record.setdefault("source", ctx.name)
        record["duration"] = record["end_time"] - record["start_time"]
    columns = ["start_time", "end_time", "label", "confidence", *EVENT_COLUMNS]
    table = pd.DataFrame(records) if records else pd.DataFrame(columns=columns)
    return EventSeries(name=ctx.name, table=table)


def event_table(events: EventSeries):
    """An event table with the detection columns, filled for any source."""
    table = events.table
    if "timestamp" not in table.columns:
        table = table.assign(timestamp=table["start_time"])
    if "duration" not in table.columns:
        table = table.assign(duration=table["end_time"] - table["start_time"])
    return table.sort_values("timestamp", kind="stable").reset_index(drop=True)


def _truth_1d(ctx, condition):
    truth = _truth(condition)
    if truth.ndim != 1:
        raise ValueError(f"{ctx.label}: events come from one true/false per instant")
    return truth


def _runs(truth, time, refractory, min_duration, last=-np.inf, run=None):
    """The events of the stretches where `truth` holds, carried on from a
    block that ended with `run` = (start, end) still holding and the last
    event starting at `last`.

    Returns (records, last, run, going): the events of the stretches that
    ended, and of the one still holding at the end, if it counts yet --
    it is not final, for it may yet go on.
    """
    spans = [] if run is None else [run]
    if len(truth):
        edges = np.diff(np.concatenate(([int(run is not None)], truth.astype(np.int8), [0])))
        starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
        if run is not None:
            spans = [(run[0], float(time[stops[0]]) if stops[0] >= 0 else run[1])]
            stops = stops[1:]
        spans += [(float(time[a]), float(time[b])) for a, b in zip(starts, stops)]
        run = spans[-1] if truth[-1] else None
    ended = spans[:-1] if run is not None else spans

    def counts(start, end, last):
        return end - start >= (min_duration or 0.0) and start - last >= (refractory or 0.0)

    records = []
    for start, end in ended:
        if counts(start, end, last):
            last = start
            records.append({"start_time": start, "end_time": end, "timestamp": start})
    going = None
    if run is not None and counts(*run, last):
        going = {"start_time": run[0], "end_time": run[1], "timestamp": run[0]}
    return records, last, run, going


class _Events(Rule):
    """Stretches that ended are final; the one still holding is looked at
    again as the condition goes on."""

    def _made(self, ctx, settled, going, frontier):
        records = [dict(r) for r in settled] + ([going] if going else [])
        return Part(_events(ctx, records), frontier=frontier)

    def fresh(self, ctx, inputs, params, live):
        first = inputs[0]
        truth = _truth_1d(ctx, first.result)
        records, last, run, going = _runs(truth, first.result.time,
                                          params.get("refractory"), params.get("min_duration"))
        if first.pos is None:
            return self._made(ctx, records, going, -np.inf), None
        return (self._made(ctx, records, going, _frontier(first.result.time, run)),
                (records, last, run))

    def extend(self, ctx, inputs, params, live, previous, state, end):
        first = inputs[0]
        if state is None or first.pos is None:
            raise Stale("no positions to carry on from")
        settled, last, run = state
        new = int(np.searchsorted(first.pos, end, side="right"))
        tail = replace(first.result, time=first.result.time[new:],
                       values=first.result.values[new:])
        truth, time = _truth_1d(ctx, tail), tail.time
        records, last, run, going = _runs(truth, time, params.get("refractory"),
                                          params.get("min_duration"), last, run)
        start = float(ctx.window.t_start)
        settled = [r for r in settled if r["timestamp"] >= start] + records
        return (self._made(ctx, settled, going, _frontier(first.result.time, run)),
                (settled, last, run))


def _frontier(time, run) -> float:
    """Before which no event will change: the start of the stretch still
    holding, else the last instant looked at."""
    if run is not None:
        return run[0]
    return float(time[-1]) if len(time) else -np.inf


@register_op("events", output="events", durations=("refractory", "min_duration"),
             incremental=_Events(),
             delay=lambda p: p.get("min_duration") or 0.0,
             label=lambda p: "events(" + format_params(
                 {k: v for k, v in p.items() if v}, ("refractory", "min_duration")) + ")",
             description="Each stretch where a condition holds, as an event")
def _to_events(ctx, condition, refractory=0.0, min_duration=0.0):
    records, _, _, going = _runs(_truth_1d(ctx, condition), condition.time,
                                 refractory, min_duration)
    return _events(ctx, records + ([going] if going else []))


def _gap_label(name):
    def label(p):
        low, high = p.get("min_delay") or 0.0, p.get("max_delay")
        gap = f"{format_seconds(low)}..{format_seconds(high) if high is not None else ''}"
        lead = f"{p['n']}, " if "n" in p else ""
        return f"{name}({lead}gap {gap})"
    return label


def _follow_on(ctx, inputs, params, kept):
    taken = () if kept is None or not len(kept) else kept["second"].to_numpy(float)
    return _follow(ctx, *inputs, params.get("min_delay"), params.get("max_delay"), taken)


def _max_delay(p):
    return np.inf if p.get("max_delay") is None else p["max_delay"]


@register_op("followed_by", output="events", pattern=True, durations=("min_delay", "max_delay"),
             incremental=Frontier(_max_delay, _follow_on, needs_history=True),
             delay=lambda p: float("inf") if p.get("max_delay") is None else p["max_delay"],
             label=_gap_label("followed_by"),
             description="An event of the first kind, then one of the second")
def _followed_by(ctx, first, second, min_delay=0.0, max_delay=None):
    """Each first event takes the earliest second event from `min_delay`
    to `max_delay` after it (strictly after, when both are the same
    stream); each event takes part in one match at most."""
    return _follow(ctx, first, second, min_delay, max_delay)


def _follow(ctx, first, second, min_delay=0.0, max_delay=None, taken=()):
    """`_followed_by`, the second events at the times `taken` already
    matched by earlier first events."""
    a_table, b_table = event_table(first), event_table(second)
    same = ctx.inputs[0].key == ctx.inputs[1].key
    a_times = a_table["timestamp"].to_numpy(float)
    b_times = b_table["timestamp"].to_numpy(float)
    low = min_delay or 0.0
    high = np.inf if max_delay is None else max_delay
    used_second: set[int] = set(np.flatnonzero(np.isin(b_times, np.asarray(taken, float))).tolist())
    used_first = used_second if same else set()
    records = []
    for i, t in enumerate(a_times):
        if i in used_first:
            continue
        for j in range(int(np.searchsorted(b_times, t + low, side="left")), len(b_times)):
            gap = b_times[j] - t
            if gap > high:
                break
            if (same and j <= i) or j in used_second:
                continue
            used_first.add(i)
            used_second.add(j)
            a_row, b_row = a_table.iloc[i], b_table.iloc[j]
            records.append({
                "start_time": float(a_row["start_time"]),
                "end_time": float(max(b_row["end_time"], a_row["end_time"])),
                "timestamp": float(t),
                "first": float(t),
                "second": float(b_times[j]),
                "gap": float(gap),
            })
            break
    return _events(ctx, records)


@register_op("repeated", output="events", pattern=True, durations=("min_delay", "max_delay"),
             incremental=Frontier(lambda p: 0.0 if p["n"] <= 1 else (p["n"] - 1) * _max_delay(p)),
             delay=lambda p: (p["n"] - 1) * (float("inf") if p.get("max_delay") is None
                                             else p["max_delay"]),
             label=_gap_label("repeated"),
             description="n events in a row, each from min to max after the last")
def _repeated(ctx, events, n, min_delay=0.0, max_delay=None):
    """From each event, a chain taking the earliest next event from
    `min_delay` to `max_delay` later, until it has `n`. Chains may share
    events: four knocks in rhythm hold two runs of three."""
    table = event_table(events)
    times = table["timestamp"].to_numpy(float)
    low = min_delay or 0.0
    high = np.inf if max_delay is None else max_delay
    records = []
    for i in range(len(times)):
        chain = [i]
        while len(chain) < n:
            last = chain[-1]
            j = int(np.searchsorted(times, times[last] + low, side="left"))
            j = max(j, last + 1)
            if j >= len(times) or times[j] - times[last] > high:
                break
            chain.append(j)
        if len(chain) < n:
            continue
        first, final = table.iloc[chain[0]], table.iloc[chain[-1]]
        records.append({
            "start_time": float(first["start_time"]),
            "end_time": float(final["end_time"]),
            "timestamp": float(times[chain[0]]),
            "times": [float(times[k]) for k in chain],
        })
    return _events(ctx, records)


@register_op("lasting", output="events", durations=("min", "max"),
             incremental=Frontier(lambda p: 0.0),
             label=lambda p: "lasting(" + format_params(p, ("min", "max")) + ")",
             description="Only the events lasting from min to max")
def _lasting(ctx, events, min=None, max=None):
    table = event_table(events)
    keep = np.ones(len(table), dtype=bool)
    if min is not None:
        keep &= table["duration"].to_numpy(float) >= min
    if max is not None:
        keep &= table["duration"].to_numpy(float) <= max
    kept = table[keep].reset_index(drop=True)
    return EventSeries(name=ctx.name, table=kept.assign(label=ctx.name, source=ctx.name))
