"""NumPy arrays saved with `np.save` and `np.savez` as time series.

    .npy   one array: (samples,) or (samples, channels); needs the rate
    .npz   named arrays: a channel per 1-D array, `name0`.. per column of
           a 2-D one; time from a `time` or `t` array in seconds, or the
           rate from a `rate`, `fs`, `sample_rate` or `sample_rate_hz`
           scalar, else given

Pickled objects are never loaded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.data import TimeSeries

__all__ = ["read_numpy", "NUMPY_SUFFIXES"]

NUMPY_SUFFIXES = frozenset({".npy", ".npz"})

_TIME_KEYS = ("time", "t")
_RATE_KEYS = ("rate", "fs", "sample_rate", "sample_rate_hz")


def read_numpy(path: str | Path, rate: float | None = None) -> TimeSeries:
    """An `.npy` or `.npz` file as a TimeSeries; `rate` in Hz when the
    file does not say when its samples were taken."""
    path = Path(path)
    try:
        loaded = np.load(path, allow_pickle=False)
    except (ValueError, EOFError, OSError) as exc:
        raise ValueError(f"{path}: not an array file that can be read safely ({exc})") from None
    if isinstance(loaded, np.ndarray):
        return _series(path, {"": loaded}, None, rate)
    with loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    time = next((arrays.pop(k) for k in _TIME_KEYS if k in arrays), None)
    stored = next((arrays.pop(k) for k in _RATE_KEYS if k in arrays), None)
    if rate is None and stored is not None and np.ndim(stored) == 0:
        rate = float(stored)
    return _series(path, arrays, time, rate)


def _series(path: Path, arrays: dict, time, rate) -> TimeSeries:
    columns: list[np.ndarray] = []
    names: list[str] = []
    for key, array in arrays.items():
        array = np.asarray(array)
        if array.dtype.kind not in "biuf" or array.ndim not in (1, 2):
            continue
        if array.ndim == 1:
            columns.append(array)
            names.append(key or "ch0")
        else:
            for i in range(array.shape[1]):
                columns.append(array[:, i])
                names.append(f"{key or 'ch'}{i}")
    if not columns:
        raise ValueError(f"{path}: no 1-D or 2-D numeric arrays in it")
    length = len(columns[0])
    if any(len(c) != length for c in columns):
        raise ValueError(f"{path}: its arrays are of different lengths")
    if time is not None:
        time = np.asarray(time, dtype=float)
        if time.shape != (length,):
            raise ValueError(f"{path}: {len(time)} times for {length} samples")
        sample_rate = float(rate) if rate else None
    elif rate:
        time = np.arange(length) / float(rate)
        sample_rate = float(rate)
    else:
        raise ValueError(f"{path}: no time or rate in it; give the sample rate")
    return TimeSeries(
        name=path.stem,
        time=time,
        values=np.stack(columns, axis=1).astype(float),
        channels=names,
        sample_rate=sample_rate,
        metadata={"source": str(path)},
    )
