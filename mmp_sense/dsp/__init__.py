"""Signal processing as plain functions over NumPy arrays.

Nothing here knows about mmp-sense's data types, modules or graph: these take
arrays and a sample rate and return arrays. Modules wrap them, and that
is where names, units and provenance are attached. Keeping the two apart
means a function can be checked against a textbook without constructing
anything, and a module can be read without wondering where the maths is.
"""

from .comb import (
    GUARD_DB,
    align_frames,
    chance_fits,
    comb_counts,
    cut_cycles,
    cycle_count,
    fit_two_rates,
    line_peaks,
    order_resample,
    over_floor_db,
    place_lines,
    repetition_rate,
    spectrum_lines,
)
from .array import FRACTION_S, delay_and_sum, far_field, near_field, srp_map, steering
from .correlation import METHODS, check_method, coherence, cross_correlation, peak_lag, robust_line
from .envelope import envelope, envelope_delay, hilbert_taps
from .filters import magnitude
from .iir import KINDS as FILTERS
from .iir import design, group_delay, sos_filter
from .iir import response as filter_response
from .peaks import PEAK_SPREADS, nearest_bin, next_peak, refine_peak, snap_to_peak, step_bin
from .spectral import amplitude_spectrum, spectrogram
from .spectral_metrics import (
    band_power,
    enbw,
    line_power,
    power_bins,
    spectral_centroid,
    spectral_flatness,
    spectral_flux,
    spectral_rolloff,
    spectral_spread,
)
from .synth import synthetic_accelerometer
from .windowed import (
    dominant_frequency,
    frame_starts,
    framed_spectrum,
    highpass1,
    highpass1_alpha,
    lowpass1,
    lowpass1_alpha,
    trailing_max,
    trailing_min,
    trailing_sum,
    zero_crossing_rate,
)

__all__ = [
    "GUARD_DB",
    "align_frames",
    "chance_fits",
    "comb_counts",
    "cut_cycles",
    "cycle_count",
    "fit_two_rates",
    "line_peaks",
    "order_resample",
    "over_floor_db",
    "place_lines",
    "repetition_rate",
    "spectrum_lines",
    "PEAK_SPREADS",
    "nearest_bin",
    "next_peak",
    "refine_peak",
    "snap_to_peak",
    "step_bin",
    "FRACTION_S",
    "FILTERS",
    "delay_and_sum",
    "design",
    "far_field",
    "near_field",
    "srp_map",
    "steering",
    "envelope",
    "envelope_delay",
    "hilbert_taps",
    "filter_response",
    "group_delay",
    "sos_filter",
    "METHODS",
    "amplitude_spectrum",
    "band_power",
    "check_method",
    "coherence",
    "cross_correlation",
    "dominant_frequency",
    "enbw",
    "frame_starts",
    "framed_spectrum",
    "highpass1",
    "highpass1_alpha",
    "lowpass1",
    "line_power",
    "lowpass1_alpha",
    "magnitude",
    "peak_lag",
    "robust_line",
    "power_bins",
    "spectral_centroid",
    "spectral_flatness",
    "spectral_flux",
    "spectral_rolloff",
    "spectral_spread",
    "spectrogram",
    "synthetic_accelerometer",
    "trailing_max",
    "trailing_min",
    "trailing_sum",
    "zero_crossing_rate",
]
