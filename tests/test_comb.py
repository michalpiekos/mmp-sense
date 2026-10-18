"""Combs: a repetition rate, lines followed and listed, drift divided out,
and spacings and rates scored against chance -- on sounds whose answers
are known."""

import numpy as np
import pandas as pd
import pytest

import mmp_sense
from mmp_sense import signal
from mmp_sense.analysis import comb_scan, fit_rates, heights, permutation_lift

RATE = 16000
SECONDS = 12.0
#: A light plane, made up: an engine's cycle X and a
#: propeller's blade rate B, geared so 2X / (B / 3) is 2.428, both
#: falling 3 % over the recording.
X0 = 37.73
B0 = 6 * X0 / 2.428
FALL = 0.03
LATTICE = list(dict.fromkeys(
    [(n, 0) for n in range(4, 25)] + [(14, k) for k in (-1, 1, 2, 3, 4)]
    + [(12, 1), (0, 2), (10, -1)]))


def _glide(t):
    return 1.0 - FALL * t / SECONDS


def _session(values, time):
    return mmp_sense.from_frame(pd.DataFrame({"mic": values}, index=time))


def _phase(hz):
    return 2 * np.pi * np.cumsum(hz) / RATE


@pytest.fixture(scope="module")
def plane():
    rng = np.random.default_rng(3)
    t = np.arange(int(SECONDS * RATE)) / RATE
    x_phase, b_phase = _phase(X0 * _glide(t)), _phase(B0 * _glide(t))
    values = rng.standard_normal(len(t))
    for n, k in LATTICE:
        values += rng.uniform(0.15, 0.4) * np.sin(n * x_phase + k * b_phase + rng.uniform(0, 6))
    return _session(values, t)


def _comb(f0_of_t, harmonics, *, seconds=4.0, noise=1.0, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    phase = _phase(np.broadcast_to(f0_of_t(t), t.shape))
    values = noise * rng.standard_normal(len(t))
    for k, a in harmonics.items():
        values += a * np.sin(k * phase + rng.uniform(0, 6))
    return _session(values, t)


RATE_OF_THE_PLANE = signal("mic").frames(1.0, 0.25).repetition_hz(
    12.5, 33, low_hz=150, high_hz=1000)


@pytest.fixture(scope="module")
def ref(plane):
    """What the plane's rate reads at 0 s. Not quite X / 2: the comb it
    finds is the two rates nearly in step, a fixed fraction off."""
    rate = plane.values(RATE_OF_THE_PLANE).dropna()
    middle = rate.index.to_numpy() - 0.5
    return float(np.median(rate.to_numpy() / _glide(middle)))


# -- the rate ----------------------------------------------------------------


def test_a_comb_repeats_at_its_spacing():
    session = _comb(lambda t: 18.25, {k: 0.3 for k in range(8, 51)})
    rate = session.repetition_hz("mic", min_hz=12.5, max_hz=33, low_hz=150, high_hz=1000)
    assert rate == pytest.approx(18.25, abs=0.02)


@pytest.mark.parametrize("harmonics, low, expected", [
    ({k: (0.6 if k % 2 == 0 else 0.12) for k in range(8, 50)}, 12.5, 20.0),  # odd ones weaker
    ({k: (0.6 if k % 2 == 0 else 0.02) for k in range(8, 50)}, 12.5, 20.0),  # odd ones faint
    ({k: 0.4 for k in range(8, 50, 2)}, 15.0, 40.0),                        # no odd ones
    ({k: (0.6 if k % 2 == 0 else 0.008) for k in range(8, 50)}, 12.5, 40.0),  # lost in noise
], ids=["odd weaker", "odd faint", "no odd", "odd lost"])
def test_the_rate_is_neither_half_nor_double(harmonics, low, expected):
    session = _comb(lambda t: 20.0, harmonics)
    rate = session.repetition_hz("mic", min_hz=low, max_hz=60, low_hz=150, high_hz=1000)
    assert rate == pytest.approx(expected, rel=0.002)


def test_the_guard_keeps_to_the_range():
    # Lines at a sixth of a ladder's spacing make half of it a comb too.
    session = _comb(lambda t: 120.0, {**{k: 0.4 for k in range(2, 12)},
                                      **{k / 6: 0.15 for k in range(7, 66) if k % 6}})
    rate = session.repetition_hz("mic", min_hz=90, max_hz=140, low_hz=150, high_hz=1500)
    assert rate == pytest.approx(120.0, rel=0.002)
    assert session.repetition_hz("mic", min_hz=12.5, max_hz=33, low_hz=150,
                                 high_hz=1500) == pytest.approx(20.0, rel=0.002)


def test_a_rate_outside_the_range_is_none():
    session = _comb(lambda t: 130.0, {k: 0.4 for k in range(2, 9)})
    assert np.isnan(session.repetition_hz("mic", min_hz=90, max_hz=110, low_hz=150,
                                          high_hz=1500))


def test_noise_has_no_rate():
    found = [_comb(lambda t: 20.0, {}, seconds=2.0, seed=seed).repetition_hz(
        "mic", min_hz=12.5, max_hz=33, low_hz=150, high_hz=1000) for seed in range(40)]
    assert np.isnan(found).mean() >= 0.95


def test_the_plane_repeats_at_half_its_engine_cycle(plane):
    # 18.94 and 18.25 Hz: the two rates nearly in step.
    for t0, t1 in ((0.0, 4.0), (8.0, 12.0)):
        rate = plane.repetition_hz("mic", t0, t1, min_hz=12.5, max_hz=33, low_hz=150,
                                   high_hz=1000)
        assert rate == pytest.approx(X0 * _glide((t0 + t1) / 2) / 2, rel=0.004)


def test_frames_follow_a_gliding_rate():
    # 133 Hz falling to 103 Hz, as the second plane's comb does.
    session = _comb(lambda t: 133.0 - 30.0 * t / 12.0, {k: 0.4 for k in range(2, 9)},
                    seconds=12.0)
    rates = session.values(signal("mic").frames(0.5, 0.25).repetition_hz(
        60, 200, low_hz=150, high_hz=1500))
    truth = 133.0 - 30.0 * (rates.index.to_numpy() - 0.25) / 12.0
    assert np.nanmedian(np.abs(rates.to_numpy() / truth - 1)) < 0.01
    assert np.isfinite(rates.to_numpy()).mean() > 0.9


# -- a line, followed --------------------------------------------------------


def test_a_line_is_followed_near_a_recipe_while_it_glides(plane):
    line = signal("mic").frames(1.0, 0.25).peak_hz(100, 1000, near=RATE_OF_THE_PLANE * 28,
                                                   within=3)
    found = plane.values(line)
    middle = found.index.to_numpy() - 0.5
    error = found.to_numpy() - 14 * X0 * _glide(middle)
    assert np.isfinite(error).mean() > 0.9
    assert np.nanmedian(np.abs(error)) < 0.1


def test_a_line_near_a_number_and_in_a_band():
    session = _comb(lambda t: 100.0, {3: 1.0, 5: 0.5})
    frames = signal("mic").frames(1.0)
    near = session.values(frames.peak_hz(0, 1000, near=497.0, within=10)).dropna()
    assert near.to_numpy() == pytest.approx(500.0, abs=0.1)
    strongest = session.values(frames.peak_hz(0, 1000)).dropna()
    assert strongest.to_numpy() == pytest.approx(300.0, abs=0.1)


def test_harmonics_of_a_drone_keep_their_ratio():
    session = _comb(lambda t: 200.0 - 20.0 * t / 4.0, {1: 0.5, 2: 0.5}, seconds=4.0)
    frames = signal("mic").frames(0.5, 0.25)
    first = frames.peak_hz(150, 260)
    second = frames.peak_hz(300, 520, near=first * 2, within=10)
    ratio = (session.values(second) / session.values(first)).dropna()
    assert np.abs(ratio.to_numpy() - 2.0).max() < 0.01


def test_ops_refuse_what_is_not_frames():
    session = _comb(lambda t: 100.0, {1: 1.0}, seconds=1.0)
    with pytest.raises(Exception, match="frames"):
        session.values(signal("mic").peak_hz(0, 500))


# -- lines, listed and measured -------------------------------------------------


def test_lines_are_listed_with_their_prominence_and_height(plane):
    lines = plane.lines("mic", 2.2, 4.5, low_hz=100, high_hz=1000, prominence=10)
    assert list(lines.columns) == ["hz", "prominence_db", "over_floor_db", "level_db"]
    assert (lines["prominence_db"] >= 10).all()
    assert 22 <= len(lines) <= 30
    assert (lines["over_floor_db"] > 6).mean() > 0.9


def test_lines_near_given_frequencies_are_precise():
    session = _comb(lambda t: 100.0, {3: 0.5, 7: 0.5}, seconds=1.5)
    table = session.lines("mic", near=[301.0, 698.6, 950.0], within=1.5)
    assert list(table["near"]) == [301.0, 698.6, 950.0]
    assert table["hz"].iloc[:2].to_numpy() == pytest.approx([300.0, 700.0], abs=0.05)
    assert table["over_floor_db"].iloc[2] < 10                       # nothing at 950


# -- drift divided out ---------------------------------------------------------


def test_aligned_frames_keep_a_gliding_comb_steady(plane, ref):
    frames = signal("mic").frames(1.0, 0.25)
    plain = plane.values(frames.steady(4.0).line_share(150, 1000)).median()
    aligned = plane.values(frames.align(RATE_OF_THE_PLANE, ref).steady(4.0)
                           .line_share(150, 1000)).median()
    assert aligned > plain + 0.1


def test_an_aligned_psd_gathers_a_gliding_line(plane, ref):
    plain = plane.psd("mic", 1, 12, segment=2**15)
    aligned = plane.psd("mic", 1, 12, segment=2**15, align=RATE_OF_THE_PLANE, ref_hz=ref)
    line = 14 * X0
    assert aligned.loc[line - 5:line + 5].idxmax() == pytest.approx(line, abs=0.5)
    assert aligned.loc[500:560].max() > 4 * plain.loc[500:560].max()


def test_orders_index_the_spectrum_by_multiples_of_the_rate(plane, ref):
    orders = plane.orders("mic", RATE_OF_THE_PLANE, 1, 12)
    assert orders.index.name == "order"
    # The 14th multiple of X, in multiples of the rate as measured.
    assert orders.loc[27.5:28.5].idxmax() == pytest.approx(14 * X0 / ref, abs=0.005)


def test_a_stretch_is_aligned_by_the_middle_of_each_frame(plane, ref):
    # A frame's rate describes its middle; read at its end it would lag.
    line = 14 * X0
    measured = plane.lines("mic", 1, 12, near=[line], align=RATE_OF_THE_PLANE, ref_hz=ref)
    assert measured["hz"].iloc[0] == pytest.approx(line, abs=0.1)


def test_a_gap_is_filled_with_the_last_value_for_so_long():
    time = np.arange(10) / 10.0
    values = np.array([1.0, np.nan, np.nan, 2.0, np.nan, np.nan, np.nan, np.nan, 3.0, np.nan])
    session = mmp_sense.from_frame(pd.DataFrame({"r": values}, index=time))
    held = session.values(signal("r").fill(0.25)).to_numpy()
    np.testing.assert_array_equal(held, [1, 1, 1, 2, 2, 2, np.nan, np.nan, 3, 3])
    assert np.isfinite(session.values(signal("r").fill()).to_numpy()).all()


# -- spacings and rates against chance ----------------------------------------------


@pytest.fixture(scope="module")
def aligned_lines(plane, ref):
    return plane.lines("mic", 1, 12, low_hz=100, high_hz=1000, prominence=10,
                       align=RATE_OF_THE_PLANE, ref_hz=ref)


def test_two_rates_explain_the_lines_far_beyond_chance(aligned_lines):
    fit = fit_rates(aligned_lines, 37.7, 93.2, band=(100, 1000))
    assert fit.x == pytest.approx(X0, abs=0.05)
    assert fit.b == pytest.approx(B0, abs=0.1)
    assert fit.fitted >= 0.9 * len(aligned_lines)
    assert fit.chance < fit.fitted / 3 and fit.chance_max < fit.fitted
    assert set(fit.table.columns) == {"hz", "n", "k", "off", "fits"}


def test_the_rates_keep_their_gearing_stretch_after_stretch(plane, aligned_lines):
    fit = fit_rates(aligned_lines, 37.7, 93.2, band=(100, 1000))
    usable = fit.table[fit.table["fits"]]
    for t in (2.0, 6.0, 10.0):
        moved = _glide(t)
        measured = plane.lines("mic", t - 1.0, t + 1.0, near=usable["hz"] * moved, within=1.5)
        again = fit_rates(measured, fit.x * moved, fit.b * moved,
                          nk=usable[["n", "k"]].to_numpy())
        assert again.x == pytest.approx(X0 * moved, rel=0.002)
        assert 2 * again.x / (again.b / 3) == pytest.approx(2.428, abs=0.005)


def test_rates_refitted_with_a_placing_take_every_line_given():
    nk = np.array([(14, 0), (14, 1), (14, 2), (4, 0), (0, 2)])
    lines = nk @ (36.0, 90.0)
    again = fit_rates(lines, 37.73, 93.25, nk=nk)
    assert (again.x, again.b) == pytest.approx((36.0, 90.0))
    with pytest.raises(ValueError):
        fit_rates([500.0, np.nan], 37.0, 93.0, nk=[(14, 0), (14, 1)])


def test_a_scan_puts_the_spacings_that_explain_most_first(aligned_lines):
    scan = comb_scan(aligned_lines, np.arange(25, 120, 0.01), top=3)
    assert list(scan.columns) == ["spacing", "lines", "chance", "p"]
    assert scan["spacing"].iloc[0] == pytest.approx(X0, abs=0.1)
    assert scan["p"].is_monotonic_increasing
    assert np.diff(np.sort(scan["spacing"])).min() > 0.5


def test_the_largest_spacing_that_fits_beats_its_half():
    lines = 18.25 * np.array([16, 22, 23, 24, 28, 29, 31, 33, 38]) + 0.1
    scan = comb_scan(lines, [9.125, 18.25, 36.5], apart=None).set_index("spacing")
    assert scan.loc[18.25, "lines"] == scan.loc[9.125, "lines"] == 9
    assert scan.loc[18.25, "p"] < scan.loc[9.125, "p"]
    assert scan.loc[36.5, "lines"] == 5                              # the even ones


def test_heights_and_a_permutation_test():
    frequency = np.arange(0, 1000, 0.25)
    rng = np.random.default_rng(0)
    power = rng.exponential(1.0, len(frequency))
    for m in range(1, 19):
        power[np.argmin(np.abs(frequency - 31.0 * m))] *= 30 if m % 3 == 0 else 1.5
    psd = pd.Series(power, index=frequency)
    stand = heights(psd, 31.0 * np.arange(1, 19))
    assert stand.index.name == "hz" and len(stand) == 18
    by_multiple = pd.Series(stand.to_numpy(), index=np.arange(1, 19))
    result = permutation_lift(by_multiple, lambda m: m % 3 == 0, tries=2000)
    assert result["lift"] > 6 and result["p"] < 0.01
    null = permutation_lift(by_multiple, lambda m: m % 3 == 1, tries=2000)
    assert null["p"] > 0.5


def test_a_permutation_test_needs_a_choice():
    with pytest.raises(ValueError):
        permutation_lift([1.0, 2.0], [True, True])
