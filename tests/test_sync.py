"""Two signals in time: correlation, delay, coherence, and lining up clocks."""

import numpy as np
import pandas as pd
import pytest
from scipy import signal as sp

import mmp_sense
from mmp_sense import dsp
from mmp_sense.core.data import TimeSeries
from mmp_sense.core.session import Session
from mmp_sense.examples.two_mics import DRIFT, OFFSET_S, synthesize
from mmp_sense.sources.recording import SeriesSource

A, B = "mic:amplitude", "phoneB_mic:amplitude"


def _session(*series):
    session = Session()
    for one in series:
        session.add(SeriesSource(one))
    while session.poll():
        pass
    return session


@pytest.fixture(scope="module")
def pair():
    return synthesize(20.0)


@pytest.fixture
def mics(pair):
    return _session(*pair)


def _fractional(delay_samples, n=8000, seed=0, noise=0.0):
    """Noise, and the same noise `delay_samples` later (between samples)."""
    rng = np.random.default_rng(seed)
    # Drawn ten times finer, and band-limited as a microphone would be.
    fine = sp.sosfiltfilt(sp.butter(8, 0.04, output="sos"), rng.standard_normal(n * 10 + 4000))
    whole = int(round(delay_samples * 10))
    x = fine[2000:2000 + n * 10:10]
    y = fine[2000 - whole:2000 - whole + n * 10:10] + noise * rng.standard_normal(n)
    return x, y


# -- the arithmetic -----------------------------------------------------------


@pytest.mark.parametrize("method", ["phat", "plain"])
@pytest.mark.parametrize("delay", [0.0, 3.0, -12.4, 37.3])
def test_a_delay_is_found_to_a_quarter_sample(method, delay):
    x, y = _fractional(delay, noise=0.2)
    lags, correlation = dsp.cross_correlation(x, y, 16000.0, max_lag=0.005, method=method)
    lag, peak, snr = dsp.peak_lag(lags, correlation)
    assert lag * 16000 == pytest.approx(delay, abs=0.25)
    assert snr > 10


def test_plain_correlation_is_one_for_a_copy():
    x, _ = _fractional(0.0)
    lags, correlation = dsp.cross_correlation(x, x, 1000.0, max_lag=0.01, method="plain")
    assert correlation.max() == pytest.approx(1.0)
    assert lags[np.argmax(correlation)] == 0.0


@pytest.mark.parametrize("lags", [1, 2])
def test_a_tight_search_still_finds_its_peak(lags):
    x, y = _fractional(0.0, noise=0.1)
    found = dsp.peak_lag(*dsp.cross_correlation(x, y, 1000.0, max_lag=lags / 1000.0,
                                                method="plain"))
    assert found[0] == pytest.approx(0.0, abs=0.25e-3) and found[1] > 0.5


def test_windows_sharing_a_clap_count_once():
    from mmp_sense.core.query import _apart
    assert _apart(np.array([0.5, 1.0, 1.5, 2.0, 2.5]), 1.0) == 3


def test_a_line_through_points_ignores_the_odd_one_out():
    t = np.arange(20.0)
    v = 0.5 + 0.01 * t
    v[7] = 9.0
    slope, intercept, kept = dsp.robust_line(t, v)
    assert slope == pytest.approx(0.01) and intercept == pytest.approx(0.5)
    assert not kept[7] and kept.sum() == 19


# -- the questions ------------------------------------------------------------


def test_the_delay_is_how_much_later_the_second_hears(mics):
    found = mics.delay(A, B, 4.0, 6.0, max_lag="100ms")
    planted = OFFSET_S + DRIFT * 4.5            # the clap is at about 4.5 s
    assert found["delay"] == pytest.approx(planted, abs=1 / 16000 / 4 + DRIFT * 0.2)
    back = mics.delay(B, A, 4.0, 6.0, max_lag="100ms")
    assert back["delay"] == pytest.approx(-found["delay"], abs=1 / 16000 / 4)


def test_the_correlation_is_a_series_by_lag(mics):
    correlation = mics.correlation(A, B, 4.0, 6.0, max_lag=0.05)
    assert correlation.index.name == "lag_s"
    assert correlation.index[0] == pytest.approx(-0.05, abs=1 / 16000)
    assert correlation.idxmax() == pytest.approx(0.037, abs=0.001)


def test_the_selection_is_the_stretch(mics):
    mics.set_selection((4.0, 6.0))
    assert mics.delay(A, B, max_lag="100ms")["delay"] == pytest.approx(
        mics.delay(A, B, 4.0, 6.0, max_lag="100ms")["delay"])


def test_streams_of_two_rates_are_compared_on_the_first_ones_times():
    session = _session(*synthesize(10.0, 16000.0, rate_b=8000.0))
    found = session.delay(A, B, 4.0, 6.0, max_lag="100ms")
    assert found["delay"] == pytest.approx(OFFSET_S + DRIFT * 4.5, abs=1 / 8000)


def test_coherence_is_high_where_the_sound_is_shared():
    rng = np.random.default_rng(1)
    rate, n = 8000.0, 80000
    shared = sp.sosfiltfilt(sp.butter(8, 1000, fs=rate, output="sos"), rng.standard_normal(n))
    time = np.arange(n) / rate
    session = mmp_sense.from_frame(pd.DataFrame({
        "x": shared + 0.05 * rng.standard_normal(n),
        "y": 0.7 * shared + 0.05 * rng.standard_normal(n)}, index=time))
    coherence = session.coherence("x", "y", segment=512)
    assert coherence.loc[100:800].min() > 0.9
    assert coherence.loc[2500:3800].max() < 0.3


# -- clocks -------------------------------------------------------------------


def test_the_offset_and_drift_of_a_clock_are_estimated(mics):
    estimate = mics.estimate_sync(A, B, max_lag="100ms")
    planted = OFFSET_S + DRIFT * estimate.at
    assert estimate.offset == pytest.approx(planted, abs=50e-6)
    assert estimate.drift_ppm == pytest.approx(DRIFT * 1e6, abs=5)
    assert estimate.used >= 15
    assert set(estimate.windows.columns) >= {"time", "delay", "peak", "snr", "used"}


def test_synchronizing_puts_the_two_on_one_time_line(mics):
    estimate = mics.synchronize(A, B, 0.0, 12.0, max_lag="100ms")
    assert estimate.used >= 9
    clock = mics.clock_of("phoneB_mic")
    assert clock.drift * 1e6 == pytest.approx(80, abs=5)
    for start in (13.0, 17.0):                   # claps not used for the fit
        assert abs(mics.delay(A, B, start, start + 2.0, max_lag="100ms")["delay"]) < 1 / 16000
    assert mics.clock_of("mic") is None


def test_synchronizing_again_refines_rather_than_starts_over(mics):
    mics.synchronize(A, B, max_lag="100ms")
    first = mics.clock_of("phoneB_mic")
    again = mics.synchronize(A, B, max_lag="10ms")
    assert abs(again.offset) < 20e-6
    second = mics.clock_of("phoneB_mic")
    assert second.offset + second.drift * (again.at - second.at) == pytest.approx(
        first.offset + first.drift * (again.at - first.at), abs=20e-6)


def test_a_clock_put_back_gives_the_source_times_exactly(mics, pair):
    buffer = mics.streams["phoneB_mic"].buffer
    rate = buffer.rate
    mics.set_clock("phoneB_mic", 0.037, 80.0, at=1.0)
    time, _ = buffer.snapshot()
    np.testing.assert_allclose(time, pair[1].time - 0.037 - 80e-6 * (pair[1].time - 1.0))
    assert buffer.rate == pytest.approx(rate / (1 - 80e-6))
    mics.set_clock("phoneB_mic")
    time, _ = buffer.snapshot()
    np.testing.assert_array_equal(time, pair[1].time)
    assert buffer.rate == rate


def test_samples_arriving_after_a_correction_are_corrected_too(pair):
    clock = [0.0]
    sources = [SeriesSource(x, replay=True, clock=lambda: clock[0]) for x in pair]
    session = Session()
    for source in sources:
        session.add(source)
    clock[0] = 5.0
    session.poll()
    session.set_clock("phoneB_mic", 0.037)
    clock[0] = 10.0
    session.poll()
    time, _ = session.streams["phoneB_mic"].buffer.snapshot()
    np.testing.assert_allclose(time, pair[1].time[:len(time)] - 0.037)
    assert len(time) > 5.0 * 16000


def test_a_weak_estimate_is_not_applied():
    rng = np.random.default_rng(3)
    time = np.arange(5 * 8000) / 8000.0
    unrelated = [TimeSeries(name=f"/{n}/mic", time=time, channels=["amplitude"],
                            values=rng.standard_normal((len(time), 1)), sample_rate=8000.0)
                 for n in ("p", "q")]
    session = _session(*unrelated)
    with pytest.raises(ValueError, match="not applied"):
        session.synchronize("mic:amplitude", "q_mic:amplitude", max_lag="50ms")
    assert session.clock_of("q_mic") is None


def test_silence_has_no_delay(mics):
    lags, correlation = dsp.cross_correlation(np.zeros(100), np.zeros(100), 100.0)
    assert np.isnan(dsp.peak_lag(lags, correlation)).all()
    quiet = mmp_sense.from_frame(pd.DataFrame({"x": np.zeros(400), "y": np.ones(400)},
                                        index=np.arange(400) / 100.0))
    assert np.isnan(quiet.delay("x", "y")["delay"])
    from mmp_sense.mcp.tools import call
    assert call(quiet, "get_delay", {"x": "x", "y": "y"})["delay"] is None


def test_a_mistake_in_asking_is_said_not_swallowed(mics):
    with pytest.raises(ValueError, match="unknown method"):
        mics.estimate_sync(A, B, method="foo")
    with pytest.raises(ValueError):
        mics.estimate_sync(A, B, max_lag="soon")
    with pytest.raises(ValueError, match="one signal"):
        mics.estimate_sync(A, mmp_sense.signal("phoneB_mic:amplitude", "phoneB_mic:amplitude"))


def test_a_delay_recipe_follows_a_corrected_clock(pair):
    clock = [0.0]
    sources = [SeriesSource(x, replay=True, clock=lambda: clock[0]) for x in pair]
    live = Session()
    for source in sources:
        live.add(source)
    delay = mmp_sense.signal(A).delay(mmp_sense.signal(B), "200ms", "50ms", max_lag="60ms").named("lag")
    live.show(delay)
    for _ in range(40):
        clock[0] += 0.25
        live.poll()
        live.window().home("lag")["lag"]                 # a tick draws it
        if clock[0] == 5.0:
            live.set_clock("phoneB_mic", 0.037, 80.0)
    whole = _session(*pair)
    whole.set_clock("phoneB_mic", 0.037, 80.0)
    got = live.values(delay)
    expected = whole.values(delay).loc[:got.index[-1]]
    np.testing.assert_array_equal(got.index, expected.index)
    np.testing.assert_allclose(got.to_numpy(), expected.to_numpy(), equal_nan=True)
    assert abs(np.nanmedian(live.values(delay).to_numpy())) < 1e-3


def test_one_stream_has_one_clock(mics):
    with pytest.raises(ValueError, match="one stream"):
        mics.synchronize(A, A)


# -- the delay as a recipe ------------------------------------------------------


def test_the_delay_recipe_follows_the_claps(mics):
    delay = mmp_sense.signal(A).delay(mmp_sense.signal(B), "200ms", "50ms", max_lag="60ms")
    values = mics.values(delay)
    # Frames both microphones heard a clap through: B's level over a frame.
    level = mics.values(mmp_sense.signal(B).rms("200ms")).reindex(values.index, method="ffill")
    at_claps = values[(level > 2 * level.median()).to_numpy(bool)]
    assert len(at_claps) > 20
    assert (at_claps - (OFFSET_S + DRIFT * at_claps.index)).abs().median() < 1 / 16000


def test_which_microphone_heard_first_is_a_detector():
    # Two microphones of one array: claps from the left, then the right.
    rate, n = 16000.0, 16000 * 8
    rng = np.random.default_rng(5)
    left = 0.002 * rng.standard_normal(n)
    right = 0.002 * rng.standard_normal(n)
    clap = rng.standard_normal(1600) * np.exp(-np.arange(1600) / 300)
    for i, at in enumerate(np.arange(0.5, 8.0, 1.0)):
        k = int(at * rate)
        late = 16                                  # 1 ms
        if i % 2 == 0:                             # nearer the left one
            left[k:k + 1600] += clap
            right[k + late:k + late + 1600] += clap
        else:
            right[k:k + 1600] += clap
            left[k + late:k + late + 1600] += clap
    time = np.arange(n) / rate
    array = mmp_sense.from_frame(pd.DataFrame({"left": left, "right": right}, index=time))
    delay = mmp_sense.signal("left").delay(mmp_sense.signal("right"), "60ms", "10ms", max_lag="5ms")
    loud = mmp_sense.signal("left").rms("20ms") > 0.2
    left_first = ((delay > 0.0005) & loud).events(refractory="500ms", name="left_first")
    right_first = ((delay < -0.0005) & loud).events(refractory="500ms", name="right_first")
    assert np.round(array.events(left_first)["start"].to_numpy() - 0.5) .tolist() == [0, 2, 4, 6]
    assert np.round(array.events(right_first)["start"].to_numpy() - 0.5).tolist() == [1, 3, 5, 7]


# -- the window and the assistant ---------------------------------------------


def test_the_correlation_pane_shows_the_delay(qapp, mics):
    pytest.importorskip("pyqtgraph")
    from mmp_sense.ui.main_window import MainWindow

    window = MainWindow(mics, console=False)
    try:
        pane = window.add_pane("correlation", A)
        assert pane.other.currentText() == B             # another stream's, by default
        mics.set_selection((4.0, 6.0))
        window.tick()
        assert pane.lag == pytest.approx(OFFSET_S + DRIFT * 4.5, abs=1e-4)
        assert "delay +37." in pane.status.text()
        pane.set_other(A)
        assert pane.lag == pytest.approx(0.0, abs=1 / 16000)
        assert pane.describe()["other"] == A
    finally:
        window.close()


def test_the_assistant_can_ask_for_a_delay(mics):
    from mmp_sense.mcp.tools import call

    found = call(mics, "get_delay", {"x": A, "y": B, "t0": 4.0, "t1": 6.0, "max_lag": 0.1})
    assert found["delay"] == pytest.approx(0.0374, abs=2e-4)
    assert found["snr"] > 10
