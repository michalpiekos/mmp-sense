"""The DSP functions, judged against signals whose answers are known."""

import numpy as np
import pytest

from mmp_sense import dsp


def test_magnitude_of_a_right_triangle():
    values = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
    assert list(dsp.magnitude(values)) == pytest.approx([5.0, 5.0])


def test_fft_finds_a_known_sinusoid_and_its_amplitude():
    rate = 1000.0
    time = np.arange(int(2 * rate)) / rate
    values = 3.0 * np.sin(2 * np.pi * 137.0 * time)

    frequency, amplitude = dsp.amplitude_spectrum(values, rate)

    peak = frequency[int(np.argmax(amplitude[:, 0]))]
    assert peak == pytest.approx(137.0, abs=0.5)
    assert amplitude[:, 0].max() == pytest.approx(3.0, rel=0.02)


def test_fft_separates_two_tones():
    rate = 500.0
    time = np.arange(int(4 * rate)) / rate
    values = np.sin(2 * np.pi * 20 * time) + 0.5 * np.sin(2 * np.pi * 80 * time)

    frequency, amplitude = dsp.amplitude_spectrum(values, rate)
    column = amplitude[:, 0]

    # The two largest *bins* are neighbours of one peak, so the peaks are
    # found as peaks and then ranked.
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(column, height=0.1)
    two = peaks[np.argsort(column[peaks])[-2:]]
    assert set(np.round(frequency[two])) == {20.0, 80.0}
    # Amplitudes come back as they went in, 1.0 and 0.5.
    assert sorted(np.round(column[two], 2)) == [0.5, 1.0]


def test_fft_rejects_an_unknown_window():
    with pytest.raises(ValueError, match="unknown window"):
        dsp.amplitude_spectrum(np.zeros(100), 100.0, window="triangle")


def test_spectrogram_shape_and_coordinates():
    rate = 200.0
    time = np.arange(int(10 * rate)) / rate
    values = np.sin(2 * np.pi * 30 * time)

    frequency, times, magnitude = dsp.spectrogram(
        values, rate, nperseg=128, overlap=0.5
    )

    assert magnitude.shape == (len(frequency), len(times))
    assert frequency[0] == 0.0
    assert frequency[-1] == pytest.approx(rate / 2)
    assert times[-1] < 10.0
    # The tone is where it should be, in every column.
    dominant = frequency[np.argmax(magnitude, axis=0)]
    assert np.median(dominant) == pytest.approx(30.0, abs=rate / 128)


def test_synthetic_accelerometer_reports_its_events():
    time, values, events = dsp.synthetic_accelerometer(
        duration=10.0, rate=100.0, n_events=3, seed=1
    )
    assert values.shape == (1000, 3)
    assert len(events) == 3
    assert all(0 < start < end < 10 for start, end in events)
    # Gravity sits on z and nowhere else.
    assert values[:, 2].mean() == pytest.approx(9.81, abs=0.1)
    assert abs(values[:, 0].mean()) < 0.5


# -- spectral metrics ------------------------------------------------------------


def one_frame(x, rate=1000.0, taper="hann"):
    frequency, amplitude = dsp.framed_spectrum(np.asarray(x, float), rate, np.array([0]), len(x), taper)
    return frequency, amplitude, dsp.enbw(taper, len(x))


def sines(*parts, rate=1000.0, n=1000):
    t = np.arange(n) / rate
    return sum(a * np.sin(2 * np.pi * f * t) for f, a in parts)


@pytest.mark.parametrize("taper, n", [("hann", 1000), ("hann", 250), ("hamming", 400), ("boxcar", 500)])
def test_a_sine_carries_half_its_amplitude_squared(taper, n):
    f, a, bw = one_frame(sines((90, 3.0), n=n), taper=taper)
    assert dsp.band_power(f, a, 70, 110, bw)[0] == pytest.approx(4.5, rel=0.02)
    assert dsp.band_power(f, a, 200, 400, bw)[0] < 0.01
    assert dsp.band_power(f, a, 70, 110, bw, relative=True)[0] == pytest.approx(1.0, abs=0.01)


def test_band_power_rejects_an_upside_down_band():
    f, a, bw = one_frame(sines((90, 1.0)))
    with pytest.raises(ValueError):
        dsp.band_power(f, a, 100, 50, bw)


def test_centroid_spread_and_rolloff_of_two_sines():
    f, a, bw = one_frame(sines((50, 1.0), (200, 2.0)))
    # powers 0.5 and 2: centre (25 + 400) / 2.5, spread sqrt((0.5*120^2 + 2*30^2) / 2.5)
    assert dsp.spectral_centroid(f, a, bw)[0] == pytest.approx(170.0, abs=1.0)
    assert dsp.spectral_spread(f, a, bw)[0] == pytest.approx(60.0, abs=1.0)
    assert dsp.spectral_rolloff(f, a, bw, 0.85)[0] == pytest.approx(200.0, abs=2.0)
    assert dsp.spectral_rolloff(f, a, bw, 0.1)[0] == pytest.approx(50.0, abs=2.0)


def test_noise_is_flat_and_a_tone_is_not():
    noise = np.random.default_rng(3).normal(size=1000)
    assert dsp.spectral_flatness(*one_frame(noise))[0] > 0.5
    assert dsp.spectral_flatness(*one_frame(sines((90, 1.0))))[0] < 0.05


def test_silence_has_no_shape():
    f, a, bw = one_frame(np.zeros(100))
    for metric in (dsp.spectral_centroid, dsp.spectral_spread, dsp.spectral_flatness):
        assert np.isnan(metric(f, a, bw)[0])
    assert np.isnan(dsp.spectral_rolloff(f, a, bw)[0])
    assert np.isnan(dsp.band_power(f, a, 0, 500, bw, relative=True)[0])
    assert dsp.band_power(f, a, 0, 500, bw)[0] == 0.0


def test_flux_is_still_for_a_steady_tone_and_jumps_at_a_burst():
    rate, width = 1000.0, 100
    x = sines((50, 1.0), n=3000)
    x[1500:1600] += sines((300, 5.0), n=100)
    starts = dsp.frame_starts(len(x), width, width)
    f, a = dsp.framed_spectrum(x, rate, starts, width)
    bw = dsp.enbw("hann", width)
    for normalize in (False, True):
        flux = dsp.spectral_flux(f, a, bw, normalize)
        assert np.isnan(flux[0])
        assert np.nanmax(flux[:14]) < 0.05
        assert np.argmax(np.nan_to_num(flux)) == 15
    onset = dsp.spectral_flux(f, a, bw)
    assert onset[15] == pytest.approx(5.0, rel=0.05)     # the burst's amplitude arrives
    assert onset[16] < 0.05                               # it leaving is no onset


def test_normalised_flux_ignores_loudness_and_onset_flux_does_not():
    rate, width = 1000.0, 100
    x = sines((50, 1.0), n=2000)
    x[1000:] *= 4.0
    starts = dsp.frame_starts(len(x), width, width)
    f, a = dsp.framed_spectrum(x, rate, starts, width)
    bw = dsp.enbw("hann", width)
    assert dsp.spectral_flux(f, a, bw, normalize=True)[10] < 0.05
    assert dsp.spectral_flux(f, a, bw)[10] == pytest.approx(3.0, rel=0.05)


# -- lines and what stays ----------------------------------------------------


def test_a_tone_over_noise_is_a_line_and_carries_its_power():
    noise = 0.01 * np.random.default_rng(4).normal(size=1000)
    f, a, bw = one_frame(sines((90, 3.0), (300, 1.0)) + noise)
    assert dsp.line_power(f, a, 50, 150, bw)[0] == pytest.approx(4.5, rel=0.02)
    assert dsp.line_power(f, a, 50, 400, bw)[0] == pytest.approx(5.0, rel=0.02)
    assert dsp.line_power(f, a, 50, 400, bw, relative=True)[0] > 0.95


@pytest.mark.parametrize("colour", ["white", "brown"])
def test_noise_has_almost_no_lines_however_its_floor_slopes(colour):
    rng = np.random.default_rng(6)
    x = rng.normal(size=4000)
    if colour == "brown":
        x = np.cumsum(x)            # falls 6 dB an octave: low bins hold the power
    starts = dsp.frame_starts(len(x), 1000, 100)
    f, a = dsp.framed_spectrum(x, 1000.0, starts, 1000)
    share = dsp.line_power(f, a, 50, 400, dsp.enbw("hann", 1000), relative=True)
    assert np.max(share) < 0.1


def test_loud_noise_over_a_line_is_floor_not_line():
    rng = np.random.default_rng(7)
    f, a, bw = one_frame(sines((200, 0.2)) + rng.normal(size=1000))
    loud = dsp.line_power(f, a, 100, 300, bw)[0]
    assert loud < 0.05 + 0.02          # only the tone (0.02), and noise by chance
    with pytest.raises(ValueError):
        dsp.line_power(f, a, 300, 100, bw)
    with pytest.raises(ValueError):
        dsp.line_power(f, a, 100, 300, bw, above=1.0)


def test_lines_of_silence_and_of_a_gap():
    f, a, bw = one_frame(np.zeros(100))
    assert dsp.line_power(f, a, 10, 400, bw)[0] == 0.0
    assert np.isnan(dsp.line_power(f, a, 10, 400, bw, relative=True)[0])
    a = np.vstack([a, a])
    a[1, 3] = np.nan
    assert np.isnan(dsp.line_power(f, a, 10, 400, bw)[1])


def test_trailing_min_looks_only_backwards_bin_by_bin():
    x = np.random.default_rng(2).normal(size=(60, 4))
    for width in (1, 3, 7):
        out = dsp.trailing_min(x, width)
        assert np.isnan(out[:width - 1]).all()
        for i in range(width - 1, len(x)):
            assert np.allclose(out[i], x[i - width + 1:i + 1].min(axis=0))
    x[30, 2] = np.nan
    out = dsp.trailing_min(x, 3)
    assert np.isnan(out[30:33]).all() and np.isfinite(out[33]).all()
