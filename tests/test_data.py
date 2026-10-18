"""The data model: construction, validation and slicing."""

import numpy as np
import pytest

from mmp_sense.core.data import EventSeries, Spectrogram, Spectrum, TimeSeries


def test_timeseries_promotes_one_channel_to_two_dimensions():
    series = TimeSeries(time=[0.0, 0.1, 0.2], values=[1.0, 2.0, 3.0])
    assert series.values.shape == (3, 1)
    assert series.channels == ["c0"]
    assert series.n_samples == 3


def test_timeseries_names_and_units_survive():
    series = TimeSeries(
        name="accel",
        time=np.arange(4) / 100,
        values=np.zeros((4, 3)),
        channels=["ax", "ay", "az"],
        units="m/s^2",
    )
    assert series.n_channels == 3
    assert series.units == "m/s^2"
    assert series.preferred_view == "timeseries"


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(time=[0, 1], values=[[1.0]]), "timestamps"),
        (dict(time=[1, 0], values=[1.0, 2.0]), "non-decreasing"),
        (dict(time=[0, 1], values=[1.0, 2.0], channels=["a", "b"]), "channel names"),
    ],
)
def test_timeseries_rejects_inconsistent_input(kwargs, message):
    with pytest.raises(ValueError, match=message):
        TimeSeries(**kwargs)


def test_timeseries_rate_is_declared_or_derived():
    time = np.arange(100) / 50.0
    derived = TimeSeries(time=time, values=np.zeros(100))
    assert derived.rate == pytest.approx(50.0)
    assert derived.is_regular()

    declared = TimeSeries(time=time, values=np.zeros(100), sample_rate=48.0)
    assert declared.rate == 48.0


def test_timeseries_channel_by_name_and_index():
    series = TimeSeries(
        time=[0, 1], values=[[1.0, 2.0], [3.0, 4.0]], channels=["a", "b"]
    )
    assert list(series.channel("b")) == [2.0, 4.0]
    assert list(series.channel(0)) == [1.0, 3.0]
    with pytest.raises(KeyError, match="no channel"):
        series.channel("z")


def test_timeseries_slicing_is_inclusive_and_keeps_metadata():
    series = TimeSeries(
        name="s", time=np.arange(10) / 10.0, values=np.arange(10), units="g"
    )
    sliced = series.time_slice(0.2, 0.5)
    assert list(sliced.time) == pytest.approx([0.2, 0.3, 0.4, 0.5])
    assert sliced.units == "g"
    assert sliced.name == "s"
    assert series.n_samples == 10, "slicing must not mutate the original"


def test_timeseries_round_trips_through_a_dataframe():
    series = TimeSeries(
        time=[0.0, 0.5], values=[[1.0, 2.0], [3.0, 4.0]], channels=["x", "y"]
    )
    back = TimeSeries.from_dataframe(series.to_dataframe())
    assert back.channels == ["x", "y"]
    assert np.allclose(back.values, series.values)
    assert np.allclose(back.time, series.time)


def test_spectrum_reports_peak_and_resolution():
    spectrum = Spectrum(frequency=[0, 1, 2, 3], values=[0.1, 0.2, 5.0, 0.3])
    assert spectrum.peak_frequency() == 2.0
    assert spectrum.resolution == 1.0


def test_spectrogram_insists_on_its_dimension_order():
    data = Spectrogram(
        time=np.arange(5) * 0.1, frequency=np.arange(3.0), values=np.zeros((3, 5))
    )
    assert data.shape == (3, 5)
    assert data.time_slice(0.0, 0.2).shape == (3, 3)
    with pytest.raises(ValueError, match="do not match"):
        Spectrogram(
            time=np.arange(5.0), frequency=np.arange(3.0), values=np.zeros((5, 3))
        )


def test_spectrogram_decibels_are_floored_below_the_peak():
    data = Spectrogram(
        time=np.arange(2.0), frequency=np.arange(2.0),
        values=np.array([[1.0, 0.0], [0.5, 0.0]]),
    )
    decibels = data.decibels()
    assert decibels[0, 0] == pytest.approx(0.0)
    # Zero has no logarithm; it is floored relative to the peak, not dropped.
    assert np.isfinite(decibels).all()
    assert decibels[0, 1] == pytest.approx(-120.0)


def test_events_add_returns_a_copy():
    empty = EventSeries(name="events")
    one = empty.add(1.0, 2.0, label="step", confidence=0.8)
    assert len(empty) == 0 and len(one) == 1
    assert one.table.loc[0, "label"] == "step"

    two = one.add(5.0, 6.0, label="drone")
    assert list(two.in_range(4.0, 7.0).table["label"]) == ["drone"]
    # An overlapping region counts, not only a contained one.
    assert len(two.in_range(1.5, 1.6)) == 1


def test_events_fill_in_missing_columns():
    events = EventSeries.from_records([{"start_time": 1.0, "end_time": 2.0}])
    assert events.table.loc[0, "confidence"] == 1.0
    assert events.table.loc[0, "label"] == ""
