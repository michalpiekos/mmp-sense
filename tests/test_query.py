"""The MCP tools: compact answers about what the user is looking at."""

import json

import numpy as np
import pytest

import mmp_sense
from mmp_sense.mcp import tools


def show_strength(session):
    strength = mmp_sense.signal().highpass(10).magnitude().peak("20ms", name="strength")
    session.show(strength, (strength > 1).events(name="knock"))


def test_statistics_default_to_the_selection_else_everything(session):
    session.set_selection((4.9, 5.1))
    answer = tools.call(session, "get_statistics", {"signal": "accel_z"})
    assert answer["stretch"] == [4.9, 5.1]
    assert answer["stats"]["n"] == 101
    session.clear_selection()
    answer = tools.call(session, "get_statistics", {"signal": "accel_z"})
    assert answer["stretch"] == pytest.approx(list(session.span))


def test_the_spectrum_summary_finds_the_knock(session):
    answer = tools.call(session, "get_spectrum_summary",
                        {"signal": "accel_z", "t0": 4.9, "t1": 5.15,
                         "bands": [[80, 100], [300, 400]]})
    summary = answer["summary"]
    assert summary["dominant_hz"] == pytest.approx(90.0, abs=5.0)
    assert summary["share 80-100 Hz"] > 0.3 > summary["share 300-400 Hz"]
    quiet = tools.call(session, "get_spectrum_summary",
                       {"signal": "accel_z", "t0": 1.0, "t1": 1.25})["summary"]
    assert quiet["flatness"] > summary["flatness"]


def test_values_are_an_envelope_that_keeps_the_peak(session):
    show_strength(session)
    answer = tools.call(session, "get_values", {"signal": "strength", "points": 50})
    assert len(answer["t"]) == 50
    peak = int(np.argmax(answer["max"]))
    assert answer["t"][peak] == pytest.approx(5.0, abs=0.2)


def test_events_of_a_detector(session):
    show_strength(session)
    answer = tools.call(session, "get_events", {"detector": "knock"})
    assert answer["count"] == 1
    assert answer["events"][0]["start"] == pytest.approx(5.0, abs=0.1)


def test_the_context_and_the_signals_name_the_detectors(session):
    show_strength(session)
    assert tools.call(session, "get_current_context")["detectors"] == ["knock"]
    listed = tools.call(session, "list_signals")
    assert "strength" in [s["name"] for s in listed["signals"]]


def test_every_answer_is_json_with_no_not_a_number(session):
    show_strength(session)
    args = {"get_statistics": {"signal": "strength"}, "get_values": {"signal": "strength"},
            "get_events": {"detector": "knock"},
            "get_spectrum_summary": {"signal": "accel_z", "bands": [[1, 2]]},
            "get_delay": {"x": "accel_x", "y": "accel_z", "max_lag": 0.05}}
    for name in tools.TOOLS:
        json.dumps(tools.call(session, name, args.get(name, {})), allow_nan=False)


def test_an_empty_session_answers_rather_than_raises():
    session = mmp_sense.Session(channels=["a"], rate=100.0)
    assert tools.call(session, "get_statistics", {"signal": "a"})["stats"]["n"] == 0
    assert tools.call(session, "get_values", {"signal": "a"})["t"] == []
    assert np.isnan(session.spectrum_summary("a")["dominant_hz"])
