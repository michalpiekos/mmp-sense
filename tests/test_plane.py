"""The plane example: engine lines heard through wind and shouts."""

import numpy as np
import pytest

import mmp_sense
from mmp_sense.examples import plane


@pytest.fixture(scope="module")
def fields():
    out = {}
    for name, there in (("plane", (8.0, 22.0)), ("none", None)):
        table, truth = plane.synthesize(plane=there, seed=2)
        out[name] = (mmp_sense.from_frame(table), truth)
    return out


def test_the_plane_is_heard_while_it_is_there(fields):
    session, truth = fields["plane"]
    nodes = plane.detector()
    result = plane.score(session, nodes["plane"], truth)
    assert result["recall"] > 0.9
    assert result["false"] == 0.0
    assert result["events"] == 1


def test_wind_and_shouts_alone_are_no_plane(fields):
    session, truth = fields["none"]
    nodes = plane.detector()
    assert session.events(nodes["plane"]).empty
    assert session.values(nodes["plane_share"]).max() < 0.22


def test_band_power_jumps_at_shouts_where_steady_lines_do_not(fields):
    session, truth = fields["plane"]
    nodes = plane.detector()
    raw = session.values(nodes["band_power"]).loc[12:20]
    lines = session.values(nodes["plane_power"]).loc[12:20]
    spread = lambda v: np.nanpercentile(v, 95) / np.nanpercentile(v, 5)   # noqa: E731
    assert spread(raw) > 5 * spread(lines)


def test_the_example_runs_and_reports(capsys):
    assert plane.main([]) == 0
    out = capsys.readouterr().out
    assert "no plane" in out and "plane heard" in out
