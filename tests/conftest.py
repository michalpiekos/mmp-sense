"""Settings and fixtures the suite needs.

Qt's offscreen platform is chosen before any Qt import, so the GUI tests
run on a machine with no display -- a build server, or an ssh session.
Set QT_QPA_PLATFORM yourself to override it.

The `session` fixture is a simulated accelerometer against a clock of
the test's own, so a test says how much time passed rather than waiting
for it.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# The in-process kernel the console tests start checks for frozen modules
# and says so at length; nothing here is being debugged.
os.environ.setdefault("PYDEVD_DISABLE_FILE_VALIDATION", "1")

import pytest


# -- a session to work on ----------------------------------------------


class Clock:
    """A clock a test winds itself."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = float(t)

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> float:
        self.t += float(seconds)
        return self.t


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def source(clock):
    """A simulated accelerometer that knocks only when told to."""
    from mmp_sense import SyntheticAccelerometer

    return SyntheticAccelerometer(rate=500.0, knock_interval=0.0, clock=clock, seed=7)


@pytest.fixture
def session(source, clock):
    """Ten seconds of it, with one knock at 5 s."""
    from mmp_sense import Session

    session = Session(source, buffer_s=60.0, clock=clock).start()
    source.knock(at=5.0)
    clock.advance(10.0)
    session.poll()
    try:
        yield session
    finally:
        session.stop()


def show_basics(session):
    """What the GUI tests look at besides the channels: the magnitude and
    its energy, shown as a user would show them."""
    import mmp_sense

    magnitude = mmp_sense.signal().magnitude(name="magnitude")
    session.show(magnitude, magnitude.energy("50ms", name="energy"), plot=False)
    return session


@pytest.fixture(scope="session")
def qapp():
    """One QApplication for the GUI tests; Qt allows no more."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    application = QApplication.instance() or QApplication([])
    yield application
