"""The GUI: panes over one session, and nothing else.

Qt lives here and nowhere else in mmp-sense. A pane draws what the session
gives it; `bridge.py` is the one place where the core's plain callbacks
become Qt signals.
"""

__all__ = ["main"]


def __getattr__(name):
    if name == "main":
        from .app import main

        return main
    raise AttributeError(name)
