"""An IPython console on the session, in the window.

Not a REPL of mmp-sense's own: an in-process IPython kernel, so `session` in
the console is the very object the panes are drawing. Assign a signal
there and it is drawn before the next tick; move the cursor
in the GUI and `session.cursor` in the console says so.

The console is optional. Where qtconsole or ipykernel is missing, the
window says so in the dock and everything else works.
"""

from __future__ import annotations

import os

__all__ = ["make_console", "BANNER"]

BANNER = """mmp-sense -- `session` is what the charts draw; `signal` starts a recipe.

    strength = signal().highpass(2.0).magnitude().peak("60ms", name="strength")
    show(strength)                     # a chart appears
    session.values(strength)           # pandas: the selection, or everything
    session.spectrum_summary(session.channels[0], 4.0, 4.2)
    run("experiment.py")               # its main(session), on this session

    knock = (strength > 0.7).events(refractory="145ms", name="knock")
    show(knock)                        # marked on every chart
    session.events(knock)

Ctrl+Space shows the call tip of the function the cursor is in; Tab completes.
In a chart: h and l move the cursor, space plays, n and p jump between peaks.
"""


def make_console(namespace: dict, banner: str = BANNER):
    """A console widget with `namespace` in it, or None if unavailable.

    Returns (widget, shutdown) where shutdown closes the kernel.
    """
    # qtpy has to be told which binding is already loaded, or it may pick
    # another one and two Qts in one process is a crash, not a warning.
    os.environ.setdefault("QT_API", "pyside6")
    # The in-process kernel checks for frozen modules on the way in and
    # says so at length; nothing here is being debugged.
    os.environ.setdefault("PYDEVD_DISABLE_FILE_VALIDATION", "1")
    try:
        from qtconsole.inprocess import QtInProcessKernelManager
        from qtconsole.rich_jupyter_widget import RichJupyterWidget
        from qtpy import QtCore
    except Exception:                                        # noqa: BLE001
        return None, lambda: None

    class Console(RichJupyterWidget):
        """qtconsole's widget, with the call tip -- a function's signature
        and docstring -- shown on Ctrl+Space rather than at every `(`."""

        def _auto_call_tip(self):
            pass

        def _event_filter_console_keypress(self, event):
            if (event.key() == QtCore.Qt.Key_Space
                    and self._control_key_down(event.modifiers(), include_command=False)):
                self._call_tip()
                return True
            return super()._event_filter_console_keypress(event)

    manager = QtInProcessKernelManager()
    manager.start_kernel(show_banner=False)
    kernel = manager.kernel
    kernel.gui = "qt"
    kernel.shell.push(namespace)

    client = manager.client()
    client.start_channels()

    console = Console()
    console.kernel_manager = manager
    console.kernel_client = client
    console.banner = banner
    console.set_default_style("linux")

    def shutdown() -> None:
        try:
            client.stop_channels()
            manager.shutdown_kernel()
        except Exception:                                    # noqa: BLE001
            pass

    return console, shutdown
