"""Running a Python file against a session.

    # experiment.py, anywhere on disk
    from mmp_sense import Session

    def main(session: Session) -> None:
        samples = session.samples("ax")
        ...

    mmp_sense.run_script("experiment.py", session)

The file is read and run afresh every time, so a rerun sees its edits,
and the session is an argument rather than a global, so the file is
ordinary Python to an editor and works on a window's session or on one
from `mmp_sense.load()`.
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .core.session import Session

__all__ = ["run_script"]

#: A module must be in sys.modules while it runs, or `dataclass` fails in it.
MODULE = "__mmp_sense_script__"


def _top_level_in(module: types.ModuleType, directory: Path) -> bool:
    file = getattr(module, "__file__", None)
    if file is None:
        return False
    path = Path(file)
    return directory in (path.parent, path.parent.parent if path.stem == "__init__" else None)


def _forget_siblings(directory: Path, before: set[str]) -> None:
    """Drop what the run imported from beside the script, so an edit to it counts next time."""
    new = set(sys.modules) - before
    tops = {n for n in new if "." not in n and _top_level_in(sys.modules[n], directory)}
    for name in new:
        if name.partition(".")[0] in tops:
            del sys.modules[name]


def run_script(path: str | os.PathLike[str], session: Session) -> Any:
    """Run the file's `main(session)` and return what it returns.

    Modules beside the file can be imported while it runs. It runs here
    and now, on the calling thread, with this user's privileges.
    """
    script = Path(path).expanduser().resolve()
    source = script.read_bytes()
    directory = script.parent
    module = types.ModuleType(MODULE)
    module.__file__ = str(script)
    before = set(sys.modules)
    sys.modules[MODULE] = module
    sys.path.insert(0, str(directory))
    importlib.invalidate_caches()
    try:
        exec(compile(source, str(script), "exec"), module.__dict__)
        main = module.__dict__.get("main")
        if not callable(main):
            raise ValueError(f"{script} defines no main(session)")
        return main(session)
    finally:
        sys.path.remove(str(directory))
        sys.modules.pop(MODULE, None)
        _forget_siblings(directory, before)
