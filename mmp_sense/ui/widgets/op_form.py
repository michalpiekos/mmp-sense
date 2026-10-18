"""A form for one graph op, from the feature menu: its parameters, then Add.

It is built from the op's own signature and the starting values it was
registered with, so an op registered with `menu=` gets a form without any
code here. Adding does what the console line would --
`show(signal(x).op(**params, name=...))` -- and a value the op refuses is
said in the form, which stays open. An op that measures frames (band
power, centroid...) gets the frames' length and hop too, and is added as
`signal(x).frames(length, hop).op(...)`.
"""

from __future__ import annotations

import inspect

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)

from ...core.graph.node import signal as graph_signal
from ...core.graph.ops import op as graph_op

__all__ = ["OpForm", "op_parameters", "FRAMES"]

#: The frames an op of frames is given, until the form says otherwise.
FRAMES = {"length": "200ms", "hop": "25ms"}


def op_parameters(name: str) -> list[tuple[str, object]]:
    """An op's parameters after its input, with their starting values."""
    spec = graph_op(name)
    params = list(inspect.signature(spec.fn).parameters.values())[2:]
    out = []
    for p in params:
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        default = spec.menu.get(p.name, None if p.default is p.empty else p.default)
        out.append((p.name, default))
    return out


class OpForm(QDialog):
    """Parameters for one op on one signal; Add shows the result."""

    added = Signal(str)

    def __init__(self, session, op_name: str, signal: str | None = None, parent=None) -> None:
        super().__init__(parent)
        self.session = session
        self.op_name = op_name
        spec = graph_op(op_name)
        self.setWindowTitle(f"Add {op_name}")
        self.setMinimumWidth(360)

        self.input = QComboBox()
        self.input.addItems(session.signals.names)
        if signal in session.signals:
            self.input.setCurrentText(signal)
        self.input.currentTextChanged.connect(self._suggest_name)

        self.fields: dict[str, QLineEdit | QCheckBox] = {}
        self._optional: set[str] = set()
        form = QFormLayout()
        form.addRow("signal", self.input)
        self.frames: dict[str, QLineEdit] = {}
        if spec.on_frames:
            for name, default in FRAMES.items():
                field = QLineEdit(default)
                field.setToolTip("seconds, or text such as 200ms")
                self.frames[name] = field
                form.addRow(f"frames {name}", field)
        for name, default in op_parameters(op_name):
            if isinstance(default, bool):
                field = QCheckBox()
                field.setChecked(default)
            else:
                field = QLineEdit("" if default is None else str(default))
                if default is None:
                    self._optional.add(name)
                    field.setPlaceholderText("default")
                if name in spec.durations:
                    field.setToolTip("seconds, or text such as 80ms")
            self.fields[name] = field
            form.addRow(name, field)
        self.name = QLineEdit()
        form.addRow("name", self.name)
        self._suggest_name()

        self.error = QLabel("")
        self.error.setStyleSheet("color: #e03c31;")
        self.error.setWordWrap(True)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        buttons.addButton("Add", QDialogButtonBox.AcceptRole)
        buttons.accepted.connect(self.try_add)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        description = QLabel(spec.description)
        description.setWordWrap(True)
        layout.addWidget(description)
        layout.addLayout(form)
        layout.addWidget(self.error)
        layout.addWidget(buttons)

    def _suggest_name(self) -> None:
        stem = f"{self.op_name}_{self.input.currentText()}"
        name, n = stem, 1
        taken = lambda x: (x in self.session.signals or x in self.session.shown  # noqa: E731
                           or x in self.session.groups)
        while taken(name):
            n += 1
            name = f"{stem}{n}"
        self.name.setText(name)

    def values(self) -> dict:
        """The parameters as the op takes them; ValueError if one is unreadable."""
        spec = graph_op(self.op_name)
        out = {}
        for name, field in self.fields.items():
            if isinstance(field, QCheckBox):
                out[name] = field.isChecked()
                continue
            text = field.text().strip()
            if not text:
                if name in self._optional:
                    continue
                raise ValueError(f"{name}: needs a value")
            try:
                out[name] = float(text)
            except ValueError:
                if name not in spec.durations:
                    raise ValueError(f"{name}: not a number: {text!r}") from None
                out[name] = text
        return out

    def try_add(self) -> bool:
        """Show the op's result if it works; else say why and stay open."""
        try:
            name = self.name.text().strip()
            if not name:
                raise ValueError("name: needs one")
            source = graph_signal(self.input.currentText())
            if self.frames:
                source = source.frames(self.frames["length"].text().strip(),
                                       self.frames["hop"].text().strip() or None)
            node = source.apply(self.op_name, name=name, **self.values())
            self.session.values(node)            # refused values are refused here
            self.session.show(node)
        except Exception as exc:                              # noqa: BLE001
            self.error.setText(str(exc))
            return False
        self.error.setText("")
        self.added.emit(name)
        self.accept()
        return True
