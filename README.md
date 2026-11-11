# mmp-sense

Open a sensor's data, and work out what in it is worth measuring.

mmp-sense puts a signal on the screen with a cursor you can move through
it: drop the cursor on a transient and the FFT is of that instant, drag
out a region and the statistics are of that region, replay it, and when
you have an idea, write it as a line of Python in the console built into
the window and see it plotted next to the raw data. An assistant can ask
the same questions over MCP, about the same cursor.

```
   sources                 the session                    what looks at it

  a data file ─┐       ┌───────────────────┐            ┌──────────────┐
  MCAP, WAV,   │       │  a stream each    │            │ panes        │
  CSV, NumPy   ├──────▶│  signals:         │───────────▶│ raw, signal, │
               │       │   measured        │            │ FFT, image   │
  simulated ───┘       │   derived         │            └──────────────┘
                       │                   │            ┌──────────────┐
                       │  cursor ──────────┼───────────▶│ IPython      │
                       │  selection        │◀───────────│ console      │
                       │  analysis window  │            └──────────────┘
                       │                   │            ┌──────────────┐
                       │  MCP tools        │───────────▶│ MCP server   │
                       └───────────────────┘            └──────────────┘
```

Whichever of the three changes something, the other two see it.

## Install

```sh
python3 -m venv .venv
.venv/bin/pip install --config-settings editable_mode=compat -e '.[gui,mcp]'
```

Python 3.12 or later. The core needs NumPy, SciPy, pandas and mcap; the
`gui` extra adds PySide6, pyqtgraph, qtconsole and ipykernel, and the
`mcp` extra adds the MCP server. A script that only reads signals needs
neither extra. The commands below assume the environment is active
(`. .venv/bin/activate`).

## Open a file

```sh
mmp-sense walk.mcap                         # every plottable topic, a stream each
mmp-sense clip.wav                          # sound, as the file holds it
mmp-sense voice.m4a                         # a clip from the phone's recorder
mmp-sense table.tsv                         # a column per channel
mmp-sense capture.npy --rate 400            # an array, with its sample rate
mmp-sense bench.csv --replay --speed 2      # played back in time, twice as fast
mmp-sense walk.mcap --topic /sensor/accelerometer --topic /sensor/gyroscope
mmp-sense export/<session-id>              # a session srv-senselab exported with --raw: mics by name, its clocks
```

The file is named on its own or with `--file`. Without `--replay` the
whole file is there at once, however long -- the buffer grows to hold
it; with it, the samples arrive at the rate they were recorded at, so a
recording behaves like the bench it came from. Several streams of one
file are read on one clock and replayed together; each is named after
its topic, and its channels `stream:channel` (`accelerometer:x`).

Which formats open, and what each becomes -- srv-senselab's files
included -- is in [doc/formats.md](doc/formats.md).

## Watch a simulated accelerometer

```sh
mmp-sense
mmp-sense --rate 2000 --window 10 --analysis-window 0.1
```

Three axes, noise and a knock every few seconds, so there is something to
practise on with no data to hand.

## Work on a signal from a script

```sh
python -c '
import mmp_sense
session = mmp_sense.load("bench.mcap")       # any file the window opens
size = mmp_sense.signal("x", "y", "z").magnitude()
print(session.stats(size, 4.0, 6.0))            # a pandas Series
print(session.values(size, 4.0, 6.0).idxmax())  # when it was largest
'
```

No window is started and no Qt is imported. [API.md](doc/API.md) is the rest
of it.

## Find events in a signal

```sh
python -m mmp_sense.examples.double_knock          # a detector backtested, printed
python -m mmp_sense.examples.double_knock --gui    # the same detector, in the window
```

A detector is a graph built in the console from the signals: filter,
measure, compare, turn a condition into events, and compose events in
time. `session.show(...)` draws every step and marks every detection on
the charts, and `mmp_sense.load("bench.mcap").events(detector)` runs the
same object over a whole recording with no window:

```python
hp = mmp_sense.signal().magnitude().highpass(10)
knock = ((hp.energy("80ms") > 5)
         & hp.frames("80ms").dominant_hz().between(120, 350)
        ).events(refractory="100ms", name="knock")
session.show(knock, knock.repeated(2, gap=("80ms", "500ms"), name="double_knock"))
```

The demo plants knocks, double knocks and two disturbances in a minute of
synthetic data and shows that energy alone or frequency alone mis-fires
where the two together do not. [API.md](doc/API.md#detectors) has the rest.

## Hear an engine through wind and voices

```sh
python -m mmp_sense.examples.plane                     # a synthetic field, scored
python -m mmp_sense.examples.plane recording.mcap      # the detector on a recording
```

An engine is a comb of steady spectral lines; wind is a broadband rumble
and a voice harmonics that glide and stop. `frames.steady("2s")` keeps,
bin by bin, only what has sounded for the last two seconds, and
`.line_share(150, 1000)` says how much of that band stands out as lines:
wind is floor, a shout has come and gone, the engine stays.

## Line up two microphones

```sh
python -m mmp_sense.examples.two_mics              # two phones, two clocks, lined up by claps
```

Each phone stamps samples with its own clock: off by some milliseconds,
and drifting by tens of ppm. Clap a few times, select that stretch, and
in the console

```python
session.synchronize("phoneA_mic:amplitude", "phoneB_mic:amplitude")
```

measures B's offset and drift from the claps and corrects B's clock, for
what is held and what is still to come. A **correlation** pane (*Add
pane*) shows two signals' correlation against lag over the selection,
and their delay; `mmp_sense.signal(a).delay(mmp_sense.signal(b), "50ms")` gives it
frame by frame, live, so "which one heard it first" is a detector.
[API.md](doc/API.md#two-signals-two-clocks) has the rest.

## Find where a sound came from

```sh
python -m mmp_sense.examples.array                 # four microphones, a source to find
```

Given where the microphones are (`mmp_sense.Array`, in metres),
`session.direction`, `session.locate` and `session.direction_map` find
a sound's direction or point from the delays between them, and
`mmp_sense.beam(array, azimuth)` listens in one direction, live. Add the
**direction** pane for the power by azimuth, the **psd** pane for an
averaged spectrum in dB, and filter with `.bandpass(lo, hi)`,
`.notch(50)` or `.lowpass(fc, order=4)`.
[API.md](doc/API.md#arrays-where-a-sound-came-from) has the rest.

## Let an assistant ask

```sh
mmp-sense --serve                           # the window, answering on a socket
mmp-sense-mcp                                # attaches to it, speaks MCP on stdio
mmp-sense-mcp --no-attach walk.mcap         # or a session of its own, on a file
mmp-sense-mcp --list                         # which sessions are serving
```

Point the assistant's MCP configuration at `mmp-sense-mcp`. Attached, its tools
answer about the window in front of you: `get_current_context` is your
cursor, and `get_statistics` with no interval is your selection. The
socket is local to your machine and your user, the tools only read, and
nothing is served unless `--serve` asks for it.

## Run code from your editor

```sh
mmp-sense --listen                          # the window takes code for its console
printf 'x = 42\n' | mmp-sense-send            # runs there, shown as if typed
mmp-sense-send --list                        # which windows are listening
```

In Neovim, with no plugin, `:'<,'>w !mmp-sense-send` sends the selected lines
and `:.w !mmp-sense-send` the current one. `mmp-sense-send` strips the common
indentation, so a block from inside a function runs as it is. To send
any selection, lines or part of one, with `<leader>dv`, put this in
`init.lua`:

```lua
-- Run code in an mmp-sense window started with --listen.
local function mmp_sense_send(text)
  local out = vim.fn.system({ "mmp-sense-send" }, text)
  if vim.v.shell_error ~= 0 then
    vim.notify(vim.trim(out), vim.log.levels.ERROR)
  end
end

vim.keymap.set("x", "<leader>dv", function()
  local lines = vim.fn.getregion(vim.fn.getpos("v"), vim.fn.getpos("."), { type = vim.fn.mode() })
  vim.api.nvim_feedkeys(vim.keycode("<Esc>"), "nx", false)
  mmp_sense_send(table.concat(lines, "\n") .. "\n")
end, { desc = "Run the selection in mmp-sense" })

vim.keymap.set("n", "<leader>dv", function()
  mmp_sense_send(vim.api.nvim_get_current_line() .. "\n")
end, { desc = "Run the line in mmp-sense" })
```

Begin a script with

```python
from mmp_sense.live import session, signal, show, ui
```

and the editor knows what those are: its linter checks them and it
completes `session.` like anything else. In the console the line gives
the window's own objects, so the whole file can be sent again and again,
and a selection without it runs too. `pyproject.toml` points pyright at
the project's `.venv`, for scripts kept in the repository.

If sent code fails, the **Python** title turns red and bold, the error
in its tooltip, until you open the console -- or, with it open, until a
send works. The code runs in the window's console, in order, after
anything still running; what you had half typed there is put back. `mmp-sense-send` goes to
the window started last, or to `--socket PATH` / `$MMP_SENSE_SOCKET`. The
socket runs what it is sent, so nothing listens unless `--listen` asks,
and only your user can connect to it.

## Run a script in the window

A script is a Python file anywhere on disk with a `main` that takes the
session. It gets the window's own session, so there is nothing to look up
and your editor checks and completes it:

```python
# ~/experiments/ax_spectrum.py
import numpy as np
from mmp_sense import Session

def main(session: Session) -> None:
    s = session.samples("accel_x")             # the selection, else everything held
    frequency = np.fft.rfftfreq(len(s.values), 1 / s.rate)
    session.plot("ax spectrum", frequency, np.abs(np.fft.rfft(s.values)),
                 x_label="Frequency", x_units="Hz",
                 y_label="Magnitude", y_units=s.units)
```

```sh
mmp-sense --listen
mmp-sense-send --script ~/experiments/ax_spectrum.py     # from any directory
```

Or type `run("ax_spectrum.py")` in the console. Select another stretch and
run it again: the file is read afresh, and the chart called `ax spectrum`
updates in place. Plotting under one name never adds signals or streams.

The script runs in the window's process, so the window waits until it
returns. It can import modules beside it. If it fails, the **Python**
title turns red and the traceback is in the console. The install above
lets pyright and mypy find `mmp_sense` from any directory.

## The options those commands take

**mmp-sense** (or `python -m mmp_sense`)

| Option | Meaning |
|---|---|
| `FILE`, `--file FILE` | A data file to open; see [the formats](doc/formats.md). Without one, a simulated accelerometer. |
| `--topic TOPIC` | An MCAP topic to open; again for another beside it. Default: every plottable one. |
| `--replay`, `--speed N` | Play a file back in time, at N times its own rate. |
| `--rate HZ` | Sample rate of a file that carries no time (an array, a table without a time column); otherwise of the simulated source, default 1000. |
| `--scale FACTOR` | Units per count of a sensor's WAV; it opens in counts without it. |
| `--units UNITS` | The units of what is opened, e.g. `m/s^2`. |
| `--buffer SECONDS` | How much history the session keeps. Default 60; a file is held whole. |
| `--window SECONDS` | How much of it is on screen. Default 30. |
| `--analysis-window SECONDS` | What the FFT is taken over. Default 0.5. |
| `--peak-hold SECONDS` | How long the spectrum's peak markers stay; 0 hides them. Default 2. |
| `--serve [SOCKET]` | Answer the MCP tools on a local socket, for `mmp-sense-mcp`. |
| `--listen [SOCKET]` | Run code sent by `mmp-sense-send` in the console. Off unless given. |

**mmp-sense-mcp**

| Option | Meaning |
|---|---|
| `--endpoint SOCKET` | A particular running session to attach to. |
| `--no-attach` | Run a session of its own rather than attaching. |
| `--list` | List the sessions serving on this machine. |
| `FILE`, `--file`, `--topic`, `--rate`, `--scale`, `--units`, `--buffer` | As above, for a session of its own. |

**mmp-sense-send**

| Option | Meaning |
|---|---|
| `--script PATH` | Run this file's `main(session)` rather than code from stdin. |
| `--socket PATH` | The window to send to. Default `$MMP_SENSE_SOCKET`, else the newest. |
| `--list` | List the windows listening. |

The window's keys, buttons and docks are in [doc/window.md](doc/window.md).

## Run the tests

```sh
./scripts/run-tests.sh
./scripts/run-tests.sh -k session -v
```

It makes the virtual environment if it is missing. The GUI tests run on
Qt's offscreen platform, so no display is needed.

To see what a tick costs, in a real window:

```sh
python scripts/bench-tick.py
```

## License

MIT; see [LICENSE](LICENSE).
