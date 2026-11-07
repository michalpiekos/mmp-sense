# The window

| Key | What it does |
|---|---|
| click | Put the cursor there. On a spectrum or PSD, put the frequency cursor on the nearest peak (`Alt`+click: exactly there); on the spectrogram, set both cursors. Each frequency chart reads out its value at it, beside its title |
| shift-drag | On a time chart, select an interval; the FFT and the statistics follow it. On a spectrum or PSD, add a frequency band; drag a band's edges to change it |
| double-click a pane's header | Let that pane fill the window; again to bring the others back |
| `▾` / `▸` by a pane's `x` | Collapse it to its title, or open it; the charts take the room |
| click a legend entry | Hide or show that curve; double-click to show it alone, again for all |
| `h` / `l` | Move the cursor a quarter of the analysis window |
| `H` / `L` | Move it a whole window |
| `Ctrl+h` / `Ctrl+l` | Move it one sample |
| `n` / `p` | Jump to the next or previous peak of the focus signal |
| `j` / `k` | Move the frequency cursor one bin down or up, in the first open spectrum or PSD |
| `J` / `K` | Move it to the next peak below or above |
| `space` | Play the focus signal from the cursor, as sound with the cursor running along (also the **Play** button); again to pause, and again to resume. With a selection it starts at the cursor if that is inside, else at the selection's start, and stops at its end. With bands ticked, only those frequencies. The speed box applies, pitch with it. With no sound device the cursor still runs, and the status bar says why it is silent |
| `b` | Add a frequency band, its edges typed in Hz |
| `f` | Follow the newest data again |
| `v` | Select the analysis window around the cursor |
| `Esc` | Clear the selection |
| `[` / `]` | Halve or double the analysis window |
| `=` | Tidy: every open chart the same height (also the **Tidy** button); collapsed charts stay collapsed |
| `d` | PDF: beside each chart, how its values are distributed (also the **PDF** button) |
| **k·f** button | Mark the frequency cursor's harmonics, 2f, 3f, ..., on the spectrum and PSD |
| `Ctrl+Space` | In the console: show the call tip -- signature and docstring -- of the function the cursor is in. It no longer pops up at every `(`; `Esc` closes it |

Drag or zoom a chart and the view stops scrolling with the data until
`f`. The **Signals** dock lists what exists: double-click one to plot it,
or add a step from the box beneath. The **Detectors** dock
lists what `session.show` put on screen, with how many events each has
found and the graph of the one selected; untick a detector's **Show**
box to take its marks off the charts while it keeps counting. The
**Bands** dock lists the frequency bands, with their edges in Hz;
**Add…** and **Edit…** (or a double-click) take the edges as numbers,
and only the ticked bands are heard. It opens with the first band.
Each pane chooses its own
signal -- or, on a signal chart, a group drawn together, such as a shown
node's components or `session.group(name, [...])` -- and closes with the
`x` in its corner. **Add feature** offers the recipe steps of one
signal -- filters, peak, energy, band power, the spectral measures and
the rest: the `…` opens a form for the signal and parameters (and, for a
frequency measure, its frames' length and hop), and Add does what
`show(signal(x).frames("200ms", "25ms").band_power(80, 100, name=...))`
would in the console. **Drop** takes a shown recipe away. Any pane or dock
collapses to its title with the `▾` beside its close button: charts share
the room it frees, a collapsed chart is not drawn until opened, and a
collapsed dock keeps working -- the console still runs code. The
**Python** console starts collapsed: `▸` on its title bar opens it. With both
left docks collapsed their column becomes a thin strip. Over each spectrum bar a
short line marks the highest value that bin reached, for as long as the
toolbar's **hold** says (2 s unless `--peak-hold` sets it; 0 hides them).
The **Amp** box sets how loud Play is, and a change takes effect at once. `amp auto` raises a weak recording's louder moments to a comfortable level, by up to 60 dB, whatever one knock does; `+12`, `+24` and `+40 dB` are fixed. A limiter holds loud moments under full scale. Noise rises with the signal, so tick a band to cut hiss or rumble. `amp off` plays the recording at its own level, for comparing recordings by ear.

An `xy` pane draws a script's `session.plot`; its chooser picks among the plots.

The spectrogram's header chooses its segment -- 256 samples show when things happen, 16384 tell lines a few hertz apart -- and how high it reaches; its colours run from the typical cell up, and the psd pane's segments go to 65536.
**PDF** in the toolbar adds a narrow column to the right of every signal
chart: the distribution of its values, turned on its side so each value
sits level with the chart beside it -- zoom the chart and the column
follows. It counts the samples on screen, or those in the selection when
there is one, and each series drawn has its own curve in its colour.

**Stop** and **Start** in the toolbar decide whether anything is being
captured. Stop stops taking samples and keeps what it took: the chart
holds still, the status line says `stopped`, and the cursor, the FFT, the
statistics and the replay all still work over the buffer -- a stopped
session is the one to explore. Start throws that away and captures again
from an empty buffer, so what was on screen is gone; the file is what
keeps data for good. Neither is Play, which moves the cursor through
what is already held.
