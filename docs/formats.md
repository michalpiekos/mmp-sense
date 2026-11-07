# The file formats

What `mmp-sense FILE`, `mmp_sense.load(FILE)` and `mmp-sense-mcp FILE`
open, and what each becomes. All three read a file the same way.

A file is recognised by its first bytes where the format has a
signature, so a renamed file still opens; otherwise by its suffix.

| Format | What it becomes |
|---|---|
| **MCAP** (`.mcap`) with JSON messages | A stream per plottable topic, or those `--topic` names. Channels and unit come from each channel's JSON Schema (`properties` in order, a `unit`). A file without a summary, or cut short before its footer, is read up to where it ends. |
| **WAV** | Sound: from -1 to 1, at the file's own rate, every channel kept: one signal `audio`, or `ch0`, `ch1`, ... for several. 8-, 16-, 24- and 32-bit integer and float samples. |
| **Sensor WAV**: below 8 kHz, or with `--scale` | A sensor's counts, at the file's own rate, every channel kept -- `x`, `y`, `z` for three. `--scale` multiplies them into units per count, and `--units` names those. |
| **Compressed audio**: m4a, aac, 3gp, amr, mp3, ogg, opus, flac, webm | As WAV, decoded by the `ffmpeg` program, which must be installed. |
| **Delimited text**: `.csv`, `.tsv`, `.txt`, `.dat`, ... | The separator -- comma, tab, semicolon, bar, spaces -- is sniffed, and with a semicolon a decimal comma is understood. Time is the column named `t_ms`, `t_us`, `t_ns` (and `time_ms` ...) in those units, or `time`, `time_s`, `t` or `timestamp` in seconds -- unless its values are too large to be seconds, when they are an epoch's milliseconds, microseconds or nanoseconds -- or in ISO 8601 dates; with no time column, `--rate` gives the sample rate. Rows are put in time order. Every other numeric column is a channel; a column of text is not. |
| **NumPy** `.npy` | One array, `(samples,)` or `(samples, channels)`, at `--rate`. |
| **NumPy** `.npz` | A channel per 1-D array, named after it, and `name0`, `name1`, ... per column of a 2-D one. Time from a `time` or `t` array, or the rate from a `rate`, `fs`, `sample_rate` or `sample_rate_hz` scalar, else `--rate`. Pickled objects are never loaded. |

## srv-senselab's files

The single files srv-senselab, the MMP sampling server, stores and exports
open as they are:

- **Its MCAP**, as the mmp-sample phone app writes it:
  `/sensor/accelerometer`, `/sensor/gyroscope` and
  `/sensor/magnetometer` become `x`, `y`, `z` in m/s², rad/s and µT, and
  `/location/gnss` its latitude, longitude, altitude, accuracy, speed and
  bearing. Time is the session's. Timing points, markers and device
  status are not signals, and are left out.
- **Its WAVs**: the phone's sound as sound, and an `imu` WAV -- an
  accelerometer or gyroscope streamed by a node, three channels of
  counts -- as a sensor's, in counts, or in units with the sample's
  `encoding.scale`:
  `mmp-sense accel.wav --scale 0.0023942 --units m/s^2`.
- **Its clip-export CSVs**: `t_ns,x,y,z` in seconds and in the stream's
  unit, read from the file's name (`..._accelerometer_...` is m/s²), and
  the GNSS table with its gaps kept.

Export manifests and annotation files are not read.
