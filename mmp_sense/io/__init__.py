"""Getting data in and out: `files.read_file` opens any format read here.

    files.py    which format a file is, and its series
    mcap.py     MCAP recordings of JSON messages
    audio.py    WAV, sensor WAVs, and what ffmpeg decodes
    csv.py      delimited text
    arrays.py   NumPy .npy and .npz
"""

from .audio import AUDIO_SUFFIXES, AudioError, read_audio, read_sensor_wav
from .csv import read_timeseries, write_timeseries
from .files import FileFormatError, kind_of, read_file

__all__ = ["read_file", "kind_of", "FileFormatError", "read_timeseries",
           "write_timeseries", "read_audio", "read_sensor_wav", "AudioError",
           "AUDIO_SUFFIXES"]
