"""Putting a recording in front of a session."""

from __future__ import annotations

__all__ = ["load_into", "series_of", "BUFFER_S"]

#: The least a session holds; a recording is held whole however long.
BUFFER_S = 60.0


def load_into(session, series) -> None:
    """Make `session` hold these series, and only them, each a stream of
    its own as the command line opens a file."""
    from ..analysis import file_buffer
    from ..core.session import stream_names
    from ..sources.recording import SeriesSource

    sources = [SeriesSource(one, name=one.name) for one in series]
    if not sources:
        raise ValueError("the recording holds nothing to show")
    session.buffer_s, session.max_samples = file_buffer(sources, BUFFER_S)
    names = stream_names(sources) if len(sources) > 1 else [None]
    if session.playing:
        session.pause()
    session.attach(sources[0], name=names[0], keep_signals=False)
    for source, name in zip(sources[1:], names[1:]):
        session.add(source, name)
    session.poll()


def series_of(workspace, recording) -> list:
    """What a workspace recording holds, with the channel names its file
    cannot carry (a live fragment's WAV)."""
    from ..io.files import read_file

    series = read_file(workspace.path_of(recording), **recording.options)
    names = recording.meta.get("channel_names")
    if names and len(series) == 1 and len(names) == series[0].n_channels:
        series[0].channels = list(names)
    return series
