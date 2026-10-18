"""The model: the live buffer, the signals over it, and the session.

    buffer.py    the ring buffer the samples land in
    signals.py   named columns over it: channels and shown recipes
    session.py   the shared state, and the questions asked of it
    query.py     the arithmetic behind the answers
    graph/       recipes: nodes, ops, evaluation
    data/        the typed datasets panes draw

Nothing under here imports Qt or knows that a GUI exists.
"""
