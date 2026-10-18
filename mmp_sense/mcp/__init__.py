"""AI access to a session: the same semantic tools, over MCP.

    tools.py      the tools, named once, read-only
    endpoint.py   a running session answering them on a local socket
    server.py     the stdio MCP server, attached or standalone

There is no AI in mmp-sense. This is the other direction: an assistant that
already exists calls mmp-sense, and gets the same answers the console does.
"""

from .tools import TOOLS, call, describe_tools

__all__ = ["TOOLS", "call", "describe_tools"]
