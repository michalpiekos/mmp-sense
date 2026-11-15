"""Workspaces: recordings and their metadata, kept together and sent to srv-senselab."""

from .model import (FORMAT, WORKSPACE_FILE, Recording, SyncState, Workspace, WorkspaceError,
                    is_workspace)

__all__ = ["Workspace", "Recording", "SyncState", "WorkspaceError", "is_workspace",
           "WORKSPACE_FILE", "FORMAT"]
