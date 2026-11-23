"""Plain domain records shared by the legacy app, the Qt app and the CLI.

Nothing here imports Tk, Qt or any widget toolkit (BLUEPRINT rule 6).
"""
from helio.domain.settings import ProcessingSettings

__all__ = ["ProcessingSettings"]
