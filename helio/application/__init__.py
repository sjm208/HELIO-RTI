"""Application services shared by the classic app, the Qt app and scripts.

GUI-free: nothing here imports Tk or Qt (BLUEPRINT rule 6).  ``pipeline``
is the photometric-stereo solve; ``jobs`` runs long work off the interface
thread.
"""
