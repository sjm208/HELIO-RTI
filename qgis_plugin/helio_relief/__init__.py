"""
Helio Relief
------------
A QGIS plugin for heritage height maps, made to sit beside Helio: one-click
import of a Helio export, Clean relief and studio renders, multi-shade and
QGIS-formula hillshading, Brown texture shading, a detail / noise separator,
measurement, line tracing, publication plates, spatial-frequency splitting,
a synthetic light dome and provenance records.

Modules
    relief_engine   numerical core (NumPy/SciPy only, unit tested in tests/)
    raster_io       GDAL raster reading/writing
    preview         background renderer for the live preview
    dialog          the dialog and its background jobs
    helio           reads a Helio export folder
    align, detail, tools, widgets, tasks

Created by: Scott Maloney
"""


def classFactory(iface):
    from .plugin import HelioReliefPlugin
    return HelioReliefPlugin(iface)
