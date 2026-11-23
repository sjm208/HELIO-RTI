"""Helio's kept sections and annotations as map layers.

Helio writes them as GeoJSON in the height map's own units.  GeoJSON has no CRS of
its own here, so each layer is given the height map's and sits exactly over it.
"""
from qgis.core import (Qgis, QgsPalLayerSettings, QgsProperty, QgsSymbol, QgsSymbolLayer, QgsTextBufferSettings,
                       QgsTextFormat, QgsVectorLayer, QgsVectorLayerSimpleLabeling, QgsWkbTypes)
from qgis.PyQt.QtGui import QColor

GEOMETRIES = ("Point", "LineString", "Polygon")


def _labels(layer, field):
    s = QgsPalLayerSettings()
    s.fieldName = field
    if layer.geometryType() == QgsWkbTypes.GeometryType.LineGeometry:
        s.placement = Qgis.LabelPlacement.Line        # the default suits points, and draws nothing on a line
    fmt = QgsTextFormat()
    fmt.setSize(10)
    buf = QgsTextBufferSettings()
    buf.setEnabled(True)
    buf.setColor(QColor("white"))
    fmt.setBuffer(buf)
    s.setFormat(fmt)
    place = s.placementSettings()              # a name beside another still shows
    place.setOverlapHandling(Qgis.LabelOverlapHandling.AllowOverlapIfRequired)
    s.setPlacementSettings(place)
    layer.setLabeling(QgsVectorLayerSimpleLabeling(s))
    layer.setLabelsEnabled(True)


def section_layer(path, name, crs):
    """The sections: one blue line each, labelled A–B."""
    lyr = QgsVectorLayer(path, name, "ogr")
    if not lyr.isValid() or lyr.featureCount() == 0:
        return None
    lyr.setCrs(crs)
    sym = QgsSymbol.defaultSymbol(lyr.geometryType())
    sym.setColor(QColor("#2b6cff"))
    sym.setWidth(0.6)
    lyr.renderer().setSymbol(sym)
    _labels(lyr, "name")
    return lyr


def annotation_layers(path, name, crs):
    """The annotations, one layer for each kind of geometry there is, in the colours they were drawn in."""
    out = []
    for geometry in GEOMETRIES:
        lyr = QgsVectorLayer(f"{path}|geometrytype={geometry}", f"{name} ({geometry.lower().replace('linestring', 'lines')})",
                             "ogr")
        if not lyr.isValid() or lyr.featureCount() == 0:
            continue
        lyr.setCrs(crs)
        sym = QgsSymbol.defaultSymbol(lyr.geometryType())
        colour = QgsProperty.fromExpression('"colour"')
        area = lyr.geometryType() == QgsWkbTypes.GeometryType.PolygonGeometry
        for sl in sym.symbolLayers():
            sl.setDataDefinedProperty(QgsSymbolLayer.Property.StrokeColor, colour)
            if area:
                sl.setDataDefinedProperty(QgsSymbolLayer.Property.FillColor,
                                          QgsProperty.fromExpression("set_color_part(\"colour\", 'alpha', 40)"))
            else:
                sl.setDataDefinedProperty(QgsSymbolLayer.Property.FillColor, colour)
        lyr.renderer().setSymbol(sym)
        _labels(lyr, "label")
        out.append(lyr)
    return out
