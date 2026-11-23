"""Helio's sections and annotations arrive over the height map."""
import json

from qgis.core import QgsCoordinateReferenceSystem

from helio_relief import helio, vectors

CRS = QgsCoordinateReferenceSystem("EPSG:32630")


def feature(shape, coords, **props):
    return dict(type="Feature", geometry=dict(type=shape, coordinates=coords), properties=props)


def write(path, feats):
    path.write_text(json.dumps(dict(type="FeatureCollection", features=feats)), encoding="utf-8")
    return str(path)


def test_sections_are_labelled_lines_with_the_height_maps_crs(tmp_path):
    p = write(tmp_path / "Seal_sections.geojson",
              [feature("LineString", [[100, 250], [200, 250]], name="A–B", length="5.0 mm", copies=3)])
    assert helio.scan(str(tmp_path))["sections"] == p
    lyr = vectors.section_layer(p, "sections", CRS)
    assert lyr.crs() == CRS and lyr.labelsEnabled()
    f = next(lyr.getFeatures())
    assert f["name"] == "A–B" and f.geometry().asPolyline()[1].x() == 200


def test_annotations_split_by_geometry_in_their_colours(tmp_path):
    p = write(tmp_path / "Seal_annotations.geojson", [
        feature("Point", [30, 260], kind="dot", label="Note", colour="#ffff00", layer=""),
        feature("LineString", [[1, 2], [3, 4]], kind="line", label="", colour="#ff0000", layer=""),
        feature("LineString", [[5, 2], [7, 4]], kind="arrow", label="", colour="#ff0000", layer=""),
        feature("Polygon", [[[0, 0], [5, 0], [5, 5], [0, 0]]], kind="rect", label="Stamp", colour="#00ff00", layer="Damage")])
    got = vectors.annotation_layers(p, "notes", CRS)
    assert sorted(g.featureCount() for g in got) == [1, 1, 2] and all(g.crs() == CRS for g in got)


def test_an_empty_file_gives_no_layer(tmp_path):
    p = write(tmp_path / "x_sections.geojson", [])
    assert vectors.section_layer(p, "s", CRS) is None and vectors.annotation_layers(p, "a", CRS) == []
