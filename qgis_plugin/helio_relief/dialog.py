"""
Helio Relief - main dialog.

Controls on the left (profile bar, surface card, five task tabs); a live
preview with a sun-position dial on the right; progress, log and Close along
the bottom. Numerics live in relief_engine, GDAL I/O in raster_io, and every
long job runs as a QgsTask so QGIS stays responsive and jobs can be cancelled.
"""

import datetime
import html
import json
import math
import os
import re
import tempfile
import time

import numpy as np
from qgis.core import (Qgis, QgsApplication, QgsColorRampShader, QgsContrastEnhancement, QgsCoordinateTransform,
                       QgsMultiBandColorRenderer, QgsProject, QgsRasterLayer, QgsRasterShader, QgsRectangle,
                       QgsSettings, QgsSingleBandGrayRenderer, QgsSingleBandPseudoColorRenderer, QgsUnitTypes)
from qgis.gui import QgsColorButton, QgsFileWidget, QgsMapLayerComboBox
from qgis.PyQt.QtCore import Qt, QTimer, QUrl
from qgis.PyQt.QtGui import QColor, QDesktopServices, QIcon, QKeySequence, QPainter, QPixmap
from qgis.PyQt.QtWidgets import (QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QFrame,
                                 QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit,
                                 QMenu, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea,
                                 QSizePolicy, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget,
                                 QToolButton, QVBoxLayout, QWidget)

from . import raster_io as rio
from . import relief_engine as eng
from .preview import DETAIL_MAX_PIXELS, OVERVIEW_DIM, PreviewRenderer
from .tasks import Job, sub_progress
from .widgets import PreviewCanvas, Section, ShadeStrip, SliderSpin, SunDial, _qimage

try:
    from qgis.PyQt.QtGui import QShortcut
except ImportError:
    from qgis.PyQt.QtWidgets import QShortcut


try:                                   # the window title and every provenance sidecar carry the plugin's version
    with open(os.path.join(os.path.dirname(__file__), "metadata.txt"), encoding="utf-8") as _f:
        VERSION = next((_l.split("=", 1)[1].strip() for _l in _f if _l.startswith("version=")), "")
except OSError:
    VERSION = ""
# These names predate the rename to Helio Relief; kept so saved presets, looks and projects still open.
PRESET_FILE = os.path.expanduser("~/.qgis_photometric_presets.json")
LOOKS_FILE = os.path.expanduser("~/.qgis_photometric_looks.json")
USER_LOOK_PREFIX = "\u2605 "
PROJECT_SCOPE = "PhotometricSuite"
TAB_RENDER, TAB_DETAIL, TAB_TOOLS, TAB_LAB, TAB_FREQUENCY, TAB_STACK, TAB_PROVENANCE = range(7)
DETAIL_SHORT = {"signed": "Signed (grey)", "signed_colour": "Signed (colour)", "hillshade": "Hillshade",
                "clean": "Clean relief", "lines": "Line strength", "energy": "Where strong", "noise": "Removed"}
PAPER_MM = {"A4": (297.0, 210.0), "A3": (420.0, 297.0), "Letter": (279.4, 215.9)}      # landscape (w, h)
PLATE_CONTENT = ["Grey relief", "Grey relief + colour composite", "Grey relief + close-up of the view",
                 "Grey relief, colour composite and close-up"]
LAB_DISPLAY_OF = {"Auto contrast (whole surface)": "auto", "gdaldem grey levels (raw)": "raw"}
MODE_BY_LABEL = {label: key for key, label in eng.RENDER_MODES.items()}
LABEL_ALIASES = {"Light dome (multi-directional)": eng.RENDER_MODES["dome"]}   # renamed in 2.2
VAT_LABELS = {"General (RVT VAT - Archaeological)": "general", "Low relief (tighter stretches)": "low-relief"}
VAT_LABEL_OF = {v: k for k, v in VAT_LABELS.items()}
COMPONENT_OF = {label: key for key, label in eng.COMPONENTS.items()}
SHADE_TYPE_OF = {label: key for key, label in eng.BAND_SHADE_TYPES.items()}
FQ_COMBO = "Band shading combination"
FQ_RECOMBINED = "Recombined relief (enabled bands × gains)"
STRIP_LOOKS = ["Clean relief · All-round (default)", "Clean relief · Extra smooth",
               "Clean relief · Crisp detail (tooling, drypoint)", "Clean relief · Archive match (north-east light)",
               "Clean relief · Lit from every side", "Clean relief · Raking all-round",
               "QGIS · Hillshade (gdaldem default)", "QGIS · Multidirectional hillshade", "QGIS · Combined shading",
               "Shade Lab · PCA 1 of 16 hillshades", "Shade Lab · Direction-free relief (PCA 1+2)",
               "Shade Lab · Fine bands, edge-preserving (drypoint)",
               "QGIS · Igor shading",
               "GIS · Archive shaded render (matched)", "GIS · Shaded render (classic)",
               "GIS · Darkest of 4 lights (incisions)", "GIS · Slope-enhanced relief",
               "Specular enhancement · Overhead light",
               "Studio · Plaster (classic)", "Studio · Raking detail", "Studio · Drypoint & incision",
               "Exaggerated · Drypoint lines", "Exaggerated · Fine detail (all scales)",
               "Multi-shade · Inscription reading",
               "Frequency bands · Soft form, crisp detail", "Frequency bands · Drypoint lines",
               "Frequency bands · Inscription", "Archaeological blend (VAT)", "Brown texture shading 50 %",
               "Sky-view factor (cracks & incisions)"]

BANDS = [("Very_Low_Freq", "Very low", "Gross warp / curl"),
         ("Low_Freq", "Low", "Page buckling, cockling"),
         ("Mid_Freq", "Mid", "Tool marks, ink"),
         ("High_Freq", "High", "Fine detail"),
         ("Ultra_High_Freq", "Ultra-high", "Grain, pores")]

# Semi-transparent diverging ramps: band-pass layers are signed (valley < 0 < ridge).
COLOR_RAMP_PRESETS = {
    "Blue - Orange - Brown": [(0.0, (30, 60, 150, 160)), (0.5, (255, 255, 255, 0)),
                              (0.75, (230, 140, 30, 160)), (1.0, (120, 70, 20, 190))],
    "Red - White - Green": [(0.0, (200, 30, 30, 170)), (0.5, (255, 255, 255, 0)), (1.0, (30, 160, 60, 170))],
    "Purple - White - Teal": [(0.0, (120, 40, 150, 160)), (0.5, (255, 255, 255, 0)), (1.0, (20, 150, 140, 160))],
    "Magenta - Yellow": [(0.0, (200, 30, 150, 150)), (0.5, (255, 255, 255, 0)), (1.0, (230, 200, 30, 150))],
    "None (Greyscale, opaque)": None,
}
FQ_PREVIEWS = (["Band shading combination", "Recombined relief (enabled bands × gains)"]
               + [f"{label} band (shaded)" for _, label, _ in BANDS]
               + [f"{label} band (signed ramp)" for _, label, _ in BANDS])
DEFAULT_BAND_RAMPS = {"Very_Low_Freq": "Purple - White - Teal", "Low_Freq": "Blue - Orange - Brown",
                      "Mid_Freq": "Red - White - Green", "High_Freq": "Magenta - Yellow",
                      "Ultra_High_Freq": "Blue - Orange - Brown"}

# Render recipes. Values use engine parameter names; texture_detail is Brown's alpha.
LOOKS = {
    # Clean relief: the drift of the depth map removed, edge-preserving smoothing, a key light blended with a ring
    # of lights round the compass, and a soft grey tone about the flat level (never black or white). Relief, grey,
    # contrast and highlight gain are fitted to archive shaded renders of tooled bindings (their normal maps against
    # their renders): one light from the north-east at 30-55 deg, the steepest 5 % shaded at ~45 deg.
    "Clean relief · All-round (default)": dict(
        mode="clean", azimuth=315, altitude=45, clean_smooth=3.0, clean_edge=0.5, clean_fine=0.3, clean_relief=45,
        clean_allround=0.3, clean_ring_alt=30, clean_lights=16, clean_mid=0.6, clean_contrast=1.0,
        clean_highlight=1.6),
    "Clean relief · Extra smooth": dict(
        mode="clean", azimuth=315, altitude=45, clean_smooth=6.0, clean_edge=0.8, clean_fine=0.1, clean_relief=42,
        clean_allround=0.35),
    "Clean relief · Crisp detail (tooling, drypoint)": dict(
        mode="clean", azimuth=315, altitude=40, clean_smooth=1.5, clean_edge=0.3, clean_fine=0.8, clean_relief=48,
        clean_allround=0.3, clean_contrast=1.1),
    "Clean relief · Archive match (north-east light)": dict(
        mode="clean", azimuth=45, altitude=45, clean_allround=0.0, clean_relief=45, clean_mid=0.6,
        clean_contrast=1.0, clean_highlight=1.6),
    "Clean relief · Lit from every side": dict(
        mode="clean", clean_allround=1.0, clean_ring_alt=35, clean_relief=35, clean_mid=0.72, clean_contrast=1.0),
    "Clean relief · Raking all-round": dict(
        mode="clean", azimuth=315, altitude=25, clean_allround=0.35, clean_ring_alt=18, clean_relief=40),
    "Clean relief · Colour composite (albedo × relief)": dict(
        mode="clean", azimuth=315, altitude=45, albedo=True, albedo_strength=1.0),
    "Studio \u00b7 Plaster (classic)": dict(
        mode="studio", azimuth=315, altitude=32, key=1.0, fill=0.2, fill_offset=180, fill_altitude=30,
        ambient=0.28, wrap=0.15, shadows=0.35, shadow_softness=4.0, shadow_reach=96, texture_detail=0.6,
        texture_mix=0.3, texture_contrast=1.0, texture_midtone=0.0, occlusion=0.2, specular=0.0, clarity=0.25,
        clarity_radius=12, base_tone=0.62, contrast=1.1, gamma=1.0, tint="Neutral grey"),
    "Studio \u00b7 Raking detail": dict(
        mode="studio", azimuth=315, altitude=14, key=1.0, fill=0.12, fill_altitude=25, ambient=0.2, wrap=0.05,
        shadows=0.55, shadow_softness=2.5, shadow_reach=128, texture_detail=0.7, texture_mix=0.4, occlusion=0.2,
        clarity=0.35, clarity_radius=10, base_tone=0.58, contrast=1.2, tint="Neutral grey"),
    "Studio \u00b7 Grazing microtexture": dict(
        mode="studio", azimuth=300, altitude=6, key=1.0, fill=0.05, ambient=0.25, shadows=0.6,
        shadow_softness=1.5, shadow_reach=192, texture_detail=0.9, texture_mix=0.5, occlusion=0.1, clarity=0.4,
        clarity_radius=8, base_tone=0.55, contrast=1.3, tint="Neutral grey"),
    "Studio \u00b7 Soft museum light": dict(
        mode="studio", azimuth=315, altitude=45, key=1.0, fill=0.3, fill_altitude=35, ambient=0.35, wrap=0.3,
        shadows=0.15, shadow_softness=8.0, texture_detail=0.5, texture_mix=0.25, occlusion=0.25, clarity=0.2,
        clarity_radius=16, base_tone=0.66, contrast=1.0, tint="Neutral grey"),
    "Studio \u00b7 Metal plate (specular)": dict(
        mode="studio", azimuth=315, altitude=40, key=0.9, fill=0.15, ambient=0.25, shadows=0.25,
        shadow_softness=3.0, texture_detail=0.6, texture_mix=0.25, occlusion=0.1, specular=0.45, shininess=40,
        clarity=0.2, clarity_radius=10, base_tone=0.6, contrast=1.15, tint="Cool silver"),
    "Studio \u00b7 Warm plaster cast": dict(
        mode="studio", azimuth=315, altitude=32, key=1.0, fill=0.25, ambient=0.35, wrap=0.25, shadows=0.3,
        shadow_softness=6.0, texture_detail=0.55, texture_mix=0.3, occlusion=0.2, specular=0.05, shininess=12,
        clarity=0.15, clarity_radius=14, base_tone=0.66, contrast=1.05, tint="Warm plaster"),
    # GIS hill shading, as photometric-stereo studios render their depth maps in GIS software (Global Mapper,
    # GRASS, QGIS): one neutral colour, relief from the light, shadows capped, a touch of highlight.
    # Exactly QGIS's Raster > Analysis > Hillshade (gdaldem hillshade), shown the way QGIS shows a new layer.
    "QGIS · Hillshade (gdaldem default)": dict(
        mode="gdal", azimuth=315, altitude=45, gdal_variant="standard", gdal_z=1.0, gdal_scale=1.0, gdal_zt=False,
        gdal_edges=False, gdal_display="qgis"),
    "QGIS · Multidirectional hillshade": dict(
        mode="gdal", azimuth=315, altitude=45, gdal_variant="multidirectional", gdal_z=1.0, gdal_scale=1.0,
        gdal_zt=False, gdal_edges=False, gdal_display="qgis"),
    "QGIS · Combined shading": dict(
        mode="gdal", azimuth=315, altitude=45, gdal_variant="combined", gdal_z=1.0, gdal_scale=1.0, gdal_zt=False,
        gdal_edges=False, gdal_display="qgis"),
    "QGIS · Igor shading": dict(
        mode="gdal", azimuth=315, altitude=45, gdal_variant="igor", gdal_z=1.0, gdal_scale=1.0, gdal_zt=False,
        gdal_edges=False, gdal_display="qgis"),
    "Shade Lab · PCA 1 of 16 hillshades": dict(
        mode="lab", lab_output="pc1", lab_dirs=16, lab_weights=[1.0, 1.0, 1.0, 1.0, 1.0], lab_alt=45, lab_z=1.0),
    "Shade Lab · Direction-free relief (PCA 1+2)": dict(
        mode="lab", lab_output="pcmag", lab_dirs=16, lab_weights=[1.0, 1.0, 1.0, 1.0, 1.0], lab_alt=45, lab_z=1.0),
    "Shade Lab · Multi-scale PCA of band hillshades": dict(
        mode="lab", lab_output="pc1", lab_dirs=8, lab_pca_bands=True, lab_weights=[0.0, 0.3, 0.7, 1.0, 1.0],
        lab_alt=45, lab_z=1.0),
    "Shade Lab · Mean of 16 directions": dict(
        mode="lab", lab_output="mean", lab_dirs=16, lab_weights=[1.0, 1.0, 1.0, 1.0, 1.0], lab_alt=45, lab_z=1.0),
    "Shade Lab · Fine bands, edge-preserving (drypoint)": dict(
        mode="lab", lab_output="hillshade", lab_weights=[0.0, 0.0, 0.6, 1.0, 1.0], lab_smoothing="edge",
        lab_smooth_radius=1.5, lab_edge=0.3, lab_variant="multidirectional", lab_alt=45, lab_z=1.0),
    # Fitted, pixel by pixel, to published photometric-stereo shaded renders (their depth maps, normal maps and
    # shaded renders of the same objects): one light from 45 deg (north-east) at about 60 deg, a face turned full
    # to the light only ~0.76 grey, ambient ~0.17. The exaggeration varies per object - use Auto or Match reference.
    "GIS · Archive shaded render (matched)": dict(
        mode="gis", azimuth=45, altitude=60, gis_lights=1, gis_ambient=0.17, gis_brightness=0.765, gis_shadow=0.0,
        gis_highlight=0.0, gis_slope=0.0, gis_ve=1.0, contrast=1.0, tint="Neutral grey"),
    "GIS · Shaded render (classic)": dict(
        mode="gis", azimuth=315, altitude=45, gis_lights=1, gis_ambient=0.15, gis_shadow=0.2, gis_highlight=0.15,
        contrast=1.0, tint="Neutral grey"),
    "GIS · Raking shaded render": dict(
        mode="gis", azimuth=315, altitude=25, gis_lights=1, gis_ambient=0.25, gis_shadow=0.25, gis_highlight=0.1,
        contrast=1.2, tint="Neutral grey"),
    "GIS · Four lights, weighted": dict(
        mode="gis", azimuth=315, altitude=45, gis_lights=4, gis_blend="weighted", gis_ambient=0.1, gis_shadow=0.15,
        gis_highlight=0.1, contrast=1.1, tint="Neutral grey"),
    "GIS · Darkest of 4 lights (incisions)": dict(
        mode="gis", azimuth=315, altitude=55, gis_lights=4, gis_blend="min", gis_ambient=0.2, gis_shadow=0.0,
        gis_highlight=0.0, contrast=1.2, tint="Neutral grey"),
    "GIS · Slope-enhanced relief": dict(
        mode="gis", azimuth=315, altitude=60, gis_lights=1, gis_ambient=0.0, gis_shadow=0.14, gis_highlight=0.0,
        gis_slope=0.8, gis_slope_range=10, contrast=1.0, tint="Neutral grey"),
    "Specular enhancement · Overhead light": dict(
        mode="specular", azimuth=315, altitude=80, spec_depth=1.0, shininess=30, key=1.0, contrast=1.0),
    "Specular enhancement · Raking glint": dict(
        mode="specular", azimuth=315, altitude=45, spec_depth=1.5, shininess=20, key=1.4, contrast=1.0),
    "Multi-shade \u00b7 Raking ring (inscriptions)": dict(
        mode="multi", multi_preset="Raking ring, 8 lights (inscriptions)", ambient=0.15, texture_mix=0.25,
        occlusion=0.15, clarity=0.3, clarity_radius=10, base_tone=0.6, contrast=1.2),
    "Multi-shade \u00b7 RGB orientation": dict(
        mode="multi", multi_preset="RGB orientation (3 coloured lights)", ambient=0.1, texture_mix=0.0,
        clarity=0.15, base_tone=0.6, contrast=1.2),
    "Multi-shade \u00b7 Warm key / cool fill": dict(
        mode="multi", multi_preset="Warm key / cool fill", ambient=0.2, wrap=0.15, texture_mix=0.3,
        occlusion=0.2, clarity=0.2, base_tone=0.62, contrast=1.1),
    "Studio · Drypoint & incision": dict(
        mode="studio", azimuth=315, altitude=20, key=1.0, fill=0.1, fill_altitude=25, ambient=0.25, wrap=0.05,
        shadows=0.4, shadow_softness=2.0, shadow_reach=64, detail_boost=0.9, detail_radius=2.0,
        texture_detail=0.8, texture_mix=0.3, occlusion=0.15, curvature=0.4, curvature_radius=1.5,
        slope_shade=0.35, slope_range=35, clarity=0.25, clarity_radius=8, base_tone=0.6, contrast=1.2),
    "Multi-shade · Inscription reading": dict(
        mode="multi", multi_preset="Raking ring, 8 lights (inscriptions)", ambient=0.15, detail_boost=0.8,
        detail_radius=2.0, texture_mix=0.2, occlusion=0.15, curvature=0.5, curvature_radius=1.5, slope_shade=0.4,
        slope_range=35, clarity=0.3, clarity_radius=10, base_tone=0.62, contrast=1.25),
    "Exaggerated · Fine detail (all scales)": dict(
        mode="exaggerated", azimuth=315, altitude=35, exag_contrast=8, exag_scales=5, exag_detail=0.5,
        exag_base=0.2, texture_mix=0.0, curvature=0.2, clarity=0.15, base_tone=0.6, contrast=1.0),
    "Exaggerated · Drypoint lines": dict(
        mode="exaggerated", azimuth=315, altitude=30, exag_contrast=14, exag_scales=4, exag_detail=1.0,
        exag_base=0.1, detail_boost=0.8, detail_radius=2.0, texture_mix=0.0, curvature=0.35,
        curvature_radius=1.0, slope_shade=0.25, slope_range=35, clarity=0.2, base_tone=0.6, contrast=1.1),
    "Archaeological blend (VAT)": dict(mode="vat", azimuth=315, altitude=35, svf_radius=16, vat_variant="general"),
    "Archaeological blend (VAT, low relief)": dict(mode="vat", azimuth=315, altitude=35, svf_radius=16,
                                                   vat_variant="low-relief"),
    "Frequency bands · Soft form, crisp detail": dict(
        mode="bands", band_preset="Soft form, crisp detail (resolution bumping)", texture_mix=0.0, clarity=0.15,
        base_tone=0.62, contrast=1.05),
    "Frequency bands · Drypoint lines": dict(
        mode="bands", band_preset="Drypoint lines", texture_mix=0.0, clarity=0.2, base_tone=0.6, contrast=1.15),
    "Frequency bands · Inscription": dict(
        mode="bands", band_preset="Inscription (lit from every side)", texture_mix=0.0, clarity=0.2,
        base_tone=0.6, contrast=1.2),
    "Brown texture shading 50 %": dict(mode="texture", texture_detail=0.5, texture_contrast=1.0,
                                       texture_midtone=0.0, gamma=1.0),
    "Brown texture shading 100 % (steep relief)": dict(mode="texture", texture_detail=1.0, texture_contrast=1.2),
    "Texture + hillshade blend (Patterson)": dict(
        mode="studio", altitude=45, key=1.0, fill=0.0, ambient=0.0, texture_detail=0.5, texture_mix=0.5,
        base_tone=0.62, contrast=1.0),
    "Soft light dome (16 lights)": dict(mode="dome", dome_lights=16, dome_altitude=35, contrast=1.0),
    "Sky-view factor (cracks & incisions)": dict(mode="svf", svf_radius=24, svf_dirs=16, contrast=1.2),
}

SURFACE_PRESETS = {  # sigmas, gains, median, pre-blur, detrend
    "Neutral Scientific": ([64, 32, 16, 8], [1, 1, 1, 1, 1], 0, 0, False),
    "Smooth Vellum Dry-Point": ([96, 40, 14, 4], [0.8, 1.0, 1.25, 1.45, 0.9], 3, 0.35, True),
    "Parchment Cockling": ([128, 48, 16, 5], [1.3, 1.35, 1.0, 0.9, 0.7], 3, 0.5, True),
    "Iron Gall Ink": ([80, 32, 12, 3], [0.8, 0.95, 1.45, 1.25, 0.75], 0, 0.2, False),
    "Incised Metal": ([64, 24, 8, 2.5], [0.7, 0.9, 1.25, 1.7, 1.0], 0, 0.0, True),
    "Engraved Plate": ([80, 28, 9, 2.5], [0.7, 0.9, 1.3, 1.6, 0.9], 0, 0.0, True),
    "Painted Surface": ([128, 48, 16, 5], [1.0, 1.1, 1.2, 0.9, 0.6], 3, 0.45, True),
    "Craquelure Inspection": ([96, 36, 10, 2.5], [0.7, 0.9, 1.15, 1.7, 1.15], 0, 0.0, False),
    "Paper Fibre": ([80, 30, 10, 3], [0.7, 0.9, 1.1, 1.15, 0.65], 3, 0.35, False),
    "Wax Seal": ([96, 36, 12, 3], [0.8, 1.0, 1.3, 1.45, 0.85], 0, 0.15, True),
    "Stone Tool Marks": ([128, 48, 14, 4], [0.8, 1.1, 1.4, 1.5, 0.8], 0, 0.1, True),
    "Low-Noise Documentation": ([96, 40, 16, 5], [1, 1, 1, 1, 0.7], 3, 0.6, False),
}

# What the Presets menu offers first: the object in front of you, as a look and a surface preset together.
OBJECT_PRESETS = {
    "Manuscript · ink and drypoint": ("Studio · Drypoint & incision", "Smooth Vellum Dry-Point"),
    "Manuscript · parchment cockling": ("Clean relief · All-round (default)", "Parchment Cockling"),
    "Paper · fibre and watermark": ("Brown texture shading 50 %", "Paper Fibre"),
    "Seal · wax impression": ("Multi-shade · Inscription reading", "Wax Seal"),
    "Coin or medal": ("Studio · Metal plate (specular)", "Incised Metal"),
    "Engraved gem": ("Studio · Raking detail", "Engraved Plate"),
    "Inscription · stone or metal": ("Multi-shade · Raking ring (inscriptions)", "Stone Tool Marks"),
    "Binding · tooling": ("Clean relief · Crisp detail (tooling, drypoint)", "Incised Metal"),
    "Copper plate · engraved lines": ("Exaggerated · Drypoint lines", "Engraved Plate"),
    "Painting · craquelure": ("Sky-view factor (cracks & incisions)", "Craquelure Inspection"),
    "Painting · impasto": ("Studio · Raking detail", "Painted Surface"),
    "Small find · tool marks": ("Clean relief · Raking all-round", "Stone Tool Marks"),
    "Landscape · archaeology": ("Archaeological blend (VAT)", "Neutral Scientific"),
    "Neutral documentation": ("Clean relief · All-round (default)", "Low-Noise Documentation"),
}
HELIO_OBJECTS = {"coin": "Coin or medal", "seal": "Seal · wax impression", "gem": "Engraved gem",
                 "inscription": "Inscription · stone or metal", "manuscript": "Manuscript · ink and drypoint",
                 "binding": "Binding · tooling", "find": "Small find · tool marks"}   # Helio's object types

STACK_PRESETS = {"Documentation (8 × 40°)": (8, 40.0), "Raking relief (8 × 15°)": (8, 15.0),
                 "Fine incision (16 × 10°)": (16, 10.0), "Broad form (8 × 35°)": (8, 35.0),
                 "Grazing microtexture (16 × 7°)": (16, 7.0)}

BLEND_MODES = {"Normal": QPainter.CompositionMode.CompositionMode_SourceOver,
               "Multiply": QPainter.CompositionMode.CompositionMode_Multiply,
               "Screen": QPainter.CompositionMode.CompositionMode_Screen,
               "Overlay": QPainter.CompositionMode.CompositionMode_Overlay,
               "Soft light": QPainter.CompositionMode.CompositionMode_SoftLight,
               "Hard light": QPainter.CompositionMode.CompositionMode_HardLight}

LEGACY_KEYS = {"w_u": "mix_w_u", "z_u": "mix_z_u", "w_h": "mix_w_h", "z_h": "mix_z_h", "w_l": "mix_w_l",
               "z_l": "mix_z_l", "z_master": "z_factor", "azimuth_offset": "ls_offset",
               "light_altitude_offset": "ls_altitude", "export_dir": "export_dir"}

STYLE = """
QPushButton[primary="true"] { background: #2b5797; color: white; font-weight: 600; padding: 6px 18px;
                              border: 1px solid #1f4175; border-radius: 4px; }
QPushButton[primary="true"]:hover { background: #3567ae; }
QPushButton[primary="true"]:disabled { background: #8a9bb5; border-color: #8a9bb5; }
QLabel[hint="true"] { color: #7d8594; }
"""


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------

def _raster_combo(allow_empty=False):
    cb = QgsMapLayerComboBox()
    try:
        cb.setFilters(Qgis.LayerFilter.RasterLayer)
    except (AttributeError, TypeError):
        from qgis.core import QgsMapLayerProxyModel
        cb.setFilters(QgsMapLayerProxyModel.Filter.RasterLayer)
    if allow_empty:
        try:
            cb.setAllowEmptyLayer(True, "— none —")
        except TypeError:
            cb.setAllowEmptyLayer(True)
        cb.setLayer(None)
    return cb


def _hbox(*widgets, stretch_first=True):
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    for i, item in enumerate(widgets):
        lay.addWidget(item, 1 if (i == 0 and stretch_first) else 0)
    return w


def _hint(text):
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setProperty("hint", True)
    return lbl


def style_exact(layer, lo=0, hi=255):
    """Display 8/16-bit renders 1:1. QGIS's default 2-98 % cumulative stretch
    would otherwise change the tones of a finished render."""
    prov = layer.dataProvider()
    algo = QgsContrastEnhancement.ContrastEnhancementAlgorithm.StretchToMinimumMaximum

    def ce(band):
        c = QgsContrastEnhancement(prov.dataType(band))
        c.setContrastEnhancementAlgorithm(algo)
        c.setMinimumValue(lo)
        c.setMaximumValue(hi)
        return c

    if layer.bandCount() >= 3:
        r = QgsMultiBandColorRenderer(prov, 1, 2, 3)
        r.setRedContrastEnhancement(ce(1))
        r.setGreenContrastEnhancement(ce(2))
        r.setBlueContrastEnhancement(ce(3))
    else:
        r = QgsSingleBandGrayRenderer(prov, 1)
        r.setContrastEnhancement(ce(1))
    layer.setRenderer(r)


def style_cumulative_cut(layer, lower=0.02, upper=0.98):
    """Grey stretch between the 2 % and 98 % points of the histogram, as QGIS's 'Cumulative count cut'."""
    lo, hi = layer.dataProvider().cumulativeCut(1, lower, upper, QgsRectangle(), 250000)
    style_exact(layer, lo, max(hi, lo + 1))


def apply_pseudocolor_ramp(layer, ramp_key, band=1):
    """Semi-transparent diverging ramp scaled to the band's min/max."""
    stops = COLOR_RAMP_PRESETS.get(ramp_key)
    if not stops:
        return
    prov = layer.dataProvider()
    stats = prov.bandStatistics(band)
    vmin, vmax = stats.minimumValue, stats.maximumValue
    span = (vmax - vmin) or 1.0
    shader = QgsColorRampShader()
    shader.setColorRampType(QgsColorRampShader.Type.Interpolated)
    shader.setColorRampItemList([QgsColorRampShader.ColorRampItem(vmin + f * span, QColor(*c), f"{vmin + f * span:.3g}")
                                 for f, c in stops])
    rs = QgsRasterShader()
    rs.setRasterShaderFunction(shader)
    layer.setRenderer(QgsSingleBandPseudoColorRenderer(prov, band, rs))


def style_overlay(layer, stops, band=1):
    """Pseudocolour with fixed (value, rgba) stops - for Helio's uncertainty and metal layers, which mean the same
    numbers on every object, so the colours should too."""
    shader = QgsColorRampShader()
    shader.setColorRampType(QgsColorRampShader.Type.Interpolated)
    shader.setColorRampItemList([QgsColorRampShader.ColorRampItem(v, QColor(*c), label) for v, c, label in stops])
    rs = QgsRasterShader()
    rs.setRasterShaderFunction(shader)
    layer.setRenderer(QgsSingleBandPseudoColorRenderer(layer.dataProvider(), band, rs))


# Helio's extra layers: the fit residual (0-1, high = least believable normal) and where the metal-aware solve
# re-solved the normals (1 = gilding / metal). Both are loaded hidden, to switch on over a render.
HELIO_OVERLAYS = (
    ("uncertainty", "uncertainty", [(0.0, (255, 255, 255, 0), "0 · well fitted"), (0.15, (255, 190, 0, 110), ""),
                                    (0.4, (215, 25, 28, 200), "0.4+ · least believable")]),
    ("metal", "metal (re-solved)", [(0.0, (0, 0, 0, 0), "not metal"), (1.0, (212, 165, 40, 160), "re-solved as metal")]),
)


def _read_json(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)


def _read_presets():
    return _read_json(PRESET_FILE)


def _write_presets(data):
    _write_json(PRESET_FILE, data)


def _physical_ram():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return None


def _safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "output"


def _unique_base(folder, base, ext="tif"):
    """Path without extension that does not collide with an existing file
    (overwriting a GeoTIFF that QGIS still has open is unreliable)."""
    candidate, i = os.path.join(folder, base), 2
    while os.path.exists(f"{candidate}.{ext}"):
        candidate = os.path.join(folder, f"{base}_{i}")
        i += 1
    return candidate


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _write_sidecar(path, data):
    data = dict(data)
    data["generated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    data["plugin_version"] = VERSION
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)
    return path


def _tiles(bounds, tile, margin, width, height):
    """Yield (core, window) over bounds = (x0, y0, x1, y1): core is the block
    written to the output, window = (xoff, yoff, xsize, ysize) is read with up
    to `margin` px of context, clipped to the raster (width, height)."""
    bx0, by0, bx1, by1 = bounds
    for y0 in range(by0, by1, tile):
        for x0 in range(bx0, bx1, tile):
            x1, y1 = min(bx1, x0 + tile), min(by1, y0 + tile)
            rx0, ry0 = max(0, x0 - margin), max(0, y0 - margin)
            rx1, ry1 = min(width, x1 + margin), min(height, y1 + margin)
            yield (x0, y0, x1, y1), (rx0, ry0, rx1 - rx0, ry1 - ry0)


def _decimate(z, max_dim):
    k = max(1, int(math.ceil(max(z.shape) / float(max_dim))))
    if k == 1:
        return z, 1
    h, w = (z.shape[0] // k) * k, (z.shape[1] // k) * k
    return z[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3)), k


# ----------------------------------------------------------------------------
# Worker-thread jobs. They receive plain dicts and never touch widgets or the
# project; the dialog adds their outputs to the map when they finish.
# ----------------------------------------------------------------------------

def _plugin_version():
    try:
        with open(os.path.join(os.path.dirname(__file__), "metadata.txt"), encoding="utf-8") as f:
            return next((line.split("=", 1)[1].strip() for line in f if line.startswith("version=")), "")
    except OSError:
        return ""


def tiled_job(task, job):
    """Copy a slow-to-read colour image / normal map to a tiled GeoTIFF (on the height map's georeferencing)."""
    t0 = time.time()
    rio.make_tiled_copy(job["source"], job["out"], job.get("ref"), lambda pc, msg: task.report(pc, msg))
    return dict(path=job["out"], key=job["key"], seconds=time.time() - t0)


def trace_job(task, job):
    """Trace the height map's lines (tools.trace_lines) and write them to a GeoPackage, georeferenced."""
    from osgeo import ogr, osr
    from . import tools
    t0 = time.time()
    info = rio.RasterInfo(job["source"])
    win = job["window"] or (0, 0, info.width, info.height)
    k = max(1.0, math.sqrt(win[2] * win[3] / float(job["max_px"])))
    d = job.get("detail")
    read = win
    if d and job["window"]:                                  # the band needs context round the view
        from . import detail
        m = detail.margin_px(d)
        x0, y0 = max(0, win[0] - m), max(0, win[1] - m)
        read = (x0, y0, min(info.width, win[0] + win[2] + m) - x0, min(info.height, win[1] + win[3] + m) - y0)
    out = (max(2, int(round(read[2] / k))), max(2, int(round(read[3] / k))))
    kx, ky = read[2] / float(out[0]), read[3] / float(out[1])
    task.report(5, "Reading heights")
    z = rio.read_array(info.path, read, out)[0]
    if d:
        from . import detail
        task.report(12, "Isolating the Detail tab's band")
        st = detail.detail_stats(job["ref"], d, job["dx"], job["dy"], job["plane"], job["ref_origin"])
        z = detail.isolate(z, d, kx, st, job["plane"], (read[0], read[1]))[0]
    task.report(20, "Finding the lines")
    lines, _ = tools.trace_lines(z, job["width"] / kx, job["sensitivity"], job["min_length"] / kx, job["kind"],
                                 job["simplify"] / kx, bridge_px=job.get("bridge", 0.0) / kx,
                                 smooth=int(job.get("smooth", 0)), min_strength=job.get("min_strength", 0.0))
    if read != win:                                          # keep the lines of the view itself
        lines = [l for l in lines if win[0] <= read[0] + (np.mean([p[0] for p in l["points"]]) + 0.5) * kx
                 <= win[0] + win[2] and win[1] <= read[1] + (np.mean([p[1] for p in l["points"]]) + 0.5) * ky
                 <= win[1] + win[3]]
    task.report(85, "Writing the GeoPackage")
    gt = info.geotransform
    if os.path.exists(job["out"]):
        os.remove(job["out"])
    ds = ogr.GetDriverByName("GPKG").CreateDataSource(job["out"])
    srs = None
    if info.projection:
        srs = osr.SpatialReference()
        srs.ImportFromWkt(info.projection)
    lyr = ds.CreateLayer("lines", srs, ogr.wkbLineString)
    for name, typ in (("kind", ogr.OFTString), ("length", ogr.OFTReal), ("strength", ogr.OFTReal),
                      ("depth", ogr.OFTReal), ("orientation", ogr.OFTReal)):
        lyr.CreateField(ogr.FieldDefn(name, typ))
    px, total, overlay = job["pixel"], 0.0, []
    for line in lines:
        pts = [(read[0] + (x + 0.5) * kx, read[1] + (y + 0.5) * ky) for x, y in line["points"]]   # full-res corners
        g = ogr.Geometry(ogr.wkbLineString)
        for fx, fy in pts:
            g.AddPoint_2D(gt[0] + fx * gt[1] + fy * gt[2], gt[3] + fx * gt[4] + fy * gt[5])
        length = sum(math.hypot((b[0] - a[0]) * px[0], (b[1] - a[1]) * px[1]) for a, b in zip(pts, pts[1:]))
        f = ogr.Feature(lyr.GetLayerDefn())
        f.SetGeometry(g)
        f.SetField("kind", line["kind"])
        f.SetField("length", length)
        f.SetField("strength", line["strength"])
        f.SetField("depth", line["depth"])                  # height units, against the surface round the mark
        f.SetField("orientation", line["orientation"])      # degrees anticlockwise from east (image x), 0-180
        lyr.CreateFeature(f)
        f = None
        total += length
        overlay.append((line["kind"], pts))
    ds = None
    return dict(path=job["out"], count=len(lines), total=total, unit=job["unit"], overlay=overlay, k=k,
                seconds=time.time() - t0)


def detail_job(task, job):
    """The Detail tab's band of the height map (float32 heights) or its display (8-bit), tiled, georeferenced."""
    from . import detail
    t0 = time.time()
    info = rio.RasterInfo(job["source"])
    W, H = info.width, info.height
    ax0, ay0, ow, oh = job.get("window") or (0, 0, W, H)
    gt = info.geotransform
    out_gt = (gt[0] + ax0 * gt[1] + ay0 * gt[2], gt[1], gt[2], gt[3] + ax0 * gt[4] + ay0 * gt[5], gt[4], gt[5])
    d, what = detail.params(job["detail"]), job["what"]
    task.report(2, "Measuring the reference")
    st = detail.detail_stats(job["ref"], d, job["dx"], job["dy"], job["plane"], job["ref_origin"])
    whole = (ow, oh) == (W, H) and W * H <= 36e6
    tile = max(ow, oh) if whole else 3072
    margin = 0 if whole else detail.margin_px(d)
    if what == "band":
        w = rio.GeoTiffWriter(job["out_base"] + ".tif", ow, oh, 1, "float32", out_gt, info.projection,
                              float("nan") if info.nodata is not None else None)
    else:
        w = rio.GeoTiffWriter(job["out_base"] + ".tif", ow, oh, 3 if d["display"] == "signed_colour" else 1, "uint8",
                              out_gt, info.projection, None)
    tiles = list(_tiles((ax0, ay0, ax0 + ow, ay0 + oh), tile, margin, W, H))
    try:
        for i, (core, win) in enumerate(tiles):
            task.report(5 + 85.0 * i / len(tiles), f"Tile {i + 1}/{len(tiles)}" if len(tiles) > 1
                        else "Isolating the band")
            z, _ = rio.read_array(info.path, win)
            z, valid = eng.fill_nodata(z)
            band, removed = detail.isolate(z, d, 1.0, st, job["plane"], (win[0], win[1]))
            if what == "band":
                arr = band if valid is None else np.where(valid, band, np.float32(np.nan))
            else:
                arr = eng.to_uint8(detail.render(band, removed, d, job["params"], job["dx"], job["dy"], st, 1.0))
            y0, x0 = core[1] - win[1], core[0] - win[0]
            sl = (slice(y0, y0 + core[3] - core[1]), slice(x0, x0 + core[2] - core[0]))
            w.write(arr[sl], core[0] - ax0, core[1] - ay0)
        task.report(92, "Building overviews")
        w.close()
    except BaseException:
        w.close(overviews=False)
        if os.path.exists(w.path):
            os.remove(w.path)
        raise
    sidecar = _write_sidecar(job["out_base"] + "_provenance.json",
                             dict(job["meta"], detail=d, noise_model=st["model"], tiles=len(tiles),
                                  seconds=round(time.time() - t0, 2)))
    return dict(path=w.path, what=what, sidecar=sidecar, seconds=time.time() - t0)


def compare_job(task, job):
    """Compare two integrations of the same normals at nine places; recommend one or a fusion; thumbnails."""
    from . import tools
    t0 = time.time()
    ia, ib = rio.RasterInfo(job["a"]), rio.RasterInfo(job["b"])
    if (ia.width, ia.height) != (ib.width, ib.height):
        raise RuntimeError("A and B are not the same size - compare two integrations of the same capture.")
    W, H = ia.width, ia.height
    P = int(min(512, W // 4, H // 4))
    metrics = []
    for gy in range(3):
        for gx in range(3):
            x = int(min(max(W * (0.2 + 0.3 * gx) - P / 2, 0), W - P))
            y = int(min(max(H * (0.2 + 0.3 * gy) - P / 2, 0), H - P))
            za = eng.fill_nodata(rio.read_array(job["a"], (x, y, P, P))[0])[0]
            zb = eng.fill_nodata(rio.read_array(job["b"], (x, y, P, P))[0])[0]
            metrics.append(tools.compare_integrations(za, zb))
            task.report(5 + 60 * len(metrics) / 9.0, "Comparing detail, noise and drift")
    choice, why = tools.recommend(metrics, job["names"])
    cw = (max(0, W // 2 - 512), max(0, H // 2 - 512), min(1024, W), min(1024, H))
    gain = tools.fuse_gain(eng.fill_nodata(rio.read_array(job["a"], cw)[0])[0],
                           eng.fill_nodata(rio.read_array(job["b"], cw)[0])[0], job["sigma"])
    summary = {s: {k: float(np.median([m[s][k] for m in metrics])) for k in ("detail", "noise", "snr", "drift")}
               for s in ("a", "b")}
    task.report(75, "Thumbnails")
    view = job["view"]
    k = max(1.0, max(view[2], view[3]) / 300.0)
    out = (max(2, int(view[2] / k)), max(2, int(view[3] / k)))
    kx = view[2] / float(out[0])
    za = eng.fill_nodata(rio.read_array(job["a"], view, out)[0])[0]
    zb = eng.fill_nodata(rio.read_array(job["b"], view, out)[0])[0]
    fused = tools.fuse_heights(za, zb, job["sigma"] / kx, gain)
    thumbs = [eng.render(z, job["dx"] * kx, job["dx"] * kx, job["params"], kx)[0] for z in (za, zb, fused)]
    return dict(metrics=summary, agreement=float(np.median([m["agreement"] for m in metrics])), choice=choice, why=why,
                gain=gain, sigma=job["sigma"], thumbs=thumbs, names=job["names"], seconds=time.time() - t0)


def fuse_job(task, job):
    """A's broad form + B's fine detail (scaled to A's heights) as a float32 GeoTIFF, tile by tile."""
    from . import tools
    t0 = time.time()
    ia = rio.RasterInfo(job["a"])
    W, H, sigma = ia.width, ia.height, float(job["sigma"])
    gain = job.get("gain")
    if gain is None:
        cw = (max(0, W // 2 - 512), max(0, H // 2 - 512), min(1024, W), min(1024, H))
        gain = tools.fuse_gain(eng.fill_nodata(rio.read_array(job["a"], cw)[0])[0],
                               eng.fill_nodata(rio.read_array(job["b"], cw)[0])[0], sigma)
    nodata = ia.nodata
    w = rio.GeoTiffWriter(job["out"], W, H, 1, "float32", ia.geotransform, ia.projection, nodata)
    tiles = list(_tiles((0, 0, W, H), 2048, int(4 * sigma) + 4, W, H))
    try:
        for i, (core, win) in enumerate(tiles):
            za, va = eng.fill_nodata(rio.read_array(job["a"], win)[0])
            zb, vb = eng.fill_nodata(rio.read_array(job["b"], win)[0])
            f = tools.fuse_heights(za, zb, sigma, gain)
            valid = np.ones(f.shape, bool) if va is None else va
            if vb is not None:
                valid = valid & vb
            f = np.where(valid, f, np.float32(nodata) if nodata is not None else np.float32(np.nan))
            y0, x0 = core[1] - win[1], core[0] - win[0]
            w.write(f[y0:y0 + core[3] - core[1], x0:x0 + core[2] - core[0]].astype(np.float32), core[0], core[1])
            task.report(95.0 * (i + 1) / len(tiles), f"Tile {i + 1}/{len(tiles)}")
        w.close(overviews=True)
    except BaseException:
        w.close(overviews=False)
        raise
    return dict(path=job["out"], sigma=sigma, gain=gain, seconds=time.time() - t0)


def plate_job(task, job):
    """Render the plate's panels: the whole surface (grey, or colour composite) and a close-up of the view."""
    t0 = time.time()
    info = rio.RasterInfo(job["source"])
    W, H, p = info.width, info.height, job["params"]
    panels = []
    for i, pan in enumerate(job["panels"]):
        task.report(10 + 80 * i / len(job["panels"]), f"Rendering the {pan['kind']} panel")
        if pan["kind"] in ("grey", "colour"):
            win = (0, 0, W, H)
            margin = 0
        else:
            win = job["view"] or (0, 0, W, H)
            margin = eng.spectral_margin_px(p)
        k = max(win[2] / float(pan["w"]), win[3] / float(pan["h"]), 1.0)
        mk = int(margin)
        full = (max(0, win[0] - mk), max(0, win[1] - mk), min(W, win[0] + win[2] + mk), min(H, win[1] + win[3] + mk))
        rw = (full[0], full[1], full[2] - full[0], full[3] - full[1])
        out = (max(2, int(round(rw[2] / k))), max(2, int(round(rw[3] / k))))
        kx, ky = rw[2] / float(out[0]), rw[3] / float(out[1])
        z = rio.read_array(info.path, rw, out)[0]
        nm = rio.read_normals(job["normals"], (W, H), rw, out) if job.get("normals") else None
        img = eng.render(z, job["dx"] * kx, job["dy"] * ky, p, kx, job["stats"], job["plane"], (rw[0], rw[1]),
                         normals=nm)[0]
        if pan["kind"] == "colour" and job.get("colour"):
            grey = img if img.ndim == 2 else img.mean(axis=2)
            img = eng.colour_composite(grey, rio.read_rgb(job["colour"], (W, H), rw, out), p, job["albedo_strength"])
        ox, oy = int(round((win[0] - rw[0]) / kx)), int(round((win[1] - rw[1]) / ky))
        ow, oh = max(1, int(round(win[2] / kx))), max(1, int(round(win[3] / ky)))
        panels.append(dict(img=np.ascontiguousarray(img[oy:oy + oh, ox:ox + ow]), k=kx))
    return dict(panels=panels, seconds=time.time() - t0)


def align_job(task, job):
    """Register the colour / albedo image onto the height map and write the aligned VRT (align.py)."""
    from . import align
    t0 = time.time()
    res = align.align_to(job["ref"], job["img"], job["out"], lambda pc, msg: task.report(pc, msg))
    res["seconds"] = time.time() - t0
    return res


def render_job(task, job):
    t0 = time.time()
    info = rio.RasterInfo(job["source"])
    p, W, H = job["params"], info.width, info.height
    ax0, ay0, ow, oh = job.get("window") or (0, 0, W, H)    # optional sub-area, e.g. the preview view
    gt = info.geotransform
    out_gt = (gt[0] + ax0 * gt[1] + ay0 * gt[2], gt[1], gt[2], gt[3] + ax0 * gt[4] + ay0 * gt[5], gt[4], gt[5])
    whole = ow * oh / 1e6 <= job.get("tile_mp", 36)
    tile = max(ow, oh) if whole else 3072
    margin = eng.spectral_margin_px(p) if (not whole or (ow, oh) != (W, H)) else 0
    rgb = eng.is_rgb(p) or bool(job["albedo"])
    bands = 3 if rgb else 1
    gd = p.get("mode") == "gdal"                  # gdaldem: nodata 0, grey levels 1-255 kept exactly
    nodata = 0 if (info.nodata is not None or gd) else None
    base = job["out_base"]
    layout = [] if gd else None                  # gdaldem's plain strips, so QGIS samples and styles it identically
    w8 = rio.GeoTiffWriter(base + ".tif", ow, oh, bands, "uint8", out_gt, info.projection, nodata, layout)
    w16 = (rio.GeoTiffWriter(base + "_16bit.tif", ow, oh, bands, "uint16", out_gt, info.projection, nodata, layout)
           if job["out16"] else None)
    tiles = list(_tiles((ax0, ay0, ax0 + ow, ay0 + oh), tile, margin, W, H))
    try:
        for i, (core, win) in enumerate(tiles):
            prefix = f"Tile {i + 1}/{len(tiles)}: " if len(tiles) > 1 else ""
            prog = sub_progress(task, 90.0 * i / len(tiles), 90.0 * (i + 1) / len(tiles), prefix)
            prog(0, "reading heights")
            z, _ = rio.read_array(info.path, win)
            valid = np.isfinite(z)
            nm = (rio.read_normals(job["normals"], (W, H), win)
                  if job.get("normals") and p.get("mode") == "clean" and p.get("clean_source") == "normals" else None)
            img, _ = eng.render(z, job["dx"], job["dy"], p, 1.0, job["stats"], job["plane"], (win[0], win[1]), prog,
                                normals=nm)
            if job["albedo"]:
                img = eng.colour_composite(img, rio.read_rgb(job["albedo"], (W, H), win), p, job["albedo_strength"])
            y0, x0 = core[1] - win[1], core[0] - win[0]
            sl = (slice(y0, y0 + core[3] - core[1]), slice(x0, x0 + core[2] - core[0]))
            img, valid = img[sl], valid[sl]
            out8 = eng.to_uint8(img)
            if nodata is not None:
                m = valid if out8.ndim == 2 else valid[..., None]
                out8 = np.where(m, out8 if gd else np.maximum(out8, 1), 0).astype(np.uint8)
            w8.write(out8, core[0] - ax0, core[1] - ay0)
            if w16:
                w16.write(eng.to_uint16(img), core[0] - ax0, core[1] - ay0)
        task.report(91, "Building overviews")
        w8.close(overviews=not gd)               # gdaldem writes no overviews - nor do we, so QGIS styles it alike
        if w16:
            w16.close(overviews=not gd)
    except BaseException:
        for w in (w8, w16):
            if w:
                w.close(overviews=False)
                if os.path.exists(w.path):
                    os.remove(w.path)
        raise
    outputs = [w8.path] + ([w16.path] if w16 else [])
    if job["png"]:
        task.report(96, "Writing PNG")
        outputs.append(rio.export_png(w8.path, base + ".png"))
    meta = dict(job["meta"], outputs=outputs, tiles=len(tiles), normalisation=job["stats"],
                seconds=round(time.time() - t0, 2))
    sidecar = _write_sidecar(base + "_provenance.json", meta)
    return {"outputs": outputs, "tif": w8.path, "sidecar": sidecar, "seconds": time.time() - t0}


def split_job(task, job):
    t0 = time.time()
    info = rio.RasterInfo(job["source"])
    s, W, H = job["settings"], info.width, info.height
    whole = info.megapixels <= 64
    tile = max(W, H) if whole else 3072
    margin = 0 if whole else int(min(1024, 3 * max(s["sigmas"]) + 16))
    nodata = float("nan") if info.nodata is not None else None
    writers = {k: rio.GeoTiffWriter(_unique_base(job["out_dir"], job["prefix"] + k) + ".tif", W, H, 1, "float32",
                                    info.geotransform, info.projection, nodata) for k in s["enabled"]}
    tiles = list(_tiles((0, 0, W, H), tile, margin, W, H))
    try:
        for i, (core, win) in enumerate(tiles):
            task.report(2 + 85.0 * i / len(tiles), f"Splitting tile {i + 1}/{len(tiles)}" if len(tiles) > 1
                        else "Splitting into five bands")
            z, _ = rio.read_array(info.path, win)
            z, valid = eng.fill_nodata(z)
            z = eng.condition_surface(z, s, 1.0, job["plane"], (win[0], win[1]))
            bands = eng.post_split(eng.frequency_split(z, s["sigmas"]), s)
            y0, x0 = core[1] - win[1], core[0] - win[0]
            sl = (slice(y0, y0 + core[3] - core[1]), slice(x0, x0 + core[2] - core[0]))
            for k, w in writers.items():
                arr = bands[k] * np.float32(s["gains"][k])
                if valid is not None:
                    arr = np.where(valid, arr, np.float32(np.nan))
                w.write(arr[sl], core[0], core[1])
        task.report(90, "Building overviews")
        for w in writers.values():
            w.close()
    except BaseException:
        for w in writers.values():
            w.close(overviews=False)
            if os.path.exists(w.path):
                os.remove(w.path)
        raise
    files = [(k, w.path) for k, w in writers.items()]
    sidecar = _write_sidecar(os.path.join(job["out_dir"], job["prefix"] + "frequency_split_provenance.json"),
                             dict(job["meta"], settings=s, bands_written=dict(files),
                                  exact_reconstruction=(len(files) == 5 and all(g == 1 for g in s["gains"].values())
                                                        and all(t["mode"] == "None" and t["threshold"] == 0
                                                                for t in s["treat"].values()))))
    return {"files": files, "sidecar": sidecar, "seconds": time.time() - t0}


def _load_stack_surface(job):
    """Full-resolution surface for light-stack jobs, plus the reference RasterInfo."""
    info = rio.RasterInfo(job["source"])
    if job["source_mode"] == "mix":
        surface = np.zeros((info.height, info.width), np.float32)
        for path, weight, zf in job["mix"]:
            a, _ = rio.read_array(path, None, (info.width, info.height))  # resampled onto the first map's grid
            surface += eng.fill_nodata(a)[0] * np.float32(weight * zf)
        return surface * np.float32(job["z_master"]), info
    z, _ = rio.read_array(info.path)
    z, _ = eng.fill_nodata(z)
    p = eng.merged_params(job["params"])
    if job["source_mode"] == "prepared":
        return eng.prepare_surface(z, job["dx"], job["dy"], p, 1.0, job["plane"])[0], info
    return z * np.float32(p["z_factor"]), info


def stack_job(task, job):
    t0 = time.time()
    task.report(1, "Reading surface")
    surface, info = _load_stack_surface(job)
    H, W = surface.shape
    gt, proj, dx, dy = info.geotransform, info.projection, job["dx"], job["dy"]
    lights, out, pre = job["lights"], job["out_dir"], job["prefix"]
    n = len(lights)
    layers = []

    def new_path(name, ext="tif"):
        return _unique_base(out, pre + name, ext) + "." + ext

    lift = np.float32(job["lift"] / 100.0)

    def tone_fn(img, alt):
        if job["tone"] == 1:
            return np.maximum(img, lift)
        if job["tone"] == 2:
            return eng.tone(img, math.sin(math.radians(alt)), 1.25, 1.0, soft=True)
        return img

    individual = {}
    if job["individual"] or job["vrt"]:
        for i, (suffix, _az, _alt) in enumerate(lights):
            individual[i] = rio.GeoTiffWriter(new_path(f"Light_{_safe_name(suffix)}"), W, H, 1, "uint8", gt, proj)
    multi = rio.GeoTiffWriter(new_path(f"Light_Stack_{n}"), W, H, n, "uint8", gt, proj) if job["multiband"] else None
    if multi:
        multi.set_descriptions([f"{s} (az {a:.1f}, alt {e:.1f})" for s, a, e in lights])
    mean_w = rio.GeoTiffWriter(new_path("Light_Dome_Mean"), W, H, 1, "uint8", gt, proj) if job["mean"] else None
    nrm_w = (rio.GeoTiffWriter(new_path("Normal_Map_" + job["normal_conv"].split()[0]), W, H, 3, "uint8", gt, proj)
             if job["normals"] else None)
    s_sum, gram, count = np.zeros(n), np.zeros((n, n)), 0
    dirs = [(az, alt) for _, az, alt in lights]
    for r0, r1, stack, normals in eng.light_stack_chunks(surface, dx, dy, dirs):
        toned = np.stack([tone_fn(stack[i], lights[i][2]) for i in range(n)])
        q = eng.to_uint8(toned)
        for i, w in individual.items():
            w.write(q[i], 0, r0)
        if multi:
            multi.write(np.moveaxis(q, 0, -1), 0, r0)
        if mean_w:
            mean_w.write(eng.to_uint8(toned.mean(axis=0)), 0, r0)
        if nrm_w:
            nrm_w.write(eng.normal_map_rgb(normals, job["normal_conv"]), 0, r0)
        if job["pca"]:
            x = stack.reshape(n, -1).astype(np.float64)
            s_sum += x.sum(axis=1)
            gram += x @ x.T
            count += x.shape[1]
        task.report(3 + 57.0 * r1 / H, f"Lighting {n} directions: rows {r1:,}/{H:,}")

    task.report(61, "Building overviews")
    for i, w in individual.items():
        w.close()
        if job["individual"]:
            layers.append(dict(path=w.path, name=f"Light_{lights[i][0]}", kind="light"))
    for w, kind in ((multi, "multiband"), (mean_w, "mean"), (nrm_w, "normals")):
        if w:
            w.close()
            layers.append(dict(path=w.path, name=os.path.splitext(os.path.basename(w.path))[0], kind=kind))
    if job["vrt"] and individual:
        vrt = rio.build_vrt(new_path("Light_Stack", "vrt"), [w.path for w in individual.values()])
        layers.append(dict(path=vrt, name="Light_Stack_VRT", kind="vrt"))

    meta = dict(job["meta"])
    if job["pca"] and count:
        task.report(65, "PCA of the light stack")
        mean = s_sum / count
        cov = gram / count - np.outer(mean, mean)
        eig, vec = np.linalg.eigh(cov)
        order = np.argsort(eig)[::-1]
        eig, vec = eig[order], vec[:, order]
        k = min(job["pca_n"], n)
        v32, m32 = vec[:, :k].astype(np.float32), mean.astype(np.float32)[:, None]
        sd = np.sqrt(np.maximum(eig[:3], 1e-12)).astype(np.float32)
        pw = rio.GeoTiffWriter(new_path("PCA_Components"), W, H, k, "float32", gt, proj)
        pw.set_descriptions([f"PC{i + 1}" for i in range(k)])
        cw = rio.GeoTiffWriter(new_path("PCA_Composite_RGB"), W, H, 3, "uint8", gt, proj)
        for r0, r1, stack, _ in eng.light_stack_chunks(surface, dx, dy, dirs):
            comps = (v32.T @ (stack.reshape(n, -1) - m32)).reshape(k, r1 - r0, W)
            pw.write(np.moveaxis(comps, 0, -1), 0, r0)
            rgb = [eng.to_uint8(0.5 + comps[min(c, k - 1)] / (5.0 * sd[min(c, k - 1)])) for c in range(3)]
            cw.write(np.dstack(rgb), 0, r0)
        pw.close()
        cw.close()
        layers.append(dict(path=cw.path, name="PCA_Composite_RGB", kind="pca_rgb"))
        layers.append(dict(path=pw.path, name="PCA_Components", kind="float"))
        meta["pca_explained_variance_ratio"] = (eig[:k] / max(eig.sum(), 1e-12)).tolist()

    if job["svf"]:
        svf = eng.sky_view_factor(surface, dx, dy, job["svf_radius"], 16, sub_progress(task, 70, 82))
        low = float(np.percentile(svf[::4, ::4], 1.0))
        path = new_path("Sky_View_Factor")
        rio.write_raster(path, eng.to_uint8((svf - low) / max(1e-6, 1.0 - low)), gt, proj, "uint8")
        layers.append(dict(path=path, name="Sky_View_Factor", kind="svf"))
        del svf
    if job["tpi"]:
        task.report(83, "Topographic position index")
        path = new_path("TPI_Ridge_Valley")
        rio.write_raster(path, eng.topographic_position_index(surface, job["tpi_radius"]), gt, proj, "float32")
        layers.append(dict(path=path, name="TPI_Ridge_Valley", kind="float"))

    meta.update(lights=lights, outputs=[l["path"] for l in layers], seconds=round(time.time() - t0, 2))
    sidecar = _write_sidecar(os.path.join(out, pre + "light_stack_provenance.json"), meta)
    return {"layers": layers, "sidecar": sidecar, "seconds": time.time() - t0}


# ----------------------------------------------------------------------------
# Dialog
# ----------------------------------------------------------------------------

class HelioReliefDialog(QDialog):

    def __init__(self, parent=None, iface=None):
        super().__init__(parent)
        self.iface = iface
        self.setWindowTitle(f"Helio Relief {VERSION}")
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowMaximizeButtonHint |
                            Qt.WindowType.WindowMinimizeButtonHint)
        self._w, self._layer_keys, self._run_buttons = {}, set(), []
        self._suspend = True
        self._task = None
        self._run_sync = False          # tests run jobs inline instead of via the task manager
        self._ls = None
        self._last_split = None
        self._look_name, self._look_dirty = None, False
        self._gen, self._shown_gen, self._want_overview = 0, {}, True
        self._last_preview_error = None
        self._to_map_busy = False
        self.renderer = PreviewRenderer(self)
        self.renderer.ready.connect(self._on_preview_ready)
        self.renderer.failed.connect(self._on_preview_failed)
        self._timer = QTimer(self)            # settings changed -> overview + detail
        self._timer.setSingleShot(True)
        self._timer.setInterval(60)
        self._timer.timeout.connect(self._request_previews)
        self._view_timer = QTimer(self)       # pan / zoom -> detail only
        self._view_timer.setSingleShot(True)
        self._view_timer.setInterval(90)
        self._view_timer.timeout.connect(self._request_previews)
        self._strip_timer = QTimer(self)      # side-panel thumbnails, once things settle
        self._strip_timer.setSingleShot(True)
        self._strip_timer.setInterval(450)
        self._strip_timer.timeout.connect(self._request_strip)
        self._strip_sig, self._strip_current, self._strip_gen_shown = None, None, -1
        self._undo_stack, self._redo_stack, self._undo_base = [], [], None    # settings undo / redo
        self._undo_timer = QTimer(self)       # a burst of edits becomes one undo step once it settles
        self._undo_timer.setSingleShot(True)
        self._undo_timer.setInterval(600)
        self._undo_timer.timeout.connect(self._commit_undo)
        self._lines_timer = QTimer(self)      # live line tracing of the view
        self._lines_timer.setSingleShot(True)
        self._lines_timer.setInterval(350)
        self._lines_timer.timeout.connect(self._request_lines)
        self._lines_gen, self._trace_history, self._detail_model = 0, [], None

        self._build_ui()
        self.renderer.busyChanged.connect(self.canvas.setBusy)
        self._labels = self._collect_labels()
        self._last_change_t = 0.0
        for sec in self.findChildren(Section):
            sec.resetRequested.connect(lambda s=sec: self._reset_section(s))
        self._regen_table()
        self._apply_multi_preset(next(iter(eng.MULTI_PRESETS)))
        self._defaults = self.state(include_layers=False)
        self._reload_profiles()
        self._reload_looks()
        self._load_project_state()
        self._suspend = False
        self._update_mode_visibility()
        self._update_source_visibility()
        self._sync_dial()
        self._on_surface_changed()
        if iface is not None:
            try:
                iface.mapCanvas().extentsChanged.connect(self._on_canvas_extent)
            except Exception:
                pass
        self._fit_to_screen()
        self._restore_geometry()
        QShortcut(QKeySequence("Ctrl+Return"), self, self._run_current_tab)
        QShortcut(QKeySequence("Ctrl+R"), self, lambda: self._schedule_preview(force=True))
        QShortcut(QKeySequence("Ctrl+E"), self, lambda: self._quick_export(False))
        QShortcut(QKeySequence("Ctrl+Z"), self, lambda: self._undo_settings())
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, lambda: self._undo_settings(True))
        QShortcut(QKeySequence("Ctrl+Y"), self, lambda: self._undo_settings(True))
        self._update_overlay()
        self._undo_base = self._undo_snapshot()

    # ------------------------------------------------------------------ registry
    def _reg(self, key, widget, layer=False):
        self._w[key] = widget
        if layer:
            self._layer_keys.add(key)
        if isinstance(widget, (SliderSpin, QDoubleSpinBox, QSpinBox)):
            sig = widget.valueChanged
        elif isinstance(widget, QCheckBox):
            sig = widget.toggled
        elif isinstance(widget, QgsMapLayerComboBox):
            sig = widget.layerChanged
        elif isinstance(widget, QComboBox):
            sig = widget.currentIndexChanged
        elif isinstance(widget, QgsFileWidget):
            sig = widget.fileChanged
        elif isinstance(widget, QLineEdit):
            sig = widget.textChanged
        else:
            sig = None
        if sig is not None:
            sig.connect(lambda *_a, k=key: self._on_changed(k))
        return widget

    @staticmethod
    def _value(w):
        if isinstance(w, (SliderSpin, QDoubleSpinBox, QSpinBox)):
            return w.value()
        if isinstance(w, QCheckBox):
            return w.isChecked()
        if isinstance(w, QgsMapLayerComboBox):
            lyr = w.currentLayer()
            return lyr.id() if lyr else ""
        if isinstance(w, QComboBox):
            return w.currentText()
        if isinstance(w, QgsFileWidget):
            return w.filePath()
        if isinstance(w, QLineEdit):
            return w.text()
        return None

    @staticmethod
    def _set_value(w, v):
        try:
            if isinstance(w, (SliderSpin, QDoubleSpinBox)):
                w.setValue(float(v))
            elif isinstance(w, QSpinBox):
                w.setValue(int(v))
            elif isinstance(w, QCheckBox):
                w.setChecked(bool(v))
            elif isinstance(w, QgsMapLayerComboBox):
                lyr = QgsProject.instance().mapLayer(v) if v else None
                if lyr is not None or w.allowEmptyLayer():
                    w.setLayer(lyr)
            elif isinstance(w, QComboBox):
                i = w.findText(str(v))
                if i >= 0:
                    w.setCurrentIndex(i)
            elif isinstance(w, QgsFileWidget):
                w.setFilePath(str(v))
            elif isinstance(w, QLineEdit):
                w.setText(str(v))
        except (TypeError, ValueError):
            pass

    def state(self, include_layers=True):
        d = {k: self._value(w) for k, w in self._w.items() if include_layers or k not in self._layer_keys}
        d["light_table"] = self._table_rows()
        d["multi_lights"] = self._multi_rows()
        return d

    def set_state(self, d):
        d = {LEGACY_KEYS.get(k, k): v for k, v in d.items()}
        prev, self._suspend = self._suspend, True
        try:
            for k, v in d.items():
                if k == "r_mode":
                    v = LABEL_ALIASES.get(v, v)
                if k in self._w:
                    self._set_value(self._w[k], v)
            if d.get("light_table"):
                self._set_table_rows(d["light_table"])
            else:
                self._regen_table()
            if d.get("multi_lights"):
                self._set_multi_rows(d["multi_lights"])
        finally:
            self._suspend = prev
        if not prev:
            self._update_mode_visibility()
            self._update_source_visibility()
            self._sync_dial()
            self._on_surface_changed()
            self._update_overlay()

    # ------------------------------------------------------------------ widget factories
    def _dspin(self, key, lo, hi, val, dec=2, step=0.1, suffix="", tip=None):
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setDecimals(dec)
        w.setSingleStep(step)
        w.setValue(val)
        w.setSuffix(suffix)
        w.setKeyboardTracking(False)
        if tip:
            w.setToolTip(tip)
        return self._reg(key, w)

    def _ispin(self, key, lo, hi, val, step=1, suffix="", tip=None):
        w = QSpinBox()
        w.setRange(lo, hi)
        w.setSingleStep(step)
        w.setValue(val)
        w.setSuffix(suffix)
        w.setKeyboardTracking(False)
        if tip:
            w.setToolTip(tip)
        return self._reg(key, w)

    def _combo(self, key, items, current=None, tip=None):
        w = QComboBox()
        w.addItems(list(items))
        if current is not None:
            w.setCurrentText(current)
        if tip:
            w.setToolTip(tip)
        return self._reg(key, w)

    def _check(self, key, text, checked=False, tip=None):
        w = QCheckBox(text)
        w.setChecked(checked)
        if tip:
            w.setToolTip(tip)
        return self._reg(key, w)

    def _slider(self, key, lo, hi, val, dec=2, step=0.05, log=False, suffix="", tip=None):
        w = SliderSpin(lo, hi, val, dec, step, log, suffix)
        if tip:
            w.setToolTip(tip)
        return self._reg(key, w)

    def _primary(self, text, slot):
        b = QPushButton(text)
        b.setProperty("primary", True)
        b.setMinimumHeight(34)
        b.clicked.connect(slot)
        self._run_buttons.append(b)
        return b

    def _page(self, sections, actions):
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 4, 0, 0)
        inner = QWidget()
        iv = QVBoxLayout(inner)
        iv.setContentsMargins(2, 2, 10, 2)
        iv.setSpacing(2)
        for s in sections:
            iv.addWidget(s)
        iv.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(inner)
        v.addWidget(scroll, 1)
        row = QHBoxLayout()
        row.setContentsMargins(2, 6, 10, 2)
        row.addStretch(1)
        for a in actions:
            row.addWidget(a)
        v.addLayout(row)
        return page

    # ------------------------------------------------------------------ layout
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 8)
        root.setSpacing(6)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 4, 0)
        lv.addWidget(self._build_profile_bar())
        lv.addWidget(self._build_surface_card())
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        for title, tip, builder in (
                ("Render", "Studio relief renders, multi-shade and single hillshades, Brown texture shading, sky-view "
                 "factor, normal maps",
                 self._build_render_tab),
                ("Detail", "Isolate chosen feature sizes of the height map and separate detail from noise, live - "
                 "shown as relief, hillshade, Clean relief, lines or where it is strong", self._build_detail_tab),
                ("Tools", "Measure profiles and areas in real units, trace lines to a vector layer, compare "
                 "integrations, make publication plates", self._build_tools_tab),
                ("Shade Lab", "Frequency bands x QGIS-formula hillshades x PCA of many directions - pick an output "
                 "above the preview", self._build_lab_tab),
                ("Frequency", "Split the height map into five spatial-frequency bands (and set the render equaliser)",
                 self._build_frequency_tab),
                ("Light Stack", "N-directional light dome: per-light layers, composites, sky-view factor, normal map, PCA",
                 self._build_stack_tab),
                ("Provenance", "Multispectral overlay, validation and provenance records",
                 self._build_provenance_tab)):
            self.tabs.addTab(builder(), title)
            self.tabs.setTabToolTip(self.tabs.count() - 1, tip)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        lv.addWidget(self.tabs, 1)
        left.setMinimumWidth(430)

        self.splitter.addWidget(left)
        self.splitter.addWidget(self._build_preview_panel())
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([530, 800])
        root.addWidget(self.splitter, 1)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setMaximumHeight(150)
        self.log.setPlaceholderText("Job messages and errors appear here.")
        self.log.setVisible(False)
        root.addWidget(self.log)
        root.addLayout(self._build_bottom_bar())
        self.setStyleSheet(STYLE)

    def _build_profile_bar(self):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 4)
        self.profile_combo = QComboBox()
        self.profile_combo.setToolTip(f"Saved settings profiles ({PRESET_FILE})")
        load = QPushButton("Load")
        load.clicked.connect(self._load_profile)
        save = QPushButton("Save…")
        save.clicked.connect(self._save_profile)
        more = QToolButton()
        more.setText("⋯")
        more.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(more)
        menu.addAction("Delete selected profile", self._delete_profile)
        menu.addSeparator()
        menu.addAction("Reset every setting to defaults", self._reset_defaults)
        more.setMenu(menu)
        start = QPushButton("Start here")
        start.setToolTip("A short guide: from a scan to a publishable relief in seven steps, with buttons that do each")
        start.clicked.connect(self._show_start_here)
        h.addWidget(start)
        self.btn_undo_settings, self.btn_redo_settings = QToolButton(), QToolButton()
        for b, text, tip, redo in ((self.btn_undo_settings, "↶", "Undo the last settings change (Ctrl+Z)", False),
                                   (self.btn_redo_settings, "↷", "Redo (Ctrl+Shift+Z)", True)):
            b.setText(text)
            b.setToolTip(tip)
            b.setStyleSheet("font-size: 16px; padding: 0 3px;")
            b.setEnabled(False)
            b.clicked.connect(lambda _c=False, r=redo: self._undo_settings(r))
            h.addWidget(b)
        QTimer.singleShot(600, self._maybe_start_here)
        self.btn_presets = QToolButton()
        self.btn_presets.setText("Presets")
        self.btn_presets.setToolTip("Set the render and the surface for the object in front of you, or choose any look")
        self.btn_presets.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.presets_menu = QMenu(self.btn_presets)
        self.presets_menu.aboutToShow.connect(self._fill_presets_menu)
        self.btn_presets.setMenu(self.presets_menu)
        h.insertWidget(0, self.btn_presets)
        h.addWidget(QLabel("Profile"))
        h.addWidget(self.profile_combo, 1)
        h.addWidget(load)
        h.addWidget(save)
        h.addWidget(more)
        return w

    def _build_surface_card(self):
        box = QGroupBox("Surface")
        f = QFormLayout(box)
        f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        cb = self._reg("surface", _raster_combo(), layer=True)
        cb.setToolTip("Height / depth map from photometric stereo or photogrammetry (one band, higher = raised).")
        imp = QToolButton()
        imp.setText("Import HELIO…")
        imp.setToolTip("Choose a Helio export folder: its height maps, colour image and normal map are loaded in a "
                       "group, slow PNG / JPEG files copied to tiled GeoTIFFs, the colour aligned, the two "
                       "integrations compared and the better one chosen - and, if you like, the standard renders "
                       "made.")
        imp.clicked.connect(lambda: self._import_helio())
        f.addRow("Height map", _hbox(cb, imp))
        self.lbl_info = _hint("")
        f.addRow("", self.lbl_info)

        auto_px = self._check("pixel_auto", "From raster", True,
                              "Use the raster's own pixel size (geotransform).")
        px = self._dspin("pixel_size", 0.000001, 100000.0, 1.0, 6, 0.001,
                         tip="Horizontal size of one pixel in the same units as the heights - e.g. 0.0244 "
                             "for a 1040 ppi scan with heights in mm.")
        unit = self._combo("pixel_unit", ["auto", "mm", "\u00b5m", "cm", "m", "px"],
                           tip="Unit of the pixel size and heights - used for the preview scale bar and the "
                               "provenance record. 'auto' uses the layer CRS when georeferenced.")
        unit.setMaximumWidth(72)
        f.addRow("Pixel size", _hbox(px, unit, auto_px))

        z = self._slider("z_factor", 0.0001, 10000.0, 1.0, dec=4, step=0.1, log=True,
                         tip="Vertical exaggeration applied to the heights before shading.")
        auto_z = QToolButton()
        auto_z.setText("Auto")
        auto_z.setToolTip("Pick a Z-factor so the steepest 5 % of the (flattened) surface reaches 30° - "
                          "a good start whatever the height units.")
        auto_z.clicked.connect(self._auto_z)
        f.addRow("Z-factor", _hbox(z, auto_z))

        col = self._reg("colour", _raster_combo(allow_empty=True), layer=True)
        col.setToolTip("Optional colour / albedo image of the same object, for the colour composite and the "
                       "colour comparison. If it is not on the height map's grid (another size, or not "
                       "georeferenced, so QGIS draws it apart), press Align.")
        align_btn = QToolButton()
        align_btn.setText("Align")
        align_btn.setToolTip("Find where the colour / albedo image sits on the height map (scale and offset, from the "
                             "structure both show) and add an aligned copy - a VRT on the height map's exact pixel "
                             "grid, georeferenced like it, 16 bits kept - that QGIS draws right over the height "
                             "map. The original layer is hidden, not changed.")
        align_btn.clicked.connect(self._align_colour)
        f.addRow("Colour / albedo", _hbox(col, align_btn))
        self.lbl_colour = _hint("")
        self.lbl_colour.setVisible(False)
        f.addRow(self.lbl_colour)
        nm = self._reg("normals", _raster_combo(allow_empty=True), layer=True)
        nm.setToolTip("Optional photometric-stereo normal map of the same object (RGB: R = x, G = y, B = z). Clean "
                      "relief can be shaded from it directly - every detail the normals hold, and no integration "
                      "drift. It must be on the height map's grid (HELIO writes them so).")
        green = self._combo("normals_green", ["auto", "up (OpenGL)", "down (DirectX)"], tip=(
            "Which way the normal map's green channel points. Auto reads it the way that agrees with the height "
            "map's own slopes - labels are not always right."))
        green.setMaximumWidth(128)
        f.addRow("Normal map", _hbox(nm, green))
        f.addRow("", self._check("r_clean_normals", "Shade Clean relief from the normal map", False, (
            "Clean relief integrates the normal map window by window instead of reading the height map - sharper "
            "where the depth map lost detail, and nothing to flatten. The height map is still used for measuring "
            "and for the other modes.")))
        self.lbl_normals = _hint("")
        self.lbl_normals.setVisible(False)
        f.addRow(self.lbl_normals)
        fw = QgsFileWidget()
        fw.setStorageMode(QgsFileWidget.StorageMode.GetDirectory)
        fw.setDialogTitle("Export folder")
        fw.setToolTip("Where renders, bands, stacks, viewers and provenance JSON are written. "
                      "Without one, outputs go to a temporary folder.")
        self._reg("export_dir", fw)
        f.addRow("Export folder", fw)
        return box

    def _build_render_tab(self):
        s_look = Section("Look")
        s_look.addRow("Mode", self._combo("r_mode", eng.RENDER_MODES.values(), tip=(
            "Clean relief: the broad drift of the depth map removed, edge-preserving smoothing, a key light "
            "blended with a ring of lights round the compass and a soft grey tone - clean, smooth, detailed relief "
            "that reads from every side, never black or white.\n"
            "Studio relief: key / fill / ambient lighting with soft cast shadows, Brown texture shading, occlusion "
            "and clarity - flattened, plaster-like relief renders.\n"
            "Multi-shade: several weighted, optionally coloured lights; aspect-adaptive = GDAL multidirectional.\n"
            "Exaggerated: multi-scale shading - fine lines read everywhere; ideal for drypoint and inscriptions.\n"
            "Archaeological blend (VAT): hillshade + slope + openness + sky view; direction-independent.\n"
            "Frequency band shading: a different shade for each frequency band, combined (set on the Frequency tab).\n"
            "Shade Lab: frequency bands x QGIS-formula hillshades x PCA of many directions (set up on the Shade "
            "Lab tab).\n"
            "QGIS hillshade: exactly QGIS's Raster > Analysis > Hillshade (gdaldem hillshade) - standard, "
            "combined, multidirectional or Igor shading, Horn or Zevenbergen-Thorne, Z factor, scale and edges - "
            "shown the way QGIS shows a new hillshade layer.\n"
            "GIS hillshade: hill shading as in GIS software such as Global Mapper - one or several lights "
            "(weighted / average / maximum / minimum), ambient, shadow-darkness cap, highlight and slope-enhanced "
            "relief; the clean shaded render photometric-stereo studios make from depth maps.\n"
            "Specular enhancement: RTI-style glints only (black diffuse, white highlight) on exaggerated normals - "
            "the faintest change of slope flashes light or dark.\n"
            "Hillshade: single light, identical to GDAL.\nLight dome: diffuse slope shading.\n"
            "Texture shading: Leland Brown's isotropic fractional-Laplacian shading.\n"
            "Sky-view factor: openness to the sky - cracks and incisions dark.\nNormal map: tangent-space RGB.")))
        self.look_combo = QComboBox()
        self.look_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.look_combo.setMinimumContentsLength(14)
        self.look_combo.addItems(list(LOOKS))
        self.look_combo.setToolTip("Built-in looks start from the default settings, so switching looks never "
                                   "leaves stale values behind. Surface preparation is kept.")
        apply_look = QPushButton("Apply")
        apply_look.clicked.connect(self._apply_look)
        save_look = QToolButton()
        save_look.setText("Save…")
        save_look.setToolTip(f"Save the current render settings as your own look ({LOOKS_FILE})")
        save_look.clicked.connect(self._save_look)
        gallery = QToolButton()
        gallery.setText("Gallery\u2026")
        gallery.setToolTip("See every look side by side on your own surface, then click one to use it")
        gallery.clicked.connect(self._open_gallery)
        s_look.addRow("Preset", _hbox(self.look_combo, apply_look, save_look, gallery))

        s_prep = Section("Surface preparation")
        s_prep.addRow(self._check("r_detrend", "Remove tilt (best-fit plane)", True,
                                  "Subtracts one global plane so a slightly tilted object shades evenly."))
        flat = self._slider("r_flatten_sigma", 0, 1500, 0, dec=0, step=1, suffix=" px", tip=(
            "Removes warp, bowing and cockling broader than this (Gaussian high-pass, full-resolution pixels), so "
            "only the surface texture remains - the flattened look of studio relief renders. 0 = off."))
        auto_flat = QToolButton()
        auto_flat.setText("Auto")
        auto_flat.setToolTip("2 % of the long edge - a typical flattening for relief renders.")
        auto_flat.clicked.connect(self._auto_flatten)
        s_prep.addRow("Flatten broader than", _hbox(flat, auto_flat))
        s_prep.addRow(self._check("r_eq_enabled", "Apply the Frequency tab equaliser (band gains)", False,
                                  "Re-weights spatial-frequency bands before shading - e.g. boost tool marks, "
                                  "suppress grain. Crossovers and gains are set on the Frequency tab."))
        s_prep.addRow("Detail boost", self._slider("r_detail_boost", 0, 3, 0.0, tip=(
            "Lifts fine relief - drypoint burr, fine incisions, tool marks - before shading (an unsharp mask on "
            "the heights). 0 = off.")))
        s_prep.addRow("Detail size", self._slider("r_detail_radius", 0.5, 16, 2.0, 1, 0.5, suffix=" px",
                                                  tip="Relief finer than about this many full-resolution px is lifted."))
        s_prep.addRow("Smooth heights", self._slider("r_smooth_sigma", 0, 8, 0.0, 1, 0.1, suffix=" px", tip=(
            "Low-pass the heights before shading (Gaussian, full-resolution px): removes the terracing of 8-bit "
            "height maps and photometric-stereo noise. 0 = off.")))

        s_key = Section("Key light")
        s_key.addRow("Azimuth", self._slider("r_azimuth", 0, 360, 315, 1, 1, suffix="°", tip=(
            "Clockwise from image-up; 315° = top-left, the convention for relief renders. In Multi-shade "
            "and Light dome modes this rotates the whole set of lights (315° = as listed).")))
        s_key.addRow("Altitude", self._slider("r_altitude", 0, 90, 45, 1, 1, suffix="°",
                                              tip="Height of the light above the surface. Lower = more raking."))
        s_key.addRow("Intensity", self._slider("r_key", 0, 2, 1.0))

        s_clean = Section("Clean relief")
        s_clean.addRow(_hint("Clean, smooth grey relief lit from every side. The broad drift of a photometric-stereo "
                             "depth map is removed, noise and grain are smoothed away while the walls of tooling and "
                             "incisions stay sharp, and the key light above is blended with a ring of lights round "
                             "the compass. The steepness is set from the surface itself, so any height units shade "
                             "alike."))
        cflat = self._slider("r_clean_flatten", 0, 1000, 150, dec=0, step=1, suffix=" px", tip=(
            "Removes drift, warp and bowing broader than this (Gaussian high-pass, full-resolution px). Integrating "
            "photometric-stereo normals bends a depth map into a ramp or bowl far taller than the relief on it; this "
            "takes it away so the whole surface shades evenly. 0 = only the overall tilt is removed."))
        cauto = QToolButton()
        cauto.setText("Auto")
        cauto.setToolTip("1.5 % of the long edge of the height map.")
        cauto.clicked.connect(self._auto_clean_flatten)
        s_clean.addRow("Remove drift broader than", _hbox(cflat, cauto))
        s_clean.addRow("Smoothing", self._slider("r_clean_smooth", 0, 12, 3.0, 1, 0.5, suffix=" px", tip=(
            "Edge-preserving smoothing (guided filter), full-resolution px: photometric-stereo noise, grain and "
            "pores are averaged away, the walls of tooling, lines and incisions are kept. 0 = off.")))
        s_clean.addRow("Keep edges above", self._slider("r_clean_edge", 0.05, 3, 0.5, 2, 0.05, log=True, tip=(
            "Relief steps bigger than this share of the typical local relief stay sharp; smaller ones are smoothed. "
            "Lower = crisper (more grain kept), higher = smoother.")))
        s_clean.addRow("Fine detail", self._slider("r_clean_fine", 0, 2, 0.3, tip=(
            "Lifts the finest relief (under about 6 full-resolution px) after smoothing - tool marks, drypoint "
            "lines. 0 = off.")))
        s_clean.addRow("Relief", self._slider("r_clean_relief", 5, 75, 45, 0, 1, suffix="°", tip=(
            "How steep the steepest 5 % of the surface is shaded - the vertical exaggeration, set from the surface "
            "itself so the height units do not matter. Higher = stronger relief.")))
        s_clean.addRow("All-round light", self._slider("r_clean_allround", 0, 1, 0.3, tip=(
            "Share of the ring of lights round the compass. 0 = the key light alone (one direction); 1 = lit "
            "evenly from every side (direction-free). In between every edge reads whichever way it runs while "
            "the key light keeps the sense of depth.")))
        s_clean.addRow("Ring height", self._slider("r_clean_ring_alt", 5, 80, 30, 0, 1, suffix="°", tip=(
            "Height of the ring of lights. Lower = more raking, edges darker all round.")))
        s_clean.addRow("Ring lights", self._ispin("r_clean_lights", 4, 32, 16, tip=(
            "Lights in the ring (the ring turns with the key-light azimuth).")))
        s_clean.addRow("Brightness", self._slider("r_clean_mid", 0.3, 0.85, 0.6, tip=(
            "Grey of the flat surface. Relief swings lighter and darker round it on a soft curve - never pure "
            "black or white.")))
        s_clean.addRow("Contrast", self._slider("r_clean_contrast", 0.3, 3, 1.0, log=True))
        s_clean.addRow("Highlights", self._slider("r_clean_highlight", 0.5, 3, 1.6, tip=(
            "Extra brightness for faces turned to the light: they brighten far less than faces turned away "
            "darken, so without it highlights look dull. Archive shaded renders of tooled bindings fit 1.4-1.8.")))

        s_gdal = Section("QGIS hillshade (gdaldem)")
        s_gdal.addRow(_hint("Exactly what QGIS's Raster ▸ Analysis ▸ Hillshade makes (gdaldem hillshade), pixel for "
                            "pixel, straight from the raw heights - the surface preparation above is not used. "
                            "Azimuth and altitude are the light above."))
        s_gdal.addRow("Z factor", self._slider("r_gdal_z", 0.01, 100, 1.0, 3, 0.01, log=True, tip=(
            "-z: vertical exaggeration, as in QGIS's Hillshade dialog. 1 = true relief.")))
        scale = self._slider("r_gdal_scale", 0.001, 200000, 1.0, 3, 0.001, log=True, tip=(
            "-s: ratio of vertical units to horizontal. 1 when heights and pixels are both in metres; 111120 "
            "when the raster is in degrees and the heights in metres."))
        auto_scale = QToolButton()
        auto_scale.setText("Auto")
        auto_scale.setToolTip("111120 for a raster in degrees (geographic CRS), otherwise 1.")
        auto_scale.clicked.connect(self._auto_gdal_scale)
        s_gdal.addRow("Scale", _hbox(scale, auto_scale))
        s_gdal.addRow("Shading", self._combo("r_gdal_variant", eng.GDAL_VARIANTS.values(), tip=(
            "Standard: one light (-az, -alt).\nCombined: slope and oblique shading together (-combined).\n"
            "Multidirectional: four lights at 225, 270, 315 and 360°, each weighted by how squarely it faces the "
            "slope (-multidirectional; the azimuth is fixed).\nIgor: Igor Brejc's shading, slope and aspect only "
            "(-igor; no altitude).")))
        s_gdal.addRow(self._check("r_gdal_zt", "Use Zevenbergen-Thorne formula instead of Horn's", False,
                                  "-alg ZevenbergenThorne: a smoother gradient on smooth surfaces."))
        s_gdal.addRow(self._check("r_gdal_edges", "Compute edges", False,
                                  "-compute_edges: shade the outer ring of pixels too (otherwise it is nodata)."))
        s_gdal.addRow("Display", self._combo("r_gdal_display", eng.GDAL_DISPLAY.values(), tip=(
            "As QGIS shows it: QGIS styles a new hillshade layer with 'stretch to min/max' between the minimum and "
            "maximum of ~250 000 sampled pixels - this preview does the same.\nPunchier: the 2 % and 98 % points of "
            "the histogram (Layer Properties ▸ Symbology ▸ Cumulative count cut).\nRaw: gdaldem's grey levels as "
            "they are.\nThe saved GeoTIFF always holds gdaldem's exact grey levels; this only sets how it is shown.")))
        self.lbl_gdal_cmd = QLabel("")
        self.lbl_gdal_cmd.setWordWrap(True)
        self.lbl_gdal_cmd.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl_gdal_cmd.setStyleSheet("font-family: Menlo, Consolas, monospace; font-size: 11px;")
        s_gdal.addRow(self.lbl_gdal_cmd)

        s_lab = Section("Shade Lab")
        s_lab.addRow(_hint("Frequency bands x QGIS-formula hillshades x PCA of many directions. Everything is set up "
                           "on the Shade Lab tab; the Output selector above the preview picks what is rendered."))
        open_lab = QPushButton("Open the Shade Lab tab")
        open_lab.clicked.connect(lambda: self.tabs.setCurrentIndex(TAB_LAB))
        s_lab.addRow(open_lab)

        s_gis = Section("GIS hill shading")
        s_gis.addRow(_hint("Hill shading the way GIS software renders a depth map (Global Mapper's Daylight shader "
                           "with hill shading): one neutral colour, the relief from the light alone. Light direction "
                           "and height are the key light above."))
        s_gis.addRow("Light sources", self._ispin("r_gis_lights", 1, 8, 1, tip=(
            "Several lights spaced evenly round the circle, starting at the azimuth (4 lights from 315°: 315, 45, "
            "135, 225) - multi-directional hill shading.")))
        s_gis.addRow("Combine", self._combo("r_gis_blend", eng.GIS_BLENDS.values(), tip=(
            "Weighted average: the key light counts as much as all the others together.\nAverage: every light "
            "equal.\nMaximum: the brightest light at each point - soft, open shading.\nMinimum: the darkest light "
            "at each point - every slope reads dark whichever way it faces, so incisions stand out.")))
        ve = self._slider("r_gis_ve", 0.05, 20, 1.0, 2, 0.05, log=True, tip=(
            "Vertical exaggeration for the hill shading, on top of the Z-factor (Global Mapper's 'Vertical "
            "exaggeration'). Scanner shaded renders vary from object to object - use Auto or Match reference."))
        auto_ve = QToolButton()
        auto_ve.setText("Auto")
        auto_ve.setToolTip("Sets the exaggeration so the render has the typical contrast of scanner shaded renders "
                           "(grey spread about 0.10), measured on a full-resolution crop.")
        auto_ve.clicked.connect(lambda: self._auto_gis_ve())
        s_gis.addRow("Exaggeration", _hbox(ve, auto_ve))
        s_gis.addRow("Brightness", self._slider("r_gis_brightness", 0.2, 1.2, 1.0, tip=(
            "Surface colour: the grey of a face turned full to the light. Scanner shaded renders top out near "
            "0.76 - never pure white.")))
        s_gis.addRow("Ambient lighting", self._slider("r_gis_ambient", 0, 1, 0.15,
                                                      tip="Lifts every tone - brightens dark-looking surfaces."))
        s_gis.addRow("Shadow darkness cap", self._slider("r_gis_shadow", 0, 0.9, 0.2, tip=(
            "How dark the darkest shadows may get. 0 = shadows can reach black; higher caps them on steep relief "
            "so the low relief keeps its detail.")))
        s_gis.addRow("Highlight", self._slider("r_gis_highlight", 0, 1, 0.15, tip=(
            "Highlight from direct light: whitens faces turned squarely to the key light.")))
        s_gis.addRow("Slope-enhanced", self._slider("r_gis_slope", 0, 1, 0.0, tip=(
            "Screens a slope mask over the shade (Kent Brown, Utah Geological Survey): near-level surface turns "
            "white and only slopes keep their shading - clean, drawing-like relief. 0 = off.")))
        s_gis.addRow("Slope range", self._slider("r_gis_slope_range", 1, 45, 10, 1, 0.5, suffix="°",
                                                 tip="Slope at which the full shading returns (needs Slope-enhanced "
                                                     "> 0)."))
        s_gis.addRow(_hint("Match a reference: pick a grey render of the same area - for example the shaded "
                           "render a photometric-stereo scanner delivers with its depth map - and the light, "
                           "exaggeration and tone are fitted to it pixel by pixel over the current view."))
        s_gis.addRow("Reference render", self._reg("gis_ref", _raster_combo(allow_empty=True), layer=True))
        fit_ref = QPushButton("Match reference")
        fit_ref.setToolTip("Fit this mode's settings to the reference render (same coverage as the height map).")
        fit_ref.clicked.connect(self._fit_gis_reference)
        s_gis.addRow(fit_ref)
        self.lbl_gis_fit = QLabel("")
        self.lbl_gis_fit.setWordWrap(True)
        s_gis.addRow(self.lbl_gis_fit)

        s_multi = Section("Multi-shade lights")
        self.multi_preset = QComboBox()
        self.multi_preset.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.multi_preset.setMinimumContentsLength(14)
        self.multi_preset.addItems(list(eng.MULTI_PRESETS))
        apply_multi = QPushButton("Apply")
        apply_multi.clicked.connect(lambda: self._apply_multi_preset())
        s_multi.addRow("Light rig", _hbox(self.multi_preset, apply_multi))
        s_multi.addRow("Combine", self._combo("r_multi_method", eng.MULTI_METHODS.values(), tip=(
            "Aspect-adaptive: each slope is lit mainly by the light that faces it most squarely (Mark 1992; "
            "with the 4-light rig this is exactly GDAL's multidirectional hillshade).\n"
            "Weighted average: fixed weights.\nBrightest light: the maximum of all lights.")))
        self.multi_table = QTableWidget(0, 5)
        self.multi_table.setHorizontalHeaderLabels(["On", "Azimuth °", "Altitude °", "Weight", "Colour"])
        self.multi_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.multi_table.verticalHeader().setVisible(False)
        self.multi_table.setMinimumHeight(150)
        self.multi_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.multi_table.itemChanged.connect(self._on_multi_edit)
        s_multi.addRow(self.multi_table)
        add_light, remove_light = QPushButton("Add light"), QPushButton("Remove light")
        add_light.clicked.connect(lambda: (self._add_multi_row(), self._on_multi_edit()))
        remove_light.clicked.connect(self._remove_multi_row)
        s_multi.addRow(_hbox(add_light, remove_light, stretch_first=False))
        s_multi.addRow(_hint("Coloured lights give an orientation-coded colour relief; white lights give grey."))

        s_exag = Section("Exaggerated shading")
        s_exag.addRow(_hint("Every scale is lit from a grazing angle relative to the smoother surface around it, so "
                            "fine lines read everywhere regardless of the overall form (Rusinkiewicz et al. 2006)."))
        s_exag.addRow("Exaggeration", self._slider("r_exag_contrast", 1, 40, 8.0, 1, 0.5, log=True,
                                                   tip="Higher = crisper, more graphic light / dark separation."))
        s_exag.addRow("Scales", self._ispin("r_exag_scales", 2, 7, 5,
                                            tip="Number of detail scales (1, 2, 4, 8 ... full-resolution px)."))
        s_exag.addRow("Fine-detail emphasis", self._slider("r_exag_detail", -1, 2, 0.5, tip=(
            "Above 0 favours the finest scales (lines, grain); below 0 favours broad form.")))
        s_exag.addRow("Base shading", self._slider("r_exag_base", 0, 1, 0.2,
                                                   tip="Share of ordinary shading of the broad form, for depth."))

        s_spec = Section("Specular enhancement")
        s_spec.addRow(_hint("Only the glossy highlight is kept (black diffuse, white highlight), as in the specular "
                            "enhancement of RTI viewers (Malzbender et al. 2006). "
                            "With an overhead light a level surface is white and every "
                            "incision a dark line; lower the light for raking glints. Gloss and intensity are "
                            "under Key light and Material & tone."))
        s_spec.addRow("Normal depth", self._slider("r_spec_depth", 0.2, 10, 1.0, 2, 0.1, log=True, tip=(
            "Multiplies every slope before the highlight is worked out - raise it for very shallow marks "
            "(Z-factor already scales this surface).")))

        s_vat = Section("Archaeological blend (VAT)")
        s_vat.addRow(_hint("Hillshade + inverted slope + positive openness + sky-view factor, blended as in the Relief "
                           "Visualization Toolbox 'VAT - Archaeological' recipe (Kokalj & Somrak 2019). "
                           "Direction-independent and good for faint, low relief."))
        s_vat.addRow("Variant", self._combo("r_vat_variant", list(VAT_LABELS)))

        s_bands = Section("Frequency band shading")
        s_bands.addRow(_hint("Each spatial-frequency band is shaded its own way and the shades are combined. Choose "
                             "the shade, strength and light height per band on the Frequency tab; the key-light "
                             "azimuth turns every band's light together."))
        edit_bands = QPushButton("Edit the bands on the Frequency tab")
        edit_bands.clicked.connect(lambda: (self.tabs.setCurrentIndex(TAB_FREQUENCY),
                                            self._w["fq_preview_band"].setCurrentText(FQ_COMBO)))
        s_bands.addRow(edit_bands)

        s_fill = Section("Fill, ambient & softness")
        s_fill.addRow("Fill intensity", self._slider("r_fill", 0, 1, 0.2, tip="Second, softer light that opens "
                                                                              "up shadows."))
        s_fill.addRow("Fill direction", self._slider("r_fill_offset", 0, 360, 180, 0, 5, suffix="°",
                                                     tip="Relative to the key light (180° = opposite)."))
        s_fill.addRow("Fill altitude", self._slider("r_fill_altitude", 0, 90, 30, 0, 1, suffix="°"))
        s_fill.addRow("Ambient", self._slider("r_ambient", 0, 1, 0.3, tip="Direction-less light: lifts the "
                                                                          "darkest areas."))
        s_fill.addRow("Soft light (wrap)", self._slider("r_wrap", 0, 0.5, 0.0, tip=(
            "Softer roll-off from light into shade, like a large diffused studio light on plaster. 0 = hard.")))

        s_shadow = Section("Cast shadows")
        s_shadow.addRow("Strength", self._slider("r_shadows", 0, 1, 0.0, tip=(
            "Raised edges throw shadows away from the key light - essential for raking-light realism. 0 = off.")))
        s_shadow.addRow("Softness", self._slider("r_shadow_softness", 0.5, 20, 3.0, 1, 0.5, suffix="°",
                                                 tip="Apparent size of the light: bigger = softer shadow edges."))
        s_shadow.addRow("Reach", self._ispin("r_shadow_reach", 8, 1024, 96, 8, " px",
                                             "How far (full-resolution px) to look for shadow-casting relief."))

        s_dome = Section("Light dome")
        s_dome.addRow("Lights", self._ispin("r_dome_lights", 3, 64, 16))
        s_dome.addRow("Altitude", self._slider("r_dome_altitude", 1, 89, 35, 0, 1, suffix="°"))

        s_tex = Section("Texture shading (Brown 2014)")
        s_tex.addRow(_hint("Isotropic and scale-invariant: raised areas light, hollows dark, fine detail at "
                           "progressively lower contrast - the direction-free component a single light misses."))
        s_tex.addRow("Detail", self._slider("r_texture_detail", 0, 200, 60, 0, 5, suffix=" %", tip=(
            "Fractional-Laplacian order (Brown's percentage). 0 % = plain height map, 200 % = ordinary "
            "Laplacian. 50-100 % suits most surfaces; raise it when a few deep features dominate.")))
        s_tex.addRow("Mix into render", self._slider("r_texture_mix", 0, 1, 0.35,
                                                     tip="0 = off (and the texture controls below grey out)."))
        s_tex.addRow("Contrast", self._slider("r_texture_contrast", 0.1, 6, 1.0, log=True,
                                              tip="S-curve steepness: trades contrast in the strongest ridges "
                                                  "and cuts against the mid-tones (Brown eq. 7/8)."))
        s_tex.addRow("Midtone", self._slider("r_texture_midtone", -1, 1, 0.0))
        s_tex.addRow("Curve", self._combo("r_texture_curve", ["tanh (Brown eq. 7)", "x / √(1 + x²) (eq. 8)"]))

        s_occ = Section("Occlusion (sky-view factor)", expanded=False)
        s_occ.addRow("Strength", self._slider("r_occlusion", 0, 1, 0.0,
                                              tip="Darkens enclosed places - cracks, incisions, undercuts. 0 = off."))
        s_occ.addRow("Search radius", self._ispin("r_svf_radius", 2, 512, 24, suffix=" px"))
        s_occ.addRow("Directions", self._ispin("r_svf_dirs", 4, 64, 16))

        s_edge = Section("Line & edge enhancement")
        s_edge.addRow(_hint("Makes drawn and incised lines read like a drawing: ridges and burr lighter, grooves "
                            "darker (curvature, in the spirit of radiance scaling); steep walls darker (slope)."))
        s_edge.addRow("Curvature", self._slider("r_curvature", 0, 1, 0.0,
                                                tip="Ridges and burr lighter, incisions darker. 0 = off."))
        s_edge.addRow("Curvature size", self._slider("r_curvature_radius", 0.5, 8, 1.5, 1, 0.5, suffix=" px",
                                                     tip="Width of the lines to pick out, in full-resolution px."))
        s_edge.addRow("Slope darkening", self._slider("r_slope_shade", 0, 1, 0.0, tip=(
            "Steep walls - the sides of incisions and cuts - darkened, like ink lines. 0 = off.")))
        s_edge.addRow("Slope range", self._slider("r_slope_range", 5, 80, 45, 0, 1, suffix="\u00b0",
                                                  tip="Slope that reaches full darkening."))

        s_mat = Section("Material & tone")
        s_mat.addRow("Specular", self._slider("r_specular", 0, 1, 0.0, tip="Glossy highlight (metal, varnish)."))
        s_mat.addRow("Gloss", self._slider("r_shininess", 1, 200, 30, 0, 1, log=True,
                                           tip="Highlight tightness (needs Specular > 0)."))
        s_mat.addRow("Clarity", self._slider("r_clarity", 0, 1, 0.0,
                                             tip="Local contrast: crisper surface texture without harsher lighting."))
        s_mat.addRow("Clarity radius", self._slider("r_clarity_radius", 1, 64, 12, 0, 1, suffix=" px"))
        s_mat.addRow("Flat tone", self._slider("r_base_tone", 0.2, 0.9, 0.62,
                                               tip="Grey level of a perfectly flat area."))
        s_mat.addRow("Contrast", self._slider("r_contrast", 0.25, 4, 1.0, log=True))
        s_mat.addRow("Gamma", self._slider("r_gamma", 0.4, 2.5, 1.0))
        s_mat.addRow("Tint", self._combo("r_tint", eng.TINTS.keys()))
        s_mat.addRow(self._check("r_albedo", "Colour composite: albedo × relief shading", False,
                                 "The colour / albedo layer multiplied by the relief shading - a flat surface keeps "
                                 "its colour, the relief lightens and darkens it: the colour composite "
                                 "photometric-stereo archives publish beside their grey shaded renders. Needs a "
                                 "Colour / albedo layer on the height map's grid - press Align next to it first."))
        s_mat.addRow("Albedo strength", self._slider("r_albedo_strength", 0, 1, 1.0))
        s_mat.addRow("Colour relief", self._slider("r_albedo_relief", 0.25, 3, 1.0, tip=(
            "How strongly the relief lightens and darkens the colour: 1 = in proportion to the grey render. Archive "
            "colour composites of bindings fit 0.5-1.8.")))
        s_mat.addRow("Colour gamma", self._slider("r_albedo_gamma", 0.3, 1.5, 0.55, tip=(
            "Softens the colour's own contrast - stains, ink, discolouration - so the relief leads. Archive colour "
            "composites fit 0.53-0.59; 1 = the colour as it is.")))
        s_mat.addRow("Colour exposure", self._slider("r_albedo_exposure", 0.4, 2, 0.8, tip=(
            "Overall brightness of the colour composite (archive composites fit 0.76-0.82).")))
        s_mat.addRow("Convention", self._combo("r_normal_convention", ["OpenGL (green up)", "DirectX (green down)"]))

        s_out = Section("Output", expanded=False)
        name = self._reg("r_out_name", QLineEdit())
        name.setPlaceholderText("automatic: <layer>_<mode>")
        s_out.addRow("File name", name)
        s_out.addRow("Area", self._combo("r_out_area", ["Whole raster", "Current preview view"],
                                         tip="Render only what the preview shows, at full resolution and "
                                             "georeferenced - quick detail figures."))
        s_out.addRow(self._check("r_out_16", "Also write a 16-bit GeoTIFF (for print / further grading)"))
        s_out.addRow(self._check("r_out_png", "Also write a PNG"))
        s_out.addRow(self._check("r_out_add", "Add the render to the map", True))

        self._rs = dict(look=s_look, prep=s_prep, key=s_key, clean=s_clean, gdal=s_gdal, lab=s_lab, gis=s_gis, multi=s_multi,
                        exag=s_exag,
                        spec=s_spec,
                        vat=s_vat, bands=s_bands,
                        fill=s_fill,
                        shadow=s_shadow, dome=s_dome, tex=s_tex, occ=s_occ, edge=s_edge, mat=s_mat, out=s_out)
        return self._page(list(self._rs.values()), [self._primary("Render full resolution", self._run_render)])

    def _build_detail_tab(self):
        from qgis.PyQt.QtWidgets import QButtonGroup, QRadioButton
        from . import detail
        from .widgets import SpectrumView
        s_sizes = Section("Feature sizes")
        s_sizes.addRow(_hint("A height map mixes relief of every size - the object's form, tool marks, strokes, grain "
                             "and noise. The chart shows how much relief the middle of the view holds at each size; "
                             "drag its handles to keep a band of sizes and watch the preview. Up to three bands, each "
                             "with its own gain, can be mixed - fine strokes boosted and the broad form turned down, "
                             "say."))
        self.spectrum_view = SpectrumView()
        self.spectrum_view.bandChanged.connect(self._on_spectrum_drag)
        s_sizes.addRow(self.spectrum_view)
        grid_w = QWidget()
        g = QGridLayout(grid_w)
        g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(6)
        for c, head in enumerate(["Edit", "Band", "Finest", "Broadest", "Gain"]):
            g.addWidget(_hint(head), 0, c)
        self._dt_edit = QButtonGroup(self)
        for i, b in enumerate(detail.BAND_DEFAULTS):
            r = QRadioButton()
            r.setToolTip("The chart's handles edit this band")
            r.setChecked(i == 0)
            self._dt_edit.addButton(r, i)
            g.addWidget(r, i + 1, 0)
            g.addWidget(self._check(f"dt_on{i}", f"Band {i + 1}", b["on"]), i + 1, 1)
            fine = self._dspin(f"dt_fine{i}", 0, 512, b["fine"], 2, 0.5, " px", tip=(
                "The finest feature size kept, in full-resolution px (half a wavelength is the width of a groove). "
                "0: no lower limit."))
            fine.setSpecialValueText("finest")
            g.addWidget(fine, i + 1, 2)
            broad = self._dspin(f"dt_broad{i}", 0, 8192, b["broad"], 1, 4, " px", tip=(
                "The broadest feature size kept. 0: no upper limit - all the form."))
            broad.setSpecialValueText("all")
            g.addWidget(broad, i + 1, 3)
            gain = self._dspin(f"dt_gain{i}", -4, 8, b["gain"], 2, 0.1, tip=(
                "How strongly this band counts in the mix (negative turns it upside down)."))
            gain.setMaximumWidth(72)
            g.addWidget(gain, i + 1, 4)
        g.setColumnStretch(2, 1)
        g.setColumnStretch(3, 1)
        self._dt_edit.idToggled.connect(lambda _i, on: on and self._update_detail_chart())
        s_sizes.addRow(grid_w)
        self.dt_preset = QComboBox()
        self.dt_preset.addItems(list(detail.PRESETS))
        self.dt_preset.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.dt_preset.setMinimumContentsLength(14)
        apply_p = QPushButton("Apply")
        apply_p.setMaximumWidth(70)
        apply_p.setToolTip("Sets the band being edited")
        apply_p.clicked.connect(self._apply_detail_preset)
        s_sizes.addRow("Preset", _hbox(self.dt_preset, apply_p))
        s_sizes.addRow("Band edges", self._slider("dt_soft", 0.1, 2.0, 0.5, 2, 0.05, suffix=" oct", tip=(
            "How gently each band fades out at its limits, in octaves (doublings of size): small is a sharp cut, which "
            "can ring beside strong edges; large a gentle blend.")))

        s_sep = Section("Separate detail from noise")
        s_sep.addRow(_hint("All off to begin with. Suppress noise uses the noise fitted on the chart - where none is "
                           "resolvable it changes nothing. The texture gate keeps strong local structure (tool marks, "
                           "strokes) and damps faint even texture such as grain. Clean-up smooths within surfaces but "
                           "not across edges. Show 'What the separation removed' to check: it should look like "
                           "structureless grain - if marks show in it, back off."))
        s_sep.addRow("Suppress noise", self._slider("dt_wiener", 0, 4, 0.0, 2, 0.1, tip=(
            "Wiener filtering from the fitted spectrum: 1 is the statistically best amount, more removes more.")))
        s_sep.addRow("Suppress faint texture", self._slider("dt_gate", 0, 3, 0.0, 2, 0.05, tip=(
            "Damps the band wherever it is weaker than this multiple of its typical local strength; strong marks "
            "pass untouched. 0 = off.")))
        s_sep.addRow("Edge-preserving clean-up", self._slider("dt_clean", 0, 12, 0.0, 1, 0.5, suffix=" px", tip=(
            "Guided-filter radius: smooths grain inside surfaces while keeping edges sharp. 0 = off.")))

        s_show = Section("Show")
        s_show.addRow("As", self._combo("dt_display", detail.DISPLAYS.values(), tip=(
            "How the preview shows the band. The side panel shows it every way at once - click one to switch.")))
        s_show.addRow("Strength", self._slider("dt_show_gain", 0.2, 5, 1.0, 2, 0.05, log=True, tip=(
            "Display contrast. Hillshade and Clean relief use the Render tab's light (drag the sun on the dial) "
            "and tone.")))

        s_use = Section("Use it")
        s_use.addRow(_hint("Save the band as a height map - a GeoTIFF on the source's georeferencing - to render, "
                           "measure or trace it anywhere, save the preview's display at full resolution, or trace the "
                           "band's lines straight away."))
        s_use.addRow("Area", self._combo("dt_area", ["Whole height map", "Current preview view"]))
        s_use.addRow(self._check("dt_use_saved", "Use the saved band as the height map", False, tip=(
            "After saving, choose the band as the Surface's height map - every tab then works on it.")))
        save_img = QPushButton("Save the display…")
        save_img.setToolTip("Writes the band shown as the preview shows it (8-bit GeoTIFF) at full resolution.")
        save_img.clicked.connect(lambda: self._run_detail_save("image"))
        trace = QPushButton("Trace lines in this band")
        trace.setToolTip("Line tracing (Tools tab) on this band, with the Tools tab's tracing settings.")
        trace.clicked.connect(self._trace_in_band)
        s_use.addRow(_hbox(save_img, trace, stretch_first=False))
        self.lbl_detail = _hint("")
        s_use.addRow(self.lbl_detail)
        return self._page([s_sizes, s_sep, s_show, s_use],
                          [self._primary("Save band as a height map", lambda: self._run_detail_save("band"))])

    def _build_tools_tab(self):
        from . import tools
        from .widgets import ProfileView
        s_meas = Section("Measure")
        s_meas.addRow(_hint("Choose Measure (above the preview) and drag across the preview: the height profile "
                            "along the line - a groove's depth and width, a ridge's height. Shift-drag a box for its "
                            "relief and the volume of its hollows. Distances use the pixel size above, heights the "
                            "height map's own units."))
        go = QPushButton("Measure tool")
        go.clicked.connect(lambda: self.btn_measure_tool.setChecked(True))
        clear = QPushButton("Clear")
        clear.clicked.connect(self._clear_measure)
        s_meas.addRow(_hbox(go, clear, stretch_first=False))
        s_meas.addRow("Heights", self._combo("ms_heights", ["As in the height map", "Broad drift removed"], tip=(
            "As in the height map: the depths as measured. Broad drift removed: minus the warp broader than Clean "
            "relief's drift setting - marks measured against the surface right round them.")))
        s_meas.addRow("Height scale", self._slider("ms_zscale", 0.0001, 1000.0, 1.0, 4, 0.01, log=True, tip=(
            "Multiplies the height map's values in every measurement - for heights not in the unit above. Photometric-"
            "stereo exports often label their heights 'mm' without a real calibration; measure something of known "
            "depth and set this so it comes out right. 1 = the values as exported.")))
        self.profile_view = ProfileView()
        s_meas.addRow(self.profile_view)
        self.lbl_measure = QLabel("")
        self.lbl_measure.setWordWrap(True)
        self.lbl_measure.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl_measure.setMinimumHeight(self.fontMetrics().lineSpacing() * 6 + 8)   # room for every result line
        self.lbl_measure.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        s_meas.addRow(self.lbl_measure)
        csv = QPushButton("Export CSV…")
        csv.clicked.connect(self._export_profile)
        cal = QPushButton("Set the scale from this line…")
        cal.setToolTip("Tell the plugin how long the measured line really is (e.g. across a ruler in the scan): the "
                       "pixel size and unit are set from it, for every measurement, scale bar and plate.")
        cal.clicked.connect(self._calibrate_from_line)
        s_meas.addRow(_hbox(csv, cal, stretch_first=False))

        s_trace = Section("Line tracing")
        s_trace.addRow(_hint("Draws the surface's incised (or raised) lines - tooling, drypoint, inscription strokes, "
                             "cracks - as a vector layer, like an archaeological line drawing you can edit, style and "
                             "measure in QGIS. The lines also show over the preview."))
        s_trace.addRow("Lines", self._combo("tr_kind", tools.TRACE_KINDS.values()))
        tw = self._slider("tr_width", 1, 40, 3.0, 1, 0.5, suffix=" px", tip=(
            "About how wide the lines are, in full-resolution px - marks much finer or broader are ignored."))
        presets = []
        for label, width, tip in (("Fine", tools.TRACE_PRESETS["Fine"], "Fine lines - hatching, drypoint, grain of "
                                   "the tooling (4 px)."),
                                  ("Main", tools.TRACE_PRESETS["Main"], "Main lines - the outlines of tool impressions "
                                   "and strokes (16 px).")):
            b = QToolButton()
            b.setText(label)
            b.setToolTip(tip + " A surface carries lines at several widths at once, so choose which to trace.")
            b.clicked.connect(lambda _c=False, v=width: self._w["tr_width"].setValue(v))
            presets.append(b)
        s_trace.addRow("Line width", _hbox(tw, *presets))
        s_trace.addRow("Sensitivity", self._slider("tr_sens", 0.05, 0.95, 0.5, tip=(
            "Higher traces fainter lines (and more grain); lower keeps only the clearest.")))
        s_trace.addRow("Shortest line", self._slider("tr_min", 2, 300, 15, 0, 1, suffix=" px"))
        s_trace.addRow("Simplify", self._slider("tr_simplify", 0, 4, 0.7, 1, 0.1, suffix=" px", tip=(
            "Straightens the traced pixel chains to within this many px (fewer vertices).")))
        s_trace.addRow("Area", self._combo("tr_area", ["Current preview view (full resolution)",
                                                        "Whole height map (up to 16 MP)"]))
        s_trace.addRow("Trace in", self._combo("tr_source", ["Height map", "Detail tab band"], tip=(
            "Height map: all of it, as before. Detail tab band: only the feature sizes (and the noise separation) set "
            "on the Detail tab - strokes without the grain, say, or only the broad outlines.")))
        s_trace.addRow("Bridge gaps", self._slider("tr_bridge", 0, 60, 0.0, 0, 1, suffix=" px", tip=(
            "Joins line ends closer than this that point the same way - a groove broken by wear or by a crossing "
            "line comes back as one line. 0 = off.")))
        s_trace.addRow("Smooth curves", self._ispin("tr_smooth", 0, 4, 0, tip=(
            "Rounds off the pixel staircase of the traced lines (this many passes of corner cutting). 0 = off.")))
        s_trace.addRow("Weakest line", self._slider("tr_minstr", 0, 1, 0.0, 2, 0.01, tip=(
            "Drops lines whose average strength is below this share of the strongest. 0 = keep them all.")))
        s_trace.addRow(self._check("tr_live", "Live: trace the view as settings or the view change", False, tip=(
            "Re-traces what is on screen, at screen resolution, whenever a setting or the view changes - to tune "
            "the settings before tracing for real.")))
        s_trace.addRow(self._primary("Trace lines", self._run_trace))
        clear = QPushButton("Clear lines")
        clear.setToolTip("Removes the traced lines from the preview (the layers stay in the project).")
        clear.clicked.connect(self._clear_lines)
        self.btn_undo_trace = QPushButton("Undo last trace")
        self.btn_undo_trace.setToolTip("Removes the last traced layer from the project and deletes its GeoPackage.")
        self.btn_undo_trace.setEnabled(False)
        self.btn_undo_trace.clicked.connect(self._undo_trace)
        reset = QPushButton("Reset")
        reset.setToolTip("Every tracing setting back to its default, and the lines cleared from the preview.")
        reset.clicked.connect(lambda: (self._reset_section(s_trace), self._clear_lines()))
        s_trace.addRow(_hbox(clear, self.btn_undo_trace, reset, stretch_first=False))
        self.lbl_trace = _hint("")
        s_trace.addRow(self.lbl_trace)

        s_pick = Section("Integration picker")
        s_pick.addRow(_hint("HELIO integrates one set of normals two ways: Poisson (smooth, clean form) and "
                            "Frankot-Chellappa (more fine detail, more noise). Compare them on this object - fine "
                            "detail, pixel noise and drift at nine places - then use one, or fuse A's form with B's "
                            "detail."))
        s_pick.addRow("A", self._reg("tp_a", _raster_combo(allow_empty=True), layer=True))
        s_pick.addRow("B", self._reg("tp_b", _raster_combo(allow_empty=True), layer=True))
        s_pick.addRow("Fuse finer than", self._slider("tp_sigma", 1, 16, 3.0, 1, 0.5, suffix=" px", tip=(
            "The fused height map takes relief broader than this from A and finer than this from B.")))
        s_pick.addRow(self._primary("Compare", self._run_compare))
        self.lbl_pick = QLabel("")
        self.lbl_pick.setWordWrap(True)
        self.lbl_pick.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl_pick.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        from qgis.PyQt.QtWidgets import QSizePolicy
        loose = (QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)   # never widen the panel
        self.lbl_pick.setSizePolicy(*loose)
        s_pick.addRow(self.lbl_pick)
        self.pick_thumbs = []
        thumbs = QHBoxLayout()
        for _ in range(3):
            lab = QLabel()
            lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lab.setMinimumHeight(10)
            lab.setSizePolicy(*loose)
            self.pick_thumbs.append(lab)
            thumbs.addWidget(lab)
        s_pick.addRow(thumbs)
        self.btn_use_a, self.btn_use_b = QPushButton("Use A"), QPushButton("Use B")
        self.btn_use_a.clicked.connect(lambda: self._use_integration("tp_a"))
        self.btn_use_b.clicked.connect(lambda: self._use_integration("tp_b"))
        fuse = QPushButton("Make a fused height map")
        fuse.setToolTip("Writes a GeoTIFF with A's broad form and B's fine detail (B scaled to A's heights) and uses "
                        "it as the height map.")
        fuse.clicked.connect(self._run_fuse)
        s_pick.addRow(_hbox(self.btn_use_a, self.btn_use_b, fuse, stretch_first=False))

        s_plate = Section("Publication plate")
        s_plate.addRow(_hint("A print-ready page: the render as the preview shows it - with the colour composite and a "
                             "close-up of the current view if you like - a true scale bar, a title and a caption with "
                             "the settings, scale and date."))
        s_plate.addRow("Paper", _hbox(self._combo("pl_paper", list(PAPER_MM)),
                                      self._combo("pl_orient", ["Landscape", "Portrait"]), stretch_first=False))
        s_plate.addRow("Resolution", self._combo("pl_dpi", ["300 dpi", "600 dpi", "150 dpi"]))
        s_plate.addRow("Content", self._combo("pl_content", PLATE_CONTENT, tip=(
            "The colour panel needs a Colour / albedo layer on the height map's grid (press Align).")))
        title = self._reg("pl_title", QLineEdit())
        title.setPlaceholderText("automatic: the object or layer name")
        s_plate.addRow("Title", title)
        s_plate.addRow(self._check("pl_caption", "Caption with the settings, scale and date", True))
        s_plate.addRow("Format", self._combo("pl_format", ["PDF", "PNG", "TIFF"]))
        s_plate.addRow(self._primary("Make plate", self._run_plate))
        return self._page([s_meas, s_trace, s_pick, s_plate], [])

    def _build_lab_tab(self):
        s_in = Section("Shade Lab")
        s_in.addRow(_hint("Build a grey relief from its pieces: choose the frequency bands (crossovers on the "
                          "Frequency tab), smooth them, shade them with QGIS's own hillshade formula - or light them "
                          "from every direction round the compass and let PCA find the strongest relief. Pick the "
                          "output with the Output selector above the preview, or click its thumbnail in the side "
                          "panel; Render full resolution saves exactly that output."))
        s_bands = Section("Frequency bands feeding the lab")
        for key, label, desc in BANDS:
            s_bands.addRow(label, self._slider(f"lab_w_{key}", 0, 2, 1.0, tip=(
                f"{desc}. 0 = left out, 1 = as recorded, 2 = doubled. With every band at 1 the lab surface is the "
                "true relief - the hillshade is then exactly QGIS's.")))
        presets = []
        for name, ws in (("All", [1, 1, 1, 1, 1]), ("Fine (drypoint)", [0, 0, 0.6, 1, 1]),
                         ("Mid + fine", [0, 0.3, 1, 1, 0.6]), ("Broad form", [1, 1, 0.3, 0, 0])):
            b = QToolButton()
            b.setText(name)
            b.setToolTip("Band mix: " + ", ".join(f"{lab} {w:g}" for (_k, lab, _d), w in zip(BANDS, ws)))
            b.clicked.connect(lambda _c=False, ws=ws: self._set_lab_weights(ws))
            presets.append(b)
        s_bands.addRow("Mix", _hbox(*presets, stretch_first=False))
        s_bands.addRow(self._check("lab_autolevel", "Auto-level (fine bands shade as clearly as broad ones)", True,
                                   "Band hillshades - and any mix that leaves bands out - are scaled so their "
                                   "steepest 5 % reach 30°."))
        s_bands.addRow(self._check("lab_detrend", "Remove tilt (best-fit plane)", False))
        to_freq = QPushButton("Crossovers on the Frequency tab")
        to_freq.clicked.connect(lambda: self.tabs.setCurrentIndex(TAB_FREQUENCY))
        s_bands.addRow(to_freq)

        s_sm = Section("Smoothing")
        s_sm.addRow("Method", self._combo("lab_smoothing", eng.LAB_SMOOTHING.values(), tip=(
            "Gaussian: a plain low-pass - evens out noise and grain, softens everything.\nEdge-preserving: the guided "
            "filter - smooths noise and grain but keeps the walls of incisions and raised lines sharp.")))
        s_sm.addRow("Radius", self._slider("lab_smooth_radius", 0.5, 16, 1.5, 1, 0.5, suffix=" px",
                                           tip="Size of the smoothing, in full-resolution px."))
        s_sm.addRow("Edge threshold", self._slider("lab_edge", 0.02, 2, 0.3, 2, 0.01, log=True, tip=(
            "Edge-preserving only: relief smaller than this share of the surface's typical local relief is smoothed "
            "away; larger steps are kept. Lower keeps more.")))

        s_hs = Section("Hillshade (QGIS / gdaldem formula)")
        s_hs.addRow("Shading", self._combo("lab_variant", eng.GDAL_VARIANTS.values(), tip=(
            "For the single-light outputs. Multidirectional and Igor ignore the azimuth, as in QGIS.")))
        s_hs.addRow("Z factor", self._slider("lab_z", 0.01, 100, 1.0, 3, 0.01, log=True, tip=(
            "Vertical exaggeration on top of any auto-level.")))
        s_hs.addRow("Altitude", self._slider("lab_alt", 1, 89, 45, 0, 1, suffix="°",
                                             tip="Light height for every output, including the ring of lights."))
        s_hs.addRow(self._check("lab_zt", "Zevenbergen-Thorne gradient instead of Horn's", False))
        s_hs.addRow(_hint("The azimuth of single-light outputs is the key light - drag the sun on the dial or on "
                          "the preview."))

        s_pca = Section("Directions & PCA")
        s_pca.addRow("Directions", self._combo("lab_dirs", ["4", "8", "16", "32"], "16", tip=(
            "Hillshades lit from this many directions round the compass feed the mean and the PCA.")))
        s_pca.addRow(self._check("lab_pca_bands", "Multi-scale ring (each light shades every band)", False,
                                 "Every light of the ring shades each included band (auto-levelled) and mixes them by "
                                 "their weights, so fine and broad relief read in every image before the mean and PCA."))
        self.lbl_lab_evr = QLabel("")
        self.lbl_lab_evr.setWordWrap(True)
        s_pca.addRow(self.lbl_lab_evr)
        s_pca.addRow(_hint("PCA 1 is the hillshade lit from the most informative direction, PCA 2 the one across "
                           "it; together they hold almost every slope, so 'PCA 1+2 strength' shows relief whichever "
                           "way it faces. After Devereux, Amable & Crow (2008), PCA of hillshades for LiDAR."))

        s_out = Section("Output")
        s_out.addRow(_hint("Choose the output with the Output selector above the preview (or a thumbnail)."))
        s_out.addRow("Display", self._combo("lab_display", list(LAB_DISPLAY_OF), tip=(
            "Auto contrast: stretched between the 0.5 % and 99.5 % points of the whole surface (same for the "
            "preview and the saved file). Raw: gdaldem's grey levels.")))
        use_main = QPushButton("Use as the main render (Render tab)")
        use_main.clicked.connect(self._use_lab_as_main)
        s_out.addRow(use_main)
        return self._page([s_in, s_bands, s_sm, s_hs, s_pca, s_out],
                          [self._primary("Render full resolution", self._run_render_lab)])

    def _build_frequency_tab(self):
        s_intro = Section("Five-band decomposition")
        s_intro.addRow(_hint("Splits the height map into five spatial-frequency bands, like an audio crossover. "
                             "At gain 1 with no treatment the bands sum back to the height map exactly. The same "
                             "crossovers and gains drive the Render tab equaliser."))
        self.fq_preset = QComboBox()
        self.fq_preset.addItems(list(SURFACE_PRESETS))
        apply_p = QPushButton("Apply")
        apply_p.setMaximumWidth(70)
        apply_p.clicked.connect(self._apply_surface_preset)
        s_intro.addRow("Surface preset", _hbox(self.fq_preset, apply_p))

        s_cond = Section("Conditioning (before splitting)", expanded=False)
        s_cond.addRow("Median despeckle", self._ispin("fq_median", 0, 9, 0, 2, " px", "Kernel size; 0 = off."))
        s_cond.addRow("Gaussian pre-blur", self._dspin("fq_preblur", 0, 5, 0, 2, 0.05, " px"))
        s_cond.addRow(self._check("fq_detrend", "Remove best-fit plane"))
        s_cond.addRow("Ultra-high denoise", self._dspin("fq_denoise", 0, 100, 0, 0, 5, " %"))

        s_x = Section("Crossover scales (Gaussian sigma, full-resolution px)")
        for i, (lbl, lo, hi, val) in enumerate((("Very low / low", 2, 2000, 64), ("Low / mid", 1, 1000, 32),
                                                ("Mid / high", 0.5, 500, 16), ("High / ultra-high", 0.25, 250, 8))):
            s_x.addRow(lbl, self._dspin(f"fq_sigma{i}", lo, hi, val, 2, 1))

        s_b = Section("Bands")
        grid_w = QWidget()
        g = QGridLayout(grid_w)
        g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(6)
        for c, head in enumerate(["Band", "Gain", "Treatment", "Radius", "Threshold"]):
            g.addWidget(_hint(head), 0, c)
        g.setColumnStretch(2, 1)
        for r, (key, label, desc) in enumerate(BANDS, start=1):
            g.addWidget(self._check(f"fq_on_{key}", label, True, desc), r, 0)
            gain = self._dspin(f"fq_gain_{key}", 0, 4, 1.0, 2, 0.1)
            gain.setMaximumWidth(62)
            g.addWidget(gain, r, 1)
            mode = self._combo(f"fq_mode_{key}", ["None", "Gaussian smooth", "Median despeckle", "Local contrast"])
            mode.setMinimumContentsLength(6)
            mode.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            g.addWidget(mode, r, 2)
            radius = self._dspin(f"fq_radius_{key}", 0, 12, 0, 2, 0.25, " px")
            radius.setMaximumWidth(72)
            g.addWidget(radius, r, 3)
            thresh = self._dspin(f"fq_thresh_{key}", 0, 1e6, 0, 6, 0.001,
                                 tip="Zero out band values weaker than this (height units).")
            thresh.setMaximumWidth(84)
            g.addWidget(thresh, r, 4)
        s_b.addRow(grid_w)
        s_b.addRow("Preview", self._combo("fq_preview_band", FQ_PREVIEWS, tip=(
            "What the preview shows on this tab: the band-shading combination, the recombined relief, one band's "
            "shade, or one band as a signed colour ramp.")))

        s_bs = Section("Band shading (a different shade per band)")
        s_bs.addRow(_hint("Give each band its own kind of shading - soft light for the broad form, raking or "
                          "multi-directional light for tool marks, curvature for lines, texture for grain - and "
                          "combine them. Click a band in the side panel to see it alone; render the result with the "
                          "Render tab's 'Frequency band shading' mode."))
        self.band_preset = QComboBox()
        self.band_preset.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.band_preset.setMinimumContentsLength(14)
        self.band_preset.addItems(list(eng.BAND_PRESETS))
        apply_bands = QPushButton("Apply")
        apply_bands.setMaximumWidth(70)
        apply_bands.clicked.connect(lambda: self._apply_band_preset())
        s_bs.addRow("Preset", _hbox(self.band_preset, apply_bands))
        bs_w = QWidget()
        gb = QGridLayout(bs_w)
        gb.setContentsMargins(0, 0, 0, 0)
        gb.setHorizontalSpacing(6)
        for c, head in enumerate(["Band", "Shade", "Strength", "Light alt."]):
            gb.addWidget(_hint(head), 0, c)
        gb.setColumnStretch(1, 1)
        start = eng.BAND_PRESETS["Balanced (form, marks, lines, grain)"]
        for r, ((key, label, desc), (t, wgt, alt)) in enumerate(zip(BANDS, start), start=1):
            name = QLabel(label)
            name.setToolTip(desc)
            gb.addWidget(name, r, 0)
            shade = self._combo(f"fq_shade_{key}", eng.BAND_SHADE_TYPES.values(), eng.BAND_SHADE_TYPES[t])
            shade.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            shade.setMinimumContentsLength(9)
            gb.addWidget(shade, r, 1)
            strength = self._dspin(f"fq_sw_{key}", 0, 2, wgt, 2, 0.05, tip="How strongly this band's shade shows.")
            strength.setMaximumWidth(70)
            gb.addWidget(strength, r, 2)
            height = self._dspin(f"fq_salt_{key}", 2, 85, alt, 0, 1, "\u00b0",
                                 tip="Light height for hillshade / multi-directional shades (low = raking).")
            height.setMaximumWidth(70)
            gb.addWidget(height, r, 3)
        s_bs.addRow(bs_w)
        s_bs.addRow(self._check("fq_autolevel", "Auto-level each band (fine bands shade as clearly as broad ones)",
                                True))
        use_bands = QPushButton("Render this combination (Render tab)")
        use_bands.clicked.connect(self._use_band_shading)
        s_bs.addRow(use_bands)
        to_lab = QPushButton("Hillshade && PCA the ticked bands (Shade Lab)")
        to_lab.setToolTip("Send the bands ticked on this tab to the Shade Lab: QGIS-formula hillshades of them, and "
                          "PCA of hillshades from every direction.")
        to_lab.clicked.connect(self._send_bands_to_lab)
        s_bs.addRow(to_lab)

        s_ramps = Section("Map colour ramps", expanded=False)
        s_ramps.addRow(self._check("fq_colorize", "Colour the band layers with semi-transparent ramps", True))
        for key, label, _ in BANDS:
            s_ramps.addRow(label, self._combo(f"fq_ramp_{key}", COLOR_RAMP_PRESETS.keys(), DEFAULT_BAND_RAMPS[key]))

        self.btn_undo = QPushButton("Undo last split")
        self.btn_undo.setEnabled(False)
        self.btn_undo.setToolTip("Removes the last split's layer group and deletes its files.")
        self.btn_undo.clicked.connect(self._undo_split)
        return self._page([s_intro, s_bs, s_b, s_x, s_cond, s_ramps],
                          [self.btn_undo, self._primary("Split into bands", self._run_split)])

    def _build_stack_tab(self):
        s_src = Section("Source surface")
        s_src.addRow("Heights", self._combo("ls_source", [
            "Height map with Render-tab surface preparation", "Height map, raw heights × Z-factor",
            "Mix of three pre-split height maps"]))
        self.mix_box = QWidget()
        mf = QFormLayout(self.mix_box)
        mf.setContentsMargins(0, 0, 0, 0)
        for tag, label, weight in (("u", "Ultra-high", 0.99), ("h", "High / mid", 1.0), ("l", "Low", 0.10)):
            cb = self._reg(f"ls_mix_{tag}", _raster_combo(), layer=True)
            w = self._dspin(f"mix_w_{tag}", 0, 10, weight, 2, 0.05)
            w.setPrefix("w ")
            zf = self._dspin(f"mix_z_{tag}", 0, 20, 1.0, 2, 0.1)
            zf.setPrefix("z ")
            mf.addRow(label, _hbox(cb, w, zf))
        s_src.addRow(self.mix_box)

        s_dome = Section("Light dome")
        s_dome.addRow("Lights", self._combo("ls_count", ["4", "6", "8", "12", "16", "24", "32", "64"], "8"))
        s_dome.addRow("Rotation", self._dspin("ls_offset", -360, 360, 0, 1, 5, "°"))
        s_dome.addRow("Altitude", self._dspin("ls_altitude", 0, 90, 15, 1, 1, "°"))
        self.stack_preset = QComboBox()
        self.stack_preset.addItems(list(STACK_PRESETS))
        ap = QPushButton("Apply")
        ap.clicked.connect(self._apply_stack_preset)
        s_dome.addRow("Preset", _hbox(self.stack_preset, ap))
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Name", "Azimuth °", "Altitude °"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(180)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setToolTip("Select a light to preview it; names, azimuths and altitudes are editable.")
        self.table.currentCellChanged.connect(self._on_table_edit)
        s_dome.addRow(self.table)

        s_out = Section("Outputs")
        s_out.addRow(self._check("ls_multiband", "Multi-band GeoTIFF (one band per light)", True))
        s_out.addRow(self._check("ls_individual", "Individual light layers"))
        s_out.addRow(self._check("ls_vrt", "Virtual raster (VRT) of the individual lights"))
        s_out.addRow(self._check("ls_mean", "Balanced composite (mean of all lights)"))
        svf_r = self._ispin("ls_svf_radius", 2, 512, 24, suffix=" px")
        s_out.addRow(_hbox(self._check("ls_svf", "Sky-view factor (horizon-based)"), svf_r))
        conv = self._combo("ls_normal_conv", ["OpenGL (green up)", "DirectX (green down)"])
        s_out.addRow(_hbox(self._check("ls_normals", "RGB normal map"), conv))
        tpi_r = self._ispin("ls_tpi_radius", 1, 256, 1, suffix=" px")
        s_out.addRow(_hbox(self._check("ls_tpi", "Topographic position index (ridges / valleys)"), tpi_r))
        pca_n = self._ispin("ls_pca_n", 1, 8, 3)
        pca_n.setPrefix("components ")
        s_out.addRow(_hbox(self._check("ls_pca", "PCA composite of the stack"), pca_n))

        s_tone = Section("Per-light tone & display", expanded=False)
        s_tone.addRow("Tone", self._combo("ls_tone", ["None (raw Lambert)", "Shadow lift", "Soft S-curve (Brown)"]))
        s_tone.addRow("Shadow lift", self._ispin("ls_lift", 0, 100, 0, suffix=" %"))
        s_tone.addRow("Layer blend", self._combo("ls_blend", BLEND_MODES.keys(), "Overlay"))
        s_tone.addRow(self._check("ls_group", "Group outputs in a layer folder", True))
        s_tone.addRow(_hbox(self._check("ls_colorize", "Colour the composite / SVF"),
                            self._combo("ls_ramp", COLOR_RAMP_PRESETS.keys(), "Blue - Orange - Brown")))
        s_tone.addRow(self._check("ls_colour_add", "Add the colour layer to the group"))
        s_tone.addRow("Colour position", self._combo("ls_colour_pos", ["Bottom (shadows over colour)",
                                                                       "Top (colour tints shadows)"]))
        s_tone.addRow("Colour blend", self._combo("ls_colour_blend", BLEND_MODES.keys(), "Multiply"))
        s_tone.addRow("Colour opacity", self._ispin("ls_colour_opacity", 0, 100, 100, suffix=" %"))

        return self._page([s_src, s_dome, s_out, s_tone],
                          [self._primary("Run light stack", self._run_stack)])

    def _build_provenance_tab(self):
        s_spec = Section("Multispectral fusion")
        s_spec.addRow("UV / IR layer", self._reg("prov_uvir", _raster_combo(allow_empty=True), layer=True))
        s_spec.addRow(self._check("prov_uvir_add", "Overlay it on light-stack results"))
        s_spec.addRow("Opacity", self._ispin("prov_uvir_opacity", 0, 100, 50, suffix=" %"))

        s_val = Section("Validation")
        s_val.addRow(self._check("prov_warn_geo", "Warn on geographic (degree) CRS", True))
        s_val.addRow(self._check("prov_warn_mismatch", "Warn when mixed maps differ in size", True))

        s_log = Section("Provenance record")
        s_log.addRow(_hint("Every output gets a *_provenance.json sidecar with the source, pixel size, every "
                           "parameter, normalisation constants, software versions and timing."))
        self.meta_view = QPlainTextEdit()
        self.meta_view.setReadOnly(True)
        self.meta_view.setMinimumHeight(220)
        s_log.addRow(self.meta_view)
        open_btn = QPushButton("Open export folder")
        open_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self._output_dir(False))))
        s_log.addRow("", open_btn)
        return self._page([s_spec, s_val, s_log], [])

    def _build_preview_panel(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 0, 0, 0)
        bar = QHBoxLayout()
        bar.setSpacing(4)

        def tool(text, tip, slot=None, checkable=False):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.setCheckable(checkable)
            if slot is not None:
                b.clicked.connect(slot)
            bar.addWidget(b)
            return b

        tool("Fit", "Fit the whole surface (F or 0)", lambda: self.canvas.fit())
        for label, sc in (("1:2", 0.5), ("1:1", 1.0), ("2:1", 2.0), ("4:1", 4.0)):
            tool(label, f"Zoom to {label} (full-resolution pixels : screen pixels)",
                 lambda _c=False, v=sc: self.canvas.zoomTo(v))
        self.lbl_zoom = QLabel("")
        self.lbl_zoom.setMinimumWidth(96)
        bar.addWidget(self.lbl_zoom)
        self.btn_pan = tool("Pan", "Left-drag pans; right-drag moves the light", checkable=True)
        self.btn_light = tool("Light", "Left-drag moves the light; right-drag pans", checkable=True)
        self.btn_measure_tool = tool("Measure", "Left-drag: height profile along a line; Shift-drag: area statistics; "
                                                "right-drag pans. Results on the Tools tab.", checkable=True)
        group = QButtonGroup(self)
        tools_ = ((self.btn_pan, "pan"), (self.btn_light, "light"), (self.btn_measure_tool, "measure"))
        for b, _name in tools_:
            group.addButton(b)
        self.btn_pan.setChecked(True)                   # before connecting: the canvas does not exist yet
        for b, name in tools_:
            b.toggled.connect(lambda on, n=name: on and self.canvas.setTool(n))
        bar.addStretch(1)
        self.lbl_view = QLabel("View")
        bar.addWidget(self.lbl_view)
        comp = self._combo("pv_component", eng.COMPONENTS.values(), tip=(
            "Look at one ingredient of the render on its own - lighting, shadows, texture, occlusion, curvature, "
            "slope - to see exactly what each control does."))
        comp.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        comp.setMinimumContentsLength(10)
        bar.addWidget(comp)
        lab_out = self._combo("lab_output", eng.LAB_OUTPUTS.values(), tip=(
            "Which Shade Lab output the preview shows - and Render full resolution saves."))
        lab_out.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        lab_out.setMinimumContentsLength(18)
        lab_out.setVisible(False)
        bar.addWidget(lab_out)
        bar.addWidget(QLabel("Compare"))
        cmp_box = self._combo("pv_compare", ["Off", "Before last change", "Colour / albedo", "Plain hillshade",
                                             "Height map"], tip=(
            "Split view with a divider you can drag. 'Before last change' shows the render as it was before your "
            "latest adjustments."))
        cmp_box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        cmp_box.setMinimumContentsLength(8)
        bar.addWidget(cmp_box)
        v.addLayout(bar)
        self.canvas = PreviewCanvas()
        self.canvas.lightDragged.connect(self._set_light)
        self.canvas.viewChanged.connect(self._on_view_changed)
        self.canvas.measured.connect(self._on_measured)
        self.side = QTabWidget()
        self.side.setDocumentMode(True)
        self.strip = ShadeStrip()
        self.strip.clicked.connect(self._on_strip_clicked)
        self.strip.setToolTip("Thumbnails of what the preview shows - click one to use it.")
        self.side.addTab(self.strip, "Shades")
        self.settings_view = QLabel()
        self.settings_view.setWordWrap(True)
        self.settings_view.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.settings_view.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.settings_view.setContentsMargins(6, 6, 6, 6)
        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setFrameShape(QFrame.Shape.NoFrame)
        settings_scroll.setWidget(self.settings_view)
        self.side.addTab(settings_scroll, "Settings")
        self.side.currentChanged.connect(lambda _i: self._strip_timer.start())
        self.side.setMinimumWidth(170)
        self.view_split = QSplitter(Qt.Orientation.Horizontal)
        self.view_split.setChildrenCollapsible(True)
        self.view_split.addWidget(self.canvas)
        self.view_split.addWidget(self.side)
        self.view_split.setStretchFactor(0, 1)
        self.view_split.setStretchFactor(1, 0)
        self.view_split.setSizes([700, 205])
        v.addWidget(self.view_split, 1)

        lower = QHBoxLayout()
        self.dial = SunDial()
        self.dial.setFixedSize(172, 172)
        self.dial.lightChanged.connect(self._set_light)
        lower.addWidget(self.dial)
        side = QVBoxLayout()
        self.lbl_showing = QLabel("Preview")
        self.lbl_showing.setWordWrap(True)
        self.lbl_showing.setProperty("hint", True)
        self.lbl_showing.setToolTip("What the preview is showing right now. Tick 'Show settings' to list "
                                    "every setting in effect over the image.")
        side.addWidget(self.lbl_showing)
        self.lbl_light = QLabel()
        f = self.lbl_light.font()
        f.setPointSizeF(f.pointSizeF() + 2)
        f.setBold(True)
        self.lbl_light.setFont(f)
        side.addWidget(self.lbl_light)
        side.addWidget(_hint("Drag the sun on the dial - or drag on the preview itself - to move the key light."))
        grid = QGridLayout()
        for i, (text, az, alt) in enumerate((("↖ Classic 35°", 315, 35), ("Raking 15°", None, 15),
                                             ("Grazing 6°", None, 6), ("Overhead 70°", None, 70),
                                             ("↺ −45°", -45, None), ("↻ +45°", 45, None))):
            b = QPushButton(text)
            if text.startswith(("↺", "↻")):
                b.clicked.connect(lambda _c=False, d=az: self._set_light(self._w["r_azimuth"].value() + d, None))
            else:
                b.clicked.connect(lambda _c=False, a=az, e=alt: self._set_light(a, e))
            grid.addWidget(b, i // 3, i % 3)
        # The map, image and export buttons share the light presets' three columns, so every button lines up.
        for i, (text, tip, slot) in enumerate((
                ("From map", "Show the QGIS map view in the preview", self._view_from_map),
                ("To map", "Zoom the QGIS map to the preview view", self._view_to_map),
                ("Save view…", "Save what the preview shows as a screen image, plus a settings .txt",
                 self._save_view),
                ("Copy settings", "Copy the settings in the view to the clipboard", self._copy_settings),
                ("⤓ Export view", "Render exactly the part of the surface the preview shows, at full resolution: "
                 "a GeoTIFF (+ PNG) in the export folder, georeferenced like the height map and added to the map "
                 "right over it. No questions asked.", lambda: self._quick_export(True)))):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            if text.startswith("⤓"):
                self._run_buttons.append(b)
            grid.addWidget(b, 2 + i // 3, i % 3)
        quick = self._primary("⤓ Quick export · full resolution", lambda: self._quick_export(False))
        quick.setFixedHeight(b.sizeHint().height())          # the same height as the buttons beside it
        quick.setToolTip("One click: the whole height map rendered at full resolution with exactly the settings the "
                         "preview shows - a GeoTIFF (+ PNG) in the export folder, georeferenced like the height map "
                         "and added to the map right over it (Ctrl+E).")
        grid.addWidget(quick, 3, 2)
        side.addLayout(grid)
        row2 = QHBoxLayout()
        row2.addWidget(self._check("pv_panel", "Side panel", True,
                                   "Show the Shades / Settings panel beside the preview"))
        row2.addWidget(self._check("pv_follow", "Follow map", False, "Keep the preview on the QGIS map view"))
        row2.addWidget(self._check("pv_live", "Live", True, "Re-render the preview as settings change"))
        refresh = QToolButton()
        refresh.setText("Refresh")
        refresh.setToolTip("Re-render the preview (Ctrl+R)")
        refresh.clicked.connect(lambda: self._schedule_preview(force=True))
        row2.addWidget(refresh)
        row2.addStretch(1)
        side.addLayout(row2)
        side.addStretch(1)
        lower.addLayout(side, 1)
        v.addLayout(lower)
        return w

    def _build_bottom_bar(self):
        h = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setMaximumWidth(260)
        self.progress.setTextVisible(False)
        self.status = QLabel("Ready")
        self.status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.btn_log = QToolButton()
        self.btn_log.setText("Log")
        self.btn_log.setCheckable(True)
        self.btn_log.toggled.connect(self.log.setVisible)
        self.btn_cancel = QPushButton("Cancel job")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(lambda: self._task and self._task.cancel())
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        h.addWidget(self.progress)
        h.addWidget(self.status, 1)
        h.addWidget(self.btn_log)
        h.addWidget(self.btn_cancel)
        h.addWidget(close)
        return h

    def _fit_to_screen(self):
        scr = self.screen() or QApplication.primaryScreen()
        if scr is None:
            self.resize(1320, 860)
            return
        g = scr.availableGeometry()
        self.setMinimumSize(min(940, g.width() - 40), min(600, g.height() - 60))
        self.resize(min(1380, int(g.width() * 0.92)), min(900, int(g.height() * 0.9)))

    # ------------------------------------------------------------------ reactions
    def _on_changed(self, key):
        if self._suspend:
            return
        if not key.startswith("pv_"):
            self._note_change()
        if key.startswith("r_") and not key.startswith("r_out"):
            self._look_dirty = True
        if key in self._labels and not key.startswith(("r_out", "prov_", "export", "pv_", "mix_")):
            now = time.monotonic()
            if now - self._last_change_t > 1.0:
                self.canvas.snapshotPrevious()     # 'Before last change' = state before this burst of edits
            self._last_change_t = now
            self.canvas.flash(f"{self._labels[key]}: {self._fmt_value(self._w[key])}")
        if key.startswith("fq_mode_") and self._w[key].currentText() != "None":
            radius = self._w["fq_radius_" + key[len("fq_mode_"):]]
            if radius.value() == 0:
                radius.setValue(1.5)          # a treatment with zero radius would do nothing
        self._update_enabled()
        self._update_overlay()
        if key == "surface":
            self._on_surface_changed()
            return
        if key == "pixel_auto":
            self._update_pixel_widgets()
        if key == "r_mode":
            self._update_mode_visibility()
        if key in ("r_azimuth", "r_altitude"):
            self._sync_dial()
        if key == "ls_source":
            self._update_source_visibility()
        if key in ("ls_count", "ls_offset", "ls_altitude"):
            self._regen_table()
        if key in ("r_mode", "r_fill", "r_fill_offset", "r_fill_altitude", "r_dome_lights", "r_dome_altitude",
                   "r_azimuth", "r_clean_allround", "r_clean_lights", "r_clean_ring_alt"):
            self._update_dial_markers()
        if key in ("pixel_size", "pixel_unit"):
            self._update_scale_bar()
        if key == "colour":
            self._update_colour_state()
        if key in ("normals", "normals_green"):
            self._update_normals_state()
        if key == "pv_compare":
            text = self._w["pv_compare"].currentText()
            self.canvas.setCompare(text != "Off", text, "previous" if text == "Before last change" else "cmp")
        if key == "pv_panel":
            self.side.setVisible(self._w["pv_panel"].isChecked())
            self._strip_timer.start()
            return
        if key == "pv_follow":
            if self._w["pv_follow"].isChecked():
                self._view_from_map()
            return
        if key == "pv_live":
            if self._w["pv_live"].isChecked():
                self._schedule_preview(force=True)
            return
        if key.startswith("dt_"):
            self._update_detail_chart()
            self._schedule_lines()
            if key in ("dt_use_saved", "dt_area"):
                return
        if key.startswith("tr_"):
            if key == "tr_live" and not self._w["tr_live"].isChecked():
                self._clear_lines()
            self._schedule_lines()
        if key.startswith(("r_out", "prov_", "mix_", "export", "pixel_unit", "pv_panel", "ms_", "tr_", "tp_",
                           "pl_")):
            return
        if key.startswith("ls_") and key not in ("ls_tone", "ls_lift", "ls_source", "ls_count", "ls_offset",
                                                 "ls_altitude"):
            return
        self._schedule_preview()

    def _on_tab_changed(self, _index):
        if self.tabs.currentIndex() == TAB_DETAIL:
            self._update_detail_chart()
        self._update_output_selector()
        self._update_dial_markers()
        self._update_overlay()
        self._schedule_preview(force=True)

    def _on_canvas_extent(self):
        if self.isVisible() and self._w["pv_follow"].isChecked() and not self._to_map_busy:
            self._view_from_map()

    def _set_light(self, azimuth, altitude):
        if azimuth is not None:
            self._w["r_azimuth"].setValue(azimuth % 360.0)
        if altitude is not None:
            self._w["r_altitude"].setValue(min(max(altitude, 0.0), 90.0))

    def _sync_dial(self):
        az, alt = self._w["r_azimuth"].value(), self._w["r_altitude"].value()
        self.dial.setLight(az, alt)
        mode = self._mode()
        if mode == "multi":
            self.lbl_light.setText(f"Rig rotation  {az - 315.0:+.0f}°   (drag the sun to rotate all lights)")
        elif mode == "dome":
            self.lbl_light.setText(f"Ring rotation  {az:.0f}°")
        else:
            self.lbl_light.setText(f"Key light  {az:.0f}° / {alt:.0f}°")

    def _update_dial_markers(self):
        tab, mode = self.tabs.currentIndex(), self._mode()
        fill = None
        if tab in (TAB_STACK,):
            markers = [(az, alt) for _, az, alt in self._table_rows()]
            if markers:
                fill = markers[min(max(self.table.currentRow(), 0), len(markers) - 1)]   # highlight the selection
        elif mode == "clean" and self._w["r_clean_allround"].value() > 0:
            markers = eng.ring_lights(self._w["r_clean_lights"].value(), self._w["r_clean_ring_alt"].value(),
                                      self._w["r_azimuth"].value())
        elif mode == "dome":
            markers = eng.ring_lights(self._w["r_dome_lights"].value(), self._w["r_dome_altitude"].value(),
                                      self._w["r_azimuth"].value())
        elif (tab == TAB_LAB or mode == "lab") and self._lab_output_key() in eng.LAB_STACK_OUTPUTS:
            n = int(self._w["lab_dirs"].currentText())
            markers = [(360.0 * i / n, self._w["lab_alt"].value()) for i in range(n)]
        elif mode == "gis" and self._w["r_gis_lights"].value() > 1:
            markers = eng.gis_ring(self._w["r_gis_lights"].value(), self._w["r_azimuth"].value(),
                                   self._w["r_altitude"].value())
        elif mode == "multi":
            rot = self._w["r_azimuth"].value() - 315.0
            markers = [((az + rot) % 360.0, alt) for on, az, alt, wt, _ in self._multi_rows() if on and wt > 0]
        else:
            markers = []
        if tab not in (TAB_STACK,) and mode == "studio" and self._w["r_fill"].value() > 0:
            fill = ((self._w["r_azimuth"].value() + self._w["r_fill_offset"].value()) % 360,
                    self._w["r_fill_altitude"].value())
        self.dial.setMarkers(markers)
        self.dial.setFill(fill)

    def _mode(self):
        return MODE_BY_LABEL.get(self._w["r_mode"].currentText(), "studio")

    def _update_mode_visibility(self):
        """Show only the sections and rows the current mode uses."""
        mode, s, w = self._mode(), self._rs, self._w
        lit = mode in eng.LIT_MODES
        s["key"].setVisible(mode in ("clean", "gdal", "lab", "studio", "gis", "multi", "exaggerated", "specular", "hillshade",
                                     "dome", "vat", "bands"))
        s["key"].toggle.setText({"multi": "Rig rotation", "dome": "Ring rotation", "exaggerated": "Light direction",
                                 "vat": "Hillshade light", "bands": "Light direction (all bands)",
                                 "gis": "Light direction (first light)", "specular": "Light direction",
                                 "gdal": "Light (gdaldem -az / -alt)", "lab": "Light (single-light outputs)",
                                 "clean": "Key light (the ring turns with it)"}.get(
                                     mode,
                                                                                                         "Key light"))
        s["key"].setRowVisible(w["r_altitude"], mode in ("clean", "gdal", "studio", "gis", "specular", "hillshade",
                                                         "exaggerated", "vat"))
        s["key"].setRowVisible(w["r_key"], mode in ("studio", "specular"))
        s["multi"].setVisible(mode == "multi")
        s["exag"].setVisible(mode == "exaggerated")
        s["vat"].setVisible(mode == "vat")
        s["gis"].setVisible(mode == "gis")
        s["gdal"].setVisible(mode == "gdal")
        s["clean"].setVisible(mode == "clean")
        s["lab"].setVisible(mode == "lab")
        # gdaldem shades the raw heights; the lab and clean relief prepare their own surface and tone
        s["prep"].setVisible(mode not in ("gdal", "lab", "clean"))
        s["mat"].setVisible(mode not in ("gdal", "lab"))       # clean relief: only the colour composite rows
        s["spec"].setVisible(mode == "specular")
        s["bands"].setVisible(mode == "bands")
        s["fill"].setVisible(mode in ("studio", "multi"))
        for k in ("r_fill", "r_fill_offset", "r_fill_altitude"):
            s["fill"].setRowVisible(w[k], mode == "studio")
        s["shadow"].setVisible(mode == "studio")
        s["dome"].setVisible(mode == "dome")
        s["tex"].setVisible(lit or mode == "texture")
        s["tex"].setRowVisible(w["r_texture_mix"], lit)
        s["occ"].setVisible(lit or mode in ("svf", "vat"))
        s["occ"].setRowVisible(w["r_occlusion"], lit)
        s["edge"].setVisible(lit)
        litset = set(eng.LIT_MODES)
        grey = litset | {"gis", "specular", "hillshade", "dome", "texture", "svf", "vat"}
        rows = {"r_specular": {"studio"}, "r_shininess": {"studio", "specular"}, "r_clarity": litset,
                "r_clarity_radius": litset, "r_base_tone": litset,
                "r_contrast": litset | {"gis", "specular", "hillshade", "dome", "svf", "vat"}, "r_gamma": grey,
                "r_tint": grey, "r_albedo": litset | {"gis", "hillshade", "dome", "vat", "clean"},
                "r_albedo_strength": litset | {"gis", "hillshade", "dome", "vat", "clean"},
                "r_albedo_relief": litset | {"gis", "hillshade", "dome", "vat", "clean"},
                "r_albedo_gamma": litset | {"gis", "hillshade", "dome", "vat", "clean"},
                "r_albedo_exposure": litset | {"gis", "hillshade", "dome", "vat", "clean"},
                "r_normal_convention": {"normals"}}
        for key, modes in rows.items():
            s["mat"].setRowVisible(w[key], mode in modes)
        self._update_enabled()
        self._update_output_selector()
        self._update_dial_markers()
        self._sync_dial()

    def _update_enabled(self):
        """Grey out controls that cannot change anything with the current
        settings, so nothing looks broken when it is merely switched off."""
        w, mode = self._w, self._mode()
        lit = mode in ("studio", "multi")

        def en(key, on):
            w[key].setEnabled(bool(on))

        tex_on = mode == "texture" or (lit and w["r_texture_mix"].value() > 0)
        for k in ("r_texture_detail", "r_texture_contrast", "r_texture_midtone", "r_texture_curve"):
            en(k, tex_on)
        svf_on = mode == "svf" or (lit and w["r_occlusion"].value() > 0)
        en("r_svf_radius", svf_on)
        en("r_svf_dirs", svf_on)
        shadows = w["r_shadows"].value() > 0
        en("r_shadow_softness", shadows)
        en("r_shadow_reach", shadows)
        en("r_shininess", w["r_specular"].value() > 0 or mode == "specular")
        en("r_gis_blend", w["r_gis_lights"].value() > 1)
        en("r_gis_slope_range", w["r_gis_slope"].value() > 0)
        variant = {v: k for k, v in eng.GDAL_VARIANTS.items()}.get(w["r_gdal_variant"].currentText(), "standard")
        en("r_azimuth", not (mode == "gdal" and variant == "multidirectional"))   # gdaldem fixes the four lights
        ring = w["r_clean_allround"].value()
        # Igor shading has no altitude; a clean relief lit only by the ring has no key light
        en("r_altitude", not ((mode == "gdal" and variant == "igor") or (mode == "clean" and ring >= 1.0)))
        en("z_factor", mode not in ("gdal", "lab", "clean"))       # own Z factors; clean relief sets its own steepness
        en("r_clean_edge", w["r_clean_smooth"].value() > 0)
        has_normals = w["normals"].currentLayer() is not None
        en("r_clean_normals", has_normals and mode == "clean")
        en("normals_green", has_normals)
        en("r_clean_ring_alt", ring > 0)
        en("r_clean_lights", ring > 0)
        if "lab_variant" in w and "lab_output" in w:                              # once the lab is built
            self._update_lab_enabled(en, mode)
        en("r_clarity_radius", w["r_clarity"].value() > 0)
        en("r_curvature_radius", w["r_curvature"].value() > 0)
        en("r_slope_range", w["r_slope_shade"].value() > 0)
        en("r_detail_radius", w["r_detail_boost"].value() > 0)
        for k in eng.BAND_KEYS:
            t = SHADE_TYPE_OF.get(w[f"fq_shade_{k}"].currentText(), "off")
            on = w[f"fq_on_{k}"].isChecked() and t != "off"
            en(f"fq_sw_{k}", on)
            en(f"fq_salt_{k}", on and t in ("hillshade", "multi"))
        fill = w["r_fill"].value() > 0
        en("r_fill_offset", fill)
        en("r_fill_altitude", fill)
        has_colour = w["colour"].currentLayer() is not None
        en("r_albedo", has_colour)
        for k in ("r_albedo_strength", "r_albedo_relief", "r_albedo_gamma", "r_albedo_exposure"):
            en(k, has_colour and w["r_albedo"].isChecked())
        en("r_tint", not self._albedo_active(self._render_params_basic()))
        en("ls_lift", w["ls_tone"].currentIndex() == 1)
        for flag, dep in (("ls_svf", "ls_svf_radius"), ("ls_normals", "ls_normal_conv"), ("ls_tpi", "ls_tpi_radius"),
                          ("ls_pca", "ls_pca_n"), ("ls_colorize", "ls_ramp"), ("prov_uvir_add", "prov_uvir_opacity")):
            en(dep, w[flag].isChecked())
        for k in ("ls_colour_pos", "ls_colour_blend", "ls_colour_opacity"):
            en(k, w["ls_colour_add"].isChecked() and has_colour)
        for k, _, _ in BANDS:
            en(f"fq_radius_{k}", w[f"fq_mode_{k}"].currentText() != "None")
        if "dt_on0" in w:
            for i in range(3):
                for k in ("dt_fine", "dt_broad", "dt_gain"):
                    en(f"{k}{i}", w[f"dt_on{i}"].isChecked())

    def _update_source_visibility(self):
        self.mix_box.setVisible(self._w["ls_source"].currentIndex() == 2)

    def _update_pixel_widgets(self):
        auto = self._w["pixel_auto"].isChecked()
        self._w["pixel_size"].setEnabled(not auto)
        if auto and self._ls is not None:
            prev, self._suspend = self._suspend, True
            self._w["pixel_size"].setValue(self._ls["info"].dx)
            self._suspend = prev
        self._update_scale_bar()

    def _set_run_enabled(self, on):
        for b in self._run_buttons:
            b.setEnabled(on and self._task is None)

    # ------------------------------------------------------------------ surface & preview data
    def _surface_layer(self):
        return self._w["surface"].currentLayer()

    def _pixel_size(self, info):
        if self._w["pixel_auto"].isChecked():
            return info.dx, info.dy
        v = self._w["pixel_size"].value()
        return v, v

    def _layer_state(self, layer):
        path = rio.gdal_path(layer.source())
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0
        key = (layer.id(), path, mtime)
        if self._ls is not None and self._ls["key"] == key:
            return self._ls
        self.status.setText("Reading surface overview…")
        info = rio.RasterInfo(path)
        _, out, _ = info.window_fit(1024)
        ov, _ = rio.read_array(path, None, out)
        a, b, c = eng.fit_plane(ov)
        sx, sy = info.width / float(out[0]), info.height / float(out[1])
        plane = (a / sx, b / sy, c + a * (0.5 / sx - 0.5) + b * (0.5 / sy - 0.5))
        rw = info.centre_window(1024)
        ref, _ = rio.read_array(path, rw)
        finite = ov[np.isfinite(ov)]
        self._ls = dict(key=key, id=layer.id(), info=info, plane=plane, ref=ref, ref_origin=(rw[0], rw[1]),
                        z_range=float(finite.max() - finite.min()) if finite.size else 0.0)
        self.status.setText("Ready")
        return self._ls

    def _on_surface_changed(self):
        self._ls = None
        layer = self._surface_layer()
        if layer is None:
            self.lbl_info.setText("Add a height-map raster to the project to begin.")
            self.canvas.clear()
            self.canvas.setMessage("Add a height-map raster (GeoTIFF, PNG16, ...) to the QGIS project, then choose "
                                   "it as the Height map.")
            self._set_run_enabled(False)
            return
        try:
            info = self._layer_state(layer)["info"]
        except Exception as exc:
            self.lbl_info.setText(f"Cannot read this raster: {exc}")
            self._set_run_enabled(False)
            return
        where = "georeferenced" if info.georeferenced else "pixel grid"
        self.lbl_info.setText(f"{info.width:,} × {info.height:,} px · {info.megapixels:.1f} MP · "
                              f"{info.data_type} · {where}")
        self.canvas.setFullSize(info.width, info.height)
        self._update_pixel_widgets()
        self._set_run_enabled(True)
        self._update_colour_state()
        self._update_normals_state()
        self._schedule_preview(force=True)

    def _canvas_window(self, layer, info):
        if self.iface is None:
            return None
        try:
            canvas = self.iface.mapCanvas()
            ext = canvas.extent()
            dest = canvas.mapSettings().destinationCrs()
            if dest.isValid() and layer.crs().isValid() and dest != layer.crs():
                ext = QgsCoordinateTransform(dest, layer.crs(), QgsProject.instance()).transformBoundingBox(ext)
            lext = layer.extent()
            c0 = (ext.xMinimum() - lext.xMinimum()) / lext.width() * info.width
            c1 = (ext.xMaximum() - lext.xMinimum()) / lext.width() * info.width
            r0 = (lext.yMaximum() - ext.yMaximum()) / lext.height() * info.height
            r1 = (lext.yMaximum() - ext.yMinimum()) / lext.height() * info.height
        except Exception:
            return None
        c0, c1 = max(0, int(c0)), min(info.width, int(math.ceil(c1)))
        r0, r1 = max(0, int(r0)), min(info.height, int(math.ceil(r1)))
        if c1 - c0 < 8 or r1 - r0 < 8:
            return None
        return (c0, r0, c1 - c0, r1 - r0)

    # ------------------------------------------------------------------ render parameters
    def _render_params_basic(self):
        return {"mode": self._mode()}

    def _render_params(self):
        w = self._w
        p = {"mode": self._mode()}
        for k in ("azimuth", "altitude", "key", "fill", "fill_offset", "fill_altitude", "ambient", "wrap", "shadows",
                  "shadow_softness", "dome_altitude", "texture_mix", "texture_contrast", "texture_midtone",
                  "occlusion", "specular", "shininess", "clarity", "clarity_radius", "base_tone", "contrast", "gamma",
                  "flatten_sigma", "detail_boost", "detail_radius", "curvature", "curvature_radius", "slope_shade",
                  "slope_range", "exag_contrast", "exag_detail", "exag_base", "gis_ambient", "gis_shadow",
                  "gis_highlight", "gis_slope", "gis_slope_range", "spec_depth", "smooth_sigma", "gis_ve",
                  "gis_brightness", "clean_flatten", "clean_smooth", "clean_edge", "clean_fine", "clean_relief",
                  "clean_allround", "clean_ring_alt", "clean_mid", "clean_contrast", "clean_highlight",
                  "albedo_relief", "albedo_gamma", "albedo_exposure"):
            p[k] = w["r_" + k].value()
        p["clean_lights"] = w["r_clean_lights"].value()
        p["clean_source"] = ("normals" if w["r_clean_normals"].isChecked() and w["normals"].currentLayer() is not None
                             else "height")
        p["normals_green"] = self._normals_green()
        p["shadow_reach"] = w["r_shadow_reach"].value()
        p["exag_scales"] = w["r_exag_scales"].value()
        p["gis_lights"] = w["r_gis_lights"].value()
        p["gis_blend"] = {v: k for k, v in eng.GIS_BLENDS.items()}.get(w["r_gis_blend"].currentText(), "weighted")
        p["gdal_z"] = w["r_gdal_z"].value()
        p["gdal_scale"] = w["r_gdal_scale"].value()
        p["gdal_variant"] = {v: k for k, v in eng.GDAL_VARIANTS.items()}.get(w["r_gdal_variant"].currentText(),
                                                                            "standard")
        p["gdal_zt"] = w["r_gdal_zt"].isChecked()
        p["gdal_edges"] = w["r_gdal_edges"].isChecked()
        p["gdal_display"] = {v: k for k, v in eng.GDAL_DISPLAY.items()}.get(w["r_gdal_display"].currentText(), "qgis")
        p["lab_output"] = self._lab_output_key()
        p["lab_weights"] = [w[f"lab_w_{k}"].value() for k, _, _ in BANDS]
        p["lab_detrend"] = w["lab_detrend"].isChecked()
        p["lab_smoothing"] = {v: k for k, v in eng.LAB_SMOOTHING.items()}.get(w["lab_smoothing"].currentText(), "none")
        p["lab_smooth_radius"] = w["lab_smooth_radius"].value()
        p["lab_edge"] = w["lab_edge"].value()
        p["lab_variant"] = {v: k for k, v in eng.GDAL_VARIANTS.items()}.get(w["lab_variant"].currentText(), "standard")
        p["lab_z"] = w["lab_z"].value()
        p["lab_alt"] = w["lab_alt"].value()
        p["lab_zt"] = w["lab_zt"].isChecked()
        p["lab_autolevel"] = w["lab_autolevel"].isChecked()
        p["lab_dirs"] = int(w["lab_dirs"].currentText())
        p["lab_pca_bands"] = w["lab_pca_bands"].isChecked()
        p["lab_display"] = LAB_DISPLAY_OF.get(w["lab_display"].currentText(), "auto")
        p["vat_variant"] = VAT_LABELS.get(w["r_vat_variant"].currentText(), "general")
        p["band_types"] = [SHADE_TYPE_OF.get(w[f"fq_shade_{k}"].currentText(), "off") for k in eng.BAND_KEYS]
        p["band_weights"] = [w[f"fq_sw_{k}"].value() for k in eng.BAND_KEYS]
        p["band_altitudes"] = [w[f"fq_salt_{k}"].value() for k in eng.BAND_KEYS]
        p["band_enabled"] = [w[f"fq_on_{k}"].isChecked() for k in eng.BAND_KEYS]
        p["band_autolevel"] = w["fq_autolevel"].isChecked()
        p["dome_lights"] = w["r_dome_lights"].value()
        p["svf_radius"] = w["r_svf_radius"].value()
        p["svf_dirs"] = w["r_svf_dirs"].value()
        p["texture_detail"] = w["r_texture_detail"].value() / 100.0
        p["texture_curve"] = "tanh" if w["r_texture_curve"].currentIndex() == 0 else "algebraic"
        p["detrend"] = w["r_detrend"].isChecked()
        p["z_factor"] = w["z_factor"].value()
        p["eq_enabled"] = w["r_eq_enabled"].isChecked()
        p["eq_sigmas"] = [w[f"fq_sigma{i}"].value() for i in range(4)]
        p["eq_gains"] = [w[f"fq_gain_{k}"].value() if w[f"fq_on_{k}"].isChecked() else 0.0 for k, _, _ in BANDS]
        p["tint"] = eng.TINTS.get(w["r_tint"].currentText(), eng.WHITE)
        p["normal_convention"] = w["r_normal_convention"].currentText()
        rot = p["azimuth"] - 315.0            # the key-light azimuth rotates the whole multi-shade rig
        p["multi_lights"] = [((az + rot) % 360.0, alt, wt, tuple(col))
                             for on, az, alt, wt, col in self._multi_rows() if on and wt > 0]
        p["multi_method"] = {v: k for k, v in eng.MULTI_METHODS.items()}.get(w["r_multi_method"].currentText(),
                                                                            "adaptive")
        if p["mode"] == "gdal" and p["gdal_display"] != "raw":
            p["gdal_limits"] = self._gdal_limits(p)
        if p["mode"] == "lab":
            p = self._with_lab_limits(p)
        if self._albedo_active(p):
            p["tint"] = eng.WHITE
        return p

    def _gdal_limits(self, p):
        """Display limits for this gdaldem hillshade - QGIS's estimated min/max, or the 2-98 % cut - from exactly
        the pixels QGIS's statistics would read (cached per surface and gdaldem settings)."""
        layer = self._surface_layer()
        if layer is None:
            return None
        ls = self._layer_state(layer)
        dx, dy = self._pixel_size(ls["info"])
        key = (ls["key"], dx, dy) + tuple(p[k] for k in ("gdal_z", "gdal_scale", "gdal_variant", "gdal_zt",
                                                         "gdal_edges", "gdal_display", "azimuth", "altitude"))
        cache = self.__dict__.setdefault("_gdal_lim_cache", {})
        if key not in cache:
            if len(cache) > 64:
                cache.clear()
            vals = []
            for heights, keep in self._gdal_samples(ls):
                z, valid = eng.fill_nodata(heights)
                dn = eng.gdal_hillshade(z, dx, dy, p)[keep]
                vals.append(dn[valid[keep]] if valid is not None else dn.ravel())
            cache[key] = eng.gdal_stretch_limits(np.concatenate(vals), p["gdal_display"])
        return cache[key]

    def _gdal_samples(self, ls, max_px=16_000_000):
        """The rows QGIS's default statistics would read from the gdaldem hillshade of this surface
        (eng.qgis_sample_rows), as full-width height strips with a row of context above and below:
        [(heights, slice of the rows to keep), ...]. Very large rasters keep every k-th strip."""
        if "gdal_samples" not in ls:
            info = ls["info"]
            width, height = info.width, info.height
            rows = eng.qgis_sample_rows(width, height)
            runs = np.split(rows, np.flatnonzero(np.diff(rows) != 1) + 1)
            step = max(1, -(-sum(len(r) + 2 for r in runs) * width // max_px))
            out = []
            for run in runs[::step]:
                a, b = max(0, int(run[0]) - 1), min(height, int(run[-1]) + 2)
                z, _ = rio.read_array(info.path, (0, a, width, b - a))
                out.append((z, slice(int(run[0]) - a, int(run[-1]) + 1 - a)))
            ls["gdal_samples"] = out
        return ls["gdal_samples"]

    # ---- Shade Lab
    def _lab_output_key(self):
        w = self._w.get("lab_output")
        return {v: k for k, v in eng.LAB_OUTPUTS.items()}.get(w.currentText(), "hillshade") if w else "hillshade"

    def _lab_params(self):
        """Render parameters of the Shade Lab (whatever the Render tab's mode is)."""
        p = self._render_params()
        return p if p["mode"] == "lab" else self._with_lab_limits(dict(p, mode="lab"))

    def _lab_ref_prep(self, p):
        """The lab prepared on the full-resolution reference crop (cached): its PCA basis, levels and explained
        variance are exactly what previews and renders use."""
        layer = self._surface_layer()
        if layer is None:
            return None
        ls = self._layer_state(layer)
        dx, dy = self._pixel_size(ls["info"])
        key = json.dumps([ls["key"], dx, dy, [p[k] for k in eng.PREPARE_KEYS]], default=str)
        cache = self.__dict__.setdefault("_lab_ref_cache", {})
        if key not in cache:
            if len(cache) > 12:
                cache.clear()
            cache[key] = eng.prepare(ls["ref"], dx, dy, dict(p, lab_limits=None), 1.0, None, ls["plane"],
                                     ls["ref_origin"])
        return cache[key]

    def _with_lab_limits(self, p):
        """Auto-contrast limits of a single-light lab output, from the whole-surface reference crop."""
        out = p["lab_output"]
        if p["lab_display"] != "auto" or not (out == "hillshade" or out.startswith("band")):
            return p
        prep = self._lab_ref_prep(p)
        if prep is None:
            return p
        key = (id(prep), p["azimuth"], p["lab_variant"])
        cache = self.__dict__.setdefault("_lab_lim_cache", {})
        if key not in cache:
            if len(cache) > 64:
                cache.clear()
            dn = eng.shade(prep, dict(p, lab_display="raw", lab_limits=None)) * 255.0
            v = dn[dn > 0]
            cache[key] = tuple(float(x) for x in np.percentile(v, [0.5, 99.5])) if v.size else None
        return dict(p, lab_limits=cache[key])

    def _lab_summary(self, p):
        out = p["lab_output"]
        stack = out in eng.LAB_STACK_OUTPUTS
        sm = {"none": "no smoothing", "gaussian": f"Gaussian smoothing {p['lab_smooth_radius']:g} px",
              "edge": f"edge-preserving smoothing {p['lab_smooth_radius']:g} px (threshold {p['lab_edge']:.2f})"}
        lines = [f"Output: {eng.LAB_OUTPUTS[out]}",
                 "Bands: " + ", ".join(f"{label} {wt:.2f}" for (_k, label, _d), wt in zip(BANDS, p["lab_weights"]))
                 + (" · auto-levelled" if p["lab_autolevel"] else ""),
                 "Surface: " + sm[p["lab_smoothing"]] + (", tilt removed" if p["lab_detrend"] else "")]
        light = "" if stack or p["lab_variant"] == "multidirectional" else f", light {p['azimuth']:.0f}°"
        lines.append(f"Hillshade: {eng.GDAL_VARIANTS['standard' if stack else p['lab_variant']].split(' (')[0].lower()}, "
                     f"{'Zevenbergen-Thorne' if p['lab_zt'] else 'Horn'}, Z {p['lab_z']:g}, altitude "
                     f"{p['lab_alt']:.0f}°{light}")
        evr_text = ""
        if stack:
            prep = self._lab_ref_prep(p)
            evr = (prep.lab.get("pca") or {}).get("evr") if prep is not None else None
            if evr:
                evr_text = "PCA explains " + " / ".join(f"{v * 100:.1f} %" for v in evr) + " (components 1 / 2 / 3)"
            lines.append(f"{p['lab_dirs']} directions round the compass" +
                         (" (multi-scale: every band, mixed by weight)" if p["lab_pca_bands"] else "") +
                         (f"; {evr_text}" if evr_text else ""))
        self.lbl_lab_evr.setText(evr_text or "PCA: choose a direction output (mean or PCA) to compute it.")
        return lines

    def _update_lab_enabled(self, en, mode):
        """Grey out the Shade Lab controls the chosen output does not use."""
        w = self._w
        out = self._lab_output_key()
        stack = out in eng.LAB_STACK_OUTPUTS
        lab_var = {v: k for k, v in eng.GDAL_VARIANTS.items()}.get(w["lab_variant"].currentText(), "standard")
        smoothing = {v: k for k, v in eng.LAB_SMOOTHING.items()}.get(w["lab_smoothing"].currentText(), "none")
        en("lab_variant", not stack)
        en("lab_dirs", stack)
        en("lab_pca_bands", stack)
        en("lab_smooth_radius", smoothing != "none")
        en("lab_edge", smoothing == "edge")
        en("lab_autolevel", out.startswith("band") or (stack and w["lab_pca_bands"].isChecked())
           or any(abs(w[f"lab_w_{k}"].value() - 1.0) > 1e-9 for k, _, _ in BANDS))
        en("lab_display", out in ("hillshade", "mean") or out.startswith("band"))
        for k, _, _ in BANDS:
            en(f"lab_w_{k}", not out.startswith("band"))
        if mode == "lab":
            en("r_azimuth", not stack and lab_var != "multidirectional")

    def _update_output_selector(self):
        """Above the preview, 'View' picks a render ingredient; with the Shade Lab it picks the lab output."""
        if not hasattr(self, "lbl_view") or "lab_output" not in self._w:
            return
        lab = self.tabs.currentIndex() == TAB_LAB or self._mode() == "lab"
        self.lbl_view.setText("Output" if lab else "View")
        self._w["pv_component"].setVisible(not lab)
        self._w["lab_output"].setVisible(lab)

    def _set_lab_weights(self, ws):
        prev, self._suspend = self._suspend, True
        try:
            for (k, _l, _d), wt in zip(BANDS, ws):
                self._set_value(self._w[f"lab_w_{k}"], float(wt))
        finally:
            self._suspend = prev
        self._update_enabled()
        self._update_overlay()
        self.canvas.flash("Band mix: " + " / ".join(f"{float(wt):g}" for wt in ws))
        self._schedule_preview(force=True)

    def _send_bands_to_lab(self):
        self._set_lab_weights([1.0 if self._w[f"fq_on_{k}"].isChecked() else 0.0 for k, _, _ in BANDS])
        self.tabs.setCurrentIndex(TAB_LAB)

    def _use_lab_as_main(self):
        self._w["r_mode"].setCurrentText(eng.RENDER_MODES["lab"])
        self.tabs.setCurrentIndex(TAB_RENDER)

    def _run_render_lab(self):
        self._render_override = self._lab_params()
        self._run_render()

    def _auto_gdal_scale(self):
        layer = self._surface_layer()
        geo = layer is not None and layer.crs().isValid() and layer.crs().isGeographic()
        self._set_value(self._w["r_gdal_scale"], 111120.0 if geo else 1.0)
        self.canvas.flash("Scale 111120 (degrees → metres)" if geo else "Scale 1 (metric raster)")

    def _albedo_active(self, p):
        return (self._w["r_albedo"].isChecked() and self._w["colour"].currentLayer() is not None
                and p["mode"] in eng.LIT_MODES + ("gis", "hillshade", "dome", "vat", "clean"))

    def _reference_stats(self, p, ls, want_tex, want_svf):
        """Normalisation constants from a full-resolution centre crop, shared by
        the preview and the final render so both get the same tones."""
        dx, dy = self._pixel_size(ls["info"])
        npath, nref = self._normals_ref(ls) if p.get("clean_source") == "normals" else (None, None)
        return self.renderer.stats(ls["key"], ls["ref"], ls["ref_origin"], ls["plane"], dx, dy, p,
                                   want_tex, want_svf, eng.needs_curvature(p), nref, npath)

    def _normals_green(self):
        text = self._w["normals_green"].currentText()
        if text.startswith("up"):
            return "up"
        if text.startswith("down"):
            return "down"
        return getattr(self, "_green_auto", None) or "up"

    def _normals_ref(self, ls):
        """(path, the normal map over the height map's reference crop) - or (None, None) without a normal map."""
        lyr = self._w["normals"].currentLayer()
        if lyr is None or ls is None:
            return None, None
        path = rio.gdal_path(lyr.source())
        cache = self.__dict__.setdefault("_nref_cache", {})
        key = (path, ls["key"])
        if key not in cache:
            cache.clear()
            h, w = ls["ref"].shape
            cache[key] = rio.read_normals(path, (ls["info"].width, ls["info"].height),
                                          (ls["ref_origin"][0], ls["ref_origin"][1], w, h))
        return path, cache[key]

    def _update_normals_state(self):
        """Detect the normal map's green channel against the height map and say how well they agree."""
        self._green_auto, text = None, ""
        lyr, layer = self._w["normals"].currentLayer(), self._surface_layer()
        if lyr is not None and layer is not None:
            try:
                ls = self._layer_state(layer)
                ni, li = rio.RasterInfo(rio.gdal_path(lyr.source())), ls["info"]
                if (ni.width, ni.height) != (li.width, li.height):
                    text = (f"⚠ {ni.width:,} × {ni.height:,} px, the height map {li.width:,} × {li.height:,}: not on "
                            f"its grid - use the normal map that came with this height map.")
                else:
                    _, nref = self._normals_ref(ls)
                    green, agree = eng.detect_normals_convention(nref, eng.fill_nodata(ls["ref"])[0])
                    self._green_auto = green
                    text = (f"Green channel points {'up (OpenGL)' if green == 'up' else 'down (DirectX)'} - it agrees "
                            f"with the height map's slopes ({agree:.2f})"
                            + ("." if agree > 0.3 else " only weakly: is it the same object?"))
            except Exception as exc:
                text = f"Cannot read the normal map: {exc}"
        self.lbl_normals.setText(text)
        self.lbl_normals.setVisible(bool(text))

    def _helio_scale(self, info):
        """For a HELIO export, None otherwise: dict(unit, per_unit, z_per_unit, estimated). HELIO writes its GIS
        rasters with a pixel of 1.0 and the heights in the same pixel-units (so slopes shade right), and the real
        size in its HELIO_XY_RES_MM tag and README: one unit is per_unit mm across, and per_unit mm tall only when
        Z was calibrated (z_per_unit; otherwise heights are relative)."""
        cache = self.__dict__.setdefault("_helio_scale_cache", {})
        if info.path not in cache:
            md, out = info.metadata, None
            readme = os.path.join(os.path.dirname(info.path), "README_GIS.txt")
            from . import helio
            r = helio.parse_readme(helio._text(readme)) if os.path.exists(readme) else {}
            try:
                per = float(md["HELIO_XY_RES_MM"]) if "HELIO_XY_RES_MM" in md else None
            except ValueError:
                per = None
            if per is None and r.get("xy_size"):
                per = r["xy_size"] * {"µm": 0.001, "mm": 1.0, "cm": 10.0, "m": 1000.0}.get(r.get("xy_unit"), 1.0)
            if per and per > 0:
                z_cal = md.get("HELIO_Z_CALIBRATED", "yes" if r.get("z_calibrated") else "no") == "yes"
                out = dict(unit="mm", per_unit=per, z_per_unit=per if z_cal else None,
                           estimated=bool(r.get("xy_estimated")))
            cache[info.path] = out
        return cache[info.path]

    def _helio_unit(self, info):
        """'mm' for a HELIO export, whose placeholder CRS says metres."""
        hs = self._helio_scale(info)
        return hs["unit"] if hs else None

    def _real_pixel(self, info):
        """The pixel size in the unit _unit_label names: _pixel_size, turned into mm for a HELIO export. Lengths,
        scale bars and captions use this; shading keeps _pixel_size, which matches the heights' units."""
        dx, dy = self._pixel_size(info)
        hs = self._helio_scale(info) if self._w["pixel_auto"].isChecked() else None
        return (dx * hs["per_unit"], dy * hs["per_unit"]) if hs else (dx, dy)

    def _real_height(self, info):
        """What one height unit is in the measuring unit (None: relative heights, not a measurement)."""
        hs = self._helio_scale(info) if self._w["pixel_auto"].isChecked() else None
        if hs is None:
            return 1.0
        return hs["z_per_unit"]

    # ------------------------------------------------------------------ preview
    def _schedule_preview(self, force=False):
        if not self._suspend:
            self._note_change()                  # looks, presets and resets arrive here too
        if self._suspend or (not force and not self._w["pv_live"].isChecked()):
            return
        self._want_overview = True
        self._timer.start()

    def _on_view_changed(self):
        self._update_zoom_label()
        self._update_overlay()
        self._schedule_lines()
        if not self._suspend and self._w["pv_live"].isChecked():
            self._view_timer.start()

    def _preview_base(self, ls):
        """Everything the worker needs, as plain data (no widgets cross threads)."""
        info = ls["info"]
        dx, dy = self._pixel_size(info)
        tab = self.tabs.currentIndex()
        p = self._lab_params() if tab == TAB_LAB else self._render_params()
        kind = {TAB_FREQUENCY: "frequency", TAB_STACK: "stack",
                TAB_DETAIL: "separate"}.get(tab, "render")
        colour = self._w["colour"].currentLayer()
        req = dict(path=info.path, layer_key=ls["key"], full=(info.width, info.height), dx=dx, dy=dy,
                   plane=ls["plane"], ref=ls["ref"], ref_origin=ls["ref_origin"], params=p, tab=kind,
                   compare={"Colour / albedo": "colour", "Plain hillshade": "hillshade",
                            "Height map": "height"}.get(self._w["pv_compare"].currentText()),
                   component=self._component() if (kind == "render" and p["mode"] != "lab") else None,
                   colour=rio.gdal_path(colour.source()) if colour else None,
                   albedo=kind == "render" and self._albedo_active(p),
                   albedo_strength=self._w["r_albedo_strength"].value())
        if p.get("clean_source") == "normals":
            req["normals"], req["ref_normals"] = self._normals_ref(ls)
        if kind == "separate":
            req["detail"] = self._detail_params()
        if kind == "frequency":
            text = self._w["fq_preview_band"].currentText()
            if text == FQ_COMBO or text.endswith("(shaded)"):
                i = next((j for j, (_, label, _) in enumerate(BANDS) if text == f"{label} band (shaded)"), None)
                req.update(tab="render", params=dict(p, mode="bands"), component=None if i is None else f"band{i}")
            else:
                band = 0 if text == FQ_RECOMBINED else 1 + next(
                    j for j, (_, label, _) in enumerate(BANDS) if text == f"{label} band (signed ramp)")
                ramp = self._w[f"fq_ramp_{BANDS[band - 1][0]}"].currentText() if band else None
                req.update(split=self._split_settings(), band=band, ramp_stops=COLOR_RAMP_PRESETS.get(ramp))
        elif kind == "stack":
            rows = self._table_rows()
            _, az, alt = rows[min(max(self.table.currentRow(), 0), len(rows) - 1)] if rows else ("", 315.0, 35.0)
            req["stack"] = dict(azimuth=az, altitude=alt, tone=self._w["ls_tone"].currentIndex(),
                                lift=self._w["ls_lift"].value() / 100.0,
                                raw=self._w["ls_source"].currentIndex() == 1)
        return req

    def _request_previews(self):
        """Queue an overview (after setting changes) and a detail render of the
        current view at up to 1:1 full-resolution pixels."""
        layer, view = self._surface_layer(), self.canvas.viewWindow()
        if layer is None or view is None or not (self.isVisible() or self._run_sync):
            return
        try:
            ls = self._layer_state(layer)
        except Exception as exc:
            self.canvas.setMessage(f"Cannot read the height map: {exc}")
            return
        self.renderer.sync = self._run_sync
        self._update_overlay()
        base = self._preview_base(ls)
        W, H = base["full"]
        self._gen += 1
        ov_pxs = max(1.0, max(W, H) / float(OVERVIEW_DIM))
        pxs = max(1.0, 1.0 / (self.canvas.scale() * self.canvas.devicePixelRatioF()))
        n = view[2] * view[3] / (pxs * pxs)
        if n > DETAIL_MAX_PIXELS:
            pxs *= math.sqrt(n / DETAIL_MAX_PIXELS)
        want_detail = pxs < ov_pxs / 1.15      # otherwise the overview is already sharp enough
        if not want_detail:
            self._shown_gen["detail"] = self._gen
            self.canvas.setLayer("detail", None)
        target_px = (view[2] * view[3] / (pxs * pxs)) if want_detail else (W * H / (ov_pxs * ov_pxs))
        if self._want_overview and target_px > 250_000 and not self._run_sync:
            # A half-resolution draft of what is on screen arrives first, then the sharp render replaces it.
            self.renderer.submit(dict(base, kind="draft", gen=self._gen,
                                      view=view if want_detail else (0, 0, W, H),
                                      px_scale=(pxs if want_detail else ov_pxs) * 2.0))
        if self._want_overview or not self.canvas.hasLayer("overview"):
            self._want_overview = False
            self.renderer.submit(dict(base, kind="overview", gen=self._gen, view=(0, 0, W, H), px_scale=ov_pxs))
        if want_detail:
            self.renderer.submit(dict(base, kind="detail", gen=self._gen, view=view, px_scale=pxs))
        self._strip_timer.start()

    def _on_preview_ready(self, res):
        kind = res["kind"]
        if kind == "lines":
            self._lines_done(res)
            return
        if kind == "strip":
            if res["gen"] >= self._strip_gen_shown:
                self._strip_gen_shown = res["gen"]
                self.strip.setItems(res["items"], self._strip_current)
                self.strip.setStatus("")
            return
        if kind == "draft":
            if res["gen"] <= self._shown_gen.get("detail", -1) or res["gen"] < self._shown_gen.get("draft", -1):
                return
            self._shown_gen["draft"] = res["gen"]
            self.canvas.setLayer("draft", res["img"], res["rect"], res["cmp"], res["probe"], res["px_scale"])
            self.canvas.setInfo(f"draft \u00b7 {res['ms']:.0f} ms \u00b7 sharpening\u2026")
            return
        if res["gen"] < self._shown_gen.get(kind, -1):
            return                              # superseded while it was rendering
        self._shown_gen[kind] = res["gen"]
        self.canvas.setLayer(kind, res["img"], res["rect"], res["cmp"], res["probe"], res["px_scale"])
        if res.get("spectrum") is not None and (kind == "detail" or not self.canvas.hasLayer("detail")):
            sizes, power, model, _crop = res["spectrum"]
            self._detail_model = model
            self.spectrum_view.setSpectrum(sizes, power, model)
            self._update_detail_chart()
        if (kind == "detail" or not self.canvas.hasLayer("detail")) and res["gen"] >= self._shown_gen.get("draft", -1):
            self.canvas.setLayer("draft", None)
        if kind == "detail" or not self.canvas.hasLayer("detail"):
            h, w = res["img"].shape[:2]
            px = res["px_scale"]
            detail = "full resolution" if px <= 1.001 else f"1/{px:.2g} resolution"
            self.canvas.setInfo(f"{w}×{h} · {detail} · {res['ms']:.0f} ms · "
                                f"clipped {res['clip']:.1f} %")
        self._last_preview_error = None

    def _on_preview_failed(self, text):
        last = text.strip().splitlines()[-1] if text.strip() else "unknown error"
        self.canvas.setInfo("Preview failed: " + last[:140])
        if text != self._last_preview_error:
            self._log_line("Preview failed:\n" + text)
        self._last_preview_error = text

    def _update_zoom_label(self):
        s = self.canvas.scale()
        ratio = f"{s:.3g}:1" if s >= 1 else f"1:{1 / s:.3g}"
        self.lbl_zoom.setText(f"{ratio}  ({s * 100:.0f} %)")

    def _unit_label(self, info):
        unit = self._w["pixel_unit"].currentText()
        if unit != "auto":
            return unit
        auto = self._w["pixel_auto"].isChecked()
        layer = self._surface_layer()
        if auto and self._helio_unit(info):
            return self._helio_unit(info)
        if auto and info.georeferenced and layer is not None:
            try:
                return QgsUnitTypes.toAbbreviatedString(layer.crs().mapUnits()) or "units"
            except Exception:
                return "units"
        return "px" if auto else "units"

    def _update_scale_bar(self):
        if self._ls is None:
            self.canvas.setPixelSize(None, "")
            return
        info = self._ls["info"]
        self.canvas.setPixelSize(self._real_pixel(info)[0], self._unit_label(info))

    def _view_from_map(self):
        layer = self._surface_layer()
        if layer is None or self._ls is None:
            return
        win = self._canvas_window(layer, self._ls["info"])
        if win:
            self.canvas.showWindow(win)
        elif self.iface is None:
            self.status.setText("Map sync needs the QGIS map canvas")

    def _view_to_map(self):
        layer, view = self._surface_layer(), self.canvas.viewWindow()
        if self.iface is None or layer is None or view is None or self._ls is None:
            return
        info, lext = self._ls["info"], layer.extent()
        x0 = lext.xMinimum() + view[0] / float(info.width) * lext.width()
        x1 = lext.xMinimum() + (view[0] + view[2]) / float(info.width) * lext.width()
        y1 = lext.yMaximum() - view[1] / float(info.height) * lext.height()
        y0 = lext.yMaximum() - (view[1] + view[3]) / float(info.height) * lext.height()
        rect = QgsRectangle(x0, y0, x1, y1)
        canvas = self.iface.mapCanvas()
        dest = canvas.mapSettings().destinationCrs()
        if dest.isValid() and layer.crs().isValid() and dest != layer.crs():
            rect = QgsCoordinateTransform(layer.crs(), dest, QgsProject.instance()).transformBoundingBox(rect)
        self._to_map_busy = True
        try:
            canvas.setExtent(rect)
            canvas.refresh()
        finally:
            self._to_map_busy = False

    def _save_view(self):
        img = self.canvas.currentImage()
        if img is None:
            return
        start = os.path.join(self._output_dir(False), "preview_view.png")
        path, _ = QFileDialog.getSaveFileName(self, "Save preview image", start,
                                              "PNG image (*.png);;JPEG image (*.jpg);;TIFF image (*.tif)")
        if path:
            if img.save(path):
                with open(os.path.splitext(path)[0] + "_settings.txt", "w", encoding="utf-8") as f:
                    f.write("\n".join(self._settings_summary()) + f"\nView window (full-res px): "
                            f"{self.canvas.viewWindow()}\nZoom: {self.lbl_zoom.text()}\n\n"
                            + json.dumps(self._render_params(), indent=1, default=str))
                self._log_line(f"Preview image saved: {path} (+ _settings.txt)")
            else:
                QMessageBox.warning(self, "Save view", f"Could not write {path}")

    def _on_table_edit(self, *_args):
        self._update_dial_markers()
        self._update_overlay()
        if self.tabs.currentIndex() in (TAB_STACK,):
            self._schedule_preview()

    def _run_current_tab(self):
        action = {TAB_RENDER: self._run_render, TAB_FREQUENCY: self._run_split, TAB_STACK: self._run_stack,
                  TAB_DETAIL: lambda: self._run_detail_save("band")}.get(self.tabs.currentIndex())
        if action and self._task is None and self._surface_layer() is not None:
            action()

    def _memory_ok(self, info, arrays, what):
        """Warn before a job that holds whole-surface arrays could exhaust RAM."""
        need = info.width * info.height * 4.0 * arrays
        ram = _physical_ram()
        if ram is None or need < 0.6 * ram:
            return True
        return QMessageBox.question(
            self, "Large raster", f"{what} keeps the whole surface in memory: roughly {need / 1e9:.1f} GB of your "
            f"{ram / 1e9:.0f} GB RAM. QGIS may slow down or run out of memory.\n\nContinue?") == \
            QMessageBox.StandardButton.Yes

    def _restore_geometry(self):
        settings = QgsSettings()
        try:
            geo, split = settings.value("HelioRelief/geometry"), settings.value("HelioRelief/splitter")
            if geo:
                self.restoreGeometry(geo)
            if split:
                self.splitter.restoreState(split)
        except TypeError:
            pass

    def _save_geometry(self):
        settings = QgsSettings()
        settings.setValue("HelioRelief/geometry", self.saveGeometry())
        settings.setValue("HelioRelief/splitter", self.splitter.saveState())

    # ------------------------------------------------------------------ presets & helpers
    def _set_look_widgets(self, name):
        """Put a look's values into the widgets (the caller suspends signals)."""
        user = name.startswith(USER_LOOK_PREFIX)
        data = _read_json(LOOKS_FILE).get(name[len(USER_LOOK_PREFIX):], {}) if user else LOOKS.get(name)
        if not data:
            return False
        keep = ("r_detrend", "r_flatten_sigma", "r_eq_enabled", "r_albedo", "r_albedo_strength", "r_clean_flatten")
        if user:
            for k, v in data.items():
                if k in self._w:
                    self._set_value(self._w[k], v)
            if data.get("__multi_lights__"):
                self._set_multi_rows(data["__multi_lights__"])
            return True
        # Built-in looks start from the defaults so no stale values linger;
        # surface preparation (flatten, detrend, EQ) is kept.
        for k, v in self._defaults.items():
            if ((k.startswith("r_") and not k.startswith("r_out")) or k.startswith("lab_")) and k not in keep \
                    and k in self._w:
                self._set_value(self._w[k], v)
        for k, v in data.items():
            if k == "mode":
                self._w["r_mode"].setCurrentText(eng.RENDER_MODES[v])
            elif k == "tint":
                self._w["r_tint"].setCurrentText(v)
            elif k == "texture_detail":
                self._w["r_texture_detail"].setValue(v * 100)
            elif k == "vat_variant":
                self._w["r_vat_variant"].setCurrentText(VAT_LABEL_OF[v])
            elif k == "gis_blend":
                self._w["r_gis_blend"].setCurrentText(eng.GIS_BLENDS[v])
            elif k == "gdal_variant":
                self._w["r_gdal_variant"].setCurrentText(eng.GDAL_VARIANTS[v])
            elif k == "gdal_display":
                self._w["r_gdal_display"].setCurrentText(eng.GDAL_DISPLAY[v])
            elif k == "lab_weights":
                for (bk, _l, _d), wt in zip(BANDS, v):
                    self._set_value(self._w[f"lab_w_{bk}"], float(wt))
            elif k in ("lab_output", "lab_variant", "lab_smoothing", "lab_display", "lab_dirs"):
                table = {"lab_output": eng.LAB_OUTPUTS, "lab_variant": eng.GDAL_VARIANTS,
                         "lab_smoothing": eng.LAB_SMOOTHING,
                         "lab_display": {v2: k2 for k2, v2 in LAB_DISPLAY_OF.items()}}.get(k)
                self._w[k].setCurrentText(table[v] if table else str(v))
            elif k.startswith("lab_") and k in self._w:
                self._set_value(self._w[k], v)
            elif k == "multi_preset":
                self._apply_multi_preset(v)
            elif k == "band_preset":
                self._apply_band_preset(v)
            elif "r_" + k in self._w:
                self._set_value(self._w["r_" + k], v)
        return True

    def _apply_look(self, name=None):
        name = name or self.look_combo.currentText()
        prev, self._suspend = self._suspend, True
        try:
            ok = self._set_look_widgets(name)
        finally:
            self._suspend = prev
        if not ok:
            return
        self.canvas.snapshotPrevious()
        i = self.look_combo.findText(name)
        if i >= 0:
            self.look_combo.setCurrentIndex(i)
        self._look_name, self._look_dirty = name, False
        self._update_mode_visibility()
        self._sync_dial()
        self._update_overlay()
        self.canvas.flash(f"Look: {name}")
        self._schedule_preview(force=True)

    def _look_params(self, name):
        """Render parameters a look would give, without changing the dialog."""
        saved = self.state(include_layers=False)
        prev, self._suspend = self._suspend, True
        try:
            self._set_look_widgets(name)
            return self._render_params()
        finally:
            self.set_state(saved)
            self._suspend = prev

    def _open_gallery(self):
        """Every look rendered on the part of the surface in view; click one to use it."""
        layer = self._surface_layer()
        if layer is None:
            return
        ls = self._layer_state(layer)
        info = ls["info"]
        view = self.canvas.viewWindow() or (0, 0, info.width, info.height)
        _, out, pxs = info.window_fit(230, view)
        z, _ = rio.read_array(info.path, view, out)
        dx, dy = self._pixel_size(info)
        dx, dy = dx * pxs, dy * view[3] / float(out[1])
        names = list(LOOKS) + [USER_LOOK_PREFIX + n for n in sorted(_read_json(LOOKS_FILE))]
        dlg = QDialog(self)
        dlg.setWindowTitle("Look gallery - every look on your surface")
        grid = QGridLayout(dlg)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            for i, name in enumerate(names):
                p = self._fast_look_params(name)
                lit = p["mode"] in eng.LIT_MODES
                st = self._reference_stats(p, ls, lit or p["mode"] == "texture", eng.needs_svf(p))
                img, _ = eng.render(z, dx, dy, p, pxs, st, ls["plane"], (view[0], view[1]))
                q, _buf = _qimage(img)
                btn = QToolButton()
                btn.setIcon(QIcon(QPixmap.fromImage(q.copy())))
                btn.setIconSize(q.size())
                btn.setText(name.replace("&", "&&"))       # a single & would become a keyboard mnemonic
                btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
                btn.setAutoRaise(True)
                btn.clicked.connect(lambda _c=False, n=name: (self._apply_look(n), dlg.accept()))
                grid.addWidget(btn, i // 4, i % 4)
        finally:
            QApplication.restoreOverrideCursor()
        self._gallery = dlg
        dlg.show()

    def _collect_labels(self):
        """Row label of every registered control, for the change pop-up."""
        labels = {}
        for sec in self.findChildren(Section):
            for k, wdg in self._w.items():
                lab = sec.form.labelForField(wdg)
                if lab is not None and hasattr(lab, "text") and lab.text():
                    labels[k] = lab.text()
        for k, wdg in self._w.items():
            if isinstance(wdg, QCheckBox):
                labels.setdefault(k, wdg.text())
        labels["z_factor"] = "Z-factor"
        return labels

    @staticmethod
    def _fmt_value(w):
        if isinstance(w, SliderSpin):
            return w.spin.text()
        if isinstance(w, (QDoubleSpinBox, QSpinBox)):
            return w.text()
        if isinstance(w, QCheckBox):
            return "on" if w.isChecked() else "off"
        if isinstance(w, QComboBox):
            return w.currentText()
        return ""

    def _reset_section(self, sec):
        keys = [k for k, w in self._w.items()
                if sec.isAncestorOf(w) and k in self._defaults and k not in self._layer_keys]
        prev, self._suspend = self._suspend, True
        try:
            for k in keys:
                self._set_value(self._w[k], self._defaults[k])
            if sec is self._rs.get("multi"):
                self._apply_multi_preset(next(iter(eng.MULTI_PRESETS)))
            if sec.isAncestorOf(self.table):
                self._regen_table()
        finally:
            self._suspend = prev
        self._look_dirty = True
        self._update_mode_visibility()
        self._update_source_visibility()
        self._update_overlay()
        self.canvas.flash(f"{sec.toggle.text().replace('&&', '&')}: reset")
        self._schedule_preview(force=True)

    # ---- multi-shade light table
    def _add_multi_row(self, on=True, az=315.0, alt=45.0, weight=1.0, colour=(1.0, 1.0, 1.0)):
        t = self.multi_table
        t.blockSignals(True)          # no itemChanged until the whole row exists
        try:
            self._build_multi_row(t, on, az, alt, weight, colour)
        finally:
            t.blockSignals(False)

    def _build_multi_row(self, t, on, az, alt, weight, colour):
        i = t.rowCount()
        t.insertRow(i)
        chk = QTableWidgetItem()
        chk.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        chk.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
        t.setItem(i, 0, chk)
        for col, (lo, hi, val, dec) in enumerate(((0, 360, az, 1), (0, 90, alt, 1), (0, 5, weight, 2)), start=1):
            sb = QDoubleSpinBox()
            sb.setRange(lo, hi)
            sb.setDecimals(dec)
            sb.setSingleStep(1.0 if col < 3 else 0.1)
            sb.setWrapping(col == 1)
            sb.setValue(val % 360.0 if col == 1 else val)
            sb.setFrame(False)
            sb.setKeyboardTracking(False)
            sb.valueChanged.connect(self._on_multi_edit)
            t.setCellWidget(i, col, sb)
        btn = QgsColorButton()
        btn.setAllowOpacity(False)
        btn.setShowNoColor(False)
        btn.setColor(QColor.fromRgbF(*[float(c) for c in colour]))
        btn.colorChanged.connect(self._on_multi_edit)
        t.setCellWidget(i, 4, btn)

    def _set_multi_rows(self, rows):
        prev, self._suspend = self._suspend, True
        try:
            self.multi_table.setRowCount(0)
            for on, az, alt, wt, col in rows:
                self._add_multi_row(bool(on), float(az), float(alt), float(wt), tuple(col))
        finally:
            self._suspend = prev

    def _multi_rows(self):
        t, rows = self.multi_table, []
        for i in range(t.rowCount()):
            item, cells = t.item(i, 0), [t.cellWidget(i, c) for c in (1, 2, 3, 4)]
            if item is None or any(c is None for c in cells):
                continue                  # row still being built
            on = item.checkState() == Qt.CheckState.Checked
            az, alt, wt = (c.value() for c in cells[:3])
            c = cells[3].color()
            rows.append((on, az, alt, wt, (round(c.redF(), 4), round(c.greenF(), 4), round(c.blueF(), 4))))
        return rows

    def _on_multi_edit(self, *_args):
        if self._suspend:
            return
        self._look_dirty = True
        self._update_dial_markers()
        self._update_overlay()
        self._schedule_preview()

    def _apply_multi_preset(self, name=None):
        name = name or self.multi_preset.currentText()
        if name not in eng.MULTI_PRESETS:
            return
        method, lights = eng.MULTI_PRESETS[name]
        prev, self._suspend = self._suspend, True
        try:
            self.multi_preset.setCurrentText(name)
            self._w["r_multi_method"].setCurrentText(eng.MULTI_METHODS[method])
            self._set_multi_rows([(True, az, alt, wt, col) for az, alt, wt, col in lights])
        finally:
            self._suspend = prev
        self._on_multi_edit()

    def _remove_multi_row(self):
        t = self.multi_table
        rows = sorted({i.row() for i in t.selectedIndexes()}, reverse=True) or ([t.rowCount() - 1] if t.rowCount() else [])
        if t.rowCount() - len(rows) < 1:
            return                        # keep at least one light
        for r in rows:
            t.removeRow(r)
        self._on_multi_edit()

    # ---- what the preview shows
    def _settings_summary(self):
        """Plain-language list of the settings that shape what the preview shows."""
        w, tab = self._w, self.tabs.currentIndex()
        if self._surface_layer() is None:
            return []
        live = "" if w["pv_live"].isChecked() else "   [Live is off - press Refresh]"
        if tab == TAB_DETAIL:
            from . import detail
            d = self._detail_params()
            lines = [f"Detail · {detail.DISPLAYS[d['display']]}{live}"]
            for i, b in enumerate(d["bands"]):
                if b["on"]:
                    lines.append(f"Band {i + 1}: " + (f"{b['fine']:g}" if b["fine"] > 0 else "finest") + " to "
                                 + (f"{b['broad']:g} px" if b["broad"] > 0 else "all sizes") + f", gain {b['gain']:g}")
            if not any(b["on"] for b in d["bands"]):
                lines.append("No band is on - tick one")
            sep = [f"{label} {d[k]:g}" for k, label in (("wiener", "noise suppression"), ("gate", "texture gate"),
                                                       ("clean", "clean-up px")) if d[k] > 0]
            lines.append("Separation: " + (", ".join(sep) if sep else "off - the bands as they are"))
            m = self._detail_model
            if m:
                lines.append(f"Noise here: outweighs relief below ≈ {m['crossover']:.1f} px" if m.get("detected")
                             else "Noise here: none resolvable - the finest detail is already smooth")
            return lines
        if tab == TAB_FREQUENCY:
            s = self._split_settings()
            lines = [f"Frequency tab · preview: {w['fq_preview_band'].currentText()}{live}",
                     "Crossovers (sigma): " + " / ".join(f"{v:g}" for v in s["sigmas"]) + " px",
                     "Gains: " + "  ".join(f"{label} {s['gains'][k]:.2f}" + ("" if k in s["enabled"] else " (off)")
                                           for k, label, _ in BANDS)]
            cond = []
            if s["median"] >= 3:
                cond.append(f"median {s['median']} px")
            if s["preblur"] > 0:
                cond.append(f"pre-blur {s['preblur']:g} px")
            if s["detrend"]:
                cond.append("plane removed")
            if s["denoise"] > 0:
                cond.append(f"ultra-high denoise {s['denoise']:.0f} %")
            if cond:
                lines.append("Conditioning: " + ", ".join(cond))
            treat = []
            for k, label, _ in BANDS:
                t = s["treat"][k]
                if t["mode"] != "None" or t["threshold"] > 0:
                    treat.append(f"{label}: {t['mode'].lower()} {t['radius']:g} px"
                                 + (f", threshold {t['threshold']:g}" if t["threshold"] > 0 else ""))
            if treat:
                lines.append("Treatments: " + "; ".join(treat))
            lines.append(self._band_summary(self._render_params()))
            lines.append("Render-tab equaliser: " + ("on" if w["r_eq_enabled"].isChecked() else "off"))
            return lines
        if tab == TAB_LAB:
            return [f"Shade Lab{live}"] + self._lab_summary(self._lab_params())
        if tab in (TAB_STACK,):
            rows = self._table_rows()
            i = min(max(self.table.currentRow(), 0), len(rows) - 1) if rows else 0
            name, az, alt = rows[i] if rows else ("-", 0.0, 0.0)
            tone = w["ls_tone"].currentText() + (f" {w['ls_lift'].value()} %" if w["ls_tone"].currentIndex() == 1 else "")
            lines = [f"Light Stack · previewing light {i + 1} of {len(rows)}: {name} "
                     f"({az:.0f}° / {alt:.0f}°){live}",
                     f"Source: {w['ls_source'].currentText()}", f"Per-light tone: {tone}"]
            outs = [label for key, label in (("ls_multiband", "multi-band"), ("ls_individual", "individual lights"),
                                             ("ls_vrt", "VRT"), ("ls_mean", "mean composite"),
                                             ("ls_svf", "sky-view factor"), ("ls_normals", "normal map"),
                                             ("ls_tpi", "TPI"), ("ls_pca", "PCA")) if w[key].isChecked()]
            lines.append("Written when you run it: " + (", ".join(outs) or "nothing ticked"))
            return lines

        p = self._render_params()
        mode, lit = p["mode"], p["mode"] in ("studio", "multi")
        look = self._look_name or "custom settings"
        if self._look_name and self._look_dirty:
            look += " (modified)"
        lines = [f"Render · {eng.RENDER_MODES[mode]} · {look}{live}"]
        surf = [f"Z ×{p['z_factor']:.4g}"]
        if p["detrend"]:
            surf.append("tilt removed")
        if p["flatten_sigma"] > 0:
            surf.append(f"flattened > {p['flatten_sigma']:.0f} px")
        if p["eq_enabled"]:
            surf.append("EQ " + " / ".join(f"{g:.2g}" for g in p["eq_gains"]))
        if p["detail_boost"] > 0:
            surf.append(f"detail boost {p['detail_boost']:.2f} below {p['detail_radius']:g} px")
        if p["smooth_sigma"] > 0:
            surf.append(f"smoothed {p['smooth_sigma']:g} px")
        if mode == "clean":
            cs = [f"drift broader than {p['clean_flatten']:.0f} px removed" if p["clean_flatten"] > 0
                  else "tilt removed"]
            if p["clean_smooth"] > 0:
                cs.append(f"edge-preserving smoothing {p['clean_smooth']:g} px (edges above {p['clean_edge']:.2f} "
                          f"kept)")
            if p["clean_fine"] > 0:
                cs.append(f"fine detail +{p['clean_fine']:.2f}")
            lines.append("Surface: " + ", ".join(cs))
        elif mode != "lab":
            lines.append("Surface: raw heights, exactly as gdaldem reads them" if mode == "gdal"
                         else "Surface: " + ", ".join(surf))
        if mode == "studio":
            light = f"key {p['azimuth']:.0f}° / {p['altitude']:.0f}° ×{p['key']:.2f}"
            light += (f", fill {p['fill']:.2f} at +{p['fill_offset']:.0f}° / {p['fill_altitude']:.0f}°"
                      if p["fill"] > 0 else ", no fill")
            light += f", ambient {p['ambient']:.2f}" + (f", soft wrap {p['wrap']:.2f}" if p["wrap"] > 0 else "")
            lines.append("Light: " + light)
            lines.append(f"Shadows {p['shadows']:.2f}, softness {p['shadow_softness']:.1f}°, "
                         f"reach {p['shadow_reach']} px" if p["shadows"] > 0 else "Shadows: off")
        elif mode == "multi":
            lines.append(f"Multi-shade: {len(p['multi_lights'])} lights, {eng.MULTI_METHODS[p['multi_method']]}, "
                         f"rig {p['azimuth'] - 315.0:+.0f}°, ambient {p['ambient']:.2f}"
                         + (f", soft wrap {p['wrap']:.2f}" if p["wrap"] > 0 else ""))
        elif mode == "hillshade":
            lines.append(f"Light {p['azimuth']:.0f}° / {p['altitude']:.0f}°")
        elif mode == "clean":
            a = p["clean_allround"]
            key = f"key {p['azimuth']:.0f}° / {p['altitude']:.0f}°"
            ring = f"{p['clean_lights']} lights round the compass at {p['clean_ring_alt']:.0f}°"
            light = key if a <= 0 else (ring if a >= 1 else f"{key} + {a * 100:.0f} % from {ring}")
            lines.append(f"Light: {light}; steepest 5 % shaded as {p['clean_relief']:.0f}°")
            lines.append(f"Tone: flat grey {p['clean_mid']:.2f}, contrast {p['clean_contrast']:.2f}, highlights "
                         f"×{p['clean_highlight']:.2f} (soft curve, never black or white)")
            if self._albedo_active(p):
                lines.append(f"Colour composite: {w['colour'].currentLayer().name()} × relief shading - strength "
                             f"{w['r_albedo_strength'].value():.2f}, relief {p['albedo_relief']:.2f}, gamma "
                             f"{p['albedo_gamma']:.2f}, exposure {p['albedo_exposure']:.2f}")
        elif mode == "lab":
            lines.extend(self._lab_summary(p))
        elif mode == "gdal":
            v = p["gdal_variant"]
            light = ("four lights at 225/270/315/360°" if v == "multidirectional" else f"light {p['azimuth']:.0f}°")
            light += "" if v == "igor" else f" at {p['altitude']:.0f}°"
            lines.append(f"gdaldem hillshade: {eng.GDAL_VARIANTS[v].split(' (')[0].lower()}, "
                         f"{'Zevenbergen-Thorne' if p['gdal_zt'] else 'Horn'}, {light}, Z {p['gdal_z']:g}, "
                         f"scale {p['gdal_scale']:g}" + (", edges computed" if p["gdal_edges"] else ""))
            lim = p.get("gdal_limits")
            how = {"qgis": "as QGIS shows it (estimated min/max)", "cut": "2-98 % cumulative cut"}.get(p["gdal_display"])
            lines.append(f"Display: {how}, grey levels {lim[0]:.0f}-{lim[1]:.0f} → black-white" if how and lim
                         else "Display: gdaldem's own grey levels 1-255 (no stretch)")
            cmd = ["gdaldem hillshade", "<height map>", "<output>.tif", "-of GTiff -b 1",
                   f"-z {p['gdal_z']:g}", f"-s {p['gdal_scale']:g}"]
            if v != "multidirectional":
                cmd.append(f"-az {p['azimuth']:g}")
            if v != "igor":
                cmd.append(f"-alt {p['altitude']:g}")
            cmd += (["-compute_edges"] if p["gdal_edges"] else []) + (["-alg ZevenbergenThorne"] if p["gdal_zt"] else [])
            cmd += {"combined": ["-combined"], "multidirectional": ["-multidirectional"], "igor": ["-igor"]}.get(v, [])
            self.lbl_gdal_cmd.setText("Same as:  " + " ".join(cmd))
            lines.append("Same as: " + " ".join(cmd))
        elif mode == "gis":
            n = p["gis_lights"]
            lights = (f"{n} lights from {p['azimuth']:.0f}°, {eng.GIS_BLENDS[p['gis_blend']].split(' (')[0].lower()}"
                      if n > 1 else f"light {p['azimuth']:.0f}°")
            lines.append(f"GIS hill shading: {lights} at {p['altitude']:.0f}°, ambient {p['gis_ambient']:.2f}, "
                         f"shadow cap {p['gis_shadow']:.2f}, highlight {p['gis_highlight']:.2f}, "
                         f"exaggeration ×{p['gis_ve']:.2f}, brightness {p['gis_brightness']:.2f}"
                         + (f", slope-enhanced {p['gis_slope']:.2f} below {p['gis_slope_range']:g}°"
                            if p["gis_slope"] > 0 else ""))
        elif mode == "specular":
            lines.append(f"Specular enhancement: light {p['azimuth']:.0f}° / {p['altitude']:.0f}°, normal depth "
                         f"×{p['spec_depth']:.2f}, gloss {p['shininess']:.0f}, intensity {p['key']:.2f}")
        elif mode == "dome":
            lines.append(f"Dome: {p['dome_lights']} lights at {p['dome_altitude']:.0f}°, "
                         f"rotated {p['azimuth']:.0f}°")
        elif mode == "exaggerated":
            lines.append(f"Exaggerated shading: light {p['azimuth']:.0f}° / {p['altitude']:.0f}°, "
                         f"exaggeration {p['exag_contrast']:.1f}, {p['exag_scales']} scales, "
                         f"fine-detail emphasis {p['exag_detail']:+.2f}, base {p['exag_base']:.2f}")
        elif mode == "vat":
            lines.append(f"VAT blend, {w['r_vat_variant'].currentText()}: hillshade {p['azimuth']:.0f}° / "
                         f"{p['altitude']:.0f}° + slope + openness + sky view (radius {p['svf_radius']} px)")
        elif mode == "bands":
            lines.append(self._band_summary(p))
        if eng.needs_texture(p):
            lines.append(f"Texture shading: detail {p['texture_detail'] * 100:.0f} %"
                         + (f", mix {p['texture_mix']:.2f}" if lit else "")
                         + f", contrast {p['texture_contrast']:.2f}, midtone {p['texture_midtone']:+.2f}")
        elif lit:
            lines.append("Texture shading: off")
        if eng.needs_svf(p):
            lines.append((f"Occlusion {p['occlusion']:.2f}" if lit else "Sky-view factor")
                         + f" (radius {p['svf_radius']} px, {p['svf_dirs']} directions)")
        if lit and (p["curvature"] > 0 or p["slope_shade"] > 0):
            parts = []
            if p["curvature"] > 0:
                parts.append(f"curvature {p['curvature']:.2f} @ {p['curvature_radius']:g} px")
            if p["slope_shade"] > 0:
                parts.append(f"slope darkening {p['slope_shade']:.2f} to {p['slope_range']:.0f}°")
            lines.append("Lines: " + ", ".join(parts))
        if mode == "normals":
            lines.append(f"Normal map: {p['normal_convention']}")
        else:
            tone = []
            if mode == "studio" and p["specular"] > 0:
                tone.append(f"specular {p['specular']:.2f} (gloss {p['shininess']:.0f})")
            if lit:
                if p["clarity"] > 0:
                    tone.append(f"clarity {p['clarity']:.2f} @ {p['clarity_radius']:.0f} px")
                tone.append(f"flat tone {p['base_tone']:.2f}")
            if mode != "texture":
                tone.append(f"contrast {p['contrast']:.2f}")
            tone.append(f"gamma {p['gamma']:.2f}")
            if self._albedo_active(p):
                tone.append(f"× colour albedo {w['r_albedo_strength'].value():.2f}")
            elif tuple(p["tint"]) != eng.WHITE:
                tone.append(f"tint {w['r_tint'].currentText()}")
            if mode not in ("gdal", "lab", "clean"):   # no Material & tone controls (clean relief: its own tone)
                lines.append("Tone: " + ", ".join(tone))
        comp = self._component()
        if comp != "final":
            note = "" if self._component_available(p, comp) else " - not in use with these settings (final shown)"
            lines.insert(1, f"Viewing: {eng.COMPONENTS[comp]}{note}")
        if w["pv_compare"].currentIndex():
            lines.append(f"Left of divider: {w['pv_compare'].currentText()}")
        return lines

    def _band_summary(self, p):
        parts = []
        for i, (_k, label, _d) in enumerate(BANDS):
            t = p["band_types"][i]
            if t == "off" or not p["band_enabled"][i] or p["band_weights"][i] == 0:
                continue
            alt = f" @{p['band_altitudes'][i]:.0f}°" if t in ("hillshade", "multi") else ""
            parts.append(f"{label} {eng.BAND_SHADE_TYPES[t].split(' (')[0].lower()} {p['band_weights'][i]:.2f}{alt}")
        return ("Band shading: " + ("; ".join(parts) or "every band off")
                + (" - auto-level" if p["band_autolevel"] else ""))

    def _apply_band_preset(self, name=None):
        name = name or self.band_preset.currentText()
        rows = eng.BAND_PRESETS.get(name)
        if not rows:
            return
        prev, self._suspend = self._suspend, True
        try:
            self.band_preset.setCurrentText(name)
            for (k, _label, _d), (t, wgt, alt) in zip(BANDS, rows):
                self._w[f"fq_shade_{k}"].setCurrentText(eng.BAND_SHADE_TYPES[t])
                self._w[f"fq_sw_{k}"].setValue(wgt)
                self._w[f"fq_salt_{k}"].setValue(alt)
        finally:
            self._suspend = prev
        if not prev:
            self._look_dirty = True
            self._update_enabled()
            self._update_overlay()
            self.canvas.flash(f"Band shading: {name}")
            self._schedule_preview(force=True)

    def _auto_gis_ve(self, target_sd=0.10):
        """Exaggeration that gives the typical grey spread of scanner shaded renders, measured on the
        full-resolution reference crop with the current light and tone."""
        layer = self._surface_layer()
        if layer is None:
            return
        ls = self._layer_state(layer)
        dx, dy = self._pixel_size(ls["info"])
        p = dict(self._render_params(), mode="gis", gis_ve=1.0, contrast=1.0, gamma=1.0)
        prep = eng.prepare(ls["ref"], dx, dy, p, 1.0, None, ls["plane"], ls["ref_origin"])

        def spread(v):
            img = eng.shade(prep, dict(p, gis_ve=float(v)))
            return float((img[prep.valid] if prep.valid is not None else img).std())

        best = min(np.geomspace(0.05, 20, 25), key=lambda v: abs(spread(v) - target_sd))
        self._set_value(self._w["r_gis_ve"], round(float(best), 2))
        self.canvas.flash(f"Exaggeration ×{best:.2f} (grey spread ≈ {target_sd:.2f})")

    def _fit_gis_reference(self):
        """Fit the GIS hill shading to a reference grey render of the same area (see eng.fit_gis_to_reference)."""
        layer, ref_layer = self._surface_layer(), self._w["gis_ref"].currentLayer()
        if layer is None or ref_layer is None:
            self.lbl_gis_fit.setText("Choose a height map and a reference render first.")
            return
        ls = self._layer_state(layer)
        info = ls["info"]
        # Fit on a full-resolution crop at the centre of the view: shading an averaged-down surface is not the
        # same as averaging down a shaded render, so a decimated fit would be biased.
        vx, vy, vw, vh = self.canvas.viewWindow() or (0, 0, info.width, info.height)
        cw, ch = min(vw, 1024), min(vh, 1024)
        win, out = (vx + (vw - cw) // 2, vy + (vh - ch) // 2, cw, ch), (cw, ch)
        sx = sy = 1.0
        z, _ = rio.read_array(info.path, win, out)
        rinfo = rio.RasterInfo(ref_layer.source())
        fx, fy = rinfo.width / float(info.width), rinfo.height / float(info.height)
        rwin = (int(round(win[0] * fx)), int(round(win[1] * fy)),
                max(1, int(round(win[2] * fx))), max(1, int(round(win[3] * fy))))
        ref, _ = rio.read_array(rinfo.path, rwin, out)
        dx, dy = self._pixel_size(info)
        p = dict(self._render_params(), mode="gis")
        self.status.setText("Matching the reference render…")
        try:
            prep = eng.prepare(z, dx * sx, dy * sy, p, sx, None, ls["plane"], (win[0], win[1]))
            fit = eng.fit_gis_to_reference(prep.normals, ref, prep.valid, max_samples=60_000)
        except ValueError as e:
            self.lbl_gis_fit.setText(str(e))
            self.status.setText("Ready")
            return
        prev, self._suspend = self._suspend, True
        try:
            self._w["r_mode"].setCurrentText(eng.RENDER_MODES["gis"])
            for k in ("azimuth", "altitude", "gis_ve", "gis_ambient", "gis_brightness", "gis_lights", "gis_shadow",
                      "gis_highlight", "gis_slope"):
                self._set_value(self._w["r_" + k], fit[k])
            self._set_value(self._w["r_contrast"], 1.0)
            self._set_value(self._w["r_gamma"], 1.0)
        finally:
            self._suspend = prev
        self._look_dirty = True
        self.lbl_gis_fit.setText(
            f"Matched {ref_layer.name()}: reproduces {max(fit['r2'], 0.0) * 100:.0f} % of its detail - light "
            f"{fit['azimuth']:.0f}° / {fit['altitude']:.0f}°, exaggeration ×{fit['gis_ve']:.2f}, ambient "
            f"{fit['gis_ambient']:.2f}, brightness {fit['gis_brightness']:.2f}")
        self.status.setText("Ready")
        self._update_mode_visibility()
        self._sync_dial()
        self._update_overlay()
        self.canvas.flash(f"Matched reference · R² {fit['r2']:.2f}")
        self._schedule_preview(force=True)

    def _use_band_shading(self):
        self._w["r_mode"].setCurrentText(eng.RENDER_MODES["bands"])
        self.tabs.setCurrentIndex(TAB_RENDER)

    def _fast_look_params(self, name):
        """Render parameters of a built-in look, computed directly (no widget changes)."""
        if name.startswith(USER_LOOK_PREFIX) or name not in LOOKS:
            return self._look_params(name)
        cur = self._render_params()
        p = dict(eng.DEFAULT_PARAMS)
        for k in ("z_factor", "detrend", "flatten_sigma", "eq_enabled", "eq_sigmas", "eq_gains", "band_types",
                  "band_weights", "band_altitudes", "band_enabled", "band_autolevel", "normal_convention"):
            p[k] = cur[k]
        for k, v in LOOKS[name].items():
            if k == "tint":
                p["tint"] = eng.TINTS[v]
            elif k == "multi_preset":
                p["multi_method"], p["multi_lights"] = eng.MULTI_PRESETS[v][0], list(eng.MULTI_PRESETS[v][1])
            elif k == "band_preset":
                rows = eng.BAND_PRESETS[v]
                p["band_types"] = [r[0] for r in rows]
                p["band_weights"] = [float(r[1]) for r in rows]
                p["band_altitudes"] = [float(r[2]) for r in rows]
            else:
                p[k] = v
        return p

    def _strip_items(self):
        """What the side-panel strip shows for the current tab, and which is current."""
        tab = self.tabs.currentIndex()
        p = self._render_params()
        if tab == TAB_LAB:
            pl = dict(p, mode="lab", lab_limits=None)
            keys = (["hillshade"] + [f"band{i}" for i in range(5) if float(pl["lab_weights"][i]) > 0]
                    + list(eng.LAB_STACK_OUTPUTS))
            return ([dict(key="lab:" + k, label=eng.LAB_OUTPUTS[k], params=dict(pl, lab_output=k)) for k in keys],
                    "lab:" + pl["lab_output"])
        if tab == TAB_DETAIL:
            from . import detail
            d = self._detail_params()
            return ([dict(key="detail:" + k, label=DETAIL_SHORT[k], params=p, detail=dict(d, display=k))
                     for k in detail.DISPLAYS], "detail:" + d["display"])
        if tab == TAB_FREQUENCY:
            pb = dict(p, mode="bands")
            items = [dict(key="combo", label="Combination", params=pb)]
            for i, (_k, label, _d) in enumerate(BANDS):
                t = pb["band_types"][i]
                if t != "off" and pb["band_enabled"][i]:
                    items.append(dict(key=f"band{i}", label=f"{label}: {eng.BAND_SHADE_TYPES[t]}", params=pb,
                                      component=f"band{i}"))
            text = self._w["fq_preview_band"].currentText()
            cur = "combo" if text == FQ_COMBO else next(
                (f"band{i}" for i, (_k, label, _d) in enumerate(BANDS) if text == f"{label} band (shaded)"), None)
            return items, cur
        if tab in (TAB_STACK,):
            items = [dict(key=f"light{i}", label=f"{name}  {az:.0f}° / {alt:.0f}°",
                          params=dict(p, mode="hillshade", azimuth=az, altitude=alt, contrast=1.0, gamma=1.0,
                                      tint=eng.WHITE))
                     for i, (name, az, alt) in enumerate(self._table_rows()[:24])]
            return items, f"light{max(self.table.currentRow(), 0)}"
        names = [n for n in STRIP_LOOKS if n in LOOKS] + [USER_LOOK_PREFIX + n for n in sorted(_read_json(LOOKS_FILE))]
        items = [dict(key="look:" + n, label=n.split(" · ")[-1], params=self._fast_look_params(n))
                 for n in names]
        for it in items:
            q = it["params"]
            if q["mode"] == "gdal" and q.get("gdal_display", "qgis") != "raw":
                it["params"] = dict(q, gdal_limits=self._gdal_limits(q))
        cur = ("look:" + self._look_name) if self._look_name and not self._look_dirty else None
        return items, cur

    def _request_strip(self):
        if self._suspend or not self.side.isVisible() or self.side.currentIndex() != 0:
            return
        layer, view = self._surface_layer(), self.canvas.viewWindow()
        if layer is None or view is None:
            return
        ls = self._layer_state(layer)
        info = ls["info"]
        _, out, _ = info.window_fit(172, view)
        items, current = self._strip_items()
        dx, dy = self._pixel_size(info)
        sig = json.dumps([ls["key"], view, out, dx, dy,
                          [(it["key"], it["label"], it.get("component"), it["params"], it.get("detail"))
                           for it in items]],
                         default=str, sort_keys=True)
        self._strip_current = current
        if sig == self._strip_sig and self.strip.keys():
            self.strip.setCurrent(current)
            return
        self._strip_sig = sig
        self._gen += 1
        self.strip.setStatus("Updating thumbnails…")
        self.renderer.sync = self._run_sync
        self.renderer.submit(dict(kind="strip", gen=self._gen, path=info.path, layer_key=ls["key"], view=view,
                                  out=out, dx=dx, dy=dy, plane=ls["plane"], items=items, full=(info.width, info.height),
                                  ref=ls["ref"], ref_origin=ls["ref_origin"],
                                  normals=self._normals_ref(ls)[0] if self._w["r_clean_normals"].isChecked() else None))

    def _on_strip_clicked(self, key):
        if key.startswith("detail:"):
            from . import detail
            self._w["dt_display"].setCurrentText(detail.DISPLAYS[key[7:]])
        elif key.startswith("look:"):
            self._apply_look(key[5:])
        elif key.startswith("lab:"):
            self._w["lab_output"].setCurrentText(eng.LAB_OUTPUTS[key[4:]])
        elif key == "combo":
            self._w["fq_preview_band"].setCurrentText(FQ_COMBO)
        elif key.startswith("band"):
            self._w["fq_preview_band"].setCurrentText(f"{BANDS[int(key[4:])][1]} band (shaded)")
        elif key.startswith("light"):
            self.table.setCurrentCell(int(key[5:]), 0)

    def _component(self):
        return COMPONENT_OF.get(self._w["pv_component"].currentText(), "final")

    @staticmethod
    def _component_available(p, comp):
        if comp == "final":
            return True
        if p["mode"] not in eng.LIT_MODES:
            return False
        return {"shadows": p["mode"] == "studio", "occlusion": p["occlusion"] > 0}.get(comp, True)

    def _update_overlay(self):
        """'Preview:' line under the image and the side panel's Settings tab -
        kept off the image itself."""
        lines = self._settings_summary()
        self.lbl_showing.setText(("Preview: " + lines[0]) if lines else "Preview: choose a height map")
        scale = self.canvas.scale() * self.canvas.devicePixelRatioF()
        if lines and scale < 0.67:
            lines = lines + [f"Showing about 1/{1 / scale:.0f} resolution - zoom to 1:1 to judge fine detail."]
        self.settings_view.setText("".join(
            f"<p style='margin:0 0 7px 0'>{'<b>' if i == 0 else ''}{html.escape(t)}{'</b>' if i == 0 else ''}</p>"
            for i, t in enumerate(lines)) or "Choose a height map.")
        self.canvas.setOverlay([])

    def _copy_settings(self):
        text = "\n".join(self._settings_summary()) + "\n\n" + json.dumps(self._render_params(), indent=1, default=str)
        QApplication.clipboard().setText(text)
        self.status.setText("Settings copied to the clipboard")

    def _reload_looks(self):
        current = self.look_combo.currentText()
        self.look_combo.clear()
        self.look_combo.addItems(list(LOOKS))
        user = sorted(_read_json(LOOKS_FILE))
        if user:
            self.look_combo.insertSeparator(self.look_combo.count())
            self.look_combo.addItems([USER_LOOK_PREFIX + n for n in user])
        i = self.look_combo.findText(current)
        if i >= 0:
            self.look_combo.setCurrentIndex(i)

    def _save_look(self):
        name, ok = QInputDialog.getText(self, "Save look", "Name for this look:")
        if not (ok and name.strip()):
            return
        data = _read_json(LOOKS_FILE)
        data[name.strip()] = {k: self._value(w) for k, w in self._w.items()
                              if k.startswith("r_") and not k.startswith("r_out")}
        data[name.strip()]["__multi_lights__"] = self._multi_rows()
        _write_json(LOOKS_FILE, data)
        self._reload_looks()
        self.look_combo.setCurrentText(USER_LOOK_PREFIX + name.strip())
        self.status.setText(f"Saved look '{name.strip()}'")

    def _fill_presets_menu(self):
        """By object first; then every look, under its family; then your own."""
        m = self.presets_menu
        m.clear()
        m.addSection("By object")
        for name in OBJECT_PRESETS:
            m.addAction(name, lambda n=name: self._apply_object_preset(n))
        m.addSection("All looks")
        families = {}
        for name in LOOKS:
            families.setdefault(name.split(" · ")[0] if " · " in name else "Other", []).append(name)
        for family, names in families.items():
            sub = m.addMenu(family)
            for name in names:
                sub.addAction(name.split(" · ", 1)[-1], lambda n=name: self._apply_look(n))
        mine = sorted(_read_json(LOOKS_FILE))
        if mine:
            sub = m.addMenu("Your looks")
            for name in mine:
                sub.addAction(name, lambda n=USER_LOOK_PREFIX + name: self._apply_look(n))

    def _apply_object_preset(self, name):
        look, surface = OBJECT_PRESETS[name]
        self.fq_preset.setCurrentText(surface)
        self._apply_surface_preset()
        self._apply_look(look)
        self.status.setText(f"Preset: {name}")

    def _apply_surface_preset(self):
        sigmas, gains, median, blur, detrend = SURFACE_PRESETS[self.fq_preset.currentText()]
        prev, self._suspend = self._suspend, True
        try:
            for i, v in enumerate(sigmas):
                self._w[f"fq_sigma{i}"].setValue(v)
            for (key, _, _), g in zip(BANDS, gains):
                self._w[f"fq_gain_{key}"].setValue(g)
                self._w[f"fq_mode_{key}"].setCurrentIndex(0)
                self._w[f"fq_radius_{key}"].setValue(0)
                self._w[f"fq_thresh_{key}"].setValue(0)
            self._w["fq_median"].setValue(median)
            self._w["fq_preblur"].setValue(blur)
            self._w["fq_detrend"].setChecked(detrend)
            self._w["fq_denoise"].setValue(20.0 if "Paper" in self.fq_preset.currentText() else 0.0)
        finally:
            self._suspend = prev
        self._schedule_preview(force=True)

    def _apply_stack_preset(self):
        n, alt = STACK_PRESETS[self.stack_preset.currentText()]
        prev, self._suspend = self._suspend, True
        self._w["ls_count"].setCurrentText(str(n))
        self._w["ls_altitude"].setValue(alt)
        self._w["ls_offset"].setValue(0)
        self._suspend = prev
        self._regen_table()

    def _auto_z(self):
        layer = self._surface_layer()
        if layer is None:
            return
        ls = self._layer_state(layer)
        dx, dy = self._pixel_size(ls["info"])
        p = dict(eng.merged_params(self._render_params()), z_factor=1.0)
        surf = eng.prepare_surface(eng.fill_nodata(ls["ref"])[0], dx, dy, p, 1.0, ls["plane"], ls["ref_origin"])[0]
        k = eng.auto_z_factor(surf, dx, dy, 30.0, 95.0)
        self._w["z_factor"].setValue(k)
        self._log_line(f"Auto Z-factor: {k:.4g} (95th-percentile slope → 30° on a full-resolution crop)")

    def _auto_clean_flatten(self):
        if self._ls is not None:
            info = self._ls["info"]
            self._w["r_clean_flatten"].setValue(max(16, round(0.015 * max(info.width, info.height))))

    def _auto_flatten(self):
        if self._ls is not None:
            info = self._ls["info"]
            self._w["r_flatten_sigma"].setValue(max(16, round(0.02 * max(info.width, info.height))))

    def _split_settings(self):
        w = self._w
        return dict(sigmas=[w[f"fq_sigma{i}"].value() for i in range(4)], median=w["fq_median"].value(),
                    preblur=w["fq_preblur"].value(), detrend=w["fq_detrend"].isChecked(),
                    denoise=w["fq_denoise"].value(),
                    treat={k: dict(mode=w[f"fq_mode_{k}"].currentText(), radius=w[f"fq_radius_{k}"].value(),
                                   threshold=w[f"fq_thresh_{k}"].value()) for k, _, _ in BANDS},
                    gains={k: w[f"fq_gain_{k}"].value() for k, _, _ in BANDS},
                    enabled=[k for k, _, _ in BANDS if w[f"fq_on_{k}"].isChecked()])

    # ------------------------------------------------------------------ light table
    def _regen_table(self):
        n = int(self._w["ls_count"].currentText())
        off, alt = self._w["ls_offset"].value(), self._w["ls_altitude"].value()
        self._set_table_rows([(f"Dir_{az:05.1f}", az, a) for az, a in eng.ring_lights(n, alt, off)])

    def _set_table_rows(self, rows):
        self.table.setRowCount(len(rows))
        for i, (name, az, alt) in enumerate(rows):
            self.table.setItem(i, 0, QTableWidgetItem(str(name)))
            a = QDoubleSpinBox()
            a.setRange(0, 360)
            a.setDecimals(1)
            a.setWrapping(True)
            a.setValue(float(az) % 360)
            e = QDoubleSpinBox()
            e.setRange(0, 90)
            e.setDecimals(1)
            e.setValue(float(alt))
            for sb in (a, e):
                sb.setFrame(False)
                sb.valueChanged.connect(self._on_table_edit)
            self.table.setCellWidget(i, 1, a)
            self.table.setCellWidget(i, 2, e)
        if rows and self.table.currentRow() < 0:
            self.table.setCurrentCell(0, 0)
        self._update_dial_markers()

    def _table_rows(self):
        rows = []
        for i in range(self.table.rowCount()):
            item = self.table.item(i, 0)
            rows.append((item.text().strip() if item else f"L{i}", self.table.cellWidget(i, 1).value(),
                         self.table.cellWidget(i, 2).value()))
        return rows

    # ------------------------------------------------------------------ profiles & project state
    def _reload_profiles(self):
        self.profile_combo.clear()
        self.profile_combo.addItems(sorted(_read_presets()))

    def _save_profile(self):
        name, ok = QInputDialog.getText(self, "Save profile", "Profile name:", text=self.profile_combo.currentText())
        if not (ok and name.strip()):
            return
        data = _read_presets()
        data[name.strip()] = self.state(include_layers=False)
        _write_presets(data)
        self._reload_profiles()
        self.profile_combo.setCurrentText(name.strip())

    def _load_profile(self):
        data = _read_presets().get(self.profile_combo.currentText())
        if data:
            self.set_state(data)
            self.status.setText(f"Loaded profile '{self.profile_combo.currentText()}'")

    def _delete_profile(self):
        name = self.profile_combo.currentText()
        if not name or QMessageBox.question(self, "Delete profile", f"Delete profile '{name}'?") != \
                QMessageBox.StandardButton.Yes:
            return
        data = _read_presets()
        data.pop(name, None)
        _write_presets(data)
        self._reload_profiles()

    def _reset_defaults(self):
        if QMessageBox.question(self, "Reset to defaults", "Reset every setting to its default? Saved profiles and "
                                "the selected layers are kept.") == QMessageBox.StandardButton.Yes:
            self.set_state(self._defaults)
            self.status.setText("All settings reset to defaults")

    def _save_project_state(self):
        try:
            QgsProject.instance().writeEntry(PROJECT_SCOPE, "state_v2", json.dumps(self.state(), default=str))
        except Exception:
            pass

    def _load_project_state(self):
        proj = QgsProject.instance()
        raw, ok = proj.readEntry(PROJECT_SCOPE, "state_v2", "")
        if ok and raw:
            try:
                self.set_state(json.loads(raw))
                return
            except ValueError:
                pass
        d = {}   # migrate v1.x project entries
        for old, new in (("export_dir", "export_dir"), ("input_ultra", "ls_mix_u"), ("input_high", "ls_mix_h"),
                         ("input_low", "ls_mix_l"), ("input_color", "colour"), ("input_uv_ir", "prov_uvir")):
            val, ok = proj.readEntry(PROJECT_SCOPE, old, "")
            if ok and val:
                d[new] = val
        z, ok = proj.readDoubleEntry(PROJECT_SCOPE, "z_master", 1.0)
        if ok:
            d["z_factor"] = z
        if d:
            self.set_state(d)

    # ------------------------------------------------------------------ jobs
    def _output_dir(self, announce=True):
        d = self._w["export_dir"].filePath().strip()
        if d and os.path.isdir(d):
            return d
        d = os.path.join(tempfile.gettempdir(), "helio_relief")
        os.makedirs(d, exist_ok=True)
        if announce:
            self._log_line(f"No export folder set - writing to {d}")
        return d

    def _provenance(self, layer, info, extra):
        dx, dy = self._real_pixel(info)
        return dict(software="Helio Relief", qgis_version=Qgis.version(),
                    source_layer=layer.name(), source_path=info.path, raster_size=[info.width, info.height],
                    pixel_size=[dx, dy], pixel_unit=self._unit_label(info), **extra)

    def _require_surface(self):
        layer = self._surface_layer()
        if layer is None:
            QMessageBox.warning(self, "No height map", "Choose a height-map layer first.")
            return None
        if self._w["prov_warn_geo"].isChecked() and layer.crs().isValid() and layer.crs().isGeographic():
            if QMessageBox.question(self, "Geographic CRS", f"'{layer.name()}' uses degrees for X/Y, so slopes "
                                    "will be wrong unless the pixel size is set by hand. Continue?") != \
                    QMessageBox.StandardButton.Yes:
                return None
        return layer

    def _start(self, description, fn, job, on_done):
        if self._task is not None:
            QMessageBox.information(self, "Busy", "A job is already running - wait for it or press Cancel job.")
            return

        def done(res):
            self._task = None
            self._set_busy(False, f"{description} finished in {res.get('seconds', 0):.1f} s")
            on_done(res)
            self._save_project_state()

        def failed(text):
            self._task = None
            self._steps = []                         # a chain of jobs stops at the first failure
            self._target_group = None
            self._set_busy(False, f"{description} failed")
            self._log_line(f"[{description}] {text}", error=True)
            QMessageBox.critical(self, description, text.split("\n\n")[0] + "\n\nFull details are in the log.")

        def cancelled():
            self._task = None
            self._steps = []
            self._target_group = None
            self._set_busy(False, f"{description} cancelled")
            self._log_line(f"[{description}] cancelled")

        task = Job(description, lambda t: fn(t, job), done, failed, cancelled)
        task.progressChanged.connect(lambda v: self.progress.setValue(int(v)))
        task.message.connect(self.status.setText)
        self._task = task
        self._set_busy(True, description + "…")
        if self._run_sync:
            task.finished(task.run())
        else:
            QgsApplication.taskManager().addTask(task)
        return True

    def _set_busy(self, busy, text):
        self.status.setText(text)
        self.btn_cancel.setEnabled(busy)
        self.progress.setValue(0 if busy else (100 if "finished" in text else 0))
        self._set_run_enabled(not busy and self._surface_layer() is not None)

    def _log_line(self, text, error=False):
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log.appendPlainText(f"{stamp}  {text}")
        if error:
            self.btn_log.setChecked(True)

    def _add_layer(self, path, name, group=None, style="exact", ramp=None, blend=None, opacity=None):
        if group is None and getattr(self, "_target_group", None) is not None:
            try:
                group = self._target_group if self._target_group.parent() is not None else None
            except RuntimeError:                         # the group was deleted by the user
                group = None
        lyr = QgsRasterLayer(path, name, "gdal")
        if not lyr.isValid():
            self._log_line(f"Could not load {path}", error=True)
            return None
        if ramp and COLOR_RAMP_PRESETS.get(ramp):
            apply_pseudocolor_ramp(lyr, ramp)
        elif style == "exact":
            style_exact(lyr)
        elif style == "cut":
            style_cumulative_cut(lyr)
        if blend:
            lyr.setBlendMode(BLEND_MODES[blend])
        if opacity is not None:
            lyr.setOpacity(opacity)
        QgsProject.instance().addMapLayer(lyr, group is None)
        if group is not None:
            group.addLayer(lyr)
        return lyr

    def _show_provenance(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.meta_view.setPlainText(f.read())
        except OSError:
            pass

    # ---- render
    def _run_render(self, then=None):
        layer = self._require_surface()
        if layer is None:
            return
        ls = self._layer_state(layer)
        info = ls["info"]
        p = self.__dict__.pop("_render_override", None) or self._render_params()
        stats = self._reference_stats(p, ls, eng.needs_texture(p), eng.needs_svf(p))
        window = self.canvas.viewWindow() if self._w["r_out_area"].currentIndex() == 1 else None
        if window == (0, 0, info.width, info.height):
            window = None
        name = self._w["r_out_name"].text().strip() or (f"{layer.name()}_{p['mode']}"
                                                        + (f"_{p['lab_output']}" if p["mode"] == "lab" else "")
                                                        + ("_view" if window else ""))
        dx, dy = self._pixel_size(info)
        self._last_render_style = ({"qgis": "default", "cut": "cut"}.get(p["gdal_display"], "exact")
                                   if p["mode"] == "gdal" else "exact")
        if p["mode"] == "gdal":
            p = dict(p, gdal_display="raw", gdal_limits=None)      # write gdaldem's exact grey levels
        job = dict(source=info.path, dx=dx, dy=dy, params=p, stats=stats, plane=ls["plane"], window=window,
                   out_base=_unique_base(self._output_dir(), _safe_name(name)), out16=self._w["r_out_16"].isChecked(),
                   png=self._w["r_out_png"].isChecked(),
                   albedo=rio.gdal_path(self._w["colour"].currentLayer().source()) if self._albedo_active(p) else None,
                   albedo_strength=self._w["r_albedo_strength"].value(),
                   normals=self._normals_ref(ls)[0] if p.get("clean_source") == "normals" else None,
                   meta=self._provenance(layer, info, dict(kind="render", render_parameters=p, window=window)))
        return self._start("Render", render_job, job,
                           (lambda res: (self._render_done(res), then())) if then else self._render_done)

    # ---- job chains, HELIO import, the Start here guide
    def _run_steps(self, steps):
        """Run steps one after another. A step returns True when it started a job; that job's done callback runs
        the next step. A failed or cancelled job ends the chain."""
        self._steps = list(steps)
        self._next_step()

    def _next_step(self):
        while getattr(self, "_steps", None):
            step = self._steps.pop(0)
            try:
                if step():
                    return
            except Exception:
                self._steps = []
                raise

    def _import_helio(self, folder=None, render=None):
        """One click from a Helio export folder to a ready surface: every height map, the colour image and the
        normal map loaded in a group; slow PNG / JPEG files copied to tiled GeoTIFFs; the colour aligned to the
        height map; the two integrations compared and the better one chosen; optionally the standard renders."""
        from . import helio
        if folder is None:
            here = self._surface_layer()
            start = os.path.dirname(rio.gdal_path(here.source())) if here is not None else os.path.expanduser("~")
            folder = QFileDialog.getExistingDirectory(self, "Choose a HELIO export folder", start)
            if not folder:
                return
        s = helio.scan(folder)
        if not s["dems"]:
            QMessageBox.warning(self, "Import HELIO export", f"No HELIO height map (…_DEM_Poisson.tif, …_depth.tif) "
                                                             f"in {folder}.")
            return
        found = ", ".join([label for label, _ in s["dems"]] + (["colour image"] if s["albedo"] else [])
                          + (["normal map"] if s["normals"] else []) + (["uncertainty"] if s["uncertainty"] else [])
                          + (["metal mask"] if s["metal"] else []))
        if render is None:
            render = QMessageBox.question(
                self, "Import HELIO export", f"{s['object_id']}: {found}.\n\nAlso render the standard set at full "
                "resolution once it is set up (grey Clean relief, colour composite, QGIS multidirectional "
                "hillshade)?") == QMessageBox.StandardButton.Yes
        if not self._w["export_dir"].filePath().strip():
            out = os.path.join(folder, "Helio_Relief_output")
            os.makedirs(out, exist_ok=True)
            self._w["export_dir"].setFilePath(out)
            self._log_line(f"Export folder set to {out}")
        self._object_id, self._helio_scan = s["object_id"], s
        group = QgsProject.instance().layerTreeRoot().insertGroup(0, f"HELIO · {s['object_id']}")
        self._helio_group = self._target_group = group          # the import's outputs go into its group
        dems = [self._add_layer(path, f"{s['object_id']} · {label}", group, style="default") for label, path in s["dems"]]
        dems = [d for d in dems if d is not None]
        if not dems:
            return
        self._w["surface"].setLayer(dems[0])
        if len(dems) > 1:
            self._w["tp_a"].setLayer(dems[0])
            self._w["tp_b"].setLayer(dems[1])
        ref = rio.gdal_path(dems[0].source())
        col = self._add_layer(s["albedo"], f"{s['object_id']} · colour", group, style="default") if s["albedo"] else None
        nml = (self._add_layer(s["normals"], f"{s['object_id']} · normal map", group, style="default")
               if s["normals"] else None)
        if col is not None:
            self._w["colour"].setLayer(col)
        if nml is not None:
            self._w["normals"].setLayer(nml)
        for key, label, stops in HELIO_OVERLAYS:
            if s.get(key):
                lyr = self._add_layer(s[key], f"{s['object_id']} · {label}", group, style="none")
                if lyr is not None:
                    style_overlay(lyr, stops)
                    group.findLayer(lyr.id()).setItemVisibilityChecked(False)
        self._add_helio_marks(s, group, dems[0].crs())
        self._w["r_mode"].setCurrentText(eng.RENDER_MODES["clean"])
        steps = []
        for key, lyr in (("colour", col), ("normals", nml)):
            if lyr is not None and rio.is_slow_to_window(lyr.source()):
                steps.append(lambda key=key, lyr=lyr: self._step_tiled(key, lyr, ref))
        steps.append(self._step_align_if_needed)
        if len(dems) > 1:
            steps.append(lambda: self._run_compare(then=self._after_helio_compare))
        if render:
            steps.append(lambda: self._step_render("clean_grey", colour=False))
            if col is not None:
                steps.append(lambda: self._step_render("clean_colour", colour=True))
            steps.append(lambda: self._step_render("qgis_multidirectional", colour=False, mode="gdal",
                                                   gdal_variant="multidirectional", gdal_edges=True))
        steps.append(lambda: self._helio_ready(found))
        self._log_line(f"HELIO export {s['object_id']} ({s['kind']}): {found}")
        self._run_steps(steps)

    def _add_helio_marks(self, s, group, crs):
        """The sections and annotations made in Helio, at the top of the import's group."""
        from . import vectors
        layers = []
        if s.get("sections"):
            layers.append(vectors.section_layer(s["sections"], f"{s['object_id']} · sections", crs))
        if s.get("annotations"):
            layers += vectors.annotation_layers(s["annotations"], f"{s['object_id']} · annotations", crs)
        for lyr in layers:
            if lyr is not None:
                QgsProject.instance().addMapLayer(lyr, False)
                group.insertLayer(0, lyr)

    def _step_tiled(self, key, lyr, ref):
        src = rio.gdal_path(lyr.source())
        ri, si = rio.RasterInfo(ref), rio.RasterInfo(src)
        same = (ri.width, ri.height) == (si.width, si.height)
        out = _unique_base(self._output_dir(), _safe_name(f"{lyr.name()}_tiled")) + ".tif"
        self._tiled_src = lyr
        return self._start(f"Copying {lyr.name()} to a tiled GeoTIFF", tiled_job,
                           dict(source=src, out=out, ref=ref if same else None, key=key),
                           lambda res: (self._tiled_done(res), self._next_step()))

    def _tiled_done(self, res):
        old = getattr(self, "_tiled_src", None)
        lyr = self._add_layer(res["path"], (old.name() if old is not None else "copy") + " (tiled)",
                              getattr(self, "_helio_group", None), style="default")
        if lyr is None:
            return
        node = QgsProject.instance().layerTreeRoot().findLayer(old.id()) if old is not None else None
        if node is not None:
            node.setItemVisibilityChecked(False)
        self._w[res["key"]].setLayer(lyr)

    def _step_align_if_needed(self):
        col, layer = self._w["colour"].currentLayer(), self._surface_layer()
        if col is None or layer is None:
            return False
        ci, li = rio.RasterInfo(rio.gdal_path(col.source())), self._layer_state(layer)["info"]
        if (ci.width, ci.height) == (li.width, li.height) and (ci.georeferenced or not li.georeferenced):
            self._w["r_albedo"].setChecked(True)         # already on the height map's grid
            return False
        return self._align_colour(then=self._next_step)

    def _after_helio_compare(self):
        res = getattr(self, "_pick", None)
        if res and res["choice"] == "b" and self._w["tp_b"].currentLayer() is not None:
            self._w["surface"].setLayer(self._w["tp_b"].currentLayer())
        self._next_step()

    def _step_render(self, suffix, colour=False, **changes):
        w = self._w
        p = dict(self._render_params(), **changes)
        saved = (w["r_out_area"].currentIndex(), w["r_out_name"].text(), w["r_albedo"].isChecked())

        def put(area, name, albedo):
            prev, self._suspend = self._suspend, True
            try:
                w["r_out_area"].setCurrentIndex(area)
                w["r_out_name"].setText(name)
                w["r_albedo"].setChecked(albedo)
            finally:
                self._suspend = prev

        put(0, f"{getattr(self, '_object_id', None) or 'render'}_{suffix}",
            bool(colour and w["colour"].currentLayer() is not None))
        self._render_override = p
        self._quick_add = True
        try:
            return self._run_render(then=self._next_step)
        finally:
            put(*saved)

    def _helio_ready(self, found):
        layer = self._surface_layer()
        self._target_group = None
        self.canvas.flash(f"{getattr(self, '_object_id', '')} ready")
        self._log_line(f"HELIO import finished - {found}. Height map: {layer.name() if layer else '-'}.")
        self.status.setText(f"{getattr(self, '_object_id', '')} imported and set up")
        preset = HELIO_OBJECTS.get((getattr(self, "_helio_scan", None) or {}).get("object_type", ""))
        if preset:                          # Helio said what this is
            self._apply_object_preset(preset)
        self._on_surface_changed()          # adding layers to a group can leave the preview on a stale 'no layer'
        return False

    def _show_start_here(self):
        from qgis.PyQt.QtCore import QSettings
        from .widgets import StartHere
        settings = QSettings()
        key = "HelioRelief/start_here"
        steps = [
            ("Load your scan", "Import a HELIO export folder - height maps, colour and normal map, set up, aligned "
             "and compared for you - or choose a height map in the Surface box.",
             [("Import HELIO folder…", lambda: self._import_helio())]),
            ("Choose the integration", "Poisson or Frankot-Chellappa? The integration picker (Tools tab) compares "
             "them on your object and recommends one - or a fusion of the two.",
             [("Open the picker", lambda: self.tabs.setCurrentIndex(TAB_TOOLS))]),
            ("Pick a look", "Clean relief - smooth, detailed, lit from every side - is the default. Try the looks in "
             "the panel beside the preview, or see them all at once in the gallery.",
             [("Open the gallery", self._open_gallery)]),
            ("Separate detail from noise", "The Detail tab keeps chosen feature sizes - strokes without the grain, the "
             "marks without the warp - live, shown as relief, hillshade, Clean relief or lines; save it as a height "
             "map or trace it.", [("Open Detail", lambda: self.tabs.setCurrentIndex(TAB_DETAIL))]),
            ("Add colour", "Choose the albedo as Colour / albedo, press Align if the note under it asks, and tick "
             "Colour composite for the archive-style colour relief.", [("Align colour", lambda: self._align_colour())]),
            ("Measure and trace", "Measure: drag a profile across the preview (Shift-drag an area) for depths in real "
             "units. Trace lines draws the incised lines as a vector layer you can edit.",
             [("Measure tool", lambda: (self.btn_measure_tool.setChecked(True),
                                        self.tabs.setCurrentIndex(TAB_TOOLS))),
              ("Trace lines", lambda: self.tabs.setCurrentIndex(TAB_TOOLS))]),
            ("Export", "Quick export saves the whole scan at full resolution, georeferenced over the height map "
             "(Ctrl+E). A publication plate lays out a print-ready page with a true scale bar (Tools tab).",
             [("Quick export", lambda: self._quick_export(False)),
              ("Make a plate…", lambda: self.tabs.setCurrentIndex(TAB_TOOLS))]),
        ]
        guide = StartHere(self, steps, settings.value(key, True, type=bool),
                          lambda on: settings.setValue(key, bool(on)))
        self._start_here = guide
        guide.show()

    def _maybe_start_here(self):
        from qgis.PyQt.QtCore import QSettings
        if getattr(self, "iface", None) is None:            # headless / tests: never pop up
            return
        if QSettings().value("HelioRelief/start_here", True, type=bool):
            self._show_start_here()

    # ---- Tools tab: measuring
    def _units(self, info):
        """(xy unit, z unit) for measurements: the pixel unit; heights in the same unit unless it is px."""
        unit = self._unit_label(info)
        relative = unit in ("px", "units") or self._real_height(info) is None
        return unit, ("height units" if relative else unit)

    def _on_measured(self, kind, x0, y0, x1, y1):
        from . import tools
        layer = self._surface_layer()
        if layer is None:
            return
        ls = self._layer_state(layer)
        info = ls["info"]
        dx, dy = self._real_pixel(info)
        uxy, uz = self._units(info)
        drift = self._w["ms_heights"].currentIndex() == 1
        flat = float(self._w["r_clean_flatten"].value())
        pad = int(3 * flat) + 4 if drift else 2
        xa, xb = sorted((x0, x1))
        ya, yb = sorted((y0, y1))
        wx0, wy0 = max(0, int(xa) - pad), max(0, int(ya) - pad)
        wx1, wy1 = min(info.width, int(math.ceil(xb)) + pad + 1), min(info.height, int(math.ceil(yb)) + pad + 1)
        win = (wx0, wy0, max(2, wx1 - wx0), max(2, wy1 - wy0))
        k = max(1.0, math.sqrt(win[2] * win[3] / 25e6))                  # keep a read under ~25 MP
        out = (max(2, int(round(win[2] / k))), max(2, int(round(win[3] / k))))
        kx, ky = win[2] / out[0], win[3] / out[1]
        z = rio.read_array(info.path, win, out)[0].astype(np.float64)
        z = np.where(np.isfinite(z), z, np.nanmedian(z)) * float(self._w["ms_zscale"].value()) \
            * (self._real_height(info) or 1.0)
        if drift and flat > 0:
            z = z - eng.gaussian_blur(z.astype(np.float32), flat / kx)
        if kind == "line":
            ax, ay = (x0 - wx0) / kx - 0.5, (y0 - wy0) / ky - 0.5
            bx, by = (x1 - wx0) / kx - 0.5, (y1 - wy0) / ky - 0.5
            d_px, h = tools.profile(z, ax, ay, bx, by, 0.5)
            s = tools.profile_stats(d_px, h)
            length = math.hypot((x1 - x0) * dx, (y1 - y0) * dy)
            unit_per = length / max(float(d_px[-1]), 1e-12)
            d = d_px * unit_per
            trend = h - s["residual"]
            self.profile_view.setProfile(d, h, (trend[0], trend[-1]),
                                         (s["depth_at"] * unit_per, float(h[int(np.argmin(s['residual']))]),
                                          f"deepest {s['depth']:.4g} {uz}"), uxy, uz)
            lines = [f"Length {length:.4g} {uxy}   ·   heights {s['min']:.5g} to {s['max']:.5g} {uz} "
                     f"(range {s['range']:.4g})",
                     f"Deepest groove below the line's trend: {s['depth']:.4g} {uz} at {s['depth_at'] * unit_per:.4g} "
                     f"{uxy}, {s['width'] * unit_per:.4g} {uxy} wide at half depth",
                     f"Highest ridge above the trend: {s['height']:.4g} {uz}   ·   slope of the trend "
                     f"{s['slope'] / max(unit_per, 1e-12):.4g} {uz}/{uxy}"]
            self._last_measure = dict(kind="line", length=length, length_px=math.hypot(x1 - x0, y1 - y0), unit=uxy,
                                      zunit=uz, dist=d, heights=h, residual=s["residual"])
            self.canvas.flash(f"Deepest groove {s['depth']:.3g} {uz}")
        else:
            bx0, by0 = int((xa - wx0) / kx), int((ya - wy0) / ky)
            bx1, by1 = max(bx0 + 2, int((xb - wx0) / kx)), max(by0 + 2, int((yb - wy0) / ky))
            s = tools.area_stats(z[by0:by1, bx0:bx1])
            cell = dx * kx * dy * ky
            w_u, h_u = (xb - xa) * dx, (yb - ya) * dy
            lines = [f"Box {w_u:.4g} × {h_u:.4g} {uxy} = {w_u * h_u:.4g} {uxy}²   ·   heights mean {s['mean']:.5g}, "
                     f"{s['min']:.5g} to {s['max']:.5g} {uz}, sd {s['sd']:.3g}",
                     f"Relief after removing the surface's plane: {s['relief']:.4g} {uz} (1st-99th percentile)",
                     f"Volume of the hollows below the plane: {s['hollow_volume'] * cell:.4g} {uz}·{uxy}²   ·   of the "
                     f"bumps above it: {s['bump_volume'] * cell:.4g}"]
            self.profile_view.clear()
            self._last_measure = dict(kind="box", stats=s, unit=uxy, zunit=uz)
            self.canvas.flash(f"Relief {s['relief']:.3g} {uz}")
        if k > 1:
            lines.append(f"(read at 1/{k:.2g} resolution - measure a shorter line or smaller box for full detail)")
        self.lbl_measure.setText("\n".join(lines))
        if self.tabs.currentIndex() != TAB_TOOLS:
            self.tabs.setCurrentIndex(TAB_TOOLS)

    def _clear_measure(self):
        self.canvas.setMeasure(None)
        self.profile_view.clear()
        self.lbl_measure.setText("")
        self._last_measure = None

    def _export_profile(self):
        m = getattr(self, "_last_measure", None)
        if not m or m["kind"] != "line":
            QMessageBox.information(self, "Export profile", "Measure a line first (Measure tool, drag across the "
                                                            "preview).")
            return
        start = os.path.join(self._output_dir(False), "profile.csv")
        path, _ = QFileDialog.getSaveFileName(self, "Save the profile", start, "CSV (*.csv)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"distance_{m['unit']},height_{m['zunit']},below_trend_{m['zunit']}\n")
                for a, b, c in zip(m["dist"], m["heights"], m["residual"]):
                    f.write(f"{a:.6g},{b:.8g},{c:.6g}\n")
            self._log_line(f"Profile saved: {path}")

    def _calibrate_from_line(self):
        from qgis.PyQt.QtWidgets import QInputDialog
        m = getattr(self, "_last_measure", None)
        if not m or m["kind"] != "line":
            QMessageBox.information(self, "Set the scale", "Measure a line of known length first (e.g. along a "
                                                           "ruler in the scan).")
            return
        value, ok = QInputDialog.getDouble(self, "Set the scale", "The line you measured is really this long:", 10.0,
                                           0.000001, 1e9, 4)
        if not ok:
            return
        unit, ok = QInputDialog.getItem(self, "Set the scale", "Unit:", ["mm", "µm", "cm", "m"], 0, False)
        if not ok:
            return
        size = value / max(m["length_px"], 1e-9)
        self._w["pixel_auto"].setChecked(False)
        self._w["pixel_size"].setValue(size)
        self._w["pixel_unit"].setCurrentText(unit)
        self._update_scale_bar()
        self._log_line(f"Scale set from a measured line: 1 px = {size:.6g} {unit}")
        self.canvas.flash(f"1 px = {size:.4g} {unit}")

    # ---- Tools tab: line tracing
    def _trace_params(self):
        from . import tools
        w = self._w
        return dict(kind={v: k for k, v in tools.TRACE_KINDS.items()}.get(w["tr_kind"].currentText(), "incised"),
                    width=w["tr_width"].value(), sensitivity=w["tr_sens"].value(), min_length=w["tr_min"].value(),
                    simplify=w["tr_simplify"].value(), bridge=w["tr_bridge"].value(), smooth=w["tr_smooth"].value(),
                    min_strength=w["tr_minstr"].value())

    def _run_trace(self):
        layer = self._require_surface()
        if layer is None:
            return
        ls = self._layer_state(layer)
        info = ls["info"]
        whole = self._w["tr_area"].currentIndex() == 1
        window = None if whole else self.canvas.viewWindow()
        band = self._w["tr_source"].currentIndex() == 1
        name = f"{layer.name()}_lines" + ("_band" if band else "") + ("" if whole else "_view")
        dx, dy = self._pixel_size(info)
        job = dict(self._trace_params(), source=info.path, window=window, max_px=16e6 if whole else 25e6,
                   pixel=self._real_pixel(info), unit=self._unit_label(info),
                   out=_unique_base(self._output_dir(), _safe_name(name), "gpkg") + ".gpkg")
        if band:
            job.update(detail=self._detail_params(), ref=ls["ref"], plane=ls["plane"], ref_origin=ls["ref_origin"],
                       dx=dx, dy=dy)
        self._trace_name = name
        self._start("Trace lines", trace_job, job, self._trace_done)

    def _trace_done(self, res):
        from qgis.core import QgsCategorizedSymbolRenderer, QgsLineSymbol, QgsRendererCategory, QgsVectorLayer
        lyr = QgsVectorLayer(res["path"] + "|layername=lines", getattr(self, "_trace_name", "lines"), "ogr")
        if lyr.isValid():
            cats = [QgsRendererCategory(k, QgsLineSymbol.createSimple({"color": c, "width": "0.35"}), label)
                    for k, c, label in (("incised", "230,40,40", "Incised"), ("raised", "40,110,240", "Raised"))]
            lyr.setRenderer(QgsCategorizedSymbolRenderer("kind", cats))
            QgsProject.instance().addMapLayer(lyr)
        self._trace_history.append((lyr.id() if lyr.isValid() else None, res["path"]))
        self.btn_undo_trace.setEnabled(True)
        self._lines_gen += 1                      # a live trace still running must not replace these
        self.canvas.setLines(res["overlay"])
        text = (f"{res['count']} lines, {res['total']:.4g} {res['unit']} in all"
                + (f" (traced at 1/{res['k']:.2g} resolution)" if res["k"] > 1.01 else "") + f" → {res['path']}")
        self.lbl_trace.setText(text)
        self._log_line("Line tracing: " + text)

    def _clear_lines(self):
        self._lines_gen += 1
        self.canvas.setLines([])
        self.lbl_trace.setText("")

    def _undo_trace(self):
        if not self._trace_history:
            return
        lid, path = self._trace_history[-1]
        if QMessageBox.question(self, "Undo last trace", f"Remove the traced layer and delete "
                                f"{os.path.basename(path)}?") != QMessageBox.StandardButton.Yes:
            return
        self._trace_history.pop()
        if lid and QgsProject.instance().mapLayer(lid) is not None:
            QgsProject.instance().removeMapLayer(lid)
        for f in (path, path + "-wal", path + "-shm"):
            try:
                if os.path.exists(f):
                    os.remove(f)
            except OSError as exc:
                self._log_line(f"Could not delete {f}: {exc}", error=True)
        self._clear_lines()
        self.btn_undo_trace.setEnabled(bool(self._trace_history))
        self.lbl_trace.setText(f"Undone - {os.path.basename(path)} deleted")

    # ---- live line tracing of the view
    def _schedule_lines(self):
        if "tr_live" in self._w and self._w["tr_live"].isChecked() and not self._suspend:
            self._lines_timer.start()

    def _request_lines(self):
        layer, view = self._surface_layer(), self.canvas.viewWindow()
        if layer is None or view is None or not self._w["tr_live"].isChecked():
            return
        ls = self._layer_state(layer)
        info = ls["info"]
        dx, dy = self._pixel_size(info)
        tr = dict(self._trace_params(), max_px=1.5e6)
        k = max(1.0, math.sqrt(view[2] * view[3] / tr["max_px"]))
        self._lines_gen += 1
        if tr["width"] / k < 1.2:                   # the lines would be finer than the pixels traced
            self.canvas.setLines([])
            self.lbl_trace.setText(f"Live: zoom in to see {tr['width']:g}-px lines live (this view would be traced at "
                                   f"1/{k:.2g} resolution) - or trace it with Trace lines")
            return
        req = dict(kind="lines", gen=self._lines_gen, path=info.path, layer_key=ls["key"],
                   full=(info.width, info.height), view=view, trace=tr,
                   dx=dx, dy=dy, plane=ls["plane"], ref=ls["ref"], ref_origin=ls["ref_origin"])
        if self._w["tr_source"].currentIndex() == 1:
            req["detail"] = self._detail_params()
        self.renderer.sync = self._run_sync
        self.renderer.submit(req)

    def _lines_done(self, res):
        if res["gen"] != self._lines_gen or not self._w["tr_live"].isChecked():
            return
        self.canvas.setLines(res["lines"])
        self.lbl_trace.setText(f"Live: {res['count']} lines in view"
                               + (f" (traced at 1/{res['k']:.2g} resolution)" if res["k"] > 1.01 else "")
                               + f" · {res['ms']:.0f} ms")

    # ---- the Detail tab
    def _detail_params(self):
        from . import detail
        w = self._w
        return dict(bands=[dict(on=w[f"dt_on{i}"].isChecked(), fine=w[f"dt_fine{i}"].value(),
                                broad=w[f"dt_broad{i}"].value(), gain=w[f"dt_gain{i}"].value())
                           for i in range(detail.N_BANDS)],
                    soft=w["dt_soft"].value(), wiener=w["dt_wiener"].value(), gate=w["dt_gate"].value(),
                    clean=w["dt_clean"].value(), gain=w["dt_show_gain"].value(),
                    display={v: k for k, v in detail.DISPLAYS.items()}.get(w["dt_display"].currentText(), "signed"))

    def _update_detail_chart(self):
        from . import detail
        if not hasattr(self, "spectrum_view"):
            return
        d = self._detail_params()
        self.spectrum_view.setBands([(b["fine"], b["broad"], b["on"]) for b in d["bands"]],
                                    max(self._dt_edit.checkedId(), 0))
        lo, hi = self.spectrum_view._range()
        sizes = np.geomspace(lo, hi, 96)
        self.spectrum_view.setResponse(sizes, detail.response(sizes, d, self._detail_model))

    def _on_spectrum_drag(self, fine, broad):
        i = max(self._dt_edit.checkedId(), 0)
        if not self._w[f"dt_on{i}"].isChecked():
            self._w[f"dt_on{i}"].setChecked(True)
        self._w[f"dt_fine{i}"].setValue(fine)
        self._w[f"dt_broad{i}"].setValue(broad)

    def _apply_detail_preset(self):
        from . import detail
        pr = detail.PRESETS[self.dt_preset.currentText()]
        m = self._detail_model or {}
        cross = round(float(m["crossover"]), 2) if m.get("detected") else None
        fine = pr["fine"] if pr["fine"] >= 0 else (cross or 0.0)
        broad = pr["broad"] if pr["broad"] >= 0 else (cross or 2.0)
        i = max(self._dt_edit.checkedId(), 0)
        prev, self._suspend = self._suspend, True
        try:
            self._w[f"dt_on{i}"].setChecked(True)
            self._w[f"dt_fine{i}"].setValue(fine)
            self._w[f"dt_broad{i}"].setValue(broad)
        finally:
            self._suspend = prev
        if min(pr["fine"], pr["broad"]) < 0 and cross is None:
            self.canvas.flash("No noise resolvable in this view - the preset uses the finest sizes instead")
        else:
            self.canvas.flash(f"Band {i + 1}: {self.dt_preset.currentText()}")
        self._update_enabled()
        self._update_detail_chart()
        self._update_overlay()
        self._schedule_lines()
        self._schedule_preview(force=True)

    def _run_detail_save(self, what="band"):
        layer = self._require_surface()
        if layer is None:
            return
        d = self._detail_params()
        if not any(b["on"] for b in d["bands"]):
            QMessageBox.information(self, "Detail", "Tick at least one band first.")
            return
        ls = self._layer_state(layer)
        info = ls["info"]
        dx, dy = self._pixel_size(info)
        window = self.canvas.viewWindow() if self._w["dt_area"].currentIndex() == 1 else None
        if window == (0, 0, info.width, info.height):
            window = None
        name = f"{layer.name()}_" + ("band" if what == "band" else f"detail_{d['display']}") + ("_view" if window else "")
        job = dict(source=info.path, detail=d, what=what, params=self._render_params(), ref=ls["ref"],
                   plane=ls["plane"], ref_origin=ls["ref_origin"], dx=dx, dy=dy, window=window,
                   out_base=_unique_base(self._output_dir(), _safe_name(name)),
                   meta=self._provenance(layer, info, dict(kind="detail_" + what, detail=d, window=window)))
        self._detail_name = name
        return self._start("Save band" if what == "band" else "Save the display", detail_job, job, self._detail_done)

    def _detail_done(self, res):
        lyr = self._add_layer(res["path"], getattr(self, "_detail_name", "band"),
                              style="exact" if res["what"] == "image" else "cut")
        text = f"Saved {res['path']}"
        if lyr is not None and res["what"] == "band" and self._w["dt_use_saved"].isChecked():
            self._w["surface"].setLayer(lyr)
            text += " - now the height map"
        self.lbl_detail.setText(text)
        self._log_line("Detail: " + text)

    def _trace_in_band(self):
        self._w["tr_source"].setCurrentIndex(1)
        self.tabs.setCurrentIndex(TAB_TOOLS)
        self._run_trace()

    # ---- settings undo / redo
    def _undo_snapshot(self):
        return {k: v for k, v in self.state(include_layers=False).items() if not k.startswith("pv_")}

    def _note_change(self):
        if self._undo_base is not None:
            self._undo_timer.start()

    def _commit_undo(self):
        if self._undo_base is None:
            return
        snap = self._undo_snapshot()
        if snap != self._undo_base:
            self._undo_stack.append(self._undo_base)
            del self._undo_stack[:-100]
            self._undo_base = snap
            self._redo_stack.clear()
        self._update_undo_buttons()

    def _undo_settings(self, redo=False):
        if self._undo_base is None:
            return
        if self._undo_timer.isActive():
            self._undo_timer.stop()
            self._commit_undo()
        src, dst = (self._redo_stack, self._undo_stack) if redo else (self._undo_stack, self._redo_stack)
        if not src:
            self.canvas.flash("Nothing to redo" if redo else "Nothing to undo")
            return
        current, target = self._undo_base, src.pop()
        dst.append(current)
        changed = [k for k in target if target.get(k) != current.get(k)]
        # only the settings that differ, without set_state's surface reload - the view and the layer stay put
        prev, self._suspend = self._suspend, True
        try:
            for k in changed:
                if k in self._w and k not in self._layer_keys:
                    self._set_value(self._w[k], target[k])
            if "light_table" in changed:
                self._set_table_rows(target["light_table"]) if target.get("light_table") else self._regen_table()
            if "multi_lights" in changed and target.get("multi_lights"):
                self._set_multi_rows(target["multi_lights"])
        finally:
            self._suspend = prev
        self._update_mode_visibility()
        self._update_source_visibility()
        self._update_pixel_widgets()
        self._sync_dial()
        self._update_dial_markers()
        self._update_colour_state()
        self._update_normals_state()
        self._update_scale_bar()
        self._update_overlay()
        self._undo_timer.stop()
        self._undo_base = self._undo_snapshot()
        self._look_dirty = True
        self._update_enabled()
        self._update_detail_chart()
        self._schedule_preview(force=True)
        self._undo_timer.stop()
        what = self._labels.get(changed[0]) if len(changed) == 1 else None
        self.canvas.flash(("Redo" if redo else "Undo") + (f": {what}" if what else f" ({len(changed)} settings)"))
        self._update_undo_buttons()

    def _update_undo_buttons(self):
        if hasattr(self, "btn_undo_settings"):
            self.btn_undo_settings.setEnabled(bool(self._undo_stack))
            self.btn_redo_settings.setEnabled(bool(self._redo_stack))

    # ---- Tools tab: integration picker
    def _pick_layers(self):
        a, b = self._w["tp_a"].currentLayer(), self._w["tp_b"].currentLayer()
        if a is None or b is None or a.id() == b.id():
            QMessageBox.information(self, "Integration picker", "Choose two height maps of the same capture as A and "
                                                                "B (e.g. the Poisson and Frankot-Chellappa DEMs).")
            return None, None
        return a, b

    def _run_compare(self, then=None):
        a, b = self._pick_layers()
        if a is None:
            return
        layer = self._surface_layer() or a
        ls = self._layer_state(layer)
        info = ls["info"]
        p = dict(self._render_params(), mode="clean", clean_source="height")
        view = self.canvas.viewWindow() or (0, 0, info.width, info.height)
        job = dict(a=rio.gdal_path(a.source()), b=rio.gdal_path(b.source()), names=(a.name(), b.name()), params=p,
                   view=view, sigma=self._w["tp_sigma"].value(), dx=self._pixel_size(info)[0])
        return self._start("Compare integrations", compare_job, job,
                           (lambda res: (self._compare_done(res), then())) if then else self._compare_done)

    def _compare_done(self, res):
        m, (na, nb) = res["metrics"], res["names"]
        choice = {"a": f"use {na}", "b": f"use {nb}", "fuse": f"fuse {na}'s form with {nb}'s detail"}[res["choice"]]
        self._pick = res
        rows = [("Fine detail (1.5-8 px)", "detail", "{:.3f}"), ("Pixel noise (&lt; 1.5 px)", "noise", "{:.3f}"),
                ("Detail / noise", "snr", "{:.2f}"), ("Broad drift / relief", "drift", "{:.2f}")]
        table = "<table cellspacing=6><tr><td></td><td><b>A</b> " + na + "</td><td><b>B</b> " + nb + "</td></tr>"
        for label, key, f in rows:
            table += f"<tr><td>{label}</td><td>{f.format(m['a'][key])}</td><td>{f.format(m['b'][key])}</td></tr>"
        table += "</table>"
        self.lbl_pick.setText(f"{table}Their tooling-scale relief agrees {res['agreement']:.2f} (1 = identical). "
                              f"B's heights are ×{1 / max(res['gain'], 1e-12):.3g} A's.<br><b>Recommendation: {choice}</b>"
                              f" - {res['why']}.<br>Thumbnails: Clean relief of the view from A, B and fused.")
        width = max(80, (self.tabs.width() - 60) // 3)            # three side by side, within the panel
        for lab, img, name in zip(self.pick_thumbs, res["thumbs"], ("A", "B", "fused")):
            q, buf = _qimage(img)
            full = QPixmap.fromImage(q.copy())
            lab.setPixmap(full.scaledToWidth(min(width, full.width()), Qt.TransformationMode.SmoothTransformation))
            lab.setToolTip(f"Clean relief of the view from {name}")
        self._log_line(f"Integration picker: {choice} - {res['why']}")

    def _use_integration(self, key):
        lyr = self._w[key].currentLayer()
        if lyr is not None:
            self._w["surface"].setLayer(lyr)
            self.canvas.flash(f"Height map: {lyr.name()}")

    def _run_fuse(self):
        a, b = self._pick_layers()
        if a is None:
            return
        res = getattr(self, "_pick", None)
        sigma = self._w["tp_sigma"].value()
        name = f"{a.name()}_fused_{b.name()}"
        job = dict(a=rio.gdal_path(a.source()), b=rio.gdal_path(b.source()), sigma=sigma,
                   gain=res["gain"] if res and abs(res.get("sigma", sigma) - sigma) < 1e-9 else None,
                   out=_unique_base(self._output_dir(), _safe_name(name)) + ".tif")
        self._fuse_name = name
        self._start("Fuse height maps", fuse_job, job, self._fuse_done)

    def _fuse_done(self, res):
        lyr = self._add_layer(res["path"], getattr(self, "_fuse_name", "fused"), style="default")
        if lyr is not None:
            self._w["surface"].setLayer(lyr)
        self._log_line(f"Fused height map (form from A, detail finer than {res['sigma']:g} px from B × "
                       f"{res['gain']:.4g}) → {res['path']}")

    # ---- Tools tab: publication plate
    def _run_plate(self):
        from qgis.PyQt.QtCore import QRectF
        layer = self._require_surface()
        if layer is None:
            return
        ls = self._layer_state(layer)
        info = ls["info"]
        p = self._render_params()
        paper = PAPER_MM[self._w["pl_paper"].currentText()]
        page_mm = paper if self._w["pl_orient"].currentIndex() == 0 else (paper[1], paper[0])
        dpi = int(self._w["pl_dpi"].currentText().split()[0])
        mm = dpi / 25.4
        content = PLATE_CONTENT.index(self._w["pl_content"].currentText())
        colour = self._w["colour"].currentLayer()
        kinds = ["grey"] + (["colour"] if content in (1, 3) and colour is not None else []) + \
            (["closeup"] if content in (2, 3) else [])
        margin, top, bottom = 12 * mm, 20 * mm, (26 if self._w["pl_caption"].isChecked() else 12) * mm
        pw, ph = page_mm[0] * mm, page_mm[1] * mm
        area = QRectF(margin, top, pw - 2 * margin, ph - top - bottom)
        n, gap = len(kinds), 6 * mm
        across = page_mm[0] >= page_mm[1]
        rects = []
        for i in range(n):
            if across:
                w = (area.width() - gap * (n - 1)) / n
                rects.append(QRectF(area.left() + i * (w + gap), area.top(), w, area.height()))
            else:
                h = (area.height() - gap * (n - 1)) / n
                rects.append(QRectF(area.left(), area.top() + i * (h + gap), area.width(), h))
        stats = self._reference_stats(p, ls, eng.needs_texture(p), eng.needs_svf(p))
        title = self._w["pl_title"].text().strip() or getattr(self, "_object_id", None) or layer.name()
        fmt = self._w["pl_format"].currentText()
        base = _unique_base(self._output_dir(), _safe_name(f"{layer.name()}_plate"))
        job = dict(source=info.path, params=p, stats=stats, plane=ls["plane"], dx=self._pixel_size(info)[0],
                   dy=self._pixel_size(info)[1], normals=self._normals_ref(ls)[0] if p.get("clean_source") == "normals"
                   else None, colour=rio.gdal_path(colour.source()) if colour is not None else None,
                   albedo_strength=self._w["r_albedo_strength"].value(), view=self.canvas.viewWindow(),
                   panels=[dict(kind=k, w=int(r.width()), h=int(r.height())) for k, r in zip(kinds, rects)])
        self._plate = dict(rects=rects, page_mm=page_mm, dpi=dpi, fmt=fmt, title=title, kinds=kinds,
                           path=base + {"PDF": ".pdf", "PNG": ".png", "TIFF": ".tif"}[fmt],
                           caption=self._plate_caption(info) if self._w["pl_caption"].isChecked() else [],
                           unit=self._unit_label(info), px=self._real_pixel(info)[0])
        self._start("Publication plate", plate_job, job, self._plate_done)

    def _plate_caption(self, info):
        summary = self._settings_summary()
        hs = self._helio_scale(info) if self._w["pixel_auto"].isChecked() else None
        est = " (estimated)" if hs and hs["estimated"] else ""
        dx = self._real_pixel(info)[0]
        return [f"{self._surface_layer().name()} · {info.width:,} × {info.height:,} px · 1 px = {dx:.4g} "
                f"{self._unit_label(info)}{est} · {datetime.date.today().isoformat()}"] + \
            [s for s in summary[:4] if s] + ["Helio Relief " + _plugin_version()]

    def _plate_done(self, res):
        from qgis.PyQt.QtCore import QMarginsF, QRectF, QSizeF
        from qgis.PyQt.QtGui import QFont, QImage, QPageSize, QPdfWriter
        pl = self._plate
        dpi, mm = pl["dpi"], pl["dpi"] / 25.4
        pw, ph = int(pl["page_mm"][0] * mm), int(pl["page_mm"][1] * mm)
        if pl["fmt"] == "PDF":
            dev = QPdfWriter(pl["path"])
            dev.setResolution(dpi)
            dev.setPageSize(QPageSize(QSizeF(*pl["page_mm"]), QPageSize.Unit.Millimeter))
            dev.setPageMargins(QMarginsF(0, 0, 0, 0))
        else:
            dev = QImage(pw, ph, QImage.Format.Format_RGB32)
            dev.fill(QColor(255, 255, 255))
            dev.setDotsPerMeterX(int(dpi / 0.0254))
            dev.setDotsPerMeterY(int(dpi / 0.0254))
        p = QPainter(dev)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = QFont(self.font())
        font.setPixelSize(int(6 * mm))
        font.setBold(True)
        p.setFont(font)
        p.setPen(QColor(20, 20, 24))
        p.drawText(QRectF(12 * mm, 6 * mm, pw - 24 * mm, 10 * mm), Qt.AlignmentFlag.AlignLeft |
                   Qt.AlignmentFlag.AlignVCenter, pl["title"])
        labels = {"grey": "Clean relief" if self._mode() == "clean" else "Relief render", "colour": "Colour composite",
                  "closeup": "Close-up of the view"}
        small = QFont(self.font())
        small.setPixelSize(int(3 * mm))
        keep, lowest = [], 0.0
        for panel, rect, kind in zip(res["panels"], pl["rects"], pl["kinds"]):
            q, buf = _qimage(panel["img"])
            keep.append(buf)
            iw, ih = q.width(), q.height()
            s = min(rect.width() / iw, rect.height() / ih)
            target = QRectF(rect.left() + (rect.width() - iw * s) / 2, rect.top(), iw * s, ih * s)
            p.drawImage(target, q)
            p.setPen(QColor(150, 150, 156))
            p.drawRect(target)
            lowest = max(lowest, target.bottom() + 9 * mm)
            # scale bar: object units per page px = full-res px per image px x unit per full-res px / page px per image px
            per = panel["k"] * pl["px"] / s
            want = target.width() * 0.25 * per
            mag = 10 ** math.floor(math.log10(want)) if want > 0 else 1
            nice = max(v * mag for v in (1, 2, 5) if v * mag <= want) if want > 0 else 0
            p.setFont(small)
            p.setPen(QColor(40, 40, 46))
            p.drawText(QRectF(target.left(), target.bottom() + 1 * mm, target.width(), 4 * mm),
                       Qt.AlignmentFlag.AlignLeft, labels[kind])
            if nice > 0:
                length = nice / per
                y = target.bottom() + 7 * mm
                x0 = target.right() - length
                p.fillRect(QRectF(x0, y, length, 0.8 * mm), QColor(20, 20, 24))
                p.drawText(QRectF(x0 - 30 * mm, y - 4.5 * mm, length + 30 * mm, 4 * mm), Qt.AlignmentFlag.AlignRight,
                           f"{nice:g} {pl['unit']}")
        if pl["caption"]:
            p.setFont(small)
            p.setPen(QColor(60, 60, 66))
            top = min(lowest + 4 * mm, ph - 22 * mm)            # right under the panels, not at the page foot
            p.drawText(QRectF(12 * mm, top, pw - 24 * mm, 20 * mm),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap,
                       "\n".join(pl["caption"]))
        p.end()
        if pl["fmt"] != "PDF":
            dev.save(pl["path"])
        self._log_line(f"Publication plate ({pl['fmt']}, {dpi} dpi): {pl['path']}")
        self.status.setText(f"Plate saved: {pl['path']}")

    def _update_colour_state(self):
        """Say under the Colour / albedo row whether it sits on the height map (pixels and map)."""
        text, col, layer = "", self._w["colour"].currentLayer(), self._surface_layer()
        if col is not None and layer is not None:
            try:
                ci = rio.RasterInfo(rio.gdal_path(col.source()))
                li = self._layer_state(layer)["info"]
                if (ci.width, ci.height) != (li.width, li.height):
                    text = (f"⚠ {ci.width:,} × {ci.height:,} px, the height map {li.width:,} × {li.height:,}: not on "
                            f"its grid, so colour and relief would not line up. Press Align.")
                elif li.georeferenced and not ci.georeferenced:
                    text = ("Same pixels as the height map, but not georeferenced - QGIS draws it apart. Press Align "
                            "to put it over the height map.")
            except Exception:
                text = ""
        self.lbl_colour.setText(text)
        self.lbl_colour.setVisible(bool(text))

    def _align_colour(self, then=None):
        layer = self._require_surface()
        if layer is None:
            return
        col = self._w["colour"].currentLayer()
        if col is None:
            QMessageBox.information(self, "Align colour", "Choose the colour / albedo layer first.")
            return
        info = self._layer_state(layer)["info"]
        name = f"{col.name()}_aligned"
        job = dict(ref=info.path, img=rio.gdal_path(col.source()),
                   out=_unique_base(self._output_dir(), _safe_name(name)) + ".vrt")
        self._align_source = (col.id(), name)
        return self._start("Align colour", align_job, job,
                           (lambda res: (self._align_done(res), then())) if then else self._align_done)

    def _align_done(self, res):
        src_id, name = self.__dict__.pop("_align_source", (None, "colour_aligned"))
        lyr = self._add_layer(res["path"], name, style="default")
        if lyr is None:
            return
        node = QgsProject.instance().layerTreeRoot().findLayer(src_id) if src_id else None
        if node is not None:
            node.setItemVisibilityChecked(False)             # the misplaced original: hidden, not removed
        self._w["colour"].setLayer(lyr)
        self._w["r_albedo"].setChecked(True)
        msg = ("Colour already on the height map's pixels - georeferenced copy added" if res["identity"] else
               f"Colour aligned to the height map: scale {res['sx']:.4f} × {res['sy']:.4f}, rotation "
               f"{res['rotation']:+.2f}°, offset {res['tx']:+.1f}, {res['ty']:+.1f} px ({res['matched']} of "
               f"{res['patches']} places agree to {res['rms']:.2f} px)")
        self._log_line(f"{msg} → {res['path']}")
        self.status.setText(msg)
        self.canvas.flash("Colour aligned · colour composite on")
        self._update_colour_state()
        self._schedule_preview(force=True)

    def _quick_export(self, view_only=False):
        """One click, no questions: what the preview shows, at full resolution - the whole raster, or only the
        part in view - written to the export folder as a GeoTIFF plus PNG with the height map's georeferencing,
        and added to the map exactly over the height map. The Output section keeps its own settings."""
        w = self._w

        def setv(area, png):
            prev, self._suspend = self._suspend, True
            try:
                w["r_out_area"].setCurrentIndex(area)
                w["r_out_png"].setChecked(png)
            finally:
                self._suspend = prev

        saved = (w["r_out_area"].currentIndex(), w["r_out_png"].isChecked())
        setv(1 if view_only else 0, True)
        self._quick_add = True
        try:
            if self.tabs.currentIndex() == TAB_LAB or self._mode() == "lab":
                self._run_render_lab()
            else:
                self._run_render()
        finally:
            setv(*saved)
            if self._task is None:                   # finished already (or never started)
                self.__dict__.pop("_quick_add", None)

    def _render_done(self, res):
        quick = self.__dict__.pop("_quick_add", False)
        if quick or self._w["r_out_add"].isChecked():
            self._add_layer(res["tif"], os.path.splitext(os.path.basename(res["tif"]))[0],
                            style=getattr(self, "_last_render_style", "exact"))
        self._log_line(f"Render: {res['seconds']:.1f} s → " + ", ".join(res["outputs"]))
        self._show_provenance(res["sidecar"])

    # ---- frequency split
    def _run_split(self):
        layer = self._require_surface()
        if layer is None:
            return
        s = self._split_settings()
        if not s["enabled"]:
            QMessageBox.warning(self, "Nothing selected", "Tick at least one band.")
            return
        ls = self._layer_state(layer)
        info = ls["info"]
        job = dict(source=info.path, settings=s, plane=ls["plane"], out_dir=self._output_dir(),
                   prefix=_safe_name(layer.name()) + "_",
                   meta=self._provenance(layer, info, dict(kind="frequency_split")))
        self._split_layer_name = layer.name()
        self._start("Frequency split", split_job, job, self._split_done)

    def _split_done(self, res):
        root = QgsProject.instance().layerTreeRoot()
        name = f"Frequency Split - {self._split_layer_name}"
        group = root.insertGroup(0, name)
        colour = self._w["fq_colorize"].isChecked()
        for key, path in res["files"]:
            ramp = self._w[f"fq_ramp_{key}"].currentText() if colour else None
            self._add_layer(path, key, group, style="default", ramp=ramp)
        self._last_split = (name, [p for _, p in res["files"]] + [res["sidecar"]])
        self.btn_undo.setEnabled(True)
        self._log_line(f"Frequency split: {len(res['files'])} band(s) in {res['seconds']:.1f} s")
        self._show_provenance(res["sidecar"])

    def _undo_split(self):
        if not self._last_split:
            return
        name, files = self._last_split
        if QMessageBox.question(self, "Undo last split", f"Remove the '{name}' group and delete its "
                                f"{len(files)} file(s)?") != QMessageBox.StandardButton.Yes:
            return
        root = QgsProject.instance().layerTreeRoot()
        grp = root.findGroup(name)
        if grp:
            for child in list(grp.findLayers()):
                QgsProject.instance().removeMapLayer(child.layerId())
            root.removeChildNode(grp)
        removed = 0
        for f in files:
            try:
                if os.path.exists(f):
                    os.remove(f)
                    removed += 1
            except OSError as exc:
                self._log_line(f"Could not delete {f}: {exc}", error=True)
        self._last_split = None
        self.btn_undo.setEnabled(False)
        self.status.setText(f"Split undone - {removed} file(s) deleted")

    # ---- light stack & analysis
    def _stack_source(self):
        """Common job fields for the light-stack and analysis jobs, or None."""
        idx = self._w["ls_source"].currentIndex()
        p = self._render_params()
        if idx == 2:
            layers = [self._w[f"ls_mix_{t}"].currentLayer() for t in "uhl"]
            if not all(layers):
                QMessageBox.warning(self, "Mix inputs", "Choose all three pre-split height maps.")
                return None
            if not self._validate_mix(layers):
                return None
            ref_layer = layers[0]
            mix = [(rio.gdal_path(l.source()), self._w[f"mix_w_{t}"].value(), self._w[f"mix_z_{t}"].value())
                   for l, t in zip(layers, "uhl")]
            info = rio.RasterInfo(ref_layer.source())
            plane = None
        else:
            ref_layer = self._require_surface()
            if ref_layer is None:
                return None
            ls = self._layer_state(ref_layer)
            info, plane, mix = ls["info"], ls["plane"], []
        lights = self._table_rows()
        if not lights:
            QMessageBox.warning(self, "No lights", "The light dome table is empty.")
            return None
        dx, dy = self._pixel_size(info)
        return ref_layer, info, dict(source=info.path, source_mode=["prepared", "raw", "mix"][idx], mix=mix,
                                     z_master=self._w["z_factor"].value(), params=p, plane=plane, dx=dx, dy=dy,
                                     lights=lights)

    def _validate_mix(self, layers):
        warnings = []
        if self._w["prov_warn_geo"].isChecked():
            for lyr in layers:
                if lyr.crs().isValid() and lyr.crs().isGeographic():
                    warnings.append(f"'{lyr.name()}' uses a geographic CRS (degrees); slopes assume metric units.")
        if self._w["prov_warn_mismatch"].isChecked() and len({(l.width(), l.height()) for l in layers}) > 1:
            warnings.append("The three maps have different pixel dimensions; the high/mid and low maps will be "
                            "resampled onto the ultra-high map's grid.")
        if not warnings:
            return True
        return QMessageBox.question(self, "Input validation", "\n\n".join(warnings) + "\n\nContinue?") == \
            QMessageBox.StandardButton.Yes

    def _run_stack(self):
        src = self._stack_source()
        if src is None:
            return
        layer, info, job = src
        if not self._memory_ok(info, 9, "The light stack"):
            return
        w = self._w
        p = job["params"]
        job.update(out_dir=self._output_dir(), prefix=_safe_name(layer.name()) + "_",
                   individual=w["ls_individual"].isChecked(), multiband=w["ls_multiband"].isChecked(),
                   vrt=w["ls_vrt"].isChecked(), mean=w["ls_mean"].isChecked(), svf=w["ls_svf"].isChecked(),
                   svf_radius=w["ls_svf_radius"].value(), normals=w["ls_normals"].isChecked(),
                   normal_conv=w["ls_normal_conv"].currentText(), tpi=w["ls_tpi"].isChecked(),
                   tpi_radius=w["ls_tpi_radius"].value(), pca=w["ls_pca"].isChecked(), pca_n=w["ls_pca_n"].value(),
                   tone=w["ls_tone"].currentIndex(), lift=w["ls_lift"].value(),
                   meta=self._provenance(layer, info, dict(kind="light_stack", source_mode=job["source_mode"],
                                                           render_parameters=p)))
        if not any(job[k] for k in ("individual", "multiband", "vrt", "mean", "svf", "normals", "tpi", "pca")):
            QMessageBox.warning(self, "No outputs", "Tick at least one output.")
            return
        self._stack_layer_name = layer.name()
        self._start("Light stack", stack_job, job, self._stack_done)

    def _stack_done(self, res):
        w = self._w
        group = None
        if w["ls_group"].isChecked():
            group = QgsProject.instance().layerTreeRoot().insertGroup(0, f"Light Stack - {self._stack_layer_name}")
        ramp = w["ls_ramp"].currentText() if w["ls_colorize"].isChecked() else None
        for item in res["layers"]:
            kind = item["kind"]
            if kind in ("light", "multiband", "normals", "pca_rgb"):
                self._add_layer(item["path"], item["name"], group,
                                blend=w["ls_blend"].currentText() if kind == "light" else None)
            elif kind in ("mean", "svf"):
                self._add_layer(item["path"], item["name"], group, ramp=ramp)
            else:
                self._add_layer(item["path"], item["name"], group, style="default")
        colour = w["colour"].currentLayer()
        if w["ls_colour_add"].isChecked() and colour is not None:
            clone = self._add_layer(colour.source(), f"{colour.name()} (colour)", group, style="default",
                                    blend=w["ls_colour_blend"].currentText(),
                                    opacity=w["ls_colour_opacity"].value() / 100.0)
            if clone is not None and colour.renderer():
                clone.setRenderer(colour.renderer().clone())
                clone.setOpacity(w["ls_colour_opacity"].value() / 100.0)   # opacity lives in the renderer
            if clone is not None and group is not None and w["ls_colour_pos"].currentIndex() == 1:
                node = group.findLayer(clone.id())
                group.insertChildNode(0, node.clone())
                group.removeChildNode(node)
        uv = w["prov_uvir"].currentLayer()
        if w["prov_uvir_add"].isChecked() and uv is not None:
            self._add_layer(uv.source(), f"{uv.name()} (UV/IR overlay)", group, style="default",
                            opacity=w["prov_uvir_opacity"].value() / 100.0)
        self._log_line(f"Light stack: {len(res['layers'])} layer(s) in {res['seconds']:.1f} s")
        self._show_provenance(res["sidecar"])

    # ------------------------------------------------------------------ lifecycle
    def showEvent(self, e):
        super().showEvent(e)
        self._schedule_preview(force=True)

    def closeEvent(self, e):
        self._save_project_state()
        self._save_geometry()
        super().closeEvent(e)

    def shutdown(self):
        """Called when the plugin unloads."""
        self.renderer.shutdown()
        self._save_geometry()
        if self._task is not None:
            self._task.cancel()
        if self.iface is not None:
            try:
                self.iface.mapCanvas().extentsChanged.disconnect(self._on_canvas_extent)
            except (TypeError, RuntimeError):
                pass
        self._save_project_state()
