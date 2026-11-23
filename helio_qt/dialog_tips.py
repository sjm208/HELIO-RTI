"""Tooltips for the controls of Helio's dialogs, in one place.

A dialog's own code sets a tooltip where it builds a control, when it does;
these fill in the rest, so every button, box and choice says what it does
when the pointer rests on it.  Keyed by the window's title, then by the
control's own words: a button's or box's text, else its accessible name,
else a field's placeholder, else (a drop-down) its first choice.  COMMON
holds the buttons every dialog shares.  A control that already has a tooltip
keeps it.

``install(app)`` puts the filter on the whole application: a control gets its
tip when it is first shown, which also covers controls a dialog adds later.
tests/test_qt_menus.py opens the dialogs and checks nothing is left without a tip.
"""
from __future__ import annotations

from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import (QAbstractButton, QAbstractSlider, QAbstractSpinBox, QComboBox,
                               QLineEdit, QWidget)

COMMON = {
    "Close": "Close this window.",
    "Cancel": "Close without saving.",
    "OK": "Close this window.",
    "Done": "Close this window and keep the settings.",
    "Copy": "Copy the text to the clipboard.",
    "Run": "Run the tool with the settings above, on the current solve (or the selection).",
    "Export…": "Save the result: the pictures, and the figures as CSV.",
    "Save…": "Save the result.",
    "Save pictures and figures…": "Save the report's pictures as PNG and its figures as CSV.",
}

TIPS: dict[str, dict[str, str]] = {
    "RTI web package": {
        "A one-file quick view for e-mail (under 25 MB)": "Also a single HTML file with everything inside "
            "it, small enough to e-mail: open it anywhere, no server needed.",
        "Relight / OpenLIME data, with the frames to fit again": "Also the data RelightLab and OpenLIME "
            "read, and the photographs with their lights, so the RTI can be fitted again elsewhere.",
        "Zip the folder as well": "Also a .zip of the whole package, for sending or depositing.",
        "CC BY 4.0": "The licence written into the viewer and its metadata.",
        "The package folder": "Where the package is written.",
        "Title": "The title shown in the viewer and written into its metadata.",
        "Creator": "Who made the capture: credited in the viewer and its metadata.",
        "Institution": "The holding institution: credited in the viewer and its metadata.",
        "Object ID": "The object's shelfmark or accession number, written into the metadata.",
        "Web address": "Where the package will be published: used for the QR code and the embed code. "
                       "Leave empty if not known yet.",
        "Export package": "Write the web package to the folder.",
    },
    "Capture details": {
        "Defaults": "Fill the fields with the rig's usual values.",
        "Load JSON…": "Fill the fields from a saved capture-details file.",
        "Save JSON…": "Save these details as a file, to load for the next capture.",
        "Export CSV…": "Save these details as a spreadsheet row.",
        "Load CSV…": "Load measured positions (name, X, Y, Z in mm) of the lights, camera and origin.",
        "Export…": "Save the points of interest as CSV.",
        "Save": "Keep these details with the capture.",
        "Cancel": "Close without keeping the changes.",
    },
    "Send to QGIS": {
        "The package folder": "Where the GeoTIFFs and the QGIS project are written.",
        "Open in QGIS": "Write the package and open it in QGIS with the Helio Relief plugin.",
        "Save the package only": "Write the GeoTIFFs and the project without opening QGIS.",
    },
    "USDZ for AR": {
        "Also save a .usda scene with its lights (Blender, Houdini, Omniverse)": "Also a text USD scene "
            "with the same model and its lights, for 3D software.",
        "The USDZ file": "Where the USDZ is written.",
        "Export USDZ": "Write the USDZ and check it against Apple's AR rules.",
    },
    "Facsimile": {
        "Invert the relief (embossed ↔ debossed)": "Turn the relief inside out: bumps become hollows. "
            "For printing plates and seals the object type does this already.",
        "Draft: ~256 px grid, fast preview": "How finely the relief is sampled: finer is more detail and "
            "bigger files.",
        "Flat original (manuscript, print, map)": "What the object is: plates and seals are inverted and "
            "mirrored so the facsimile prints or impresses the right way round.",
        "STL: universal 3D printing (binary)": "The file for the machine: STL for any 3D printer, OBJ "
            "with displacement for elevated printing or CNC, a 16-bit TIFF for raster printers, "
            "or a coloured GLB.",
        "Export…": "Choose where, and write the facsimile files.",
    },
    "Upload to Sketchfab": {
        "Model name": "The model's name on Sketchfab.",
        "Sketchfab API token": "Your Sketchfab API token (Sketchfab › Settings › Password & API). "
                               "Kept for this session only.",
        "Upload": "Upload the textured model to your Sketchfab account.",
        "Open on Sketchfab": "Open the uploaded model in your browser.",
    },
    "QR code": {
        "Address to encode": "The web address the QR code opens: a published viewer, or this Mac's "
                             "address for phones on the same Wi-Fi.",
        "Save QR code…": "Save the QR code, on a card naming the object, its institution and scale.",
    },
    "Embed snippet": {
        "Viewer address": "The address where the viewer is published.",
        "Width": "The viewer's width in the page: a percentage or pixels.",
        "Height": "The viewer's height in the page, in pixels.",
        "Copy": "Copy the HTML, to paste into a web page.",
    },
    "Settings": {
        "Match the system": "Light or dark as the Mac is set.",
        "Light: Parchment": "A light window.",
        "Dark: Night Dome": "A dark window.",
    },
    "Reflectance unmixing": {
        "2": "Split into 2 channels.", "3": "Split into 3 channels.",
        "4": "Split into 4 channels.", "5": "Split into 5 channels.",
        "Save components…": "Save the channels as pictures.",
    },
    "Micro-features": {
        "Depressions only (ignore raised bumps)": "Find pits and holes only, not raised specks.",
        "Display depth (levelled)": "The height map to search: the levelled display depth (best for "
                                    "small pits) or the integrated depth (true heights).",
    },
    "Toolpaths": {
        "Normals (no integration)": "Where the grooves are read from: the normals directly (finest, no "
                                    "integration) or the depth map.",
    },
    "Inscription zones": {
        "8": "At most this many zones are kept, the strongest first.",
        "Book / Manuscript": "What kind of surface: weights the signs of writing (incision, ink, tooling) "
                             "for it.",
        "Zones": "Show the zones found.",
        "Signal map": "Show the combined signal the zones are drawn from.",
        "Save…": "Save the zones found.",
    },
    "Register two images": {
        "Load image A…": "Choose the image to lay over this one (a tracing, a pouncing pattern, another "
                         "object).",
        "B: the current view": "Use the current view as image B.",
        "Transform": "How A may be changed to fit B: Similarity turns, scales and shifts; Affine adds "
                     "shear; Projective a full perspective change.",
        "Mirror A": "Flip A left to right first (a pouncing pattern, the back of a leaf).",
        "Blend": "How much of A shows over B in the overlay.",
        "Clear A": "Forget the points clicked on A.",
        "Clear B": "Forget the points clicked on B.",
        "Register": "Fit A onto B from the matching points (at least 4).",
        "Export…": "Save the registered result.",
    },
    "Two faces": {
        "Recto / verso": "A leaf's two sides.",
        "Obverse / reverse": "A coin's or medal's two faces, with the die axis.",
        "Matrix / impression": "A seal's matrix and its impression.",
        "Recto (front)": "This capture is the front.", "Verso (back)": "This capture is the back.",
        "Obverse": "This capture is the obverse.", "Reverse": "This capture is the reverse.",
        "Matrix": "This capture is the matrix.", "Impression": "This capture is the impression.",
        "Choose the other side's file…": "Choose the other face's capture (a session or a picture).",
        "Flip the partner horizontally (usual for plates and leaves)": "Mirror the other face so the "
            "two line up, as a leaf's back does under its front.",
        "Save the link…": "Save the link between the two faces, with this capture's session.",
        "Export this side…": "Save this face's pictures, with the other face beside it.",
    },
    "Legend unwrap": {
        "e.g. + SIGILLVM ... (saved with the strip)": "Your reading of the legend: saved with the strip.",
        "Unwrap": "Unroll the legend round the centre into a strip.",
        "Save…": "Save the strip as a picture, with your reading.",
    },
    "Virtual impression": {
        "Impression": "As the matrix would print into wax: mirrored and inside out.",
        "Raised letters": "Inside out only: cut letters stand up, reading the right way round.",
        "As is": "The surface as it is, lit the same way.",
        "Plaster": "The colour of the material the impression is shown in.",
        "Make": "Make the impression with the settings above.",
        "Save…": "Save the impression as a picture.",
    },
    "Line drawing": {
        "Trace raised edges too": "Also trace the crests of raised edges, not only cut lines.",
        "Show the object under the lines": "Show the surface faintly under the traced lines.",
        "Trace": "Trace the lines with the settings above.",
        "Save SVG…": "Save the lines as SVG, in millimetres once a scale is set.",
    },
    "Focus stacking": {
        "Align the frames first": "Line up each light's focus frames before merging (a focus ring "
                                  "changes the scale a little).",
        "Number of light positions": "How many lights the capture has.",
        "Focus frames per light": "How many focus steps were taken under each light.",
        "Choose and stack…": "Choose all the photographs, sorted light by light, and merge each light's "
                             "focus frames into one sharp frame.",
        "Save merged…": "Save the merged frames, one per light, ready to import as a capture.",
    },
    "Ask AI": {
        "Claude (Anthropic)": "Which model answers: online providers need an API key; Ollama and LM "
                              "Studio run on this computer and nothing leaves it.",
        "Model": "The provider's model to use (Find lists those loaded on this computer).",
        "API key": "The provider's API key. Kept for this session only.",
        "Send the image": "Send the current view (at most 1024 px) with the question. Off: the "
                          "statistics only.",
        "Identify": "What the object may be.",
        "Condition report": "A draft condition report, for you to check.",
        "Figure caption": "A draft figure caption.",
        "Brief": "A short answer.", "Standard": "A usual-length answer.",
        "Comprehensive": "A full answer; identification adds located findings, drawn as numbered boxes.",
        "Ask": "Send the question to the chosen model.",
        "Copy": "Copy the answer.",
        "Save…": "Save the answer.",
    },
    "Save the light sweep": {
        "3 s": "How long one turn of the light takes.",
        "Save…": "Save the sweep as a GIF or MP4.",
    },
    "Helio Help": {
        "Search the help": "Type to find a topic.",
        "Getting started": "The first steps: import, find the lights, process, look, export.",
        "Report a problem": "A report on your Mac, this project and the log, to send with a problem.",
    },
    "Keyboard Shortcuts": {
        "Find a shortcut": "Type a key or what it does.",
    },
    "Getting started": {
        "Show at start": "Show these first steps each time Helio opens.",
        "Back": "The previous step.", "Next": "The next step.",
    },
    "Report a problem": {
        "Open my email": "Start an e-mail to Helio's support; attach the report file to it.",
        "Open the form": "Open the support form in your browser.",
        "Show the file": "Show the report in the Finder.",
    },
}


def key_of(w: QWidget) -> list[str]:
    """The words a control is known by, most specific first."""
    keys = []
    if isinstance(w, QAbstractButton) and w.text():
        keys.append(w.text().replace("&", ""))
    if w.accessibleName():
        keys.append(w.accessibleName())
    if isinstance(w, QLineEdit) and w.placeholderText():
        keys.append(w.placeholderText())
    if isinstance(w, QComboBox) and w.count():
        keys.append(w.itemText(0))
    return keys


def tip_for(w: QWidget) -> str:
    table = TIPS.get(w.window().windowTitle(), {})
    for k in key_of(w):
        if k in table:
            return table[k]
    for k in key_of(w):
        if k in COMMON:
            return COMMON[k]
    return ""


class _Filler(QObject):
    TYPES = (QAbstractButton, QComboBox, QAbstractSlider, QAbstractSpinBox, QLineEdit)

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Type.Show and isinstance(obj, self.TYPES) and not obj.toolTip():
            try:
                t = tip_for(obj)
                if t:
                    obj.setToolTip(t)
            except RuntimeError:                      # deleted while showing
                pass
        return False


_filler = None


def install(app) -> None:
    """Fill in dialog tooltips for the whole application (once)."""
    global _filler
    if _filler is None:
        _filler = _Filler(app)
        app.installEventFilter(_filler)
