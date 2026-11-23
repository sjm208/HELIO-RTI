"""Help, Getting Started, Report a Problem and the crash note.

The glossary, the methods and references, and the engine list are the
classic Help's own (helio_help, helio.domain.registry.ENGINE_BLURBS), so
the science is described once.  The how-to sections are written for this
window's five steps; the classic ones describe its tabs.
"""
from __future__ import annotations

import html
import os
import sys
import time

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QDialog, QHBoxLayout, QLineEdit, QListWidget,
                               QStackedWidget, QTextBrowser, QVBoxLayout, QWidget)

from helio_qt.widgets import button, label


def _start():
    return [
        ("p", "Helio works out the shape of a surface from ordinary photographs: the camera "
              "stays still, the light moves. Where the surface tilts toward a lamp it "
              "brightens; from those differences Helio finds which way every point faces, "
              "then rebuilds the relief."),
        ("h", "Five steps, left to right"),
        ("pre", "  1  Capture   photographs and where the lights were      ⌘1\n"
                "  2  Process   solve the surface                          ⌘2\n"
                "  3  Inspect   look at it under any light, in any view    ⌘3\n"
                "  4  Measure   scale, distances, profiles, roughness     ⌘4\n"
                "  5  Publish   images, 3D, web viewers, archives         ⌘5\n"),
        ("p", "The bar at the bottom always shows the next thing to do. ⌘K finds any tool by "
              "name; ⌘/ lists every keyboard shortcut. Hover over anything to see what it does."),
        ("h", "Settings bars"),
        ("pre", "  Drag sideways        change it (the bar's width is its whole range)\n"
                "  ⇧ while dragging     ten times finer\n"
                "  Click, no drag       type the number\n"
                "  Double-click         back to the default          ⌫ does the same\n"
                "  ← →                  step it (⇧ finer, ⌥ coarser); Home and End go to the ends\n"
                "  Scroll wheel         scrolls the panel, never changes a setting\n"),
        ("h", "What you need"),
        ("pre", "  At least 3 photographs   8 is comfortable, more is better\n"
                "  The camera never moves   only the light moves\n"
                "  The same exposure        for every shot\n"
                "  A shiny ball in frame    optional, but it removes all guesswork\n"),
    ]


def _capture():
    return [
        ("p", "To turn brightness into shape, Helio has to know where each lamp was. "
              "Capture offers three ways."),
        ("pre", "  Chrome ball       found automatically; drag its outline, then Fit to rim\n"
                "  Light file        File › Import Light Positions (.lp, .dome, .hjson, .csv)\n"
                "  Place by hand     Capture › Place Lights by Hand: drag each light on the dome\n"),
        ("p", "Helio compares the frames for camera movement before calibrating. Under half a "
              "pixel passes silently; more, and it names the frame and the distance."),
        ("h", "A grey card for hand-held flash"),
        ("p", "Carrying a flash round the object by hand is the easy way to light it, but no "
              "two flashes are equally strong: nearer is brighter, and on real captures the "
              "strongest frame was over twice the weakest. The solve reads the brighter frames "
              "as the surface turning towards them, so a flat object comes out tilted."),
        ("p", "Place a flat, matte grey card beside the object for better results: a QPcard, "
              "or a ColorChecker's grey patch, lying flat, facing the camera and lit by every "
              "flash. Helio finds it when you open the photographs and, in Process › Light "
              "and flash, evens out each flash's strength with it (Grey patch). On six "
              "bindings this took the cards from 7-18° of false tilt to 1-3°. If it is not "
              "found, Mark the grey patch lets you draw a box over it. With lamps fixed in "
              "place the flashes are steady and the card matters less."),
        ("p", "Better still, mark flat areas (Light and flash › Flat areas): the bare board "
              "round the object, on both sides. They measure how each flash falls off across "
              "the frame, not only how strong it was. Compare corrections solves small with "
              "each correction on its own, in seconds, and shows how flat the marked areas "
              "come out."),
        ("h", "Object type"),
        ("p", "Capture › Capture Details… starts with the object type: coin, medal or token; "
              "seal or matrix; engraved gem; inscription; manuscript, papyrus or page; binding; "
              "small find. Helio then sets a solver for it, puts its views first in Inspect (For "
              "this object), offers its exports first in Publish and adds its own fields: a "
              "coin's weight, diameter and die axis, a seal's legend, an inscription's findspot "
              "and letter height, a page's shelfmark and folio. Nothing is hidden, and every "
              "choice can still be changed. General changes nothing."),
        ("h", "Daylight and room light"),
        ("p", "Light that reaches every frame alike, outdoors or in a lit room, tells the solve "
              "nothing about direction and makes the relief come out shallower. Take one more "
              "photograph with the lamps off, everything else unchanged, and choose it with "
              "Capture › Add Lights-Off Frame… (or in Process › Light and flash). Its light is "
              "taken out of every frame before the solve, and Process says how strong it was."),
        ("h", "Two chrome balls"),
        ("p", "With a ball on each side of the object, each light is the mean of the two balls' "
              "readings: closer to the light the object itself saw, when the lamps are near. "
              "Both are used only when both are found in at least three frames in four, they "
              "agree to within 35° (median) and their outlines do not overlap; the notes say "
              "how far they disagreed. Ball 1 and Ball 2 in Capture show and adjust each."),
        ("p", "Photo tone (Process): camera JPEGs and TIFFs store gamma-encoded values. "
              "Automatic (the default) decodes them to light, by the file's own curve (Adobe RGB "
              "(1998), sRGB or its ICC profile): solved as stored, fine relief such as a punch's "
              "letter comes out at about 60% of its depth. Uneven lighting then shows as a broad "
              "tilt, which Flattened in the 3D relief or a white-board light field takes out; "
              "“As stored” keeps the old result. Choose “linear already” for raw-developed linear "
              "files. RAW files are always read as linear."),
    ]


def _process():
    return [
        ("p", "Quick look solves a small copy in seconds and scores it; Try a few compares "
              "settings side by side. Keep the best, then Process solves the full surface."),
        ("h", "Choosing a solver"),
        ("p", "Linear Least Squares suits matte paper, parchment, plaster and stone. For "
              "surfaces that shine, Sparse-Outlier LS (RPCA) sets aside the few readings a "
              "matte surface could not give, a highlight on gold or a cast shadow, however "
              "bright, and solves from the rest; its Outlier threshold sets how far a reading "
              "may stray. IRLS and Robust LS turn highlights down rather than setting them aside."),
        ("h", "Gold and metal"),
        ("p", "Metal-aware solve (Process › Gold and metal) keeps your solver for paper, "
              "leather and cloth, and solves only the places that shine again with the metal "
              "solver, blended at their edges: gold tooling, embroidered gold, gilt lines, "
              "clasps. It finds them by readings a matte surface could not give, or by how far "
              "they flare above the matte about them under one lamp. Sensitivity 0.3 suits most "
              "bindings; higher starts to mark matte areas close to a lamp."),
        ("h", "Unequal lamps"),
        ("p", "Light and flash › Unequal lamps finds each lamp's strength from the photographs "
              "themselves and evens the frames, with nothing to mark: LEDs on a dome or flashes "
              "rarely match, and a brighter one reads as the surface turning towards it. Flat "
              "areas and the grey patch measure the same thing on a known reference and are "
              "used instead when they are on."),
        ("h", "A region at full size"),
        ("p", "Process › Process a Region solves the box you draw from the photographs at full "
              "size, even when the whole capture was solved at a smaller working size, keeping "
              "only that region in memory."),
    ]


def _inspect():
    return [
        ("pre", "  ← →        previous / next view          D   depth map\n"
                "  C          clay (no colour)              A   sweep the light\n"
                "  H          histogram                     S   save the current view\n"
                "  ⌘\\         split view                    ⌘0  zoom to fit\n"
                "  G          save the light sweep (GIF or MP4; in 3D, a light-sweep video)\n"),
        ("p", "Drag on the light dome to move the light. The loupe shows full-resolution "
              "detail under the pointer; click to hold it. Annotate draws arrows, outlines, "
              "numbers and measured lines on layers, for the scholar set."),
        ("p", "Save ▾, beside Sweep: the light sweep at the height and length you choose, "
              "this view at full resolution, or the loupe's region. Files are named after "
              "the object: its title in Capture details, else the session or the photographs' "
              "shared name. The capture details go into each saved picture."),
        ("p", "View › 3D Model opens the surface in 3D: standard, flattened, inverted or as an "
              "elevation heatmap, with turntable and flyover recordings."),
        ("h", "Judging colour on screen"),
        ("p", "The bar over the image sets its surround. A dark surround makes a picture look "
              "brighter, less colourful and flatter than it is. Colour assessment follows ISO "
              "12646: a mid-grey surround, a thin white border, the image held well inside."),
    ]


def _relief():
    from helio_qt.relief3d import KEYS
    rows = []
    for group, keys in KEYS:
        rows.append(f"  {group}")
        rows += [f"    {k:<12} {what}" for k, what in keys]
    return [
        ("p", "The surface in 3D inside the window, beside the image (⌥3) or instead of it "
              "(⌥2); ⌥1 is the image alone. It is lit by the same light as the 2D view: move "
              "one and the other follows."),
        ("h", "What it shows"),
        ("pre", "  Exaggeration 1      the true shape: the height map scaled until its slopes\n"
                "                      agree with the normals.  At 4, every slope is 4 × steeper\n"
                "  Shading             by the photometric normals, at the preview's full size,\n"
                "                      so detail the photographs recorded is all there\n"
                "  Surfaces            colour, clay, roughness, trust (where the solve is\n"
                "                      unsure), height as colour, or the 2D view draped on it\n"
                "  Measure             two points: across, along the surface and the height\n"
                "                      between them; mm once a scale is set\n"
                "  Section             S, then drag a line A–B across the relief. Its profile\n"
                "                      opens under the 3D view: the cut, real axes, the\n"
                "                      vertical exaggeration, and each groove and ridge it\n"
                "                      crosses with its depth and width. Drag A or B to move\n"
                "                      it (Shift keeps it straight); move along the chart and a\n"
                "                      marker follows on the relief; click two points on the\n"
                "                      chart to measure between them. Save writes the profile\n"
                "                      and its features as CSV and the chart as SVG and PNG;\n"
                "                      the publication plate uses this line for its section\n"
                "  Mesh                Show the mesh draws its edges over the relief; Wire\n"
                "                      only hides the surface and leaves the wire. X steps\n"
                "                      through off, over the surface and wire only\n"),
        ("h", "The high-detail 3D window"),
        ("p", "View › 3D Model › Open High-Detail 3D Window: the same view built from up to 9 million "
              "points, shaded by the normals at 4.5K, 8K or the solve's full size. Its geometry is "
              "adaptive: the mesh is divided on the graphics card wherever it is close, up to 64 "
              "times, and lifted by the fine heights from the full-size depth, so it stays tight "
              "however close you come and costs no more far off (Finest, Fine or Fast says how "
              "tight). Its Expert tools panel adds what the relief in Inspect does not have: its "
              "own choice of shading by the normals (smooth by the heights when off), the wire on "
              "its own, an orthographic camera for plans and elevations, a headlight that turns "
              "with you, a slow spin, Exaggerated Shading and an eight-direction hillshade (live "
              "as the light moves), the light by number or raking from a corner, a second light, "
              "Fine detail, Cavity, Specular enhancement as in RTI, colour by slope or curvature, "
              "and saved views to come back to."),
        ("h", "Keys, while the 3D view has focus"),
        ("pre", "\n".join(rows) + "\n"),
        ("p", "The key card on the view lists these; ? hides it. Without OpenGL 4.1 (a remote "
              "session) the view says so; View › 3D Model opens the separate 3D window."),
    ]


def _quick():
    return [
        ("p", "File › Quick Export, and the Export panel in Inspect: the outputs most often "
              "wanted, one click each, with a short sheet of choices."),
        ("pre", "  RTI web package   ⌘⇧W   a folder that opens in any browser, offline:\n"
                "                          deep-zoom relighting at full size, a one-file quick\n"
                "                          view, Relight / OpenLIME data, the lights (.lp), a\n"
                "                          read-me, embed code, a QR code and launchers\n"
                "  Send to QGIS      ⌘⇧G   the GIS package (levelled heights, albedo, normals,\n"
                "                          uncertainty, metal mask) opened in QGIS: through\n"
                "                          the Helio Relief plugin when it is installed,\n"
                "                          else as plain layers\n"
                "  USDZ for AR       ⌘⇧U   the relief for an iPhone or iPad: life size once a\n"
                "                          scale is set; on a table or a wall\n"
                "  This view               the view on screen, at full resolution\n"),
        ("p", "File › Share › QR Code shows the code on screen to scan. Share a web export "
              "from this Mac opens it on a phone on the same Wi-Fi."),
        ("p", "Heights are in millimetres only when a height scale has been measured; "
              "otherwise every package says they are relative."),
    ]


def _colour():
    return [
        ("p", "Measure › Check Colour Against a ColorChecker reads a 24-patch ColorChecker "
              "in a photograph and compares each patch with X-Rite's published values (D50), "
              "as CIEDE2000 (ΔE₀₀)."),
        ("pre", "  About 2 or under    within the Met's working practice\n"
                "  2 to 3              close: a careful eye may see it side by side\n"
                "  Over 3              noticeable: profile the camera, or check the light\n"
                "                      and the white balance\n"),
        ("p", "Cyan lies outside sRGB, so no sRGB photograph matches it exactly; it is shown "
              "but read with that in mind. The sheet saves every patch as CSV, and the result "
              "goes on the publication plate."),
    ]


def _measure():
    return [
        ("pre", "  Scale            from a ruler (ticks found for you) or a line of known length\n"
                "  Height scale     from a feature of known height\n"
                "  Distance         a line, in mm once the scale is set\n"
                "  Height profile   a section with its profile roughness (Ra, Rq, Rz)\n"
                "  Area and volume  above and below the region's base plane\n"
                "  Statistics       ISO 25178 areal parameters, plane levelled\n"
                "  Compare regions  two areas side by side\n"
                "  Change           against an earlier scan of the same object\n"),
        ("p", "Values are in millimetres only when a scale has been measured; otherwise "
              "they are in pixels or relative units and say so."),
    ]


def _research():
    return [
        ("pre", "  Reflectance unmixing   materials that answer the lights differently\n"
                "  Micro-features         pinholes, pricking, stylus marks\n"
                "  Toolpaths              scratches and incisions traced as strokes\n"
                "  Inscription zones      where writing or decoration is likely\n"
                "  Register two images    overlay a tracing or another object\n"
                "  Two faces              recto and verso, obverse and reverse, matrix and\n"
                "                         impression; a coin's die axis\n"
                "  Legend unwrap          a legend round a seal, coin, stamp or volvelle\n"
                "                         straightened into a strip: circles, ovals,\n"
                "                         pointed ovals and free shapes you drag into place;\n"
                "                         several on a page found and numbered; read in\n"
                "                         relief, an even raking light or any view\n"
                "  Virtual impression     what a matrix or intaglio would stamp; raised letters\n"
                "  Line drawing           cuts and edges traced as an SVG drawing\n"
                "  Focus stacking         one sharp photograph per light from a bracket\n"),
        ("h", "Coins, seals and gems"),
        ("p", "Two Faces measures the die axis: turn each face until its design stands upright "
              "(photograph the reverse after turning the coin over left to right) and it gives "
              "the clock hours, and records them in Capture details. Register Two Images gives a "
              "relief match after registering two coins: 1 is identical. A high match is worth "
              "examining as a die link; it is a pointer, not a verdict, since wear and a "
              "different light lower it."),
        ("h", "Legends"),
        ("p", "Legend Unwrap unrolls the lettering round a seal, coin, medal, token, a round or "
              "oval library or ownership stamp, a binding's centrepiece or a volvelle into one "
              "strip. On a page with several seals or coins, Find objects rings and numbers "
              "each one: click the one to read, or choose it from the list, and Page goes back "
              "to pick another. Or select the object in Inspect first. Choose the "
              "shape (circle, oval, or pointed oval for the vesica of ecclesiastics' and women's "
              "seals), Find the edge, then Straighten: the outline is moved until the borders "
              "run flat. Drag on the picture to place it by hand: inside the outline moves it, "
              "its edge resizes it. A Free shape follows any outline (a lozenge centrepiece, a "
              "shield, a cartouche, a broken edge) through points you drag; double-click the "
              "edge to add a point, Option-click a point to remove it. Read worn or paper-covered impressions from the Relief, which shows "
              "the letters where the colour shows nothing, or the Even raking light, which "
              "turns with the legend so every letter is lit from the same side of itself. View "
              "on screen reads whatever Inspect shows now, its Colour or Clay layer included; "
              "Another view reads any view, in Colour, Clay or Roughness, with its settings. More "
              "gives a part arc for a fragment, Mirror for a matrix or die, and Whole face, "
              "which lays every ring of a volvelle, astrolabe or rota out as rows. Save writes "
              "the strip with a degree and millimetre scale (with the object above it, when "
              "ticked) and a .json record of the outline and settings; the legend you type and the diameter (or height × width) go to "
              "Capture details."),
        ("h", "AI"),
        ("p", "Research › AI asks a language model to identify the object, draft a condition "
              "report or a figure caption. It is a starting point to check, never a "
              "measurement. Ollama and LM Studio run on this computer: nothing leaves it "
              "and no key is needed. API keys are kept for the session only."),
    ]


def _publish():
    return [
        ("p", "Publish lists every output by what it is for: images, surface maps, 3D models, "
              "web viewers, reports, archives, GIS. Tick what you need, or start from a "
              "preset, then Export. File › Export Facsimile makes a relief for 3D printing, "
              "elevated printing or CNC. File › Batch Process Folders runs a whole folder of "
              "objects overnight."),
        ("p", "Every export records how it was made (the processing record), so a figure "
              "can be traced back to its photographs and settings."),
        ("h", "Records for each discipline"),
        ("p", "Coin record (Nomisma) writes a coin's details as Nomisma properties, in CSV and "
              "JSON-LD. EpiDoc record (TEI) starts an EpiDoc file from the capture details, with "
              "the images as its facsimile and an empty edition to transcribe into. IIIF image "
              "manifest puts the colour and the object's views in Mirador or the Universal "
              "Viewer; its read-me says which address to change once the folder is online. Line "
              "drawing (SVG) is a start for a published drawing, in millimetres once a scale is "
              "set. Each uses only what you typed: nothing is looked up or invented."),
        ("h", "The next object"),
        ("p", "Finished? File › Start Next Capture (⇧⌘N), or the button under the export "
              "queue. If this one has work not yet saved, Helio asks to save its session "
              "first; then you choose the next folder and it starts clean: no lights, scale, "
              "measurements or annotations from the last one. The solver, views and 3D "
              "settings stay as you had them. Dropping a folder on the window does the same."),
        ("h", "The publication plate (A3)"),
        ("p", "One designed page that stands on its own: the object with a scale bar, north "
              "and a section line; where the lights were; the normals with their colour key; "
              "raking light, exaggerated shading, multi-scale relief and curvature; height; "
              "the relief in 3D (the 3D view's own picture when it is open); how sure the "
              "solve is; the section with its roughness; and the method, the quality figures "
              "and the credit. Every panel is lettered and captioned. PDF (text stays text) "
              "and a 300 dpi PNG."),
        ("h", "Models: made for"),
        ("p", "The 3D model panel's Made for sets the mesh density and the files together: "
              "Blender and MeshLab, close study, the web and Sketchfab, phones and AR, the "
              "archive, or 3D printing. Texture size (Mesh, 2K, 4K, 8K, Full) takes the colour "
              "and normal pictures from the full photographs whatever the mesh density, so a "
              "light mesh can still look sharp; JPEG textures are about a quarter of the size, "
              "and the normal map always stays PNG. Closed solid with a base, and Colour on the "
              "model, are there for printing and plain shape."),
        ("h", "The publication sheet"),
        ("p", "On the sheet, in Publish, lists everything added: the sections with their letters "
              "and copies, the 3D pictures, pictures from files and the tools' results (every "
              "Research tool has Add to sheet at the bottom of its window). Select a section to "
              "see it bold on the picture in Inspect, where the kept sections are drawn faintly "
              "with their letters so a new line is not drawn over one. Up, Down and Remove change "
              "the sheet; double-click a picture to change its caption. A line close to one "
              "already kept, or a 3D view taken twice, asks whether to replace it, keep both or "
              "leave it; one kept before that is marked ⚠ in the list. The line still drawn goes "
              "on the sheet only when no section is kept, or when you Keep it."),
        ("p", "A3, A4 or US Letter, for print (300 dpi) or for screen and email (150 dpi, a much "
              "smaller PDF), with a lossless PNG of each page if you want one."),
    ]


def _tips():
    return [
        ("h", "When something looks wrong"),
        ("pre", "  Flat and grey             shallow relief: try a relief view and a\n"
                "                            lower light (raking)\n"
                "  A flat object looks       hand-held flash: put a grey card in the shot\n"
                "  tilted                    (Process evens the flash with it); lamps at\n"
                "                            different distances: Process › Flat object\n"
                "  Lit from the wrong side   the light positions are probably wrong\n"
                "  Smeared or doubled        something moved between frames\n"
                "  Slow on a big capture     process a region (Process › Process a Region)\n"),
        ("h", "If something is broken"),
        ("p", "Help › Report a Problem writes a file describing your Mac, the project and the "
              "tail of the diagnostic log, then offers to email it."),
    ]


def sections() -> list[tuple[str, list]]:
    import helio_help as H
    from helio.domain.registry import ENGINE_BLURBS, view_name
    terms = [(k, v.split("  Region ")[0] if k == "pre" else v) for k, v in H._TERMS_BLOCKS]
    engines = [("p", "Each view answers a different question about the surface.")]
    engines.append(("pre", "\n".join(f"  {view_name(n)}\n      {ENGINE_BLURBS[n]}\n"
                                            for n in sorted(ENGINE_BLURBS, key=view_name))))
    return [("Start here", _start()), ("Capture: the lights", _capture()),
            ("Process: solving", _process()), ("Inspect: views", _inspect()), ("Inspect: the 3D relief", _relief()),
            ("Measure", _measure()), ("Colour: checking it", _colour()),
            ("Research and AI", _research()), ("Publish", _publish()),
            ("Quick exports", _quick()), ("What the words mean", terms),
            ("Methods and references", H._methods()), ("Every view", engines),
            ("Tips and problems", _tips())]


def to_html(blocks) -> str:
    out = []
    for kind, text in blocks:
        if kind == "img":
            continue                       # the classic Help's screenshots are of the classic window
        t = html.escape(text)
        if kind == "h":
            out.append(f"<h3>{t}</h3>")
        elif kind == "pre":
            out.append(f"<pre style='font-family:Menlo,monospace;font-size:11pt'>{t}</pre>")
        else:
            out.append("<p>" + t.replace("\n\n", "</p><p>").replace("\n", "<br>") + "</p>")
    return "\n".join(out)


class HelpDialog(QDialog):
    def __init__(self, win, start: str = ""):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Helio Help")
        self.setMinimumSize(900, 620)
        self.secs = sections()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 16)
        left = QVBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search the help")
        self.search.setAccessibleName("Search the help")
        self.search.textChanged.connect(self.filter)
        left.addWidget(self.search)
        self.list = QListWidget()
        self.list.setFixedWidth(220)
        self.list.addItems([t for t, _b in self.secs])
        self.list.currentRowChanged.connect(self.show_section)
        left.addWidget(self.list, 1)
        row = QHBoxLayout()
        b = button("Getting started", "quiet")
        b.clicked.connect(win.show_welcome)
        row.addWidget(b)
        b = button("Report a problem", "quiet")
        b.clicked.connect(win.report_problem)
        row.addWidget(b)
        left.addLayout(row)
        lay.addLayout(left)
        self.view = QTextBrowser()
        self.view.setOpenExternalLinks(True)
        self.view.setAccessibleName("Help text")
        lay.addWidget(self.view, 1)
        i = next((k for k, (t, _b) in enumerate(self.secs) if t == start), 0)
        self.list.setCurrentRow(i)

    def show_section(self, i: int):
        if 0 <= i < len(self.secs):
            t, blocks = self.secs[i]
            self.view.setHtml(f"<h2>{html.escape(t)}</h2>" + to_html(blocks))

    def filter(self, text: str):
        q = text.strip().lower()
        first = -1
        for i, (t, blocks) in enumerate(self.secs):
            hit = not q or q in t.lower() or any(q in str(b).lower() for _k, b in blocks)
            self.list.item(i).setHidden(not hit)
            if hit and first < 0:
                first = i
        if first >= 0 and self.list.item(self.list.currentRow()).isHidden():
            self.list.setCurrentRow(first)


WELCOME = (
    ("Welcome to Helio", "The shape of a surface, from photographs.",
     ("Photograph the object several times; only the light moves.",
      "Helio works out which way every point faces, then the relief.",
      "Light it from any direction long after the object is put away.")),
    ("1  Capture", "Photographs and where the lights were.",
     ("Drag a folder of photographs onto the window.",
      "A chrome ball in frame is found for you; or import a light file;",
      "or place the lights by hand on the dome.")),
    ("2  Process", "Solve the surface.",
     ("Press Process (or ⌘R).",
      "The defaults suit most captures; every setting explains itself on hover.")),
    ("3  Inspect and 4  Measure", "Look, then quantify.",
     ("Arrow keys step through the views; drag the dome to move the light.",
      "Set the scale once and every measurement is in millimetres.")),
    ("5  Publish", "Everything you might hand on.",
     ("Images, 3D models, web viewers, reports and archive packages.",
      "Each records how it was made.")),
)


class WelcomeDialog(QDialog):
    """Getting Started: five short pages (the classic wizard had eleven)."""

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Getting started")
        self.setMinimumSize(560, 360)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(28, 24, 28, 18)
        self.stack = QStackedWidget()
        for title, sub, lines in WELCOME:
            pg = QWidget()
            pl = QVBoxLayout(pg)
            pl.addWidget(label(title, "title"))
            pl.addWidget(label(sub, "secondary", wrap=True))
            for ln in lines:
                pl.addWidget(label("•  " + ln, "", wrap=True))
            pl.addStretch(1)
            self.stack.addWidget(pg)
        lay.addWidget(self.stack, 1)
        self.dots = label("", "caption")
        self.dots.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.dots)
        row = QHBoxLayout()
        self.chk = QCheckBox("Show at start")
        self.chk.setChecked(not bool(win.prefs.value("welcome_done", False, type=bool)))
        row.addWidget(self.chk)
        row.addStretch(1)
        self.btn_back = button("Back", "quiet")
        self.btn_back.clicked.connect(lambda: self.go(-1))
        self.btn_next = button("Next", "primary")
        self.btn_next.clicked.connect(lambda: self.go(1))
        row.addWidget(self.btn_back)
        row.addWidget(self.btn_next)
        lay.addLayout(row)
        self.go(0)

    def go(self, d: int):
        i = self.stack.currentIndex() + d
        if i >= self.stack.count():
            self.accept()
            return
        i = max(0, i)
        self.stack.setCurrentIndex(i)
        n = self.stack.count()
        self.dots.setText("  ".join("●" if k == i else "○" for k in range(n)))
        self.btn_back.setEnabled(i > 0)
        self.btn_next.setText("Start" if i == n - 1 else "Next")

    def done(self, r):
        self.win.prefs.setValue("welcome_done", not self.chk.isChecked())
        super().done(r)


def diagnostics(win) -> str:
    """The classic report (machine, build, log tail) with this project's facts."""
    import helio_help
    from helio_qt import __version__
    p = win.project
    base = helio_help.diagnostics(None, f"{__version__} (new interface)").split("\n")
    head, tail = base[:7], base[7:]
    r = p.result
    s = p.settings
    facts = [f"images    : {len(p.images)}"]
    if p.images:
        facts.append(f"folder    : {os.path.dirname(p.images[0])}")
    facts += [f"lights    : {p.light_source or 'none'}",
              f"solver    : {s.resolved_normal_engine()}", f"depth     : {s.depth_engine}",
              f"tone      : {s.photo_tone}",
              f"work size : {'automatic' if s.work_scale == 0 else f'{s.work_scale * 100:.0f}%'}"
              + (f" (this result {r.work_scale * 100:.0f}%)" if r is not None and r.work_scale < 1.0 else ""),
              f"solved    : {'yes ' + str((r.h, r.w)) if r is not None else 'no'}",
              f"view      : {win.view or ''}"]
    return "\n".join(head + facts + tail)


def write_report(win, path: str | None = None) -> str:
    path = path or os.path.expanduser("~/Desktop/Helio_report.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(diagnostics(win))
    return path


class ReportDialog(QDialog):
    def __init__(self, win, path: str | None = None):
        import helio_help
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Report a problem")
        self.setMinimumWidth(500)
        try:
            self.path = write_report(win, path)
        except OSError:
            self.path = ""
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 16)
        lay.addWidget(label("Report a problem", "title"))
        lay.addWidget(label(
            (f"A report on your Mac, this project and the end of the diagnostic log is on your "
             f"Desktop: {os.path.basename(self.path)}. Please attach it: it usually holds the "
             f"actual error.") if self.path else "The report could not be written.",
            "secondary", wrap=True))
        row = QHBoxLayout()
        close = button("Close", "quiet")
        close.clicked.connect(self.reject)
        row.addWidget(close)
        row.addStretch(1)
        if self.path:
            b = button("Show the file", "outline")
            b.clicked.connect(self.reveal)
            row.addWidget(b)
        b = button("Open the form" if helio_help.SUPPORT_URL else "Open my email", "primary")
        b.clicked.connect(self.send)
        row.addWidget(b)
        lay.addLayout(row)

    def reveal(self):
        import subprocess
        if sys.platform == "darwin":
            subprocess.Popen(["open", "-R", self.path])

    def send(self):
        import helio_help
        from urllib.parse import quote
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        from helio_qt import __version__
        url = helio_help.SUPPORT_URL or (f"mailto:{helio_help.SUPPORT_EMAIL}?subject="
                                         + quote(f"Helio {__version__} problem report"))
        QDesktopServices.openUrl(QUrl(url))


class CrashNote:
    """Uncaught errors: logged, and a note offering Report a Problem (at
    most one every 30 s, so a repeating error cannot flood the screen)."""

    def __init__(self, win):
        self.win, self.last = win, 0.0
        self.prev = sys.excepthook
        sys.excepthook = self.hook

    def hook(self, etype, value, tb):
        import logging
        logging.getLogger("helio_qt").error("uncaught", exc_info=(etype, value, tb))
        if time.time() - self.last < 30:
            return
        self.last = time.time()
        from PySide6.QtWidgets import QMessageBox
        box = QMessageBox(QMessageBox.Icon.Warning, "Something went wrong",
                          f"{etype.__name__}: {value}\n\nYour work is not lost. A report helps fix it.",
                          QMessageBox.StandardButton.Close, self.win)
        rep = box.addButton("Report a Problem…", QMessageBox.ButtonRole.ActionRole)
        box.setModal(False)
        box.buttonClicked.connect(lambda b: self.win.report_problem() if b is rep else None)
        box.show()


def setup_logging():
    """The diagnostic log the problem report reads (the classic's file)."""
    import logging
    d = os.path.expanduser("~/Library/Logs/HELIO_Pro")
    try:
        os.makedirs(d, exist_ok=True)
        h = logging.FileHandler(os.path.join(d, "helio_pro_diagnostic.log"), encoding="utf-8")
    except OSError:
        return
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(h)
    if root.level > logging.INFO or root.level == logging.NOTSET:
        root.setLevel(logging.INFO)
