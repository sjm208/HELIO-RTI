"""What every command does, in a sentence or two: shown when the pointer rests
on it in a menu, in the ⌘K palette and on its buttons.

Kept apart from helio_qt.actions so the registry stays a list of commands; a
test (tests/test_qt_menus.py) checks every command has a tip here.  Write what it
does and when to use it, in the words the rest of Helio uses.
"""
from __future__ import annotations

TIPS: dict[str, str] = {
    # ── File ──────────────────────────────────────────────────────────
    "file.new": "Close this capture and start an empty project. The rig's choices (solver, views, "
                "settings) stay.",
    "file.open": "Open a saved session (.helio): its photographs, lights, scale, settings and notes.",
    "file.import": "Choose the photographs of one capture (or drop a folder anywhere in the window).",
    "file.next": "Save this capture if you wish, then start the next object on the same rig: lights, "
                 "scale and settings carried over.",
    "file.import_lp": "Read light directions from a .lp file (RTIBuilder, RelightLab, a dome). Lights "
                      "pair with the photographs by name, else in shot order.",
    "file.import_rti": "Open an RTIBuilder or RelightLab project folder: its photographs and its .lp "
                       "lights together.",
    "file.save": "Save this capture as a session (.helio) beside the photographs.",
    "file.save_as": "Save this capture as a session under a new name or place.",
    "file.export": "Go to Publish: every export, grouped by what it is for.",
    "export.rti_web": "A relightable web viewer of this capture (HSH or PTM), ready to put online or "
                      "open offline.",
    "export.rti_combined": "One web page to relight the image with the mouse: colour, specular, "
                           "normals and the PCA layers.",
    "export.rti_3d": "One web page with two tabs: relight the image, or turn a light 3D model of it.",
    "export.hf_viewer": "A web 3D viewer of the fine surface detail, with the colour, for close looking.",
    "export.qgis": "Send the height map and views to QGIS (the Helio Relief plugin) as georeferenced "
                   "layers.",
    "export.usdz": "A USDZ model for AR Quick Look on iPhone and iPad, checked against Apple's AR rules.",
    "export.view": "The current view at the photographs' full resolution, as a TIFF.",
    "file.export_now": "Run the exports ticked in Publish, into the export folder.",
    "file.reveal": "Show the last export's folder in the Finder.",
    "file.facsimile": "Files for a physical reproduction: 3D printing, elevated printing, CNC, plates "
                      "and seals.",
    "share.sketchfab": "Upload the textured 3D model to your Sketchfab account.",
    "share.qr": "A QR code for a viewer a phone can open: on screen to scan, and saved on a card "
                "naming the object, its institution and scale.",
    "share.embed": "The HTML to put a hosted web viewer in another page.",
    "share.manifest": "A checksum list of an export folder, so anyone can later prove the files are "
                      "unchanged.",
    "share.verify": "Check an export folder against its manifest: lists any file changed or missing.",
    "file.print_targets": "Print a calibration target as a PDF at any paper size: the lens chessboard, "
                          "detailed scale bars to cut out, or a grid to check the camera is square.",
    "file.batch": "Process many capture folders in one go, with the same settings and exports.",
    "app.quit": "Quit Helio (it asks first if there is unsaved work).",
    # ── Edit ──────────────────────────────────────────────────────────
    "edit.undo": "Undo the last change to how you are looking: the view, its settings and light, the "
                 "layer, the solver or depth method.",
    "edit.redo": "Redo what Undo took back.",
    "edit.copy": "Copy the photograph shown (the current frame) to the clipboard.",
    "app.settings": "Helio's preferences: appearance, export defaults, names and licence, API tokens.",
    # ── Capture ───────────────────────────────────────────────────────
    "capture.sphere": "Find the chrome ball in the photographs and read each frame's light direction "
                      "from its highlight.",
    "capture.mark_ball": "Draw round the chrome ball yourself when it is not found (a partly hidden or "
                         "dark ball).",
    "capture.outline": "Nudge the ball's outline when it is a little off: every light is read from it.",
    "capture.dome": "Place the lights by hand on the dome, when there is no ball and no .lp file.",
    "capture.details": "Who, when, what and with which camera: written into every export's metadata "
                       "and paradata.",
    "capture.drift": "Check whether the camera or object moved between frames (a moving frame blurs "
                     "every result).",
    "capture.align": "Line up frames that shifted a little (a nudged tripod) before the solve.",
    "capture.ambient": "A photograph with the lamps off: the room's light is subtracted from every frame.",
    "capture.ambient_clear": "Solve without the lights-off frame.",
    "capture.backlit": "A transmitted-light photograph (lit from behind): kept as an extra view for "
                       "watermarks and repairs.",
    "capture.alignment": "Photograph the printed chessboard lying flat on the baseboard: how far the "
                         "camera is off square, which way, and the scale on the baseboard.",
    "capture.save_cal": "Save this capture's lights as a calibration (.lp and .hjson) for another capture "
                        "on the same rig.",
    "capture.load_cal": "Load lights from a saved calibration onto this capture.",
    # ── Process ───────────────────────────────────────────────────────
    "process.run": "Solve the surface: normals, depth and every view, with the settings in Process.",
    "process.cancel": "Stop the processing or export that is running.",
    "process.recalc": "Integrate the depth again from the normals (after changing the depth method or "
                      "smoothing), without solving again.",
    "process.reprocess": "Solve a region you select at the photographs' full size: more detail on a "
                         "large capture.",
    "process.flat": "The older flat-field correction: a blank board under the same lights, its false tilt "
                    "subtracted from the object's normals after the solve. The white-board light field "
                    "(Process › Precision) uses the same photographs and is more exact.",
    "process.measurement": "Precision: the solve's choices for measuring, at once: the solver that "
                           "measured best on the DiLiGenT benchmark, free-boundary depth, the tone "
                           "decoded by the file's curve, and every look-only adjustment off.",
    "process.light_field": "Precision: one photograph of a matte white board under each light, from the "
                           "rig's fixed positions. Measures every light's strength and fall-off and "
                           "evens every frame by it.",
    "process.light_field_clear": "Precision: solve without the white-board light field.",
    "process.lens": "Precision: 10-20 photographs of a printed chessboard across the frame. Measures the "
                    "lens's distortion and takes it out of every frame.",
    "process.lens_clear": "Precision: solve without the lens correction.",
    "process.print_chessboard": "Precision: print the lens-calibration chessboard (9 x 6 inner corners) "
                                "at A4 to A1, Letter or Tabloid. Any laser print, mounted flat.",
    "process.refine": "Make the normals agree with a single surface (integrability): less noise, a "
                      "little less fine detail.",
    "qc.report": "How precise the normals are (from each pixel's own lights) and how much the surface "
                 "varies: a rating for measurement.",
    "qc.bias": "Splits the normals into their broad shape and the fine detail: light-calibration "
               "error sits in the broad shape. Optionally against a reference normal map.",
    "qc.compare": "Compare the normals with a reference normal map (photogrammetry, a scanner): the "
                  "angular error, mapped.",
    "qc.disagreement": "Integrate the depth two ways and map where they disagree: where the heights are "
                       "least sure.",
    "qc.uncertainty": "Save the fit residual and the normals' uncertainty as GeoTIFF maps.",
    "qc.dome3d": "The lights drawn on a 3D dome: see gaps in the coverage at a glance.",
    "process.settings": "Every processing setting in one dialog.",
    # ── Measure ───────────────────────────────────────────────────────
    "measure.scale": "Draw a line along something of known length (a ruler, a scale bar) and give its "
                     "length: millimetres per pixel.",
    "measure.apply_scale": "Use the line drawn and the length given as the scale.",
    "measure.zscale": "Draw from the base to the top of something of known height: heights in "
                      "millimetres.",
    "measure.apply_zscale": "Use the line drawn and the height given as the height scale.",
    "measure.ruler": "Drag a box along a ruler's ticks: Helio finds their spacing; you give the unit.",
    "measure.clear_zscale": "Heights back to pixels of height.",
    "measure.distance": "Drag between two points: the distance in millimetres (with a scale).",
    "measure.profile": "Drag across the surface: its height profile, ISO 4287 figures, and the roughness "
                       "and waviness the ISO 16610-21 Gaussian filter separates, with the "
                       "heights' ± where the lights allow.",
    "measure.volume": "Drag round a feature: its true surface area, and the material above and below "
                      "its base.",
    "measure.stats": "Drag over a surface: ISO 25178 areal roughness (Sa, Sq, Sz and more) and its "
                     "bearing curve.",
    "measure.compare": "Two regions side by side (a feature and its background): ISO 25178 figures "
                       "and their difference.",
    "measure.accuracy": "Precision: measure a gauge block's step or a flat plate shot on the same rig, "
                        "and see the error against its certified value.",
    "measure.zscale_lateral": "Precision: heights in millimetres from the lateral scale. For a "
                              "calibrated, linear capture; check it with the accuracy check.",
    "measure.print_scale": "Print detailed scale bars to cut out and lay beside the object: 20 mm in "
                           "half millimetres up to the longest the paper takes, an L scale, an inch rule.",
    "measure.colour": "Find a ColorChecker in a photograph and say how far its colours are off "
                      "(ΔE2000, ISO 19264).",
    "measure.load_reference": "Load an earlier capture of the same object: the depth to compare against.",
    "measure.change_map": "Map where the surface has gained or lost height since the reference.",
    "measure.change_report": "A report of the change since the reference: histograms and the areas "
                             "affected.",
    # ── Research ──────────────────────────────────────────────────────
    "research.open:unmix": "Splits how each pixel answers the lights into separate channels: materials, "
                           "layers and coatings that look alike in colour.",
    "research.open:micro": "Finds pinholes, stylus marks and small pits in the surface, with their depths.",
    "research.open:toolpaths": "Traces scratches, incisions and grooves as strokes.",
    "research.open:zones": "Marks the areas most likely to hold writing or worked decoration.",
    "research.open:register": "Line up two images: click the same features on both, and one is warped "
                              "onto the other.",
    "research.open:recto": "Link this face to the other face of the same object: a leaf's recto and "
                           "verso, a coin's obverse and reverse, a seal's matrix and impression.",
    "research.open:unwrap": "Unrolls the lettering round a seal, coin, stamp or centrepiece into a strip "
                           "that reads left to right.",
    "research.open:impression": "What a seal matrix or an intaglio would stamp, or cut letters standing "
                                "up as on a squeeze, under one raking light.",
    "research.open:linedraw": "Cut lines and raised edges traced as vector lines (SVG, in millimetres "
                             "with a scale), for an archaeological or epigraphic drawing.",
    "research.open:focus": "For coins, seals and gems at high magnification: one sharp photograph per "
                          "light from its focus bracket.",
    "ai.open:identify": "Ask a language model (online, or local with Ollama or LM Studio) what the "
                        "object may be, from the current view.",
    "ai.open:report": "Ask a language model to draft a condition report from the current view, for you "
                      "to check.",
    "ai.open:caption": "Ask a language model to draft a figure caption for the current view.",
    # ── View ──────────────────────────────────────────────────────────
    "view.ws:Capture": "Capture: the photographs and their lights.",
    "view.ws:Process": "Process: solve the surface, with every setting for it.",
    "view.ws:Inspect": "Inspect: look at the surface in every view, relight it, annotate it.",
    "view.ws:Measure": "Measure: scale, distances, heights, profiles, areas and roughness.",
    "view.ws:Publish": "Publish: every export, grouped by what it is for.",
    "view.next": "The next view in the list.",
    "view.prev": "The previous view in the list.",
    "view.depth": "The height map, light for high, dark for low.",
    "view.clay": "The surface as grey clay: shape only, no colour.",
    "view.white_balance": "Click or box a white or grey that should be neutral: the colour is balanced to "
                          "it (the colour only, nothing else).",
    "view.white_balance_reset": "The colour as photographed: no white balance.",
    "view.layer:Combined Colour": "The view lit over the photograph's colour.",
    "view.layer:Shading": "Shape only, as grey clay.",
    "view.layer:Roughness Fusion": "The view mixed with the surface's roughness map.",
    "view.layer:Confidence Overlay": "Marks pixels too dark or too bright in the photographs to trust.",
    "view.compare": "Two views side by side with a divider to drag between them.",
    "view.save_state": "Remember this view, its light and settings, to come back to (or to put on the "
                       "sheet).",
    "view.best_raking": "Find the low light that shows the most relief, and use it.",
    "view.sweep": "Turn the light slowly round the object, again and again.",
    "view.sweep_gif": "Save the light turning round the object as a GIF or MP4.",
    "view.histogram": "The picture's histogram: brightness and clipping.",
    "view.snapshot": "Save the current view at full resolution, as a TIFF.",
    "view.export_loupe": "Save the loupe's region at the photographs' full resolution.",
    "view.mode:image": "The relit picture in the centre.",
    "view.mode:3d": "The 3D relief in the centre: turn it, light it, section it.",
    "view.mode:both": "The picture and the 3D relief side by side.",
    "view.relief_picture": "Save the 3D view as a PNG.",
    "view.centre_picture": "Save what the centre shows (picture, 3D, or both) as a 4K PNG.",
    "view.relief_turntable": "Record the 3D relief turning, as a video.",
    "view.relief_flyover": "Record a flight over the 3D relief, as a video.",
    "view.relief_sweep": "Record the light turning over the 3D relief, as a video.",
    "view3d.open:standard": "Open the surface in the 3D viewer.",
    "view3d.open:flattened": "Open in 3D with the broad curve taken out (a warped page, a curved board), "
                             "so the detail lies flat.",
    "view3d.open:inverted": "Open in 3D inside out: a seal matrix or die as its impression.",
    "view3d.open:heatmap": "Open in 3D coloured by height.",
    "view3d.window": "The high-detail 3D window: full resolution, flyovers and turntables.",
    "view3d.settings": "The 3D model's height, material, sharpening and export options.",
    "view.composite": "One picture made from all the lights at once (mean or brightest): shadows "
                      "cancelled.",
    "view.rotate_left": "Turn the view 90° anticlockwise (the photographs are unchanged).",
    "view.rotate_right": "Turn the view 90° clockwise (the photographs are unchanged).",
    "view.rotate_left_fine": "Turn the view 1° anticlockwise, to straighten it.",
    "view.rotate_right_fine": "Turn the view 1° clockwise, to straighten it.",
    "view.rotate_reset": "The view as photographed.",
    "view.reset_transforms": "Undo every mirror, flip and inversion of the display.",
    "view.fit": "The whole picture in the window.",
    "view.zoom_in": "Closer.",
    "view.zoom_out": "Further away.",
    "view.sidebar": "Show or hide the panel on the left.",
    "view.inspector": "Show or hide the panel on the right.",
    "view.appearance:system": "Light or dark as the Mac is set.",
    "view.appearance:light": "A light window.",
    "view.appearance:dark": "A dark window.",
    "view.scheme:graphite": "Neutral greys, for judging colour (ISO 12646 viewing).",
    "view.scheme:nightdome": "Deep blue and parchment.",
    "window.minimize": "Put the window in the Dock.",
    "window.zoom": "Fill the screen with the window, or put it back.",
    # ── Help ──────────────────────────────────────────────────────────
    "help.help": "Helio's help: how each workspace and view works.",
    "help.search": "Find any command by name or by what it does (⌘K).",
    "help.shortcuts": "Every keyboard shortcut.",
    "help.welcome": "The first steps: import, find the lights, process, look, export.",
    "help.report": "A report on your Mac, this project and the end of the diagnostic log, to send with "
                   "a problem.",
    "app.about": "Helio's version, licences and credits.",
}


def tip(sid: str) -> str:
    """The command's description; solver and depth-method choices from their own blurbs."""
    if sid in TIPS:
        return TIPS[sid]
    from helio.domain.registry import DEPTH_BLURBS, SOLVER_BLURBS
    if sid.startswith("process.solver:"):
        return "Normal solver: " + SOLVER_BLURBS.get(sid.split(":", 1)[1], "")
    if sid.startswith("process.depth:"):
        return "Depth method: " + DEPTH_BLURBS.get(sid.split(":", 1)[1], "")
    return ""
