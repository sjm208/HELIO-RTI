"""Every command, once (UX-003): its menu, shortcut, and when it is available.

Menus, the ⌘K palette, toolbar buttons and the next-step bar all use these
QActions, so a command has one name, one shortcut and one reason for being
unavailable wherever it appears.  Menu order follows Apple's guidelines:
app, File, Edit, then the app's own menus, then View, Window, Help.  About,
Settings and Quit carry their native roles, so on a Mac Qt moves them into
the Helio Pro menu; elsewhere they stay in Help, Edit and File.
"""
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QActionGroup, QKeySequence

MENUS = ("File", "Edit", "Capture", "Process", "Measure", "Research", "View", "Window", "Help")


@dataclass(frozen=True)
class Spec:
    id: str
    text: str
    menu: str
    shortcut: str = ""
    needs: str | None = None     # Project.needs: None, images, lights, solvable, running,
                                 # solved, scale, classic
    handler: str = ""            # MainWindow method; empty = not in this preview yet
    keywords: str = ""
    sep_before: bool = False
    role: str = ""               # about / preferences / quit (native app-menu roles)
    checkable: bool = False
    group: str = ""              # exclusive group name
    image_focus: bool = False    # single-key shortcut: only while the image has focus


def _specs() -> list[Spec]:
    from helio.domain.registry import DEPTH_METHODS, NORMAL_SOLVERS
    S = Spec
    out = [
        S("file.new", "New Project", "File", "Ctrl+N", handler="new_project"),
        S("file.open", "Open…", "File", "Ctrl+O", handler="open_session",
          keywords="session .helio load"),
        S("file.import", "Import Images…", "File", "Ctrl+Shift+I", handler="import_images",
          sep_before=True, keywords="load folder photos capture"),
        S("file.next", "Start Next Capture…", "File", "Ctrl+Shift+N", handler="start_next_capture",
          keywords="next binding object another set new photographs restart again finished"),
        S("file.import_lp", "Import Light Positions…", "File", needs="images",
          handler="import_light_file", keywords=".lp dome lights calibration"),
        S("file.import_rti", "Import RTI Project Folder…", "File", handler="import_rti_project",
          keywords="rtibuilder relight .lp folder project"),
        S("file.save", "Save", "File", "Ctrl+S", needs="images", handler="save_session",
          sep_before=True, keywords="session .helio"),
        S("file.save_as", "Save As…", "File", "Ctrl+Shift+S", needs="images",
          handler="save_session_as", keywords="session .helio"),
        S("file.export", "Export…", "File", "Ctrl+E", needs="solved", handler="go_publish",
          sep_before=True, keywords="publish save output"),
        S("export.rti_web", "RTI Web Package…", "File/Quick Export", "Ctrl+Shift+W", needs="solved",
          handler="export_rti_web", keywords="web viewer relight online deep zoom openlime relight "
          "html share website offline package"),
        S("export.rti_combined", "Combined RTI Viewer…", "File/Quick Export", needs="solved",
          handler="export_rti_combined", keywords="web viewer relight one page browser share"),
        S("export.rti_3d", "Relight and 3D Viewer…", "File/Quick Export", needs="solved",
          handler="export_rti_3d", keywords="web viewer relight rti and 3d model tabs one page "
          "browser share presentation talk"),
        S("export.hf_viewer", "3D Viewer: Fine Detail…", "File/Quick Export", needs="solved",
          handler="export_hf_viewer", keywords="web viewer 3d relief fine detail normals colour albedo browser"),
        S("export.qgis", "Send to QGIS…", "File/Quick Export", "Ctrl+Shift+G", needs="solved",
          handler="send_to_qgis", keywords="gis dem geotiff helio relief plugin"),
        S("export.usdz", "USDZ for AR…", "File/Quick Export", "Ctrl+Shift+U", needs="solved",
          handler="export_usdz", keywords="openusd usd usda usdc ar quick look iphone ipad 3d"),
        S("file.export_now", "Export Selected Formats…", "File", needs="solved",
          handler="publish_run", keywords="publish queue save"),
        S("file.reveal", "Show Last Export in Finder", "File", needs="exported",
          handler="reveal_export", keywords="open folder"),
        S("file.facsimile", "Export Facsimile…", "File", needs="solved", handler="show_facsimile",
          keywords="3d print physical reproduction cnc elevated printing plate seal replica"),
        S("share.sketchfab", "Upload to Sketchfab…", "File/Share", needs="solved",
          handler="show_sketchfab", keywords="sketchfab 3d online upload model"),
        S("share.qr", "QR Code…", "File/Share", handler="show_qr",
          keywords="qr code label link catalogue"),
        S("share.embed", "Embed Snippet…", "File/Share", handler="show_embed",
          keywords="iframe html website embed"),
        S("share.manifest", "Write Folder Manifest…", "File/Share", handler="write_manifest",
          sep_before=True, keywords="checksum sha-256 deposit archive fixity"),
        S("share.verify", "Verify Folder Manifest…", "File/Share", handler="verify_manifest",
          keywords="checksum sha-256 deposit archive fixity verify"),
        S("file.print_targets", "Print Calibration Targets…", "File", handler="print_targets",
          sep_before=True, keywords="print pdf chessboard scale bar ruler grid calibration target a4 a3 "
          "a2 letter tabloid"),
        S("file.batch", "Batch Process Folders…", "File", handler="show_batch",
          keywords="overnight batch folders many objects queue"),
        S("app.quit", "Quit Helio", "File", "Ctrl+Q", handler="quit_app", role="quit",
          sep_before=True),

        S("edit.undo", "Undo", "Edit", "Ctrl+Z", needs="solved", handler="undo",
          keywords="back setting light view"),
        S("edit.redo", "Redo", "Edit", "Ctrl+Shift+Z", needs="solved", handler="redo"),
        S("edit.copy", "Copy Image", "Edit", "Ctrl+C", needs="images", handler="copy_image",
          sep_before=True),
        S("app.settings", "Settings…", "Edit", "Ctrl+,", handler="show_settings",
          role="preferences", sep_before=True, keywords="preferences appearance theme"),

        S("capture.sphere", "Find Chrome Sphere", "Capture", "Ctrl+Alt+F", needs="images",
          handler="find_sphere", keywords="lights calibrate ball highlight"),
        S("capture.mark_ball", "Mark the Chrome Ball by Hand…", "Capture", needs="images",
          handler="mark_ball", keywords="sphere ball dark manual draw circle not found outline"),
        S("capture.outline", "Adjust Sphere Outline…", "Capture", needs="sphere",
          handler="adjust_outline", keywords="sphere circle radius nudge fit rim"),
        S("capture.dome", "Place Lights by Hand", "Capture", needs="images",
          handler="place_by_hand", keywords="virtual dome calibrator manual shadows no sphere"),
        S("capture.no_ball", "No Ball: Find the Lights from the Photographs", "Capture",
          needs="images", handler="use_no_ball",
          keywords="no ball no sphere uncalibrated estimate lights pca without chrome ball"),
        S("capture.details", "Capture Details…", "Capture", needs="images",
          handler="show_capture_details", sep_before=True,
          keywords="metadata operator camera lens institution poi points of interest"),
        S("capture.drift", "Check Frames for Drift", "Capture", needs="images",
          handler="check_drift", sep_before=True, keywords="movement camera tripod shift"),
        S("capture.align", "Line Up Shifted Frames…", "Capture", needs="images",
          handler="alignment_report", keywords="align register phase correlation"),
        S("capture.ambient", "Add Lights-Off Frame…", "Capture", needs="images", handler="load_ambient",
          sep_before=True,
          keywords="ambient dark frame lamps off daylight room light outdoor field subtract stray"),
        S("capture.ambient_clear", "Remove Lights-Off Frame", "Capture", needs="images",
          handler="clear_ambient", keywords="ambient dark frame remove"),
        S("capture.backlit", "Add Backlit Image…", "Capture", needs="images",
          handler="load_backlit", keywords="transmissive transmitted light watermark"),
        S("capture.alignment", "Check Camera Alignment on a Chessboard…", "Capture",
          handler="check_alignment", sep_before=True,
          keywords="square level tilt keystone copy stand camera alignment chessboard scale"),
        S("capture.save_cal", "Save Calibration…", "Capture", needs="lights",
          handler="save_calibration", sep_before=True, keywords=".lp .hjson profile dome"),
        S("capture.load_cal", "Load Calibration…", "Capture", needs="images",
          handler="load_calibration", keywords=".lp .hjson .csv profile"),

        S("process.run", "Process", "Process", "Ctrl+R", needs="solvable", handler="process_run",
          keywords="solve normals depth run"),
        S("process.cancel", "Cancel", "Process", "Ctrl+.", needs="running",
          handler="process_cancel", keywords="stop processing export"),
        S("process.recalc", "Recalculate Depth", "Process", "Ctrl+Alt+R", needs="solved",
          handler="recalc_depth", keywords="integrate depth again smoothing"),
        S("process.reprocess", "Process a Region…", "Process", needs="solved",
          handler="process_region", keywords="crop area reprocess selected"),
        S("process.measurement", "Set Up for Measurement", "Process/Precision Measurement",
          needs="images", handler="set_up_for_measurement",
          keywords="precision measure metrology preset accurate rpca free boundary linear decode"),
        S("process.light_field", "Calibrate Lights on a White Board…", "Process/Precision Measurement",
          needs="lights", handler="calibrate_light_field",
          keywords="precision photometric stereo flat field white board fall-off falloff strength "
          "fixed rig calibration measurement"),
        S("process.light_field_clear", "White-Board Light Field Off", "Process/Precision Measurement",
          needs="images", handler="clear_light_field", keywords="precision flat field remove"),
        S("process.flat", "Load Flat Reference (Older Correction)…", "Process/Precision Measurement",
          needs="images", handler="load_flat", keywords="flat field correction blank white board"),
        S("process.lens", "Calibrate Lens from a Chessboard…", "Process/Precision Measurement",
          needs="images", handler="calibrate_lens",
          keywords="precision lens distortion chessboard checkerboard camera calibration barrel "
          "pincushion"),
        S("process.lens_clear", "Lens Correction Off", "Process/Precision Measurement", needs="images",
          handler="clear_lens", keywords="precision lens distortion remove"),
        S("process.print_chessboard", "Print a Calibration Chessboard…", "Process/Precision Measurement",
          handler="print_chessboard", keywords="print pdf chessboard checkerboard lens a4 a3"),
    ]
    from helio.domain.registry import MORE_SOLVERS
    # The main solvers first; the rest in a More Solvers submenu (as in Process).
    for name in sorted(NORMAL_SOLVERS, key=lambda n: n in MORE_SOLVERS):
        more = name in MORE_SOLVERS
        out.append(S(f"process.solver:{name}", name,
                     "Process/Normal Solver" + ("/More Solvers" if more else ""), handler="set_solver",
                     checkable=True, group="solver", keywords="normal solver engine"
                     + (" more" if more else "")))
    for name in DEPTH_METHODS:
        out.append(S(f"process.depth:{name}", name, "Process/Depth Method", handler="set_depth",
                     checkable=True, group="depth", keywords="depth integration method"))
    out += [
        S("process.refine", "Iterative Normal Refine", "Process", needs="solved",
          handler="refine_normals", sep_before=True, keywords="integrability refine normals"),
        S("qc.report", "Normal QC Report", "Process/Quality Checks", needs="solved",
          handler="qc_report", keywords="quality precision uncertainty rating"),
        S("qc.bias", "Low-Frequency Bias", "Process/Quality Checks", needs="solved",
          handler="qc_bias", keywords="calibration bowl laurent"),
        S("qc.compare", "Compare with a Reference Normal Map…", "Process/Quality Checks",
          needs="solved", handler="qc_compare", keywords="mae photogrammetry reference"),
        S("qc.disagreement", "Integrator Disagreement", "Process/Quality Checks", needs="solved",
          handler="qc_disagreement", keywords="poisson frankot depth trust"),
        S("qc.uncertainty", "Save Uncertainty Maps…", "Process/Quality Checks", needs="solved",
          handler="qc_uncertainty", keywords="residual sigma geotiff"),
        S("qc.dome3d", "3D Light Dome", "Process/Quality Checks", needs="lights",
          handler="qc_dome3d", keywords="lights hemisphere plot"),
        S("process.settings", "Processing Settings…", "Process", handler="go_process"),

        S("measure.scale", "Set Scale…", "Measure", needs="images", handler="measure_scale",
          keywords="ppi mm calibrate ruler"),
        S("measure.apply_scale", "Apply the Scale", "Measure", needs="images",
          handler="measure_apply_scale", keywords="ppi set"),
        S("measure.zscale", "Set Height Scale…", "Measure", needs="solved", handler="measure_zscale",
          keywords="z depth calibrate mm"),
        S("measure.apply_zscale", "Apply the Height Scale", "Measure", needs="solved",
          handler="measure_apply_zscale"),
        S("measure.ruler", "Scale from a Ruler…", "Measure", needs="images",
          handler="measure_ruler", keywords="ticks auto detect scale ppi"),
        S("measure.clear_zscale", "Clear the Height Scale", "Measure", needs="solved",
          handler="measure_clear_zscale", keywords="z reset"),
        S("measure.distance", "Measure Distance", "Measure", needs="images",
          handler="measure_distance", sep_before=True, keywords="ruler length"),
        S("measure.profile", "Height Profile", "Measure", needs="solved", handler="measure_profile",
          keywords="section roughness ra rz line"),
        S("measure.volume", "Area and Volume", "Measure", needs="solved", handler="measure_volume",
          keywords="surface area volume mm3 true 3d"),
        S("measure.stats", "Surface Statistics (ISO 25178)", "Measure", needs="solved",
          handler="measure_stats", keywords="roughness iso 25178 sa sq str sdr areal"),
        S("measure.compare", "Compare Two Regions", "Measure", needs="solved",
          handler="measure_compare", keywords="roughness iso 25178 two areas feature background"),
        S("measure.accuracy", "Accuracy Check on a Reference", "Measure", needs="solved",
          handler="measure_accuracy", keywords="precision gauge block step height flatness flat plate "
          "error uncertainty calibration verify reference"),
        S("measure.zscale_lateral", "Height Scale from the Lateral Scale", "Measure", needs="scale",
          handler="measure_zscale_lateral", keywords="precision height scale calibrated capture mm "
          "true heights"),
        S("measure.print_scale", "Print Scale Bars…", "Measure", handler="print_scale_bars",
          keywords="print pdf scale bar ruler mm inch l scale cut out"),
        S("measure.colour", "Check Colour Against a ColorChecker…", "Measure", needs="images",
          handler="colour_check", sep_before=True,
          keywords="colour color accuracy delta e de2000 ciede2000 colorchecker chart iso 19264 "
          "fadgi metamorfoze calibration"),
        S("measure.load_reference", "Load Change Reference…", "Measure", needs="solved",
          handler="load_change_reference", sep_before=True,
          keywords="earlier scan before after monitoring baseline depth"),
        S("measure.change_map", "Save Change Map…", "Measure", needs="solved",
          handler="save_change_map", keywords="difference before after monitoring loss gain"),
        S("measure.change_report", "Save Change Report…", "Measure", needs="solved",
          handler="save_change_report", keywords="difference before after condition histogram"),

        S("research.open:unmix", "Reflectance Unmixing…", "Research", needs="solved",
          handler="open_research", keywords="spectral hsh pca ica pigments layers materials components"),
        S("research.open:micro", "Micro-Features…", "Research", needs="solved", handler="open_research",
          keywords="pinholes pricking stylus pits blob log depressions"),
        S("research.open:toolpaths", "Toolpaths…", "Research", needs="solved", handler="open_research",
          keywords="scratches grooves incisions strokes tool marks engraving"),
        S("research.open:zones", "Inscription Zones…", "Research", needs="solved", handler="open_research",
          keywords="text writing inscription detect letters lettering"),
        S("research.open:register", "Register Two Images…", "Research", needs="solved",
          handler="open_research", sep_before=True,
          keywords="cross object overlay pouncing align homography affine similarity"),
        S("research.open:recto", "Two Faces…", "Research", needs="solved", handler="open_research",
          keywords="recto verso obverse reverse matrix impression front back link pair leaf page "
          "coin die axis"),
        S("research.open:unwrap", "Legend Unwrap…", "Research", needs="solved", handler="open_research",
          sep_before=True, keywords="coin seal medal circular legend inscription ring polar straighten "
          "read lettering rim diameter oval vesica stamp volvelle astrolabe rota centrepiece matrix"),
        S("research.open:impression", "Virtual Impression…", "Research", needs="solved",
          handler="open_research", keywords="seal matrix intaglio gem stamp wax squeeze mirror "
          "invert raised letters cast"),
        S("research.open:linedraw", "Line Drawing…", "Research", needs="solved", handler="open_research",
          keywords="svg vector drawing illustration trace archaeological epigraphic outline "
          "inkscape illustrator"),
        S("research.open:focus", "Focus Stacking…", "Research", handler="open_research", sep_before=True,
          keywords="macro bracket depth of field merge pyramid coin gem"),
        S("ai.open:identify", "Identify the Object…", "Research/AI", needs="solved", handler="open_ai",
          sep_before=True, keywords="ai recognition llm claude gpt gemini mistral ollama lm studio local"),
        S("ai.open:report", "Condition Report…", "Research/AI", needs="solved", handler="open_ai",
          keywords="ai conservation condition llm claude gpt gemini ollama"),
        S("ai.open:caption", "Figure Caption…", "Research/AI", needs="solved", handler="open_ai",
          keywords="ai caption publication llm claude gpt gemini ollama"),

        S("view.ws:Capture", "Capture", "View", "Ctrl+1", handler="go_workspace", group="workspace",
          checkable=True),
        S("view.ws:Process", "Process", "View", "Ctrl+2", handler="go_workspace", group="workspace",
          checkable=True),
        S("view.ws:Inspect", "Inspect", "View", "Ctrl+3", handler="go_workspace", group="workspace",
          checkable=True),
        S("view.ws:Measure", "Measure", "View", "Ctrl+4", handler="go_workspace", group="workspace",
          checkable=True),
        S("view.ws:Publish", "Publish", "View", "Ctrl+5", handler="go_workspace", group="workspace",
          checkable=True),
        S("view.next", "Next View", "View", "Right", needs="solved", handler="view_next",
          sep_before=True, image_focus=True, keywords="engine render"),
        S("view.prev", "Previous View", "View", "Left", needs="solved", handler="view_prev",
          image_focus=True, keywords="engine render"),
        S("view.depth", "Depth Map", "View", "D", needs="solved", handler="view_depth",
          image_focus=True, keywords="height"),
        S("view.clay", "Clay (No Colour)", "View", "C", needs="solved", handler="view_clay",
          image_focus=True, keywords="standard pbr relight shading grey colour"),
        S("view.white_balance", "Pick White Balance", "View", needs="solved",
          handler="pick_white_balance",
          keywords="white balance wb neutral grey card colour cast temperature tint picker eyedropper"),
        S("view.white_balance_reset", "Reset White Balance", "View", needs="solved",
          handler="reset_white_balance", keywords="white balance wb colour as photographed"),
        S("view.layer:Combined Colour", "Colour", "View/Layer", needs="solved", handler="set_layer",
          checkable=True, group="layer", keywords="combined colour albedo"),
        S("view.layer:Shading", "Clay", "View/Layer", needs="solved", handler="set_layer",
          checkable=True, group="layer", keywords="shading grey"),
        S("view.layer:Roughness Fusion", "Roughness", "View/Layer", needs="solved",
          handler="set_layer", checkable=True, group="layer", keywords="roughness fusion"),
        S("view.layer:Confidence Overlay", "Confidence", "View/Layer", needs="solved",
          handler="set_layer", checkable=True, group="layer",
          keywords="confidence overlay shadow highlight trust"),
        S("view.compare", "Split View", "View", "Ctrl+\\", needs="solved", handler="toggle_compare",
          sep_before=True, keywords="compare a/b wipe side by side"),
        S("view.save_state", "Save View State…", "View", "Ctrl+B", needs="solved",
          handler="save_state", keywords="bookmark safe state preset"),
        S("view.best_raking", "Best Raking Light", "View", needs="solved", handler="best_raking",
          sep_before=True, keywords="light contrast raking"),
        S("view.sweep", "Sweep the Light", "View", "A", needs="solved", handler="toggle_sweep",
          image_focus=True, keywords="auto sweep rotate light animate"),
        S("view.sweep_gif", "Save the Light Sweep…", "View", "G", needs="solved",
          handler="save_sweep_gif", image_focus=True,
          keywords="gif mp4 video animation light sweep moving rti"),
        S("view.histogram", "Histogram", "View", "H", needs="solved", handler="show_histogram",
          image_focus=True, keywords="levels clipping statistics"),
        S("view.snapshot", "Save View…", "View", "Ctrl+Alt+S", needs="solved", handler="save_view",
          keywords="snapshot tiff export current visual picture full resolution 3d relief "
          "screenshot png jpeg 4k 8k centre side by side high resolution"),
        S("view.export_loupe", "Save Loupe Region…", "View", needs="solved", handler="export_loupe",
          keywords="magnifier region full resolution crop"),
        S("view.mode:image", "Image", "View/Centre", "Alt+1", handler="set_centre_mode",
          group="centre", checkable=True, keywords="2d picture view centre"),
        S("view.mode:3d", "3D Relief", "View/Centre", "Alt+2", needs="solved",
          handler="set_centre_mode", group="centre", checkable=True,
          keywords="3d relief mesh opengl centre embedded"),
        S("view.mode:both", "Image and 3D Side by Side", "View/Centre", "Alt+3", needs="solved",
          handler="set_centre_mode", group="centre", checkable=True,
          keywords="3d side by side compare split centre"),
        S("view.relief_turntable", "Record a 3D Turntable…", "View/Centre", needs="solved",
          handler="record_relief_turntable", keywords="3d turntable spin video mp4 gif relief"),
        S("view.relief_flyover", "Record a 3D Flyover…", "View/Centre", needs="solved",
          handler="record_relief_flyover",
          keywords="3d flyover fly orbit loop video mp4 gif relief animation"),
        S("view.relief_sweep", "Record a 3D Light Sweep…", "View/Centre", needs="solved",
          handler="record_relief_sweep",
          keywords="3d light sweep rti moving light video mp4 gif relief animation"),
        S("view3d.open:standard", "Open in 3D", "View/3D Model", "Ctrl+Shift+3", needs="solved",
          handler="open_3d", keywords="3d viewer mesh opengl model"),
        S("view3d.open:flattened", "Open Auto-Flattened in 3D", "View/3D Model", needs="solved",
          handler="open_3d", keywords="3d flatten warp quadric taco"),
        S("view3d.open:inverted", "Open Inverted in 3D", "View/3D Model", needs="solved",
          handler="open_3d", keywords="3d invert intaglio seal die mould"),
        S("view3d.open:heatmap", "Open Elevation Heatmap in 3D", "View/3D Model", needs="solved",
          handler="open_3d", keywords="3d heatmap height colour"),
        S("view3d.window", "Open High-Detail 3D Window", "View/3D Model", needs="solved",
          handler="open_3d_window", sep_before=True,
          keywords="3d viewer window high detail flyover turntable classic"),
        S("view3d.settings", "3D Settings (Sharpening, Export)…", "View/3D Model", needs="solved",
          handler="show_3d_settings",
          keywords="3d mesh height relief clay material sharpening flip export"),
        S("view.composite", "Multi-Light Composite…", "View", needs="solved",
          handler="multi_light_composite", sep_before=True,
          keywords="average all lights shadow cancelled mean max"),
        S("view.rotate_left", "Turn View Left", "View/Orientation", "Ctrl+[", needs="images",
          handler="rotate_left", keywords="rotate canvas 90"),
        S("view.rotate_right", "Turn View Right", "View/Orientation", "Ctrl+]", needs="images",
          handler="rotate_right", keywords="rotate canvas 90"),
        S("view.rotate_left_fine", "Turn View Left 1°", "View/Orientation", "Ctrl+Shift+[",
          needs="images", handler="rotate_left_fine", keywords="rotate canvas fine degree straighten"),
        S("view.rotate_right_fine", "Turn View Right 1°", "View/Orientation", "Ctrl+Shift+]",
          needs="images", handler="rotate_right_fine", keywords="rotate canvas fine degree straighten"),
        S("view.rotate_reset", "Original Orientation", "View/Orientation", needs="images",
          handler="rotate_reset", keywords="rotate 0"),
        S("view.reset_transforms", "Reset Display Transforms", "View/Orientation", needs="solved",
          handler="reset_transforms", keywords="mirror flip invert depth normals reset"),
        S("view.fit", "Zoom to Fit", "View", "Ctrl+0", needs="images", handler="zoom_fit",
          sep_before=True),
        S("view.zoom_in", "Zoom In", "View", "Ctrl+=", needs="images", handler="zoom_in"),
        S("view.zoom_out", "Zoom Out", "View", "Ctrl+-", needs="images", handler="zoom_out"),
        S("view.sidebar", "Show Sidebar", "View", "Ctrl+Meta+S", handler="toggle_sidebar",
          checkable=True, sep_before=True),
        S("view.inspector", "Show Inspector", "View", "Ctrl+Alt+I", handler="toggle_inspector",
          checkable=True),
        S("view.appearance:system", "Match the System", "View/Appearance", handler="set_appearance",
          checkable=True, group="appearance", keywords="theme dark light"),
        S("view.appearance:light", "Light", "View/Appearance", handler="set_appearance",
          checkable=True, group="appearance", keywords="theme paper parchment"),
        S("view.appearance:dark", "Dark", "View/Appearance", handler="set_appearance",
          checkable=True, group="appearance", keywords="theme graphite night dome"),
        S("view.scheme:graphite", "Graphite and Paper (neutral, for judging colour)",
          "View/Appearance", handler="set_scheme", checkable=True, group="scheme",
          sep_before=True, keywords="theme colour scheme grey neutral iso 12646"),
        S("view.scheme:nightdome", "Night Dome and Parchment", "View/Appearance",
          handler="set_scheme", checkable=True, group="scheme", keywords="theme colour scheme blue"),

        S("window.minimize", "Minimize", "Window", "Ctrl+M", handler="showMinimized"),
        S("window.zoom", "Zoom", "Window", handler="toggle_zoomed"),

        S("help.help", "Helio Help", "Help", "Ctrl+?", handler="show_help",
          keywords="manual guide documentation glossary methods references"),
        S("help.search", "Search Tools…", "Help", "Ctrl+K", handler="show_palette",
          keywords="command palette find"),
        S("help.shortcuts", "Keyboard Shortcuts", "Help", "Ctrl+/", handler="show_shortcuts"),
        S("help.welcome", "Getting Started", "Help", handler="show_welcome",
          keywords="welcome tour introduction wizard"),
        S("help.report", "Report a Problem…", "Help", handler="report_problem", sep_before=True,
          keywords="bug crash diagnostic log support"),
        S("app.about", "About Helio", "Help", handler="show_about", role="about",
          sep_before=True),
    ]
    return out


SPECS: list[Spec] = _specs()

# The classic app's single keys, kept as second shortcuts while the image has
# focus, so hands that learned HELIO 2 still work (key, command).
ALIASES = (("R", "process.run"), ("E", "view.next"), ("Shift+E", "view.prev"),
           ("F", "view.fit"), ("?", "help.shortcuts"), ("S", "view.snapshot"))


def location(spec: Spec) -> str:
    """Where a command lives, as a person would say it: 'Process › Depth Method'."""
    return " › ".join(spec.menu.split("/"))


class Registry:
    """Builds one QAction per Spec and keeps availability current."""

    def __init__(self, window, project):
        self.window, self.project = window, project
        self.actions: dict[str, QAction] = {}
        self.specs = {s.id: s for s in SPECS}
        groups: dict[str, QActionGroup] = {}
        for s in SPECS:
            a = QAction(s.text, window)
            a.setObjectName(s.id)
            if s.shortcut:
                a.setShortcut(QKeySequence(s.shortcut))
                if s.image_focus:
                    a.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            if s.role:
                a.setMenuRole({"about": QAction.MenuRole.AboutRole,
                               "preferences": QAction.MenuRole.PreferencesRole,
                               "quit": QAction.MenuRole.QuitRole}[s.role])
            else:
                a.setMenuRole(QAction.MenuRole.NoRole)
            a.setCheckable(s.checkable)
            if s.group:
                g = groups.setdefault(s.group, QActionGroup(window))
                g.setExclusive(True)
                g.addAction(a)
            a.triggered.connect(lambda checked=False, sid=s.id: self.trigger(sid))
            self.actions[s.id] = a
        project.changed.connect(self.refresh)
        self.refresh()

    def reason(self, sid: str) -> str:
        s = self.specs[sid]
        if not s.handler:
            return self.project.needs("classic")
        return self.project.needs(s.needs)

    def trigger(self, sid: str):
        if self.reason(sid):
            return
        s = self.specs[sid]
        arg = sid.split(":", 1)[1] if ":" in sid else None
        fn = getattr(self.window, s.handler)
        fn(arg) if arg is not None else fn()

    def tip(self, sid: str) -> str:
        """What the command does (helio_qt.action_tips), or ''."""
        from helio_qt.action_tips import tip
        return tip(sid)

    def refresh(self):
        for sid, a in self.actions.items():
            why = self.reason(sid)
            what = self.tip(sid)
            a.setEnabled(not why)
            a.setStatusTip(why or what)
            # Shown when the pointer rests on it in a menu: what it does, and why not when it cannot.
            a.setToolTip("\n\n".join(t for t in (what or a.text(), f"Not available: {why}" if why else "")
                                      if t))
        self._native_soon()

    # ── the Mac's own menu bar (helio_qt.native_menu) ─────────────────
    def _native_soon(self):
        """Copy the tooltips to the native menu bar, once per turn of the event loop."""
        import sys
        if sys.platform != "darwin" or getattr(self, "_native_pending", False):
            return
        from PySide6.QtCore import QTimer
        self._native_pending = True
        QTimer.singleShot(0, self.sync_native_tooltips)

    def native_tip(self, path, title) -> str:
        """The tooltip of the command at *path* › *title* in the native menu bar, or ''."""
        if not hasattr(self, "_by_place"):
            self._by_place, by_title = {}, {}
            for sp in SPECS:
                self._by_place[(tuple(sp.menu.split("/")), sp.text)] = sp.id
                by_title.setdefault(sp.text, []).append(sp.id)
            self._by_title = {t: ids[0] for t, ids in by_title.items() if len(ids) == 1}
        sid = self._by_place.get((tuple(path[1:]), title)) or self._by_place.get((tuple(path), title))
        if sid is None:                                  # moved to the app menu by its role
            if title.startswith("Quit "):
                sid = "app.quit"
            elif title.startswith("About "):
                sid = "app.about"
            else:
                sid = self._by_title.get(title)
        a = self.actions.get(sid) if sid else None
        return a.toolTip() if a is not None else ""

    def sync_native_tooltips(self) -> int:
        self._native_pending = False
        from helio_qt.native_menu import set_tooltips
        return set_tooltips(self.native_tip)

    def build_menus(self, menubar):
        menus = {}
        for title in MENUS:
            menus[title] = menubar.addMenu(title)
            menus[title].setToolTipsVisible(True)       # what each command does, on hover
            # The Mac's native menu exists by the time a menu opens: its tooltips are set then.
            menus[title].aboutToShow.connect(self.sync_native_tooltips)
        subs = {}
        for s in SPECS:
            top, *rest = s.menu.split("/")
            m = menus[top]
            for i in range(len(rest)):                 # submenus, nested as deep as the path goes
                key = "/".join([top, *rest[:i + 1]])
                if key not in subs:
                    if s.sep_before and i == 0:
                        m.addSeparator()
                    subs[key] = m.addMenu(rest[i])
                    subs[key].setToolTipsVisible(True)
                m = subs[key]
            if not rest and s.sep_before:
                m.addSeparator()
            m.addAction(self.actions[s.id])
        return menus
