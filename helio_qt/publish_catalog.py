"""What Publish offers: destinations, formats and what each one gives you.

Each format names the legacy export that produces it today (``legacy``), so
EXP-001 can wire the card to the same code path and tests/test_exports.py keeps
covering it.  The words say what the file is *for*, not how it is made.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Format:
    key: str
    name: str
    what: str
    legacy: str            # RTIViewerPro method that writes it today; "" when new in HELIO 3
    needs: str = "solved"  # 'solved' or 'scale'
    impl: str = ""         # helio.application.exports function, once moved (EXP-001)


DESTINATIONS: dict[str, tuple[Format, ...]] = {
    "Images": (
        Format("png_views", "Current view as an image", "The view on screen, at full "
               "resolution, as a TIFF.", "export_highres_tiff", impl="current_view"),
        Format("all_views", "Every view at once", "One image per view, for comparing later, "
               "plus the colour, normals and a depth heatmap.", "export_snapshot_all_views",
               impl="all_views"),
        Format("normal_map", "Normal map set", "Normal map, hillshade, albedo × hillshade, albedo "
               "and fine normal detail, for other software.", "export_combined_renders",
               impl="combined_renders"),
        Format("anaglyph", "Red-cyan 3D", "A stereo picture for red-cyan glasses.", "export_anaglyph",
               impl="anaglyph_image"),
        Format("sweep_gif", "Moving-light GIF", "The light going once round the surface, "
               "in the current view.", "export_sweep_gif", impl="light_sweep_gif"),
        Format("deepzoom", "Zoomable tiles", "Deep-zoom tiles for very large images, with a "
               "viewer that opens without a network.", "export_deepzoom", impl="write_deepzoom"),
        Format("scalebar", "View with a scale bar", "The current view at full resolution with "
               "a measured scale bar burned in.", "export_scale_bar_render", "scale",
               impl="scale_bar_render"),
        Format("heatmap", "Elevation heatmap", "Height as a colour map (JPEG).",
               "export_elevation_heatmap", impl="elevation_heatmap"),
    ),
    "Surface maps": (
        Format("pbr", "16-bit PBR suite", "Albedo, normal, roughness, metallic, detail normals, "
               "occlusion, cavity, curvature, displacement and specular/diffuse maps.",
               "export_pbr_suite", impl="pbr_suite"),
        Format("sds", "Specular and diffuse", "Each pixel split into matte colour and gloss, "
               "with a gloss index and a highlight mask (Mallick et al. 2005).",
               "export_specular_diffuse_separation", impl="specular_diffuse"),
        Format("pca", "Material PCA", "The first three principal components of the light "
               "stack as a TIFF.", "export_pca", impl="pca_tiff"),
        Format("hsh", "HSH coefficients", "The nine hemispherical-harmonics maps, float32 "
               "and 16-bit.", "export_hsh_coeffs", impl="hsh_coefficients"),
        Format("uncertainty", "Uncertainty map", "How far to trust each pixel: the fit's "
               "residual as a tagged GeoTIFF.", "export_uncertainty_map", impl="uncertainty_map"),
    ),
    "3D models": (
        Format("glb", "GLB web viewer", "A textured 3D model and a page that opens it in "
               "any browser.", "export_3d_web_viewer", impl="web_viewer"),
        Format("stl", "STL for printing", "Watertight solid with a base; life size once a "
               "scale is set, else fitted to the print bed.", "export_stl_3dprint",
               impl="stl_print"),
        Format("print", "3D print package", "A closed solid of the surface detail on a flat "
               "base: life size once a scale is set, else 100 mm across.", "export_3d_print",
               impl="print_mesh"),
        Format("facsimile", "Facsimile for reproduction", "OBJ relief and 16-bit "
               "displacement for elevated printing or CNC; true size once a scale is set.",
               "export_facsimile_package", impl="facsimile_package"),
        Format("glb_mesh", "GLB model", "The textured mesh (albedo and normal maps) for 3D "
               "software and the web.", "_export_3d", impl="mesh_export"),
        Format("obj", "OBJ mesh", "Mesh with its material and texture maps, for Blender, "
               "MeshLab and most 3D tools.", "_export_3d", impl="mesh_export"),
        Format("ply", "PLY mesh", "Mesh with vertex colours, for MeshLab and CloudCompare.",
               "_export_3d", impl="mesh_export"),
        Format("point_cloud", "Point cloud (PLY)", "The surface as coloured points.",
               "_export_3d", impl="mesh_export"),
        Format("sketchfab", "Sketchfab PBR package", "GLB plus base colour, normal, roughness, "
               "specular, occlusion and displacement maps, zipped for upload.", "_export_3d",
               impl="mesh_export"),
        Format("usdz", "USDZ for AR / VR", "One OpenUSD file at the object's true size: opens in "
               "Quick Look on iPhone, iPad and Vision Pro, and in Blender, Omniverse and VR tools. "
               "Checked against Apple's AR rules.", "", impl="usdz"),
        Format("e57", "E57 point cloud", "For CloudCompare and survey software.", "export_e57",
               impl="point_cloud_e57"),
        Format("lod", "Multi-LOD package", "Three detail levels (OBJ and GLB) with normal, "
               "displacement and colour maps.", "export_multi_lod_package", impl="lod_package"),
        Format("displacement", "Displacement package", "Base mesh plus a 16-bit displacement map "
               "and a Blender guide.", "export_displacement_package", impl="displacement_package"),
    ),
    "Web viewers": (
        Format("rti_web", "RTI web package", "A folder that opens in any browser, offline: "
               "deep-zoom relighting at full size, a one-file quick view, Relight / OpenLIME "
               "data, the lights, a read-me, embed code and launchers.", "", impl="rti_web"),
        Format("rti_viewer", "Relightable viewer", "One HTML file: move the light in any browser.",
               "export_selfcontained_rti", impl="selfcontained_rti"),
        Format("webrti", "WebRTI", "A PTM relighting viewer with its tiles and launchers.",
               "export_webrti", impl="webrti"),
        Format("rti_combined", "Combined RTI viewer", "One HTML file: relight, specular, "
               "normals and the PCA layers, small enough to e-mail.", "export_combined_rti_viewer",
               impl="combined_rti_viewer"),
        Format("rti_3d", "Relight and 3D viewer", "One HTML file with two tabs: the combined RTI "
               "viewer and a light 3D model to turn. Opens offline in any browser; about 20-30 MB, "
               "for a talk or an e-mail.", "", impl="rti_3d_viewer"),
        Format("relight_web", "Relight / OpenLIME package", "For RelightLab and the relight.js / "
               "OpenLIME viewers, with the frames re-saved to re-fit.", "export_relight_web_package",
               impl="relight_package"),
        Format("openlime_hsh", "OpenLIME HSH RTI (tarzoom)", "Colour HSH relighting, 27 "
               "coefficients, as RelightLab's relight-cli -b hsh -p 27 writes it, in OpenLIME's "
               "tiled tarzoom format, with HELIO's normals, a viewer that opens with a double-click, "
               "relightable shine and clay views, and the surface maps (colour, normal, specular, "
               "cavity, PCA) in maps/.", "",
               impl="openlime_hsh_package"),
        Format("rti_deepzoom", "RTI DeepZoom relight", "Zoomable HSH relighting for very large "
               "captures.", "export_rti_deepzoom_relight", impl="rti_deepzoom_relight"),
        Format("iiif", "IIIF 3D manifest", "For Mirador, Universal Viewer, Aleph and Kompakkt "
               "(3D TSG draft).", "export_iiif_manifest", impl="iiif_manifest"),
        Format("iiif_images", "IIIF image manifest", "The colour and this object's views as "
               "images with a IIIF Presentation 3 manifest, for Mirador and the Universal Viewer.",
               "", impl="iiif_image_manifest"),
        Format("compare", "Comparison viewer", "The current view and the Compare view side by "
               "side with a slider.", "export_comparison_viewer", impl="comparison_page"),
        Format("engine_sweep", "Every view, one page", "Slide or play through every view in "
               "one HTML file.", "export_engine_sweep_viewer", impl="baked_frames"),
        Format("hf_viewer", "3D viewer: fine detail", "A web 3D viewer built from the normals' "
               "fine detail, for engraving and tool marks.", "export_3d_web_viewer_hf_normals",
               impl="hf_normals_viewer"),
    ),
    "Reports & figures": (
        Format("pub_sheet", "Publication sheet", "The object and the views you choose, each "
               "captioned; the details you boxed, close up under several views; the section and "
               "measured lines, the scale, 3D pictures you add, and the capture's details. JPEG "
               "and PDF.", "", impl="publication_sheet"),
        Format("plate_a3", "Publication plate (A3)", "One designed page: the object, its lights, "
               "four views that read the surface, height, 3D, a section and how sure the solve "
               "is, every panel captioned. PDF and PNG.", "", impl="plate"),
        Format("sheet", "Contact sheet (every panel)", "Every "
               "panel on one long sheet: the views, "
               "a profile, raking and 3D renders, metrology and the capture details.",
               "generate_contact_sheet", impl="publication_sheet"),
        Format("figures", "Publication figures", "One print-resolution TIFF per view, with a scale "
               "bar and caption.", "export_publication_figures", impl="publication_figures"),
        Format("plate", "Plate figure", "Five views on one captioned plate, with a scale bar "
               "once a scale is set.", "export_plate_figure", impl="plate_figure_export"),
        Format("condition", "Condition report", "Height over colour, with levelled surface "
               "statistics.", "export_condition_report", impl="condition_map"),
        Format("line_drawing", "Line drawing (SVG)", "Cut lines and raised edges traced as "
               "vector lines, in layers, in millimetres once a scale is set: a start for an "
               "archaeological or epigraphic drawing.", "", impl="line_drawing_svg"),
        Format("rose", "Orientation rose", "Which way the surface faces: the tool-direction "
               "figure.", "export_rose_diagram", impl="rose_diagram"),
        Format("calibration", "Calibration report", "Scale, lights and solvers: what a reader "
               "needs to judge the measurement.", "export_calibration_report",
               impl="calibration_report"),
        Format("provenance", "Processing record", "Every setting and file used, for the archive "
               "(JSON and a readable text copy).", "export_processing_provenance",
               impl="write_processing_record"),
    ),
    "Archive & sharing": (
        Format("heritage", "Heritage archive", "Model, textures, surface statistics and "
               "metadata in one ZIP.", "export_heritage_archive", impl="heritage_archive"),
        Format("ptm", "PTM coefficients", "The six PTM coefficient maps, float32 and 16-bit.",
               "export_ptm_coeffs", impl="ptm_coefficients"),
        Format("lp", "Light positions (.lp)", "The calibrated light file for other RTI tools.",
               "export_light_file_lp", impl="light_file"),
        Format("deposit", "Deposit package", "Masters, web and print derivatives and metadata in "
               "their own folders, with paradata, a read-me and checksums, for a repository.", "",
               impl="deposit"),
        Format("linear_masters", "Linear OpenEXR masters", "Colour, normals and heights as "
               "floating-point OpenEXR, for keeping and for VFX and 3D software.", "",
               impl="linear_masters"),
        Format("usd_layered", "Layered USD stage", "Surface, record and colour as separate USD "
               "layers, for archive and exchange, unlike the flattened USDZ.", "", impl="usd_layered"),
        Format("colour_validation", "Colour validation report", "ΔE00 on a ColorChecker in the "
               "capture, from photograph to colour map to render.", "", impl="validate"),
        Format("lido", "LIDO record", "A museum catalogue record (XML).", "export_lido_xml",
               impl="lido_record"),
        Format("zenodo", "Zenodo package", "Deposit metadata (Zenodo and DataCite), CIDOC "
               "record, lights, provenance and a checksummed manifest.", "export_zenodo_package",
               impl="zenodo_package"),
        Format("morphosource", "MorphoSource package", "Mesh, maps, IIIF manifest and Darwin "
               "Core metadata for the 3D specimen repository.", "export_morphosource_package",
               impl="morphosource_package"),
        Format("nomisma", "Coin record (Nomisma)", "The coin's details as Nomisma properties "
               "(CSV and JSON-LD): denomination, authority, mint, weight, diameter, die axis.",
               "", impl="nomisma_record"),
        Format("epidoc", "EpiDoc record (TEI)", "A TEI EpiDoc file from the capture details, the "
               "images as its facsimile and an empty edition to transcribe into.", "",
               impl="epidoc_record"),
        Format("cidoc", "CIDOC-CRM record", "A Linked Art / CIDOC-CRM JSON-LD record of the scan.",
               "export_cidoc_metadata", impl="write_cidoc"),
        Format("dublin_core", "TIFF with Dublin Core", "The current view as a TIFF carrying "
               "Dublin Core XMP metadata.", "export_dublin_core_tiff", impl="dublin_core_tiff"),
    ),
    "GIS & data": (
        Format("geotiff", "GIS DEM package", "Levelled height rasters for QGIS and ArcGIS, "
               "with albedo, normals, uncertainty, lights and a README.", "export_geotiff",
               impl="dem_package"),
        Format("freq_maps", "Frequency maps (32-bit)", "Raw Poisson and Frankot–Chellappa "
               "heights and the shape, detail and texture bands, for QGIS and RVT.",
               "_export_frequency_maps", impl="frequency_maps"),
        Format("light_vectors", "Light vectors (CSV)", "Each frame's light direction.",
               "export_light_vectors", impl="light_vectors_csv"),
        Format("light_cube", "Light cube (ImageJ)", "The source frames registered to the solve, "
               "as one stack: each pixel's measured reflectance.", "export_light_cube",
               impl="light_cube"),
        Format("engine_stack", "Every view (ImageJ stack)", "All views as one stack, sized in "
               "mm when a scale is set.", "export_engine_stack", impl="engine_stack"),
        Format("gis_maps", "Terrain analysis maps", "Seven 32-bit GeoTIFFs: openness, relief, "
               "curvature, slope, dominance, sky-view.", "export_gis_analysis_maps",
               impl="gis_analysis_maps"),
        Format("iso_stats", "Roughness statistics", "ISO 25178 surface statistics as CSV.",
               "export_iso_stats_csv", impl="iso_stats_csv"),
        Format("qgis", "Helio Relief (QGIS plugin)",
               "The QGIS plugin, v4.1: Import Helio, relief renders, measuring, line tracing "
               "and plates. Install it from the ZIP in QGIS.",
               "export_qgis_plugin", impl="write_qgis_plugin_zip"),
    ),
}

PRESETS: dict[str, tuple[str, ...]] = {
    "Museum archive": ("heritage", "lido", "provenance", "lp"),
    "Repository deposit": ("deposit",),
    "Share on the web": ("rti_web", "glb"),
    "3D print": ("stl", "print"),
    "Publication figures": ("plate_a3", "figures", "plate"),
}


def available(f: Format) -> bool:
    return bool(f.impl)


def all_formats() -> list[Format]:
    return [f for fs in DESTINATIONS.values() for f in fs]
