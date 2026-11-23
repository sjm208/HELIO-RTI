"""Archive and deposit exports, GUI-free (HER-002…013, GEO-016…018).

The classic Archive tab's writing, moved so both apps write the same files:
the MorphoSource and Zenodo deposit packages, the CIDOC-CRM / Linked Art
record, a TIFF carrying Dublin Core XMP and the light vectors as CSV.

Fixed on the way: MorphoSource's GLB was mirrored top to bottom and faced
down (KD-039) and its manifest put the lights behind the model (KD-036); the
Dublin Core TIFF built its XMP packet and never wrote it (KD-038).
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
import time
import zipfile as _zf
from xml.sax.saxutils import escape

import cv2
import numpy as np

from helio.application.exports import NothingToExport


def morphosource_package(D, p: str, obj_id: str, *, ss: dict, depth_normalised=None,
                         normal_engine: str = "", depth_engine: str = "", mm_per_px=None,
                         z_mm_per_unit=None, n_images: int = 0, lights=None,
                         institution: str = "", heatmap: bool = False) -> str:
    """A MorphoSource deposit ZIP: GLB mesh, albedo, normals, depth TIFF,
    roughness and specular maps, an IIIF 3D manifest, Darwin Core metadata
    and a deposit walkthrough."""
    if getattr(D, "normals_full", None) is None:
        raise NothingToExport("Please process scans first.")
    obj_id = obj_id or "HELIO_OBJECT"
    tmp = tempfile.mkdtemp()
    try:
        tmp = tempfile.mkdtemp()
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        n_imgs = int(n_images)
        _mm_pp = mm_per_px
        _z_mpu = z_mm_per_unit

        def _prog(msg): print(f"[MorphoSource] {msg}")

        # ── 1. GLB mesh — build using same pipeline as _export_3d ─────────
        _prog("Building GLB mesh (this may take a moment)...")
        glb_path = os.path.join(tmp, f"{obj_id}.glb")
        try:
            import trimesh
            from PIL import Image as _PILImg
            step_g = max(1, ss['export_step'])
            sb_g   = max(0.0, min(1.0, ss['shape_blend']))
            ht_g   = ss['height']
            trim_g = max(0, int(min(D.h, D.w) * ss['edge_trim'] / 200))
            y1_g,y2_g = trim_g, D.h-trim_g; x1_g,x2_g = trim_g, D.w-trim_g
            # Depth: blend full + HP same as viewer
            d_hp   = D.depth_vis_p[y1_g:y2_g, x1_g:x2_g].astype(np.float32)
            d_full = (depth_normalised[y1_g:y2_g, x1_g:x2_g].astype(np.float32)
                      if depth_normalised is not None else d_hp)
            z_g    = d_full*(1-sb_g) + d_hp*sb_g
            # Build vertex grid
            h_g, w_g = z_g.shape
            ys_g, xs_g = np.mgrid[0:h_g:step_g, 0:w_g:step_g]
            gh_g, gw_g = ys_g.shape
            z_s = z_g[::step_g, ::step_g] * ht_g
            # North up (y = -row) so the model is not mirrored, and the triangles
            # below face +z, out of the relief: HELIO 2 put the rows on +y and
            # wound the surface downward (KD-039).
            pts = np.column_stack((
                (xs_g*(w_g/gw_g)).flatten().astype(np.float32),
                (-ys_g*(h_g/gh_g)).flatten().astype(np.float32),
                z_s.flatten()))
            rr_g,cc_g = np.meshgrid(np.arange(gh_g-1),np.arange(gw_g-1),indexing='ij')
            v0_g = rr_g*gw_g+cc_g
            faces_g = np.vstack((
                np.column_stack((v0_g.flatten(),(v0_g+gw_g).flatten(),(v0_g+1).flatten())),
                np.column_stack(((v0_g+1).flatten(),(v0_g+gw_g).flatten(),(v0_g+gw_g+1).flatten()))
            )).astype(np.uint32)
            # UVs — xs_g/ys_g are pixel coords (0, step, 2*step, …).
            # UV = pixel_coord / (image_size - 1) → always 0..1 across the whole mesh.
            us_g   = (xs_g / max(w_g - 1, 1)).flatten().astype(np.float32)
            vs_g   = (1.0 - ys_g / max(h_g - 1, 1)).flatten().astype(np.float32)
            uvs_g  = np.column_stack((us_g, vs_g))
            # Texture: sample albedo at vertex positions for correct 1:1 mapping
            alb_g  = D.albedo_rgb_full[y1_g:y2_g, x1_g:x2_g][::step_g, ::step_g].astype(np.uint8)
            alb_tex = _PILImg.fromarray(alb_g)
            nrm_g  = ((D.normals_full[y1_g:y2_g, x1_g:x2_g][::step_g, ::step_g] * 0.5 + 0.5) * 255).astype(np.uint8)
            nrm_tex = _PILImg.fromarray(nrm_g)
            pbr_mat = trimesh.visual.material.PBRMaterial(
                name='helio_pbr',
                baseColorTexture=alb_tex,
                normalTexture=nrm_tex,
                metallicFactor=0.0, roughnessFactor=0.65, doubleSided=True)
            mesh_g = trimesh.Trimesh(vertices=pts, faces=faces_g, process=False)
            mesh_g.visual = trimesh.visual.TextureVisuals(uv=uvs_g, material=pbr_mat)
            mesh_g.export(glb_path)
            from helio.io.exporters import _fix_glb_material
            _fix_glb_material(glb_path, metallic=0.0, roughness=0.65, double_sided=True)
            _prog(f"  GLB: {os.path.getsize(glb_path)//1024} KB  ({pts.shape[0]:,} verts)")
        except Exception as _ge:
            _prog(f"  GLB build failed: {_ge}")
            import traceback; traceback.print_exc()
            with open(glb_path, 'wb') as f:
                f.write(b'GLB build failed - export GLB from 3D & Web tab and rename')

        # ── 2. Albedo texture (JPEG, 95%) ─────────────────────────────────
        _prog("Saving albedo texture...")
        alb_path = os.path.join(tmp, f"{obj_id}_albedo.jpg")
        cv2.imwrite(alb_path,
            cv2.cvtColor(D.albedo_rgb_full, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 95])

        # ── 3. Normal map PNG (PS normals → [0,255] RGB) ──────────────────
        _prog("Saving normal map...")
        nrm_path = os.path.join(tmp, f"{obj_id}_normals.png")
        nrm_8 = ((D.normals_full + 1.0) / 2.0 * 255).astype(np.uint8)
        cv2.imwrite(nrm_path, cv2.cvtColor(nrm_8, cv2.COLOR_RGB2BGR))

        # ── 4. Depth TIFF (32-bit float, calibrated if available) ─────────
        _prog("Saving depth TIFF...")
        dep_path = os.path.join(tmp, f"{obj_id}_depth.tif")
        if _z_mpu:
            dep_mm = D.depth_vis_p.astype(np.float32) * float(_z_mpu) * D.h
            cv2.imwrite(dep_path, dep_mm, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
        else:
            cv2.imwrite(dep_path,
                D.depth_vis_p.astype(np.float32),
                [cv2.IMWRITE_TIFF_COMPRESSION, 1])

        # ── 5. IIIF 3D manifest (inline generation, M1-M4) ────────────────
        _prog("Generating IIIF 3D manifest...")
        base_url = f"https://example.org/iiif/{obj_id}"
        phys_note = (f"{D.w*_mm_pp:.1f} × {D.h*_mm_pp:.1f} mm"
                     if _mm_pp else "uncalibrated")
        light_vecs, lights = lights, []        # the input, then the annotations
        if light_vecs is not None:
            for i, lv in enumerate(light_vecs):
                lights.append({
                    "id": f"{base_url}/annotations/light/{i}", "type": "Annotation",
                    "motivation": "painting",
                    "body": {"type": "PointLight", "color": "#ffffff",
                             "intensity": round(float(lv[2])*0.9+0.1, 3),
                             "position": {"x": round(float(lv[0]),4),
                                          "y": round(float(lv[1]),4),
                                          "z": round(float(lv[2]),4)}},
                    "target": f"{base_url}/scene"
                })
        iiif_manifest = {
            "@context": [
                "https://iiif.io/api/presentation/3/context.json",
                "https://iiif.github.io/3d/context/iiif-3d.json"
            ],
            "id": f"{base_url}/manifest", "type": "Manifest",
            "label": {"en": [obj_id]},
            "summary": {"en": [f"Photometric Stereo scan by Helio. {phys_note}."]},
            "metadata": [
                {"label":{"en":["Object ID"]},     "value":{"en":[obj_id]}},
                {"label":{"en":["Technique"]},     "value":{"en":["Photometric Stereo"]}},
                {"label":{"en":["Software"]},      "value":{"en":["Helio"]}},
                {"label":{"en":["Date"]},          "value":{"en":[now]}},
                {"label":{"en":["Physical Size"]}, "value":{"en":[phys_note]}},
                {"label":{"en":["Image Count"]},   "value":{"en":[str(n_imgs)]}},
                {"label":{"en":["IIIF 3D TSG"]},   "value":{"en":["M1–M4 compliant"]}},
            ],
            "items": [{
                "id": f"{base_url}/scene", "type": "Scene",
                "label": {"en": [obj_id]},
                "items": [{"id": f"{base_url}/scene/page", "type": "AnnotationPage",
                    "items": [{"id": f"{base_url}/annotations/model",
                        "type": "Annotation", "motivation": "painting",
                        "body": {"id": f"{base_url}/{obj_id}.glb",
                                 "type": "Model", "format": "model/gltf-binary"},
                        "target": f"{base_url}/scene"}]}],
                "annotations": [{"id": f"{base_url}/scene/annotations",
                    "type": "AnnotationPage",
                    "items": [
                        {"id": f"{base_url}/annotations/camera", "type":"Annotation",
                         "motivation":"painting",
                         "body":{"type":"Camera","position":{"spherical":{"elevation":30,"azimuth":0,"radius":round(max(D.h,D.w)*0.8,1)}}},
                         "target": f"{base_url}/scene"}
                    ] + lights}]
            }]
        }
        iiif_path = os.path.join(tmp, f"{obj_id}_iiif_manifest.json")
        with open(iiif_path, 'w') as f: json.dump(iiif_manifest, f, indent=2)

        # ── 6. MorphoSource metadata JSON (Darwin Core + MS vocabulary) ───
        _prog("Writing MorphoSource metadata...")
        _inst_str = institution or ""
        ms_meta = {
            "_comment": "MorphoSource deposit metadata — fill in fields marked REQUIRED",
            "_schema":  "https://github.com/MorphoSource/morphosource-api",
            "_darwin_core": "https://dwc.tdwg.org/terms/",

            "physicalObject": {
                "dwc:institutionCode":      _inst_str or "REQUIRED — e.g. NHM, BL, BM",
                "dwc:collectionCode":       "REQUIRED — departmental code, e.g. MSS",
                "dwc:catalogNumber":        obj_id,
                "dwc:occurrenceID":         f"urn:catalog:{_inst_str or 'INST'}:MSS:{obj_id}",
                "dwc:scientificName":       "",
                "dwc:kingdom":              "Animalia",
                "dwc:locality":             "",
                "dwc:country":              "",
                "dwc:recordedBy":           "",
                "dwc:eventDate":            now[:10],
                "schema:description":       f"Photometric stereo 3D scan produced by Helio. {phys_note}.",
                "schema:name":              obj_id,
                "schema:identifier":        obj_id,
            },

            "media": {
                "ms:mediaType":            "Mesh",
                "ms:mediaModality":        "Photogrammetry / Structured Light",
                "ms:fileFormat":           "GLB (Binary GLTF 2.0)",
                "ms:processingApplication":"Helio",
                "ms:captureMethod":        "Photometric Stereo (RTI dome / flat LED array)",
                "ms:lightCount":           n_imgs,
                "ms:normalSolver":         normal_engine,
                "ms:depthEngine":          depth_engine,
                "ms:pixelDimensions":      f"{D.w}×{D.h} px",
                "ms:physicalDimensions":   phys_note,
                "ms:zCalibrated":          _z_mpu is not None,
                "ms:iiifManifest":         f"{base_url}/manifest",
                "ms:rights":               "REQUIRED — e.g. CC BY 4.0",
                "ms:license":              "REQUIRED — e.g. https://creativecommons.org/licenses/by/4.0/",
                "ms:accessRights":         "restricted download",
            },

            "files": {
                f"{obj_id}.glb":              "Primary 3D mesh (GLB 2.0) — upload as primary Media file",
                f"{obj_id}_albedo.jpg":        "Albedo/diffuse texture (JPEG 95%) — upload as associated texture",
                f"{obj_id}_normals.png":       "Photometric stereo normal map RGB — upload as associated file",
                f"{obj_id}_depth.tif":         "32-bit depth map TIFF" + (" (calibrated mm)" if _z_mpu else " (relative units)") + " — upload as associated file",
                f"{obj_id}_iiif_manifest.json":"IIIF 3D Presentation API v3 manifest (M1–M4) — paste URL into MorphoSource IIIF field after hosting",
            }
        }
        ms_path = os.path.join(tmp, f"{obj_id}_morphosource_metadata.json")
        with open(ms_path, 'w') as f: json.dump(ms_meta, f, indent=2, ensure_ascii=False)

        # ── 7. README ─────────────────────────────────────────────────────
        readme = f"""Helio — MorphoSource Deposit Package
{"="*55}
Object ID  : {obj_id}
Generated  : {now}
Physical   : {phys_note}
Pixels     : {D.w}×{D.h}
Images     : {n_imgs} PS capture frames
Solver     : {normal_engine} / {depth_engine}

FILES IN THIS PACKAGE
─────────────────────
{obj_id}.glb
    Primary 3D file. In MorphoSource submission wizard:
      Step 1 → Media Type: "Mesh"
      Step 3 → Upload File → select this GLB

{obj_id}_albedo.jpg
    Diffuse colour texture. Upload as "Secondary Media File"
    or associate via the Files tab after submission.

{obj_id}_normals.png
    Photometric stereo normal map (RGB encoded, PS standard).
    R=X, G=Y, B=Z in OpenGL convention.
    Upload as "Secondary Media File".

{obj_id}_depth.tif
    32-bit float depth map. {"Calibrated in mm." if _z_mpu else "Relative units — calibrate Z in Helio before depositing for physical values."}
    Upload as "Secondary Media File".

{obj_id}_iiif_manifest.json
    IIIF Presentation API v3 manifest with 3D TSG extensions
    (Milestone Groups M1–M4). Includes all {n_imgs} PS light
    vectors as PointLight annotations (M3) and camera (M2).
    After hosting your GLB, replace 'https://example.org'
    with your institution URL and enter the manifest URL
    in MorphoSource's "IIIF Manifest URL" field.

{obj_id}_morphosource_metadata.json
    Metadata template. Fill in REQUIRED fields before
    submitting (institutionCode, collectionCode, rights).
    Darwin Core fields map to MorphoSource physical object
    record. MorphoSource media fields map to the Media record.

MORPHOSOURCE SUBMISSION STEPS
───────────────────────────────
1. Register / log in at https://www.morphosource.org
2. Click "Deposit" → "New Physical Object"
3. Enter specimen metadata from morphosource_metadata.json
   (physicalObject section)
4. Create a Media record, upload {obj_id}.glb as primary file
5. Upload albedo, normals, and depth as secondary files
6. Host the IIIF manifest and enter the URL in the IIIF field
7. Set access rights and licence (see "rights" field in metadata)
8. Submit for review

IIIF 3D COMPLIANCE
────────────────────
This package targets the IIIF 3D TSG draft specification:
  M1: Single model in Scene  ✅
  M2: Camera annotation      ✅  (elevation=30°, azimuth=0°)
  M3: Light annotations      {"✅  (" + str(n_imgs) + " PS lights)" if n_imgs else "⚠  (no light data — reprocess with dome)"}
  M4: Physical scale         {"✅  (" + phys_note + ")" if _mm_pp else "⚠  (uncalibrated — draw scale bar in Metrology tab)"}

Compatible viewers: MorphoSource/Aleph, Kompakkt, Universal Viewer
Spec: https://iiif.github.io/3d/

SUPPORT
───────
Helio — https://github.com/sjm208/HELIO-RTI
"""
        readme_path = os.path.join(tmp, "README.txt")
        with open(readme_path, 'w', encoding='utf-8') as f: f.write(readme)

        # ── 7b. Extra science maps (MorphoSource accepts these as secondary files)
        _prog("Saving roughness, specular and optional heatmap...")
        rough_path = os.path.join(tmp, f"{obj_id}_roughness.png")
        if getattr(D, 'roughness_full',None) is not None:
            rough_8 = cv2.normalize(D.roughness_full,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imwrite(rough_path, rough_8)
        spec_path = os.path.join(tmp, f"{obj_id}_specular_variance.png")
        if getattr(D, 'spec_var',None) is not None:
            spec_8 = cv2.normalize(D.spec_var,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imwrite(spec_path, spec_8)

        # ── 8. Assemble ZIP ───────────────────────────────────────────────
        _prog("Assembling ZIP...")
        # Optionally include elevation heatmap if checkbox ticked
        if heatmap:
            _heat_p = os.path.join(tmp, f"{obj_id}_elevation_heatmap.png")
            _heat_d = np.clip(D.depth_vis_p[y1_g:y2_g, x1_g:x2_g], 0, 1)
            cv2.imwrite(_heat_p, cv2.applyColorMap((_heat_d*255).astype(np.uint8), cv2.COLORMAP_TURBO))
            _prog("  + elevation heatmap saved")
        else:
            _heat_p = None
        all_files = [
            (f"{obj_id}.glb",                      "primary 3D mesh"),
            (f"{obj_id}_albedo.jpg",                "albedo texture"),
            (f"{obj_id}_normals.png",               "PS normal map"),
            (f"{obj_id}_depth.tif",                 "depth map 32-bit"),
            (f"{obj_id}_roughness.png",             "roughness map"),
            (f"{obj_id}_specular_variance.png",     "specular variance"),
            (f"{obj_id}_elevation_heatmap.png",     "elevation heatmap (turbo)"),
            (f"{obj_id}_iiif_manifest.json",        "IIIF 3D manifest"),
            (f"{obj_id}_morphosource_metadata.json","MorphoSource metadata"),
            ("README.txt",                          "deposit instructions"),
        ]
        # Only include heatmap in ZIP if it was generated
        if _heat_p is None:
            all_files = [f for f in all_files if "heatmap" not in f[0]]
        with _zf.ZipFile(p, 'w', compression=_zf.ZIP_DEFLATED) as zf:
            for fn, desc in all_files:
                fp = os.path.join(tmp, fn)
                if os.path.exists(fp):
                    zf.write(fp, fn)
                    _prog(f"  + {fn} ({os.path.getsize(fp)//1024} KB) — {desc}")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return p


# ── CIDOC-CRM / Linked Art ───────────────────────────────────────────────

def cidoc_record(obj_id: str, *, normal_engine: str, depth_engine: str, light_count: int,
                 w: int, h: int) -> dict:
    obj_id = obj_id or "HELIO_OBJECT_001"
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return {
        "@context": {"crm": "http://www.cidoc-crm.org/cidoc-crm/",
                     "crmdig": "http://www.ics.forth.gr/isl/CRMdig/",
                     "la": "https://linked.art/ns/terms/", "xsd": "http://www.w3.org/2001/XMLSchema#"},
        "@type": "crm:E22_Human-Made_Object",
        "@id": f"urn:helio:{obj_id}",
        "crm:P1_is_identified_by": {"@type": "crm:E42_Identifier",
                                    "crm:P190_has_symbolic_content": obj_id},
        "crmdig:L11_had_output": {
            "@type": "crmdig:D9_Data_Object",
            "crmdig:L17_measured_thing_of_type": "PhotometricStereo3DScan",
            "crm:P4_has_time-span": {"crm:P82a_begin_of_the_begin": now},
            "helio:capture_parameters": {
                "normal_solver": normal_engine, "depth_integrator": depth_engine,
                "light_count": int(light_count), "image_width": int(w), "image_height": int(h),
                "software": "Helio", "export_timestamp": now}}}


def write_cidoc(path: str, **kw) -> str:
    with open(path, "w") as f:
        json.dump(cidoc_record(**kw), f, indent=2)
    return path


# ── light vectors ────────────────────────────────────────────────────────

def light_vectors_csv(lights, path: str) -> str:
    """Image_Index, L_X, L_Y, L_Z (+Y up, the calibrated frame)."""
    if lights is None:
        raise NothingToExport("No calibration loaded.")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Image_Index", "L_X", "L_Y", "L_Z"])
        for i, v in enumerate(lights):
            w.writerow([i, round(float(v[0]), 6), round(float(v[1]), 6), round(float(v[2]), 6)])
    return path


# ── Dublin Core TIFF ─────────────────────────────────────────────────────

def dublin_core_xmp(obj_id: str, engine: str) -> bytes:
    """A minimal XMP packet of Dublin Core (identifier, date, format,
    description, subject); text escaped so an "&" or "<" stays valid XML."""
    xmp = ('<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>'
           '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
           '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
           '<rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/">'
           f'<dc:identifier>{escape(obj_id or "")}</dc:identifier>'
           f'<dc:date>{time.strftime("%Y-%m-%d")}</dc:date>'
           '<dc:format>image/tiff</dc:format>'
           f'<dc:description>Photometric stereo render ({escape(engine)}) produced by Helio.'
           '</dc:description>'
           '<dc:subject>Photometric Stereo; Cultural Heritage; 3D Imaging</dc:subject>'
           '</rdf:Description></rdf:RDF></x:xmpmeta><?xpacket end="w"?>')
    return xmp.encode("utf-8")


def dublin_core_tiff(rgb, path: str, obj_id: str, engine: str) -> str:
    """The render as a 300-dpi TIFF with the Dublin Core XMP packet in tag
    700, where Bridge, exiftool and archives read it (KD-038)."""
    from PIL import Image, TiffImagePlugin
    info = TiffImagePlugin.ImageFileDirectory_v2()
    info[700] = dublin_core_xmp(obj_id, engine)
    info.tagtype[700] = 1                      # BYTE, as the XMP-in-TIFF spec has it
    Image.fromarray(np.asarray(rgb)).save(path, format="TIFF", dpi=(300, 300), tiffinfo=info)
    return path


# ── Zenodo deposit ───────────────────────────────────────────────────────

def zenodo_package(p: str, obj_id: str, *, normal_engine: str, depth_engine: str, lights,
                   images, meta: dict | None = None, licence: str = "CC-BY-4.0",
                   provenance_app=None) -> dict:
    """A Zenodo-ready ZIP: CIDOC record, light vectors, provenance, zenodo.json
    and datacite.json (the deposition metadata), a README naming any field
    left blank, and a checksummed manifest of the package as shipped."""
    from helio.io.deposit import (LICENCES, build_datacite_metadata, build_file_manifest,
                                  build_zenodo_metadata)
    obj_id = obj_id or "HELIO_OBJECT"
    meta = dict(meta or {})
    tmp = tempfile.mkdtemp()
    try:
        doc = {"@context": {"crm": "http://www.cidoc-crm.org/cidoc-crm/"},
               "@type": "crm:E22_Human-Made_Object", "@id": f"urn:helio:{obj_id}",
               "helio:session": {"solver": normal_engine, "depth": depth_engine,
                                 "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                 "software": "Helio"}}
        with open(os.path.join(tmp, "metadata_cidoc.jsonld"), "w") as f:
            json.dump(doc, f, indent=2)
        if lights is not None:
            with open(os.path.join(tmp, "light_vectors.csv"), "w", newline="") as f:
                cw = csv.writer(f)
                cw.writerow(["Image_Index", "L_X", "L_Y", "L_Z"])
                for i, v in enumerate(lights):
                    cw.writerow([i, v[0], v[1], v[2]])
        readme_p = os.path.join(tmp, "README.txt")
        with open(readme_p, "w") as f:
            f.write(f"Helio — Zenodo Deposit Package\n{'=' * 45}\n")
            f.write(f"Object ID    : {obj_id}\n")
            f.write(f"Date         : {time.strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"Normal solver: {normal_engine}\n")
            f.write(f"Depth engine : {depth_engine}\n")
            f.write(f"Image count  : {len(list(images or []))}\n\n")
            f.write("Files included:\n  metadata_cidoc.jsonld — CIDOC-CRM Linked Art record\n"
                    "  light_vectors.csv     — calibrated light direction matrix\n"
                    "  helio_provenance.json — full provenance: source image hashes,\n"
                    "                          light vectors, solvers and library versions\n"
                    "  README.txt            — this file\n\n")
            f.write("To add 3D/image exports to this package:\n  Export OBJ/GLB/TIFFs separately "
                    "and add them to the ZIP before depositing.\n")
        if provenance_app is not None:
            from helio.io.provenance import write_provenance
            write_provenance(provenance_app, tmp)
        with open(os.path.join(tmp, "zenodo.json"), "w") as f:
            json.dump(build_zenodo_metadata(meta, obj_id, licence), f, indent=2)
        with open(os.path.join(tmp, "datacite.json"), "w") as f:
            json.dump(build_datacite_metadata(meta, obj_id, licence), f, indent=2)
        missing = [lbl for k, lbl in
                   (("operator", "Creator (Archive → Capture Metadata Sheet)"),
                    ("institution", "Institution / publisher"),
                    ("title", "Descriptive title"), ("capture_date", "Capture date"))
                   if not str(meta.get(k, "") or "").strip()]
        if not obj_id or obj_id == "HELIO_OBJECT":
            missing.append("Object identifier (accession number)")
        with open(readme_p, "a") as f:
            f.write("\nDeposition metadata\n")
            f.write("  zenodo.json    — upload with the files; the web form reads the\n"
                    "                   same fields, so it doubles as a filling-in sheet\n")
            f.write("  datacite.json  — DataCite core, for repositories that are not Zenodo\n")
            f.write("  manifest.csv   — every file with its size and SHA-256\n\n")
            f.write(f"Licence declared: {LICENCES.get(licence, LICENCES['CC-BY-4.0'])[1]}\n\n")
            if missing:
                f.write("STILL TO SUPPLY — these were left blank rather than guessed:\n")
                for m in missing:
                    f.write(f"  - {m}\n")
                f.write("\n")
            else:
                f.write("All core deposition fields are present.\n\n")
        names = sorted(os.listdir(tmp))
        with open(os.path.join(tmp, "manifest.csv"), "w") as f:
            f.write(build_file_manifest([(os.path.join(tmp, n), n) for n in names]))
        with _zf.ZipFile(p, "w", _zf.ZIP_DEFLATED) as zf:
            for fname in os.listdir(tmp):
                zf.write(os.path.join(tmp, fname), fname)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"path": p, "missing": missing}


# ── ImageJ data: the light cube and the engine stack ─────────────────────

def light_cube(paths, out: str, *, want_shape, crop=None, shifts=None, mm_per_px: float = 0.0,
               obj: str = "helio", notes=(), lights=None, light_kind: str = "",
               progress=None, base=None) -> dict:
    """The source frames as an ImageJ cube registered to the solve: common
    shape, the processing crop and the alignment shifts replayed, the
    photometric aids deliberately not (radiometry kept).  An .lp goes beside
    it unless the lights are only PCA estimates.  Raises ValueError if the
    rebuilt frames do not register with the solve.  *base* (w, h): the size
    the solve read the frames at, when it worked below full size."""
    from helio.core.frame_read import read_frame
    from helio.core.processing import resize_to_common_shape
    from helio.io.raw_loader import is_raw
    from helio.io.stacks import write_light_cube
    paths = list(paths)
    greys = []
    for i, pth in enumerate(paths):
        if progress:
            progress(f"Reading {i + 1}/{len(paths)}: {os.path.basename(pth)[:28]}",
                     (i + 1) / (len(paths) + 1))
        if is_raw(pth):                  # linear, 16-bit: cv2 would find only the thumbnail in the file
            im = read_frame(pth, cv2.IMREAD_GRAYSCALE, linear=True, sharp=True)
            if im is not None:
                im = np.round(np.clip(im * 257.0, 0, 65535)).astype(np.uint16)
        else:
            im = cv2.imread(pth, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
        if im is None:
            raise ValueError(f"could not read {os.path.basename(pth)}")
        greys.append(im)
    if len({g.shape for g in greys}) == 1:     # kept as read: the common-shape step makes them 8-bit
        frames = greys
    else:
        arr, _bh, _bw = resize_to_common_shape(greys)
        frames = [np.asarray(f) for f in arr]
    if base and tuple(frames[0].shape[1::-1]) != tuple(base):      # the solve's working size
        frames = [cv2.resize(f, tuple(base), interpolation=cv2.INTER_AREA) for f in frames]
    if crop:
        cx1, cy1, cx2, cy2 = crop
        frames = [f[cy1:cy2, cx1:cx2] for f in frames]
    shifts = list(shifts or [])
    aligned = "not applied"
    if shifts and len(shifts) == len(frames):
        for i, (dx, dy) in enumerate(shifts):
            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                continue
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            frames[i] = cv2.warpAffine(frames[i], M, (frames[i].shape[1], frames[i].shape[0]),
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        aligned = f"applied, {len(shifts)} shifts"
    elif shifts:
        aligned = f"NOT applied — {len(shifts)} shifts recorded for {len(frames)} frames"
    got = tuple(frames[0].shape[:2])
    want = tuple(int(v) for v in want_shape)
    if got != want:
        raise ValueError(f"The rebuilt frames are {got[1]}x{got[0]} but the solve is "
                         f"{want[1]}x{want[0]}. A cube that does not register with the normals "
                         "would give a Z-profile from the wrong point on the object, so it has "
                         "not been written. Re-process the capture and export again.")
    vecs = [tuple(float(c) for c in v) for v in lights] if lights is not None else []
    info = [f"Alignment: {aligned}",
            "Photometric pre-processing used by the solve but deliberately NOT applied here "
            "(radiometry preserved):"] + [f"  - {n}" for n in notes]
    info.append(f"Light vectors: {len(vecs)} ({light_kind or 'none'})")
    write_light_cube(out, frames, [os.path.basename(p) for p in paths], mm_per_px=mm_per_px,
                     object_id=obj, info_lines=info)
    lp = ""
    if vecs and light_kind != "estimated":
        from helio.io.sidecars import write_lp
        lp = write_lp(os.path.splitext(out)[0] + ".lp", vecs, paths)
    return {"path": out, "lp": lp, "frames": len(frames), "shape": want}


def engine_stack(draw, engines, path: str, *, mm_per_px: float = 0.0, full_w: int = 0,
                 obj: str = "helio", rot_k: int = 0, progress=None) -> dict:
    """Every view as one ImageJ stack, each slice named for its view, the
    pixel size in mm (scaled to the slices) so Fiji measures in mm."""
    from helio.io.stacks import write_engine_stack
    frames, labels, failed = [], [], []
    engines = list(engines)
    for i, eng in enumerate(engines):
        if progress:
            progress(f"Rendering {i + 1}/{len(engines)}: {eng[:34]}", (i + 1) / len(engines))
        try:
            img = np.asarray(draw(eng))
            k = int(rot_k) % 4
            if k:
                img = np.rot90(img, k=k)
            if img.ndim == 2:
                img = np.repeat(img[..., None], 3, axis=2)
            frames.append(np.ascontiguousarray(img[..., :3].astype(np.uint8)))
            labels.append(eng)
        except Exception as ex:
            failed.append(f"{eng}: {ex}")
    if not frames:
        raise NothingToExport("No engine rendered. " + "; ".join(failed[:3]))
    ref = frames[0].shape
    resized = 0
    for i, f in enumerate(frames):
        if f.shape != ref:
            frames[i] = cv2.resize(f, (ref[1], ref[0]), interpolation=cv2.INTER_AREA)
            resized += 1
    mm = float(mm_per_px or 0.0)
    if mm > 0 and full_w:
        mm = mm * (full_w / float(ref[1]))
    write_engine_stack(path, frames, labels, mm_per_px=mm, object_id=obj)
    return {"path": path, "frames": len(frames), "size": (ref[1], ref[0]), "mm_per_px": mm,
            "resized": resized, "failed": failed}
