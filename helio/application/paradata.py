"""Paradata: how this was made, written with what was made.

The UK Heritage 3D Data at Risk report (August 2026) found 48% of organisations
record no paradata and that what there is lives in spreadsheets and platform
descriptions.  Helio knows all of it at export, so it goes beside the files:
which photographs (with checksums, so a deposit can be checked against them),
the lights, every processing setting, how the colour was handled, and each file
written with its role, master or derivative.  JSON for machines, a read-me for
people.  GUI-free.
"""
from __future__ import annotations

import dataclasses
import json
import os
import time

SCHEMA = "helio-paradata/1"

#: What each format is, for a reader of the deposit.
ROLES = {"linear_masters": "master", "usd_layered": "master", "pbr": "master", "heritage": "master",
         "geotiff": "master", "hsh": "master", "ptm": "master",
         "glb": "derivative: web", "glb_mesh": "derivative: web", "rti_web": "derivative: web",
         "rti_viewer": "derivative: web", "hf_viewer": "derivative: web", "relight_web": "derivative: web",
         "iiif": "derivative: web", "iiif_images": "derivative: web", "deepzoom": "derivative: web",
         "usdz": "derivative: web", "stl": "derivative: print", "print": "derivative: print",
         "facsimile": "derivative: print", "pub_sheet": "report", "plate": "report", "plate_a3": "report",
         "figures": "report", "condition": "report", "colour_validation": "report", "lido": "metadata",
         "dublin_core": "metadata", "cidoc": "metadata", "provenance": "metadata", "lp": "metadata",
         "zenodo": "metadata", "calibration": "metadata"}


def _sha256(path: str) -> str | None:
    from helio.io.manifest import sha256_of
    try:
        return sha256_of(path)
    except OSError:
        return None


def frame_records(paths) -> list[dict]:
    """Each photograph: name, size, checksum, whether RAW, its colour profile."""
    from helio.io.colour_space import profile_name
    from helio.io.raw_loader import is_raw
    out = []
    for p in paths:
        raw = is_raw(p)
        try:
            size = os.path.getsize(p)
        except OSError:
            size = None
        out.append(dict(file=os.path.basename(p), bytes=size, sha256=_sha256(p), raw=raw,
                        colour_profile="" if raw else profile_name(p)))
    return out


def colour_record(frames: list[dict], *, tone: str, solved: bool, check: dict | None) -> dict:
    """How the colour was treated, in the terms a reader needs."""
    raw = [f["raw"] for f in frames]
    profiles = sorted({f["colour_profile"] for f in frames if f["colour_profile"]})
    return dict(
        input="RAW, decoded linear with the camera's white balance, to sRGB primaries" if raw and all(raw)
        else "mixed RAW and encoded files" if any(raw) else "encoded files",
        input_profiles=profiles, working_space="sRGB",
        colour_map="solved reflectance (relative, not calibrated to a white)" if solved
        else "median of the frames",
        solve_values=tone, chart_check=check or None)


def outputs(results, base: str) -> list[dict]:
    """Every file written: (key, files, problem) from the export run."""
    out = []
    for key, files, _problem in results:
        for f in files or []:
            if os.path.isdir(f):
                continue
            out.append(dict(file=os.path.relpath(f, base), format=key, role=ROLES.get(key, "derivative"),
                            bytes=os.path.getsize(f) if os.path.exists(f) else None, sha256=_sha256(f)))
    return out


def build(*, record: dict, images, lights, light_source: str, settings, tone: str, solved: bool,
          check: dict | None, results, base: str) -> dict:
    """The paradata.  *record*: publish.processing_record, the object and capture fields."""
    frames = frame_records(images)
    return dict(
        schema=SCHEMA, generated=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        object=record.get("object", {}), capture=record.get("capture", {}), frames=frames,
        lights=dict(source=light_source or "", vectors=None if lights is None
                    else [[round(float(v), 6) for v in row] for row in lights]),
        processing=dataclasses.asdict(settings) if dataclasses.is_dataclass(settings) else {},
        colour=colour_record(frames, tone=tone, solved=solved, check=check),
        software=record.get("software", {}), outputs=outputs(results, base))


def readme(p: dict) -> str:
    """The same, for a person who has the folder and nothing else."""
    o, c, col = p["object"], p["capture"], p["colour"]
    L = [f"{o.get('title') or o.get('id') or 'Untitled'}", "", f"Made with {p['software'].get('name', 'Helio')} "
         f"{p['software'].get('version', '')} on {p['generated']}.", "",
         "HOW IT WAS MADE",
         f"  Photographs: {len(p['frames'])}" + (f", lights {p['lights']['source']}" if p['lights']['source'] else ""),
         f"  Normals: {p['processing'].get('normal_engine', '')}; heights: {p['processing'].get('depth_engine', '')}"]
    if c.get("scale_mmpx"):
        L.append(f"  Scale: {c['scale_mmpx']} mm per pixel")
    L += ["", "COLOUR", f"  Input: {col['input']}"]
    if col["input_profiles"]:
        L.append(f"  Files tagged: {', '.join(col['input_profiles'])}, converted to {col['working_space']}")
    L.append(f"  Colour map: {col['colour_map']}")
    if col["chart_check"]:
        L.append(f"  Chart check: {col['chart_check'].get('mean')} mean ΔE00")
    roles: dict[str, list] = {}
    for f in p["outputs"]:
        roles.setdefault(f["role"], []).append(f["file"])
    L += ["", "WHAT IS HERE"]
    for role in sorted(roles):
        L.append(f"  {role}:")
        L += [f"    {f}" for f in roles[role]]
    L += ["", "Every setting, light and checksum is in the paradata JSON beside this file.",
          "The photographs themselves are not included: keep them with this folder."]
    return "\n".join(L) + "\n"


def write(p: dict, base: str, stem: str) -> list[str]:
    js, txt = os.path.join(base, f"{stem}_paradata.json"), os.path.join(base, f"{stem}_README.txt")
    with open(js, "w", encoding="utf-8") as f:
        json.dump(p, f, indent=2, ensure_ascii=False, default=str)
    with open(txt, "w", encoding="utf-8") as f:
        f.write(readme(p))
    return [js, txt]
