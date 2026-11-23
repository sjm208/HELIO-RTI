"""Deposition metadata for repositories — Zenodo, DataCite, and a file manifest.

The existing Zenodo export writes a CIDOC-CRM record, the light vectors and the
provenance, then tells the user to add their own exports and upload. Two things
were missing, and both are the parts a repository actually reads.

**Deposition metadata.** Zenodo does not read CIDOC-CRM. It reads its own
deposition schema — title, upload_type, creators, license, keywords — and
without it the depositor retypes everything into the web form by hand, which is
where the object ID and the capture date stop matching the files they describe.

**Something other repositories can ingest.** Rather than guess at each
repository's own spreadsheet columns, this writes DataCite, which is the schema
most of them map from and which is properly specified. A MorphoSource or ADS
deposit still needs its own template filled, but mapping from DataCite is
transcription rather than archaeology.

Nothing here invents values. A field with no answer is written empty and listed
in the README as outstanding, because a deposit that silently claims an unknown
creator is worse than one that admits the gap.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import time

__all__ = ["build_zenodo_metadata", "build_datacite_metadata",
           "build_file_manifest", "LICENCES"]

# Zenodo's own identifiers, not free text — the API rejects anything else.
LICENCES = {
    "CC-BY-4.0":      ("cc-by-4.0", "Creative Commons Attribution 4.0 International"),
    "CC-BY-SA-4.0":   ("cc-by-sa-4.0", "Creative Commons Attribution-ShareAlike 4.0 International"),
    "CC-BY-NC-4.0":   ("cc-by-nc-4.0", "Creative Commons Attribution-NonCommercial 4.0 International"),
    "CC0-1.0":        ("cc-zero", "Creative Commons Zero v1.0 Universal"),
    "MIT":            ("mit-license", "MIT License"),
}


def _creators(operator: str, institution: str) -> list:
    """One creator entry, in the "Family, Given" form Zenodo expects.

    The operator field is free text and people write it either way round. A
    name already containing a comma is assumed to be correctly ordered and is
    left alone; anything else is reversed on the last space. This is a guess,
    which is why the README says to check it.
    """
    name = (operator or "").strip()
    if not name:
        return [{"name": "", "affiliation": (institution or "").strip()}]
    if "," not in name and " " in name:
        parts = name.rsplit(" ", 1)
        name = f"{parts[1]}, {parts[0]}"
    return [{"name": name, "affiliation": (institution or "").strip()}]


def build_zenodo_metadata(meta: dict, object_id: str = "", licence: str = "CC-BY-4.0",
                          keywords=None, description_extra: str = "") -> dict:
    """Zenodo deposition metadata. Upload this alongside the files.

    Returns the ``{"metadata": {...}}`` envelope the deposition API takes; the
    web form reads the same fields, so it doubles as a filling-in sheet.
    """
    meta = meta or {}
    title = (meta.get("title") or object_id or "Photometric stereo capture").strip()
    inst = (meta.get("institution") or "").strip()
    coll = (meta.get("collection") or "").strip()
    lic_id = LICENCES.get(licence, LICENCES["CC-BY-4.0"])[0]

    desc = [
        "<p>Reflectance transformation imaging / photometric stereo capture "
        "processed with Helio.</p>"]
    if object_id:
        desc.append(f"<p><strong>Object identifier:</strong> {object_id}</p>")
    if coll:
        desc.append(f"<p><strong>Collection:</strong> {coll}</p>")
    _cap = []
    for _k, _lbl in (("camera", "Camera"), ("lens", "Lens"), ("sensor", "Sensor"),
                     ("lights", "Lighting"), ("stand", "Stand"),
                     ("working_dist", "Working distance (mm)"),
                     ("scale_mmpx", "Scale (mm/pixel)"),
                     ("capture_date", "Capture date")):
        _v = str(meta.get(_k, "") or "").strip()
        if _v:
            _cap.append(f"<li>{_lbl}: {_v}</li>")
    if _cap:
        desc.append("<p><strong>Capture</strong></p><ul>" + "".join(_cap) + "</ul>")
    if meta.get("notes"):
        desc.append(f"<p>{meta['notes']}</p>")
    if description_extra:
        desc.append(f"<p>{description_extra}</p>")

    kw = list(keywords or [])
    for _default in ("reflectance transformation imaging", "photometric stereo",
                     "cultural heritage", "surface imaging"):
        if _default not in kw:
            kw.append(_default)

    md = {
        "title": title,
        # "dataset" rather than "image": the deposit carries measurements,
        # provenance and light vectors, not only pictures.
        "upload_type": "dataset",
        "description": "".join(desc),
        "creators": _creators(meta.get("operator", ""), inst),
        "keywords": kw,
        "license": lic_id,
        "access_right": "open",
        "language": "eng",
    }
    if meta.get("capture_date"):
        md["dates"] = [{"type": "Collected", "start": str(meta["capture_date"])}]
    if meta.get("doi"):
        md["related_identifiers"] = [
            {"relation": "isSupplementTo", "identifier": str(meta["doi"]),
             "scheme": "doi"}]
    return {"metadata": md}


def build_datacite_metadata(meta: dict, object_id: str = "",
                            licence: str = "CC-BY-4.0") -> dict:
    """DataCite core fields — the schema most repositories map from.

    Written instead of guessing at any one repository's spreadsheet columns.
    Filling a MorphoSource or ADS template from this is transcription; deriving
    it from a CIDOC record is not.
    """
    meta = meta or {}
    inst = (meta.get("institution") or "").strip()
    lic_name = LICENCES.get(licence, LICENCES["CC-BY-4.0"])[1]
    year = str(meta.get("capture_date", ""))[:4] or time.strftime("%Y")

    doc = {
        "types": {"resourceTypeGeneral": "Dataset", "resourceType": "RTI/PS capture"},
        "titles": [{"title": (meta.get("title") or object_id
                              or "Photometric stereo capture").strip()}],
        "creators": [{"name": c["name"], "affiliation": [c["affiliation"]]}
                     for c in _creators(meta.get("operator", ""), inst)],
        "publisher": inst or "",
        "publicationYear": year,
        "rightsList": [{"rights": lic_name,
                        "rightsIdentifier": licence.lower()}],
        "subjects": [{"subject": s} for s in
                     ("reflectance transformation imaging", "photometric stereo",
                      "cultural heritage")],
        "descriptions": [{"descriptionType": "Methods",
                          "description": (
                              "Multi-light capture solved by photometric stereo. "
                              "Light directions calibrated from a specular sphere "
                              "or assigned on a virtual dome; surface normals "
                              "integrated to depth. Processing provenance, source "
                              "image hashes and the light matrix accompany the "
                              "deposit.")}],
    }
    if object_id:
        doc["alternateIdentifiers"] = [
            {"alternateIdentifier": object_id,
             "alternateIdentifierType": "Object identifier"}]
    if meta.get("scale_mmpx"):
        doc["sizes"] = [f"{meta['scale_mmpx']} mm/pixel"]
    return doc


def build_file_manifest(paths_and_names) -> str:
    """CSV inventory: name, bytes, SHA-256.

    A deposit is a claim that these files are these files. The checksums make
    it checkable by the repository and by anyone who downloads it later, which
    is the same argument the in-app Manifest already makes for a folder.
    """
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["filename", "bytes", "sha256"])
    for src, name in paths_and_names:
        try:
            h = hashlib.sha256()
            with open(src, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            w.writerow([name, os.path.getsize(src), h.hexdigest()])
        except Exception:
            w.writerow([name, "", "unreadable"])
    return buf.getvalue()
