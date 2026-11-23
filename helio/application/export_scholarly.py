"""Records in each discipline's own standard, GUI-free.

* nomisma_record: a coin as a row of Nomisma (nmo:) properties, CSV and JSON-LD.
* epidoc_record: an inscription or text-bearing object as a TEI EpiDoc file:
  its description filled in from the capture details, the images as its
  facsimile, an empty edition to transcribe into.
* iiif_image_manifest: the views as a IIIF Presentation 3 manifest of still
  images, for Mirador and the Universal Viewer.  (export_web.iiif_manifest
  is the 3D one.)

Every field comes from Capture details, chiefly the object profile's own
(helio.domain.object_profiles).  Blank fields are left out, never invented.
"""
from __future__ import annotations

import csv
import datetime
import json
import os
from xml.sax.saxutils import escape

PLACEHOLDER_BASE = "https://example.org/iiif"


def _m(meta: dict, key: str) -> str:
    return str(dict(meta or {}).get(key, "") or "").strip()


# ── Nomisma ──────────────────────────────────────────────────────────────

# (column / property, capture-details key)
NOMISMA = (
    ("dcterms:identifier", "_id"), ("dcterms:title", "title"),
    ("nmo:hasDenomination", "denomination"), ("nmo:hasAuthority", "authority"),
    ("nmo:hasMint", "mint"), ("nmo:hasMaterial", "material"), ("dcterms:date", "date"),
    ("nmo:hasWeight", "weight_g"), ("nmo:hasDiameter", "diameter_mm"), ("nmo:hasAxis", "die_axis"),
    ("nmo:hasLegend", "legend"), ("face", "face"), ("nmo:hasTypeSeriesItem", "reference"),
    ("nmo:hasCollection", "collection"), ("dcterms:rightsHolder", "institution"),
)


def nomisma_record(base: str, meta: dict, obj_id: str, images=()) -> list[str]:
    """*base*.csv and *base*.jsonld: the coin's Nomisma properties.

    Values are the text as typed; weight in grams, diameter in millimetres and
    die axis in clock hours, as Nomisma states them.  Where a value should be a
    Nomisma URI (a mint, a denomination), the text is kept for you to match:
    nothing is looked up."""
    meta = dict(meta or {})
    vals = {col: (obj_id if key == "_id" else _m(meta, key)) for col, key in NOMISMA}
    with open(base + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([c for c, _k in NOMISMA] + ["images"])
        w.writerow([vals[c] for c, _k in NOMISMA] + [" ".join(os.path.basename(p) for p in images)])
    doc = {"@context": {"nmo": "http://nomisma.org/ontology#", "dcterms": "http://purl.org/dc/terms/",
                        "xsd": "http://www.w3.org/2001/XMLSchema#"},
           "@id": f"#{obj_id}", "@type": "nmo:NumismaticObject"}
    for col, _k in NOMISMA:
        v = vals[col]
        if not v or col == "face":
            continue
        if col in ("nmo:hasWeight", "nmo:hasDiameter"):
            try:
                doc[col] = {"@value": float(v), "@type": "xsd:decimal"}
                continue
            except ValueError:
                pass
        if col == "nmo:hasAxis":
            try:
                doc[col] = {"@value": int(float(v)), "@type": "xsd:integer"}
                continue
            except ValueError:
                pass
        doc[col] = v
    face = _m(meta, "face").lower()
    if images and face in ("obverse", "reverse"):
        doc["nmo:hasObverse" if face == "obverse" else "nmo:hasReverse"] = {
            "foaf:depiction": [os.path.basename(p) for p in images]}
        doc["@context"]["foaf"] = "http://xmlns.com/foaf/0.1/"
    with open(base + ".jsonld", "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    return [base + ".csv", base + ".jsonld"]


# ── EpiDoc ───────────────────────────────────────────────────────────────

def epidoc_record(path: str, meta: dict, obj_id: str, *, images=(), w: int = 0, h: int = 0,
                  mm_per_px: float = 0.0, software: str = "Helio") -> str:
    """A TEI EpiDoc file: header from the capture details, the images as the facsimile
    (one surface, its size in millimetres once a scale is set), and an empty
    edition division to transcribe into."""
    meta = dict(meta or {})
    e = lambda k: escape(_m(meta, k))                              # noqa: E731
    title = escape(_m(meta, "title") or obj_id)
    idnos = [f'<idno type="filename">{escape(obj_id)}</idno>']
    for key, typ in (("text_ref", "corpus"), ("reference", "inventory"), ("shelfmark", "shelfmark")):
        if _m(meta, key):
            idnos.append(f'<idno type="{typ}">{e(key)}</idno>')
    repo = _m(meta, "repository") or _m(meta, "institution")
    ms_id = [f"<repository>{escape(repo)}</repository>"] if repo else []
    ms_id.append(f"<idno>{e('shelfmark') or e('reference') or escape(obj_id)}</idno>")
    support = (f"<support><material>{e('material')}</material></support>" if _m(meta, "material")
               else "<support/>")
    dims = ""
    if mm_per_px and w and h:
        dims = (f'<dimensions unit="mm"><height>{h * mm_per_px:.1f}</height>'
                f'<width>{w * mm_per_px:.1f}</width></dimensions>')
    lay = f"<layoutDesc><layout>{e('field_mm')}</layout></layoutDesc>" if _m(meta, "field_mm") else ""
    hand = ""
    if _m(meta, "letter_height_mm") or _m(meta, "script"):
        hand = ("<handDesc><handNote>" + (f'<height unit="mm">{e("letter_height_mm")}</height>'
                                          if _m(meta, "letter_height_mm") else "")
                + (f"<p>{e('script')}</p>" if _m(meta, "script") else "") + "</handNote></handDesc>")
    origin = []
    if _m(meta, "findspot"):
        origin.append(f"<provenance type=\"found\"><placeName>{e('findspot')}</placeName></provenance>")
    hist = ("<history><origin>" + (f"<origDate>{e('date')}</origDate>" if _m(meta, "date") else "")
            + "</origin>" + "".join(origin) + "</history>")
    lang = _m(meta, "language")
    zoom = mm_per_px if (mm_per_px and w and h) else 0
    surface = (f'<surface ulx="0" uly="0" lrx="{w}" lry="{h}">' if w and h else "<surface>")
    graphics = "".join(f'<graphic url="{escape(os.path.basename(p))}"/>' for p in images)
    when = datetime.date.today().isoformat()
    xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<?xml-model href="https://epidoc.stoa.org/schema/latest/tei-epidoc.rng" schematypens="http://relaxng.org/ns/structure/1.0"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0" xml:lang="en">
  <teiHeader>
    <fileDesc>
      <titleStmt><title>{title}</title></titleStmt>
      <publicationStmt><authority>{e('institution') or 'Unpublished'}</authority>{''.join(idnos)}</publicationStmt>
      <sourceDesc>
        <msDesc>
          <msIdentifier>{''.join(ms_id)}</msIdentifier>
          <physDesc>
            <objectDesc>
              <supportDesc>{support}{('<extent>' + dims + '</extent>') if dims else ''}</supportDesc>
              {lay}
            </objectDesc>
            {hand}
          </physDesc>
          {hist}
        </msDesc>
      </sourceDesc>
    </fileDesc>
    <encodingDesc><p>Surface imaged by photometric stereo with {escape(software)}{
        f"; 1 px = {zoom:.5f} mm" if zoom else ""}.</p></encodingDesc>
    {f'<profileDesc><langUsage><language ident="und">{escape(lang)}</language></langUsage></profileDesc>'
     if lang else ''}
    <revisionDesc><change when="{when}">Record started by {escape(software)} from the capture details.</change></revisionDesc>
  </teiHeader>
  <facsimile>
    {surface}{graphics}</surface>
  </facsimile>
  <text>
    <body>
      <div type="edition" xml:space="preserve">
        <ab/>
      </div>
      <div type="apparatus"><p/></div>
      <div type="translation"><p/></div>
      <div type="commentary"><p>{e('notes')}</p></div>
    </body>
  </text>
</TEI>
'''
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)
    return path


# ── IIIF (still images) ──────────────────────────────────────────────────

def iiif_image_manifest(folder: str, obj_id: str, meta: dict, images, *, base_url: str = "",
                        mm_per_px: float = 0.0) -> list[str]:
    """manifest.json in *folder*: one canvas per (label, file, w, h) in *images*, the files
    already in *folder*.  IIIF wants absolute addresses: with no *base_url* the
    placeholder https://example.org/iiif/<object> is written, and the read-me says
    which line to change once the folder is on a web server."""
    meta = dict(meta or {})
    base = (base_url or f"{PLACEHOLDER_BASE}/{obj_id}").rstrip("/")
    md = [{"label": {"en": [lab]}, "value": {"none": [_m(meta, k)]}}
          for lab, k in (("Title", "title"), ("Institution", "institution"),
                         ("Collection", "collection"), ("Repository", "repository"),
                         ("Shelfmark", "shelfmark"), ("Folio", "folio"), ("Date", "date"),
                         ("Material", "material"), ("Reference", "reference"),
                         ("Text reference", "text_ref"), ("Operator", "operator"),
                         ("Captured", "capture_date")) if _m(meta, k)]
    canvases = []
    for i, (lab, fname, w, h) in enumerate(images, 1):
        cid = f"{base}/canvas/{i}"
        canvas = {"id": cid, "type": "Canvas", "label": {"en": [lab]}, "width": int(w), "height": int(h),
                  "items": [{"id": f"{cid}/page", "type": "AnnotationPage", "items": [{
                      "id": f"{cid}/page/image", "type": "Annotation", "motivation": "painting",
                      "target": cid, "body": {"id": f"{base}/{fname}", "type": "Image",
                                              "format": "image/jpeg", "width": int(w),
                                              "height": int(h)}}]}]}
        if mm_per_px:
            canvas["service"] = [{"@context": "http://iiif.io/api/annex/services/physdim/1/context.json",
                                  "profile": "http://iiif.io/api/annex/services/physdim",
                                  "physicalScale": float(mm_per_px), "physicalUnits": "mm"}]
        canvases.append(canvas)
    man = {"@context": "http://iiif.io/api/presentation/3/context.json",
           "id": f"{base}/manifest.json", "type": "Manifest",
           "label": {"en": [_m(meta, "title") or obj_id]},
           "summary": {"en": ["Views of the surface from photometric stereo (Helio): the colour, and "
                              "views that show its relief."]},
           "metadata": md, "items": canvases}
    if _m(meta, "institution"):
        man["requiredStatement"] = {"label": {"en": ["Attribution"]},
                                    "value": {"en": [_m(meta, "institution")]}}
    p = os.path.join(folder, "manifest.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(man, f, indent=2, ensure_ascii=False)
    r = os.path.join(folder, "README.txt")
    with open(r, "w", encoding="utf-8") as f:
        f.write(f"""IIIF manifest for {_m(meta, 'title') or obj_id}

manifest.json lists {len(canvases)} images in this folder, one canvas each, in
IIIF Presentation 3.  Open it in Mirador or the Universal Viewer.

IIIF needs web addresses.  Every address in manifest.json starts with
    {base}
Put this folder on a web server and replace that text, everywhere in the
file, with the folder's address (a find-and-replace does it).  The server
must allow other sites to read the files (CORS).
""" + ("\nEach canvas carries its physical size: 1 px = %.5f mm.\n" % mm_per_px if mm_per_px else ""))
    return [p, r]
