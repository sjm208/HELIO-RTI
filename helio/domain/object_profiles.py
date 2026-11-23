"""Object profiles: what kind of object this is, and so what Helio puts first.

A profile is chosen in Capture details and kept in the session's capture
details (``object_type``), so it needs nothing new in the session file.  It
does four things, all of them defaults the user can change:

* sets the solver and depth method (through the Process workspace's material
  presets, helio.application.process_tools.PRESETS, where one fits);
* puts a short list of views first in Inspect ("For this object");
* offers a matching export preset in Publish;
* adds the discipline's own fields to Capture details (a coin's weight and
  die axis, a seal's legend, a page's shelfmark).

Nothing is hidden or switched off: every view, export and tool stays where it
was.  "General" changes nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# A field is (label, key, hint, tip), as helio.domain.capture_meta.FIELDS.
_REF = ("Catalogue / reference", "reference", "e.g. RIC II 207; BM 1856,1101.21",
        "A published catalogue number, an inventory number or a database ID.")


@dataclass(frozen=True)
class Profile:
    key: str
    name: str
    blurb: str
    material: str = ""                     # a process_tools.PRESETS key, or "" for none
    settings: dict = field(default_factory=dict)   # ProcessingSettings changes on top
    views: tuple = ()                      # Inspect's "For this object" views, in order
    exports: tuple = ()                    # Publish preset: format keys
    fields: tuple = ()                     # this discipline's Capture details fields
    tools: tuple = ()                      # Research tools worth reaching for first


PROFILES: dict[str, Profile] = {p.key: p for p in (
    Profile("general", "General",
            "No changes: every setting as you left it."),
    Profile("coin", "Coin, medal or token",
            "Shiny metal with a raised design and a rim. Metal-aware solve; views that read worn "
            "relief and legends.",
            material="coin", settings={"metal_aware": True},
            views=("Robust PBR (MVS)", "Exaggerated Shading", "RBF Relight (Shiny / Translucent)",
                   "Normals Unsharp Masking", "Multi-Scale Relief Model (MSRM)",
                   "Specular Enhancement", "Depth map"),
            exports=("plate_a3", "pub_sheet", "nomisma", "iiif_images"),
            fields=(
                ("Face", "face", "Obverse or reverse", ""),
                ("Denomination", "denomination", "e.g. denarius, penny, 2 euro", ""),
                ("Issuer / authority", "authority", "e.g. Hadrian; Republic of Venice", ""),
                ("Mint", "mint", "e.g. Rome, Tower Mint", ""),
                ("Date / period", "date", "e.g. AD 125-128", ""),
                ("Material", "material", "e.g. silver (AR), bronze (AE)", ""),
                ("Weight (g)", "weight_g", "e.g. 3.21", "Weighed, not measured from the images."),
                ("Diameter (mm)", "diameter_mm", "e.g. 18.5",
                 "Measured, or filled in by Legend Unwrap once a scale is set."),
                ("Die axis (h)", "die_axis", "e.g. 6",
                 "Clock hours, 1-12. Research › Two Faces measures it from both sides."),
                ("Legend", "legend", "e.g. HADRIANVS AVGVSTVS", ""),
                _REF,
            ),
            tools=("recto", "unwrap", "register")),
    Profile("seal", "Seal, sealing or matrix",
            "Wax, lead or clay impressions, or the matrix that made them. Views for soft, "
            "translucent relief; the virtual impression of a matrix.",
            material="wax_seal",
            views=("Exaggerated Shading", "RBF Relight (Shiny / Translucent)",
                   "Albedo-Suppressed Raking (Clay Mode)", "Multi-Scale Relief Model (MSRM)",
                   "Subsurface Scattering (SSS)", "Depth map"),
            exports=("plate_a3", "pub_sheet", "line_drawing", "iiif_images"),
            fields=(
                ("Matrix or impression", "seal_kind", "Matrix, impression, or cast", ""),
                ("Seal owner", "seal_owner", "e.g. Abbey of St Albans", ""),
                ("Legend", "legend", "e.g. + SIGILLVM ...", ""),
                ("Material", "material", "e.g. beeswax, lead, clay", ""),
                ("Colour", "colour", "e.g. red, natural", ""),
                ("Attachment", "attachment", "e.g. pendant on tag, applied", ""),
                ("Document", "document", "The charter or object it is attached to", ""),
                ("Date / period", "date", "", ""),
                ("Diameter (mm)", "diameter_mm", "", ""),
                _REF,
            ),
            tools=("impression", "unwrap", "recto")),
    Profile("gem", "Engraved gem, cameo or intaglio",
            "Polished hardstone: highlights everywhere, cut detail under them. Outlier-robust "
            "solve; the virtual impression of an intaglio.",
            settings={"normal_engine": "Sparse-Outlier LS (RPCA)",
                      "depth_engine": "Fast Poisson (FFT)"},
            views=("Specular Enhancement", "RBF Relight (Shiny / Translucent)",
                   "Exaggerated Shading", "Normals Unsharp Masking",
                   "Albedo-Suppressed Raking (Clay Mode)", "Depth map"),
            exports=("plate_a3", "pub_sheet", "iiif_images"),
            fields=(
                ("Intaglio or cameo", "gem_kind", "Intaglio, cameo, or scaraboid", ""),
                ("Stone", "material", "e.g. carnelian, nicolo, glass paste", ""),
                ("Shape", "shape", "e.g. oval, convex face", ""),
                ("Size (mm)", "size_mm", "e.g. 14 x 11 x 3", ""),
                ("Setting", "setting", "e.g. loose, gold ring", ""),
                ("Subject", "subject", "e.g. Victory with wreath", ""),
                ("Date / period", "date", "", ""),
                _REF,
            ),
            tools=("impression", "unwrap")),
    Profile("inscription", "Inscription",
            "Letters cut or cast in stone, metal or clay. Shadow-rejecting solve; views that "
            "find letters by their shape, not their colour.",
            material="stone",
            views=("Ridge and Groove Detector", "Multi-Scale Relief Model (MSRM)",
                   "Negative Openness", "Exaggerated Shading", "Cast-Shadow Raking",
                   "Albedo-Suppressed Raking (Clay Mode)"),
            exports=("epidoc", "line_drawing", "plate_a3", "iiif_images"),
            fields=(
                ("Support", "material", "e.g. marble stele, bronze tablet, clay", ""),
                ("Findspot", "findspot", "e.g. Ephesos, Upper Agora", ""),
                ("Language / script", "language", "e.g. Greek; Latin", ""),
                ("Letter height (mm)", "letter_height_mm", "e.g. 18-22", ""),
                ("Text field (mm)", "field_mm", "e.g. 420 x 310", ""),
                ("Date / period", "date", "", ""),
                ("Text reference", "text_ref", "e.g. EDH HD012345; TM 12345; IG II² 1",
                 "An epigraphic database or corpus number."),
                _REF,
            ),
            tools=("zones", "impression", "toolpaths")),
    Profile("manuscript", "Manuscript, papyrus or page",
            "Flat or cockled sheets of parchment, paper or papyrus. Views for faded ink, dry-point "
            "and ruling.",
            material="manuscript",
            views=("Multi-Light Fusion", "Image Quotient (IQ)", "Surface Curvature (Scratches)",
                   "Raking Composite", "Retinex Enhancement", "Standard PBR"),
            exports=("iiif_images", "epidoc", "plate_a3"),
            fields=(
                ("Repository", "repository", "e.g. British Library", ""),
                ("Shelfmark", "shelfmark", "e.g. Cotton MS Nero D IV", ""),
                ("Folio / fragment", "folio", "e.g. f. 27; P.Oxy. 1.1", ""),
                ("Recto or verso", "side", "Recto, verso", ""),
                ("Support", "material", "e.g. parchment, paper, papyrus", ""),
                ("Script", "script", "e.g. Insular half-uncial", ""),
                ("Date / period", "date", "", ""),
                ("Text reference", "text_ref", "e.g. TM 59974", ""),
            ),
            tools=("micro", "zones", "recto")),
    Profile("binding", "Binding or book object",
            "Covers, boards, clasps and tooling. Views for blind tooling and gold.",
            material="leather", settings={"metal_aware": True},
            views=("Exaggerated Shading", "Multi-Scale Relief Model (MSRM)",
                   "RBF Relight (Shiny / Translucent)", "Surface Curvature (Scratches)",
                   "Specular Enhancement", "Standard PBR"),
            exports=("plate_a3", "pub_sheet", "line_drawing", "iiif_images"),
            fields=(
                ("Repository", "repository", "", ""),
                ("Shelfmark", "shelfmark", "", ""),
                ("Covering", "material", "e.g. calf over wooden boards", ""),
                ("Decoration", "decoration", "e.g. blind-tooled, gilt rolls", ""),
                ("Date / period", "date", "", ""),
                _REF,
            ),
            tools=("toolpaths", "micro")),
    Profile("find", "Small find or artefact",
            "Excavated or collected objects of any material. A balanced solve and views that read "
            "form and use-wear.",
            settings={"normal_engine": "Shadow-Rejection LS", "depth_engine": "Fast Poisson (FFT)"},
            views=("Standard PBR", "Exaggerated Shading", "Multi-Scale Relief Model (MSRM)",
                   "Curvedness", "Ridge and Groove Detector", "Depth map"),
            exports=("plate_a3", "line_drawing", "glb", "iiif_images"),
            fields=(
                ("Site", "site", "e.g. Vindolanda", ""),
                ("Context", "context", "e.g. SU 1043", ""),
                ("Find number", "find_no", "e.g. SF 2231", ""),
                ("Object type", "object_kind", "e.g. brooch, loom weight", ""),
                ("Material", "material", "e.g. copper alloy", ""),
                ("Date / period", "date", "", ""),
                ("Condition", "condition", "e.g. corroded, pin missing", ""),
                _REF,
            ),
            tools=("toolpaths", "micro", "register")),
)}

DEFAULT = "general"


def get(key: str | None) -> Profile:
    return PROFILES.get(str(key or ""), PROFILES[DEFAULT])


def of(meta: dict | None) -> Profile:
    """The profile a set of capture details names."""
    return get(dict(meta or {}).get("object_type"))


def all_field_keys() -> set[str]:
    return {k for p in PROFILES.values() for _l, k, _h, _t in p.fields}


def labels() -> dict[str, str]:
    """Every discipline field's label by key (the first profile to name a key wins)."""
    out: dict[str, str] = {}
    for p in PROFILES.values():
        for lab, k, _h, _t in p.fields:
            out.setdefault(k, lab)
    return out


def preset_name(p: Profile) -> str:
    """The name of the profile's export preset in Publish."""
    return f"For this object: {p.name}"


def die_axis_hours(obverse_deg: float, reverse_deg: float) -> int:
    """The die axis in clock hours, 1-12, from how far each face's design is turned.

    Each angle is how far clockwise the design's top is from the photograph's top,
    with the coin turned over left to right (about its vertical axis) between the two
    captures, as numismatists turn it.  12 h: the reverse stands upright when the
    obverse does (medal alignment); 6 h: upside down (coin alignment)."""
    d = (float(reverse_deg) - float(obverse_deg)) % 360.0
    h = int(round(d / 30.0)) % 12
    return 12 if h == 0 else h
