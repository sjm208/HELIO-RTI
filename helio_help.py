"""Help text shared by the app's Help (helio_qt.help): the glossary, the
methods and references, the problem report's diagnostics, and where reports go.

The Classic (Tk) app's Help, About and Problem-Report windows that lived here
were removed in November 2026: the Qt app has its own (helio_qt.help) and
read only the text below.  GUI-free.
"""

from __future__ import annotations

import os
import platform
import sys

# ── Where bug reports go ──────────────────────────────────────────────────
#
# Swap these two and nothing else changes. If SUPPORT_URL is set it is
# offered first and the address is never shown, which is the better answer:
# an address compiled into a shipped binary cannot be withdrawn later.
SUPPORT_EMAIL = "scott.boro@gmail.com"
SUPPORT_URL = ""            # e.g. a form or an issue tracker

APP_NAME = "HELIO Pro"


def _build_stamp() -> str:
    """The build identity, or a harmless placeholder if the module is absent."""
    try:
        import helio_build
        return helio_build.stamp()
    except Exception:
        return "unknown build"


# ── Help content ──────────────────────────────────────────────────────────
# Blocks: ("p", prose), ("pre", text whose alignment carries meaning), ("h", subheading).

_TERMS_BLOCKS = [
    ("p", "The vocabulary, in plain English."),
    ("pre",
     "  Photometric stereo   Working out surface shape from several photographs\n"
     "                       lit from different directions. What HELIO does.\n\n"
     "  RTI                  Reflectance Transformation Imaging. A capture you\n"
     "                       can relight interactively afterwards.\n\n"
     "  Normal / normal map  The direction a point on the surface faces, stored\n"
     "                       as colour. Flat areas look lilac.\n\n"
     "  Albedo               The true colour of the surface with the lighting\n"
     "                       taken out \u2014 as if lit perfectly evenly.\n\n"
     "  Depth / height map   How far each point stands proud of, or sinks below,\n"
     "                       the surface. Solved from the normals.\n\n"
     "  Raking light         A lamp close to the surface, almost side-on. The\n"
     "                       single most useful setting here.\n\n"
     "  Specular             Mirror-like reflection \u2014 the shine. Strong on\n"
     "                       gilding, glaze and metal; weak on parchment.\n\n"
     "  Lambertian           A surface that scatters light evenly, like matte\n"
     "                       paper. The assumption photometric stereo rests on.\n\n"
     "  PTM / HSH            Older RTI formats that fit a mathematical surface\n"
     "                       to each pixel's response to light.\n\n"
     "  SVF                  Sky-View Factor. How much of the sky each point can\n"
     "                       'see'. Hollows score low, ridges high.\n\n"
     "  Openness             How open or enclosed a point is. Positive picks out\n"
     "                       ridges, negative picks out hollows.\n\n"
     "  Chrome ball          The shiny sphere in the frame. Its highlight tells\n"
     "                       HELIO exactly where each lamp was.\n\n"
     "  Uncalibrated         A solve done without knowing the light positions.\n"
     "                       Approximate \u2014 use when nothing else is available.\n\n"
     "  Self-contained       One file that opens with no server and no network.\n"
     "                       Everything it needs is inside it.\n\n"
     "  Camera drift         Movement between frames. Photometric stereo assumes\n"
     "                       there is none; when there is, relief smears.\n\n"
     "  Tick spacing         The repeat distance of a ruler's marks. HELIO can\n"
     "                       measure it; only you can say what one tick means.\n\n"
     "  Region               A box you draw that the Forensic tools work inside.\n"
     "                       Separate from a crop, which changes every export.\n"),
]


def _methods():
    """The published methods HELIO implements, and where to read them.

    These were already written down - as a comment block at the top of
    Helio_Pro.py that no user has ever seen.  Conservators and researchers
    have to cite what they used when they publish, so the list belongs in
    front of them rather than in the source.
    """
    return [
        ("p", "HELIO does not invent its mathematics. Each stage implements a "
              "published method, and this page lists them so you can cite what "
              "you actually used. None of it is proprietary — the papers are "
              "the reference, and this software is one implementation of them."),

        ("h", "Recovering surface normals"),
        ("pre",
         "  Woodham, R.J. (1980)\n"
         "      Photometric method for determining surface orientation from\n"
         "      multiple images. Optical Engineering, 19(1).\n"
         "      — the basis of every normal solver in the Lights panel.\n"
         "\n"
         "  Wu, L., Ganesh, A., Shi, B., Matsushita, Y., Wang, Y. & Ma, Y. (2010)\n"
         "      Robust photometric stereo via low-rank matrix completion and\n"
         "      recovery. ACCV 2010.\n"
         "  Ikehata, S., Wipf, D., Matsushita, Y. & Aizawa, K. (2012)\n"
         "      Robust photometric stereo using sparse regression. CVPR 2012.\n"
         "      — the shine- and shadow-resistant solvers.\n"
         "\n"
         "  Cho, D., Matsushita, Y., Tai, Y.-W. & Kweon, I.S. (2018)\n"
         "      Semi-calibrated photometric stereo. IEEE TPAMI.\n"
         "      — finding each lamp's strength from the photographs.\n"
         "\n"
         "  Shi, B., Mo, Z., Wu, Z., Duan, D., Yeung, S.-K. & Tan, P. (2019)\n"
         "      A benchmark dataset and evaluation for non-Lambertian and\n"
         "      uncalibrated photometric stereo. IEEE TPAMI, 41(2).\n"
         "      — DiLiGenT, on which the solvers are measured\n"
         "        (docs/BENCHMARK_DILIGENT.md).\n"),

        ("h", "Turning normals into a height map"),
        ("pre",
         "  Frankot, R.T. & Chellappa, R. (1988)\n"
         "      A method for enforcing integrability in shape from shading\n"
         "      algorithms. IEEE TPAMI, 10(4), 439-451.\n"
         "      — the Frankot-Chellappa depth engine.\n"
         "\n"
         "  Simchony, T., Chellappa, R. & Shao, M. (1990)\n"
         "      Direct analytical methods for solving Poisson equations in\n"
         "      computer vision problems. IEEE TPAMI, 12(5), 435-446.\n"
         "      — the Simchony DCT depth engine.\n"
         "\n"
         "  Queau, Y., Durou, J.-D. & Aujol, J.-F. (2018)\n"
         "      Normal Integration: A Survey. Journal of Mathematical Imaging\n"
         "      and Vision, 60(4).\n"
         "      — the comparison these engines are judged against; its DCT\n"
         "        solver with the natural boundary is Free-Boundary Poisson (DCT).\n"
         "\n"
         "  Queau, Y., Durou, J.-D. & Aujol, J.-F. (2018)\n"
         "      Variational methods for normal integration. Journal of\n"
         "      Mathematical Imaging and Vision, 60(4).\n"
         "      — Least-Squares (Free Boundary), and depth on the object alone.\n"
         "\n"
         "  Cao, X., Santo, H., Shi, B., Okura, F. & Matsushita, Y. (2022)\n"
         "      Bilateral normal integration. ECCV 2022.\n"
         "      — Bilateral (Keeps Steps).\n"),

        ("h", "Relighting (RTI)"),
        ("pre",
         "  Malzbender, T., Gelb, D. & Wolters, H. (2001)\n"
         "      Polynomial texture maps. SIGGRAPH 2001.\n"
         "      — PTM coefficients and the PTM views.\n"
         "\n"
         "  Gautron, P., Krivanek, J., Pattanaik, S. & Bouatouch, K. (2004)\n"
         "      A novel hemispherical basis for accurate and efficient\n"
         "      rendering. Eurographics Symposium on Rendering 2004.\n"
         "      — the hemispherical harmonics (HSH) basis.\n"
         "\n"
         "  Ponchio, F., Corsini, M. & Scopigno, R. (2019)\n"
         "      RELIGHT: a compact and accurate RTI representation for the\n"
         "      web. Graphical Models, 105.\n"
         "      — the Relight / OpenLIME format the web exports write. The\n"
         "      OpenLIME viewer (CNR-ISTI Visual Computing Lab) is loaded by\n"
         "      the exported page; it is not part of HELIO.\n"
         "\n"
         "  Blinn, J.F. (1977)\n"
         "      Models of light reflection for computer synthesized pictures.\n"
         "      SIGGRAPH 1977.\n"
         "      — the shine in the relightable web views.\n"),

        ("h", "Camera and lens calibration"),
        ("pre",
         "  Zhang, Z. (2000)\n"
         "      A flexible new technique for camera calibration. IEEE TPAMI,\n"
         "      22(11), 1330-1334.\n"
         "      — lens correction and the camera-alignment check, through\n"
         "      OpenCV.\n"),

        ("h", "Separating shine from colour"),
        ("pre",
         "  Mallick, S.P. et al. (2005)\n"
         "      Beyond Lambert: reconstructing specular surfaces using colour.\n"
         "      CVPR 2005.\n"
         "\n"
         "  Tan, P. et al. (2003)\n"
         "      Highlight removal by illumination-constrained inpainting.\n"
         "      ICCV 2003.\n"),

        ("h", "Relief visualisation"),
        ("pre",
         "  Zaksek, K., Ostir, K. & Kokalj, Z. (2011)\n"
         "      Sky-View Factor as a relief visualisation technique.\n"
         "      Remote Sensing, 3(2), 398-415.\n"
         "\n"
         "  Kokalj, Z. & Hesse, R. (2017)\n"
         "      Airborne Laser Scanning Raster Data Visualization:\n"
         "      A Guide to Good Practice. IIARR / ZRC SAZU.\n"
         "\n"
         "  Yokoyama, R. et al. (2002)\n"
         "      Visualizing topography by openness. PE&RS, 68(3).\n"
         "      — the openness and horizon-scanning views.\n"
         "\n"
         "  Rusinkiewicz, S., Burns, M. & DeCarlo, D. (2006)\n"
         "      Exaggerated shading for depicting shape and detail.\n"
         "      ACM Transactions on Graphics, 25(3) (SIGGRAPH 2006).\n"
         "\n"
         "  Vergne, R., Pacanowski, R., Barla, P., Granier, X. & Schlick, C. (2010)\n"
         "      Radiance scaling for versatile surface enhancement. I3D 2010.\n"
         "\n"
         "  Orengo, H.A. & Petrie, C.A. (2018)\n"
         "      Multi-scale relief model (MSRM). Earth Surface Processes and\n"
         "      Landforms, 43(6).\n"
         "\n"
         "  Burt, P.J. & Kolczynski, R.J. (1993)\n"
         "      Enhanced image capture through fusion. ICCV 1993.\n"
         "  Mertens, T., Kautz, J. & Van Reeth, F. (2007)\n"
         "      Exposure fusion. Pacific Graphics 2007.\n"
         "      — the fused views across lights.\n"
         "\n"
         "  Crameri, F., Shephard, G.E. & Heron, P.J. (2020)\n"
         "      The misuse of colour in science communication. Nature\n"
         "      Communications, 11, 5444.\n"
         "  Borland, D. & Taylor, R.M. (2007)\n"
         "      Rainbow color map (still) considered harmful. IEEE Computer\n"
         "      Graphics and Applications, 27(2).\n"
         "      — why the height and uncertainty colour maps are perceptual.\n"
         "\n"
         "  Mara, H. & Kroemker, S. (2017)\n"
         "      Visual computing for archaeological artifacts with integral\n"
         "      invariant filters. GCH 2017.\n"
         "      — the multi-scale integral invariant (MSII) curvature view.\n"),

        ("h", "Capture and calibration"),
        ("pre",
         "  Cultural Heritage Imaging (CHI), San Francisco\n"
         "      RTI Guide v2.0 (2013), and the RTIBuilder .lp format.\n"
         "      — reflective-sphere light detection follows this standard,\n"
         "        including the 200 px minimum sphere diameter.\n"
         "      https://culturalheritageimaging.org\n"
         "\n"
         "  Gaiani, M. et al. (2025). Heritage, 8(4).\n"
         "      — the measured stand calibration method.\n"
         "\n"
         "  Laurent, L. et al. (2025)\n"
         "      Combining geometric and photometric 3D reconstruction for\n"
         "      cultural heritage. Journal of Cultural Heritage.\n"),

        ("h", "Licence"),
        ("p", "HELIO Pro is released under the MIT Licence. Implementing a "
              "published method carries no obligation to credit it, but in this "
              "field the citation is the courtesy that makes the result usable "
              "by somebody else — and it is what lets you defend the method in "
              "print."),
        ("p", "No source code has been copied from RTIBuilder, RTIViewer or "
              "Relight. The PTM and HSH fits follow the published papers. The "
              "Relight / OpenLIME export writes Relight's file format; its "
              "details (the HSH normalisation, the quantisation, the tarzoom "
              "layout) were checked against Relight's public source (its core "
              "is LGPL-3) so that the files interchange exactly, and are written "
              "in HELIO's own code. The OpenLIME viewer (GPL-3) is loaded by the "
              "exported page from the web, not shipped with HELIO."),
        ("h", "Third-party components"),
        ("p", "The libraries HELIO Pro is built on, and their licences, are "
              "listed in Third-Party Notices.md — in the repository and beside "
              "the application on the disk image."),
        ("pre",
         "  Two of them carry an obligation worth knowing about:\n"
         "\n"
         "  The interface is Qt 6 through PySide6, used under the LGPL 3.\n"
         "  The Qt and PySide6 libraries are separate files inside the app,\n"
         "  linked dynamically, so they can be replaced; their source is at\n"
         "  https://code.qt.io/. HELIO does not become LGPL by using them.\n"
         "\n"
         "  Camera raw support uses rawpy (MIT), which bundles LibRaw —\n"
         "  and LibRaw is LGPL 2.1. HELIO Pro does not become GPL: the\n"
         "  LGPL exists so a differently licensed program may link the\n"
         "  library, and it is linked dynamically as a .dylib you could\n"
         "  replace. What it does require is that the notice travels with\n"
         "  the binary, which is what Third-Party Notices.md is for.\n"
         "\n"
         "  Raw support is optional by construction — remove rawpy and\n"
         "  the build simply stops offering raw file types.\n"
         "  LibRaw source and licence: https://www.libraw.org/\n"),
    ]


def diagnostics(app=None, version="") -> str:
    """A short factual report a user can send without having to describe it.

    *app* is accepted for old callers and not read."""
    lines = [f"{APP_NAME} problem report", ""]
    lines.append(f"version   : {version}")
    lines.append(f"build     : {_build_stamp()}")
    try:
        lines.append(f"machine   : {platform.machine()} on macOS {platform.mac_ver()[0]}")
    except Exception:
        pass
    lines.append(f"python    : {sys.version.split()[0]}")
    lines.append(f"frozen    : {bool(getattr(sys, 'frozen', False))}")
    # The tail of the diagnostic log is usually the whole answer.
    for p in (os.path.expanduser("~/Library/Logs/HELIO_Pro/helio_pro_diagnostic.log"),
              "helio_pro_diagnostic.log"):
        try:
            if os.path.exists(p):
                with open(p, errors="replace") as fh:
                    tail = fh.readlines()[-60:]
                lines += ["", f"--- last 60 lines of {p} ---", ""] + [t.rstrip() for t in tail]
                break
        except Exception:
            continue
    return "\n".join(lines)
