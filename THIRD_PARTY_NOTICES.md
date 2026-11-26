# Third-party notices

Helio is MIT licensed (see `LICENSE`). It is built on, and distributes,
the components below. This file exists so that anyone who receives a Helio
disk image also receives the notices those components require.

Versions are the ones this build was made against; run
`pip show <package>` for what is on your own machine.

## Bundled Python packages

| Component | Version | Licence | What it does here |
|---|---|---|---|
| PySide6 / Qt for Python (Qt 6) | 6.11.0 | LGPL-3.0 (see below) | The application window, menus and every control. |
| numpy | 2.4.6 | BSD-3-Clause | Arrays and linear algebra throughout. |
| scipy | 1.17.0 | BSD-3-Clause | Solvers, sparse linear algebra, filtering. |
| opencv-contrib-python | 5.0.0.93 | Apache-2.0 | Image I/O, resampling, morphology, camera calibration. |
| Pillow | 12.2.0 | MIT-CMU (HPND) | Image encoding and the help screenshots. |
| scikit-image | 0.26.0 | BSD-3-Clause | Micro-features, toolpaths, image measures. |
| scikit-learn | 1.8.0 | BSD-3-Clause | PCA and clustering in the analysis views. |
| matplotlib | 3.10.8 | PSF-based (BSD-like) | Plots: profiles, histograms, the light dome. |
| trimesh | 4.11.2 | MIT | Mesh construction and OBJ/PLY/GLB writing. |
| tifffile | 2026.3.3 | BSD-3-Clause | 16- and 32-bit TIFF, and the ImageJ stacks and cubes. |
| rasterio | 1.3.11 | BSD-3-Clause | GeoTIFF writing for the GIS/DEM exports. |
| usd-core (OpenUSD) | 25.11 | Tomorrow Open Source Technology License 1.0 (Apache-2.0 based) | USDZ and layered USD exports. |
| pye57 | 0.4.19 | MIT; bundles libE57Format (BSL-1.0) and Xerces-C (Apache-2.0) | E57 point clouds. |
| PyOpenGL | 3.1.10 | BSD-style | The 3D relief window. |
| vispy | 0.16.1 | BSD-3-Clause | The interactive 3D viewer. |
| glfw | 2.10.0 | MIT (GLFW: zlib) | OpenGL context for that viewer. |
| rvt-py | 2.2.1 | Apache-2.0 | Sky-view factor, openness and local dominance. |
| qrcode | 8.2 | BSD-3-Clause | QR codes for the sharing labels. |
| rawpy | 0.27.0 | MIT | Python binding for LibRaw — see the note below. |
| PyInstaller | 6.19.0 | GPL-2.0 with the bootloader exception | Build tool. The exception lets the bootloader it puts in the app ship under any licence. |

## Qt and PySide6 — LGPL 3

The interface is built with **Qt 6 through PySide6 (Qt for Python)**, used
under the **GNU LGPL version 3**. As with LibRaw below:

* **Helio does not become GPL or LGPL.** It only uses the libraries.
* **They are dynamically linked:** the Qt frameworks and the PySide6 and
  shiboken6 libraries sit as separate files inside `Helio.app`, so a recipient
  can replace them with their own builds (LGPL-3 §4).
* **Source:** Qt's and PySide6's source code is at <https://download.qt.io/>
  and <https://code.qt.io/>; the LGPL-3 text is at
  <https://www.gnu.org/licenses/lgpl-3.0.html> and inside the PySide6
  distribution.
* Qt is a trademark of The Qt Company Ltd. Helio is not affiliated with it.

## LibRaw — the one with obligations

Camera raw support comes from **rawpy**, which is MIT, but rawpy ships
**LibRaw** as a bundled shared library (`libraw_r.dylib` inside the app), and
**LibRaw is licensed under the GNU LGPL version 2.1**. Its wheel also carries
`liblcms2` (MIT) and `libjasper` (JasPer licence).

What that means in practice:

* **Helio does not become GPL.** The LGPL exists precisely so that a
  differently licensed program may link the library. Helio remains MIT.
* **The library is dynamically linked**, as a `.dylib` inside the bundle, so a
  recipient can replace it with their own build of LibRaw. That is what
  satisfies LGPL 2.1 §6.
* **The notice must travel with the binary.** That is what this file is for,
  and it is copied into the disk image beside the application.
* LibRaw's source and licence text: <https://www.libraw.org/> — the full
  LGPL 2.1 text also ships inside the `rawpy` distribution as
  `rawpy-*.dist-info/licenses/LICENSE.LibRaw`.

If you would rather not carry an LGPL component at all, raw support is
optional by construction: `helio.io.raw_loader.available()` gates it, the
file-type lists only offer raw formats when a decoder is present, and removing
`rawpy` from `requirements.txt` and `setup.py` leaves a working build with no
LibRaw in it.

## JavaScript bundled into the exported viewers

Both of these were previously loaded from a public CDN and are now
redistributed inside the disk image and written into exported viewers, so both
are attributed here. That is the trade: an export that depends on a CDN is not
archivable, and one that bundles a library must credit it.

**three.js** r128 — MIT licence. Copyright © 2010-2021 three.js authors.
<https://github.com/mrdoob/three.js>
Inlined into the 3D Web Viewer and the HF+Normals viewer, together with the
`OrbitControls` and `GLTFLoader` example modules from the same release.

**OpenSeadragon** 4.1.0 — BSD-3-Clause licence. Copyright © 2009-2013 CodePlex
Foundation, © 2010-2023 OpenSeadragon contributors.
<https://openseadragon.github.io/>
Written beside the tiles in a DeepZoom export, with its navigation button
images, so the export opens without network access.

Neither library is modified. Both licences permit redistribution with
attribution, which is what this section is.

**OpenLIME** 1.2.6 — GPL-3.0 licence. © CNR-ISTI Visual Computing Lab.
<https://github.com/cnr-isti-vclab/openlime>
*Not* bundled: the OpenLIME HSH export's `index.html` loads it from the
jsDelivr CDN when the page is opened, and the export's README says so. Helio
neither contains nor distributes OpenLIME's code. Anyone who copies
`openlime.min.js` into an export to make it work offline is distributing
GPL-3 code and must follow that licence for it.

## Methods

The algorithms Helio implements are published research, and the papers are
cited in **Help ▸ Methods and references** inside the application. Implementing
a published method from its paper — or a documented file format from its
specification — is ordinary practice and carries no licence obligation; the
citation is there because in this field it is what makes a result defensible.

No source code has been copied from RTIBuilder, RTIViewer or Relight.
Helio's PTM and HSH fits follow the published papers (Malzbender et al. 2001;
Gautron et al. 2004). The Relight / OpenLIME export writes Relight's file
format (Ponchio, Corsini and Scopigno 2019); so that RelightLab and OpenLIME
read it exactly as their own, its details (the HSH normalisation, the
coefficient quantisation, the tarzoom layout) were checked against Relight's
public source, whose core library (`src/`) is LGPL-3.0 and whose applications
are GPL-3.0. They are written in Helio's own code: the exporter was compared
with that source in November 2026, function by function, and nothing in it is
a translation of Relight's code (see the notes at the top of
`helio/io/openlime_hsh.py`). Matching a file format so that files interchange
is ordinary practice; the credit is given here and in the export's README.

