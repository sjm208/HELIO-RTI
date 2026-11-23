# Helio: Quick Start

From photographs to a publication sheet, in Helio's five workspaces. The
workspaces run left to right along the top (⌘1 to ⌘5). The bar at the bottom
always shows the next step.

## 1. Install

Open the disk image from
[Releases](https://github.com/sjm208/HELIO-RTI/releases) and double-click
**Install Helio**. It puts Helio in Applications. Apple silicon, macOS 13
or later.

## 2. Photograph the object

- **Camera:** fixed on a tripod; the object stays still.
- **Light:** take 12–64 photographs, moving only the light, all round the
  object. Light from low angles (15–45°) shows the most relief.
- **Calibration:** put a chrome ball in the frame, beside the object. Two
  balls, one each side, are better still.
- **Scale:** include a scale bar, so you can set the scale later.
- **Raw files** give the most accurate result (Phase One IIQ, Sony, Canon,
  Nikon, Hasselblad and uncompressed DNG). Capture One sessions can't be read:
  export TIFF or uncompressed DNG, not JPEG XL DNG. JPEG works too.
- **Lights:** with only 4 frames the score stays low; use 8 or more. A TIFF's
  colour is read by its embedded ICC profile (Adobe RGB, ProPhoto, eciRGB) and
  shown in sRGB; the height is solved from its stored numbers.
- **Outdoors or in a lit room:** take one more photograph with the lamps off,
  everything else unchanged. Choose it with **Capture › Add Lights-Off
  Frame…** and its light is taken out of every frame.

## 3. Capture (⌘1)

1. **File › Import Images…** (⇧⌘I), or drop the folder on the window.
2. Give Helio the lights, in one of these ways:
   - **Chrome sphere** finds the ball and measures each light.
   - **Light file** reads a `.lp` file from a calibrated dome.
   - **By hand** puts the lights on a virtual dome.
3. **Capture › Capture Details…** (optional): choose the **Object type**
   (coin, seal, gem, inscription, manuscript, binding, small find). Helio sets
   a solver for it, puts its views first in Inspect and its exports first in
   Publish, and adds its own fields (a coin's die axis, a page's shelfmark).
4. **Save as rig…** if you'll photograph more objects on the same rig. Its
   lights, the ball's position and the scale then apply to the next capture,
   from **Rig profile**.

## 4. Process (⌘2)

1. **Quick look** solves at about 1,000 px in seconds, with a score.
   **Try a few** compares several settings at once. Show any quick look again
   to compare them.
2. **Process** (⌘R) runs the full-size solve. The defaults suit most objects:
   - Linear Least Squares for the normals;
   - Fast Poisson for the height.

   For metal or gilding, try a specular-robust solver.

## 5. Inspect (⌘3)

- **Views:** step through the 36 views with ← and →. **Raking Composite**,
  **Exaggerated Shading**, **Surface Curvature** and **MSRM** show the most
  relief.
- **Light:** drag it on the dome. **A** sweeps it round.
- **Layers:** **Colour**, **Clay**, **Roughness** and **Confidence** change how
  each view is coloured. A capture entirely in RAW also offers **Solved
  reflectance**: its colour with the shading taken out, used in the views, 3D
  textures and exports. It is relative, not calibrated to a white.
- **The 3D relief:**
  - draw a section line across it for its profile;
  - double-click to turn round a point;
  - **Record video** makes a turntable or flyover.
- **The high-detail 3D window** shows the surface at 4,500 px.
- **Annotate:** draw boxes and lines on the surface. Named boxes become the
  publication sheet's close-up details.

## 6. Measure (⌘4)

- **Set Scale…** from the scale bar in the photograph.
- **Distance:** measure lengths. Measured lines appear on the publication
  sheet.
- **Profiles:** draw a height profile. It gives the ISO 4287 figures about the
  whole profile's mean line, and the roughness and waviness separated by the
  ISO 16610-21 Gaussian filter (the cut-off is chosen by ISO 4288 once a scale
  and a height scale are set; the CSV carries both profiles).
- **Area and volume:** measure the area and volume of a region.

## 7. Publish (⌘5)

1. Tick the formats you want. Each card says what it makes.
   **Presets** tick a usual set at once: museum archive, share on the web,
   3D print, publication figures. With an object type chosen, its own preset
   comes first: for a coin, a Nomisma record; for an inscription, an EpiDoc
   file and a line drawing; for a page, a IIIF manifest.
2. **Export N formats…** asks for the folder, then writes them one by one.

**The publication sheet** is under Reports & figures. When you tick it, its
choices appear:
- **Views:** choose up to 20 views, and the large one.
- **Detail views:** each named box is shown close up in these.
- **Add this section:** draw sections on the 3D relief and keep up to six.
- **3D pictures:** **Add the 3D relief** and **Add high-detail 3D**.
- **Add pictures…** adds your own photographs from files.
- **From the tools:** every Research tool (Legend Unwrap, Line Drawing,
  Toolpaths, Inscription Zones …) has **Add to sheet** at the bottom of its
  window, with a caption.
- **On the sheet** lists everything added, in order. While the **Section**
  tool is on, kept sections are drawn on Inspect's picture faintly with their
  letters, the one selected here bold; turn the tool off, or untick **Show on
  the 2D image**, for a clean picture. **Up**, **Down** and **Remove** change it; double-click
  a picture to recaption it. A line close to one already kept, or a 3D view
  taken twice, asks whether to replace it. The line still drawn goes on only
  when no section is kept, or with **Keep**.
- **Clay 3D at an angle** is on by default. Open the 3D relief once in
  Inspect so it can be made.
- **Second page:** the surface lit from eight directions.

**Every export** writes a `_paradata.json` and a `_README.txt` beside its
files: the photographs with checksums, the lights, every setting, how the colour
was treated, and each file's role.

**For keeping (Archive & sharing):**
- **Deposit package:** masters, web and print derivatives and metadata in their
  own folders, with the paradata and a checksum manifest. The photographs are
  not copied: keep them with it.
- **Linear OpenEXR masters:** colour, normals and heights in floating point.
- **Layered USD stage:** the surface, the record and the colour as USD layers,
  with a variant for the solved reflectance on a RAW capture. A **material**
  variant switches between the photograph, the measurement (the solve's normals
  and heights as float EXR, the heights as displacement) and OpenPBR (MaterialX).
  The capture's lights come in as USD lights, kept sections and annotations as
  curves lying on the surface, and the scale and colour check in its metadata.
- **Colour validation report:** with a 24-patch ColorChecker Classic in the
  capture, ΔE₀₀ from photograph to colour map to render.

**For GIS:**
- **GIS DEM package:** the height maps for QGIS.
- **Helio Relief (QGIS):** the QGIS plugin, version 4.1,
  for QGIS 4. Install it in QGIS with **Install from ZIP**, then use its
  **Import Helio…**. Its **Presets** button sets the render and surface for the
  object in front of you, and an export from a capture with an object type
  chooses one for you. Sections kept for the sheet and annotations come in as
  labelled map layers over the height map.
- **File › Quick Export › Send to QGIS** (⇧⌘G) opens the surface in QGIS
  directly. If the plugin is missing or older, a button there installs it.

## Research

**Research** holds tools for particular objects: **Two Faces** (recto and
verso, obverse and reverse with the die axis), **Legend Unwrap** (the legend
round a seal, coin, stamp or volvelle as one strip: circles, ovals, pointed
ovals and free shapes dragged into place, several on a page found and numbered, read in relief,
an even raking light or any view in Colour or Clay), **Virtual Impression** (what a matrix stamps, or an
inscription's letters raised), **Line Drawing** (an SVG to finish by hand),
and Inscription Zones, Toolpaths and Micro-Features.

## Tips

- **Save often:** **File › Save** (⌘S) writes a `.helio` session.
- **Greyed out?** A greyed-out command says why when you hover it. ⌘K finds
  any command and says where it lives.
