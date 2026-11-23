# Precision photometric stereo in Helio

An optional way of shooting and processing for measurement. Nothing here
changes an ordinary capture: every piece is off until you set it.

## 1. Audit of the photometric-stereo pipeline (November 2026)

What the solve does, in order (`helio/application/pipeline.py`, `_solve`):

| Step | What happens | State before this work |
|---|---|---|
| Load | Frames read in parallel; RAW decoded **linear**, JPEG/TIFF as stored; ICC colour for the colour only | Sound |
| Lens | — | **Missing**: no distortion correction anywhere |
| Ambient | Lights-off frame subtracted in linear light | Sound |
| Light strength / fall-off | Grey card (hand-held flash), flat areas marked in the shot (plane per frame), semi-calibrated strengths from the photographs | Good for hand-held flash; **no per-light flat field for a fixed rig** |
| Near-field | Lights on one dome radius around the ball, inverse-square per pixel; with two balls, each light at the position where their directions meet (November 2026) | One ball: every light at the same assumed distance |
| Tone | Automatic by default: decoded by the file's own curve (ICC profile, Adobe RGB from the EXIF, else sRGB); until 2026-11-12 only when the lighting was evened, but as stored fine relief came out at 60% of its depth (DiLiGenT-Pi); stored, decode or linear on request | For measurement shoot RAW |
| Normals | Linear least squares (Rust), robust, shadow-rejecting, metal-aware, near-field solvers | Sound |
| Precision | Per-pixel 1σ angular uncertainty (`sigma_deg`, needs ≥ 4 lights), Normal QC rating | Sound, but **not carried into heights** |
| Geometry check | Conditioning and dome coverage shown in Capture | Sound |
| Integration | Poisson / Frankot-Chellappa / others; heights in true pixels (`heights.py`) | Sound; low-frequency drift is inherent |
| Height scale | Set by hand from a feature of known height | **No link to the lateral scale** for a calibrated capture |
| Verification | — | **Missing**: nothing measures the rig's error on a reference |

## 2. What was added

All in Process › **Precision measurement** and Measure:

1. **White-board light field** (`helio/core/light_field.py`). One frame of a
   matte white board per light, from the rig's fixed positions. Measures each
   light's strength and fall-off across the frame; every frame of a capture is
   divided by its light's field before the solve. On a synthetic 4-light
   capture with lamps 0.75–1.3× and ±25% fall-off, the flat base went from
   over 3° of false tilt to under 0.5°, as good as knowing the lamps exactly.
2. **Lens correction** (`helio/core/lens.py`). 10–20 photographs of a printed
   chessboard; OpenCV's calibration; every frame undistorted before the solve.
3. **Heights from the lateral scale** (Measure › Height scale). For a
   calibrated, linear capture, a pixel of height is a pixel of width.
4. **Accuracy check on a reference** (Measure › Accuracy check). A gauge
   block's step (plane fitted to the base, the top's height above it) and a
   flat plate's flatness (RMS, peak-to-valley, tilt), against the certified
   value. The result goes into the session and onto the plate.
5. **± on heights.** Profiles and the accuracy check give the 1σ height
   uncertainty from the normals' own precision along the line. Calibration
   error is not in it; the accuracy check measures that.
6. **Readiness checklist** in Process: lights, spread, RAW, board, lens,
   scale, heights, accuracy.
8. **The score is a ranking, not an accuracy** (November 2026). It is worked
   out from the photographs, which cannot see a wrong light direction, uneven
   light or a tone curve. The accuracy check's measured error is shown beside
   it once run.
7. Both calibrations travel in the **rig profile** and the session.

## 3. Shooting for measurement (four AD200 Pro II heads)

**Once per rig set-up**
- Four heads on one rigid frame, 90° apart round the object, 35–45° up, bare
  bulb (no modifiers), as far away as the power allows.
- Camera square to the object on a copy stand; a longer lens; manual
  focus, aperture, ISO, shutter, white balance; **RAW**.
- Flash power fixed, the same for all four; modelling lamps off.
- Each head in its own group (A–D) on the trigger; per shot, only that
  group on.

**Printing the targets**

File › *Print Calibration Targets…* writes exact PDFs at A4, A3, A2, A1,
Letter or Tabloid: the lens chessboard (9 × 6 inner corners), detailed scale
bars to cut out (20 mm in half millimetres up to the longest the paper takes,
an L scale, an inch rule) and a 10 mm square-to-camera grid. Any laser print
will do: print at 100% / Actual size, on matte paper, and mount it flat (foam
board, Dibond or glass). Every sheet has a 100 mm check bar: measure it. The
chessboard's square size does not matter to the lens calibration; for
publication, check a printed scale against a steel rule.

**Is the camera square?** Lay the chessboard flat on the baseboard and run
Capture › *Check Camera Alignment on a Chessboard…*: it says how many degrees
the camera is off square and which side of the frame is nearer, and the scale
on the board (which it can set). Aim for under 1°.

**Calibration frames (same rig, same camera settings)**
1. Chrome ball: four frames, one per head (Capture finds the lights).
2. White board: a matte white (or light grey) board filling the frame,
   flat, square to the camera; four frames, one per head, same order.
   Not clipped anywhere.
3. Chessboard: 10–20 frames of a printed chessboard, flat, at different
   places and tilts across the frame.
4. A reference: a gauge block (or a shim of measured thickness) on a flat
   plate, shot like an object.

In Helio: find the ball → Process › Precision measurement › *Set up for
measurement* (the DiLiGenT-best solver, free-boundary depth, the tone decoded
by the file's curve, every look-only adjustment off) → *Calibrate on a
white board…* and *Calibrate from a chessboard…* → Capture › save a **rig
profile**. Process the reference, set the scale, Measure › *Use the lateral
scale*, Measure › *Accuracy check*.

**Each object**
- Same rig, same settings: four frames (one per head) and a lights-off frame.
- Apply the rig profile, process, measure.
- Re-shoot the board if anything moves: a head, the camera, the lens, focus,
  aperture or flash power.

## 4. Limits worth knowing

- The chrome ball's own precision: Capture's calibration notes say how well
  the ball measured each light (from its rim fit and the highlight's place)
  and what that alone puts into the surface directions: about 1° for a ball
  60 px across the radius, more for a smaller one. Use the largest ball the
  frame allows.
- Two balls place every light at its own position, and Near-field correction
  then uses them. On a flat board under eight lamps at different distances,
  one dome radius left 12.6° of tilt, the positions 0.2°
  (tests/test_light_positions.py).

- Measured on the DiLiGenT benchmark (`docs/BENCHMARK_DILIGENT.md`):
  - Helio's solvers range from 13.1° mean angular error (Sparse-Outlier LS,
    the best) to 15.5° (least squares).
  - Solving an 8-bit sRGB stack as stored, without decoding it, adds 2.3° on
    average and 10° on a matte object.
  - For measurement, shoot RAW, or even the lighting (white board or flat
    areas). Photo tone's default, Automatic, decodes camera files.
- Measured on DiLiGenT-Pi (`docs/BENCHMARK_DILIGENT_PI.md`), 30 near-flat
  objects like a punch's face: least squares 9.3°; camera JPEGs solved as
  stored leave fine relief at 60% of its depth, decoded 97%.

- Four lights solve three unknowns per pixel with one to spare: a pixel in
  shadow or highlight in one frame is still solved, but less surely. More
  heads (or the frame turned 45° to a second set of fixed stops, calibrated
  as its own lights) help most on glossy or deep objects.
- Integrated heights drift over long distances: trust steps and features
  more than the height difference between opposite corners of the frame.
- The camera is treated as looking straight down (orthographic). With a
  longer lens and the camera far away this is small; it is not corrected.
- A small object on a dark surround (a steel punch on black velvet): the
  surround's surface directions are noise. Process › Depth › *Depth on the
  object alone* (off by default; Process › Set Up for Measurement turns it on
  when the last solve shows a small object, under 40 % of the frame) works the
  depth out on the object only and keeps the surround flat. On a large object
  (a binding filling 64 % of the frame) it does harm: held flat all round, the
  object comes out a raised pillow with lifted edges and softened tooling. On a 50-light Baskerville punch, the depth's slopes
  disagreed with the solved directions by 10° median over the punch with
  the whole frame, 6° on the punch alone. The score's **Depth** part (20)
  measures this for each solve. Normals that face away from the camera (a
  shadow-rejecting or shine-resistant solve leaves a few, fitted on two or
  three frames) are filled from their neighbours before integration: 2% of
  them had made that punch's depth span 16,800 px where it spans 150.
- Cross-polarisation (film on the heads, a polariser on the lens) removes
  gloss and makes the surface behave as the solve assumes; shoot a second,
  unpolarised pass for the RTI's shine.
