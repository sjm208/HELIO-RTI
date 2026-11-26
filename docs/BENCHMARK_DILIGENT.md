# Helio on the DiLiGenT benchmark (November 2026)

DiLiGenT (Shi, Wu, Tan, Ma, Ikeuchi, Tan: *A Benchmark Dataset and Evaluation
for Non-Lambertian and Uncalibrated Photometric Stereo*, CVPR 2016 / TPAMI
2019) has ten objects, each lit from 96 calibrated directions, with 16-bit
linear photographs and true surface directions from a laser scan. The usual
measure is the **mean angular error, in degrees, inside each object's mask**.
Lower is better.

To reproduce, with the dataset in `~/Downloads/DiLiGenT`:

    python tools/benchmark_diligent.py

All ten objects take under 30 s. The figures are in
`docs/benchmark_diligent.json` (mean and median per object).

## Set-up

- Helio's own solvers are called as the Process workspace calls them
  (`pipeline._solve_normals`, default settings).
- All 96 lights are used, as in the published table.
- Each frame is divided by its light's RGB intensity and averaged to grey.
- The stack is scaled so the 99.5th percentile inside the mask reads 230 of 255.
  That is a well-exposed linear (RAW-like) frame, which is the scale the robust
  solvers' thresholds are defined on.
- The published results that ship with DiLiGenT (`estNormalNonLambert`) are
  scored by the same script, with the same mask and measure.

**Check on the harness:** Helio's linear least squares scores 15.48°, against
15.39° for the published least squares. The small difference comes from
solving on grey rather than per channel.

## Results: mean angular error (°)

| Method | ball | cat | pot1 | bear | pot2 | buddha | goblet | reading | cow | harvest | **Mean** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Helio: calibrated, linear** | | | | | | | | | | | |
| Linear Least Squares | 4.29 | 8.44 | 9.12 | 8.94 | 14.94 | 15.13 | 18.66 | 19.12 | 25.61 | 30.55 | **15.48** |
| Shadow-Rejection LS | 3.23 | 7.26 | 7.84 | 7.56 | 14.25 | 11.63 | 16.94 | 23.85 | 28.89 | 32.32 | **15.38** |
| Mid-Tone Weighted LS (was "Robust Least Squares (Shiny Metal)") | 2.11 | 7.53 | 7.86 | 7.92 | 15.78 | 11.40 | 19.57 | 20.53 | 28.92 | 30.45 | **15.21** |
| IRLS (Metallic/Specular) | 3.10 | 7.77 | 8.66 | 7.80 | 14.47 | 13.66 | 17.51 | 16.58 | 25.16 | 29.79 | **14.45** |
| Iterative Specular Separation | 2.29 | 8.40 | 8.90 | 7.94 | 13.00 | 15.94 | 17.48 | 17.33 | 24.11 | 30.67 | **14.61** |
| Sparse-Outlier LS (RPCA) | 2.39 | 7.03 | 8.08 | 7.04 | 13.59 | 11.96 | 15.69 | 13.72 | 23.42 | 28.50 | **13.14** |
| **Helio: lamp strengths** | | | | | | | | | | | |
| Semi-calibrated LS (strengths estimated) | 4.76 | 9.41 | 9.33 | 9.08 | 17.66 | 15.45 | 21.18 | 19.75 | 26.91 | 43.54 | **17.71** |
| LS, strengths ignored | 16.73 | 17.57 | 18.35 | 21.15 | 21.59 | 20.99 | 23.47 | 25.29 | 27.03 | 33.83 | **22.60** |
| **Helio: tone (8-bit sRGB, as a camera JPEG)** | | | | | | | | | | | |
| LS on stored values (Helio's default) | 14.01 | 14.06 | 15.12 | 15.42 | 15.06 | 19.49 | 17.90 | 20.28 | 16.49 | 27.99 | **17.58** |
| LS on decoded values (sRGB → linear) | 3.51 | 8.44 | 9.12 | 8.92 | 14.93 | 15.02 | 18.53 | 18.67 | 25.59 | 30.47 | **15.32** |
| **Published (shipped with DiLiGenT, scored here)** | | | | | | | | | | | |
| Least squares (Woodham 1980) | 4.10 | 8.41 | 8.89 | 8.39 | 14.65 | 14.92 | 18.50 | 19.80 | 25.60 | 30.62 | **15.39** |
| Goldman et al. 2005 | 3.21 | 8.22 | 8.53 | 6.62 | 7.90 | 14.85 | 14.22 | 19.07 | 9.55 | 27.84 | **12.00** |
| Alldrin et al. 2008 | 2.71 | 6.53 | 7.23 | 5.96 | 11.03 | 12.54 | 13.93 | 14.17 | 21.48 | 30.50 | **12.61** |
| Higo et al. 2010 | 3.55 | 8.40 | 10.85 | 11.48 | 16.37 | 13.05 | 14.89 | 16.82 | 14.95 | 21.79 | **13.22** |
| Wu et al. 2010 (RPCA) | 2.06 | 6.73 | 7.18 | 6.50 | 13.12 | 10.91 | 15.70 | 15.39 | 25.89 | 30.01 | **13.35** |
| Shi et al. 2012 (ECCV) | 13.58 | 12.34 | 10.37 | 19.44 | 9.84 | 18.37 | 17.80 | 17.17 | 7.62 | 19.30 | **14.58** |
| Shi et al. 2012 (CVPR) | 1.74 | 6.12 | 6.51 | 6.12 | 8.78 | 10.60 | 10.09 | 13.63 | 13.93 | 25.44 | **10.30** |
| Ikehata et al. 2012 | 2.54 | 7.21 | 7.74 | 7.32 | 14.09 | 11.11 | 16.25 | 16.17 | 25.70 | 29.26 | **13.74** |
| Ikehata and Aizawa 2014 | 3.34 | 6.74 | 6.64 | 7.11 | 8.77 | 10.47 | 9.71 | 14.19 | 13.05 | 25.95 | **10.60** |

Learning-based methods published since 2018 are not in the shipped set. They
report means of roughly 5–8° in their own papers.

## What it says

1. **Helio's best solver, Sparse-Outlier LS (RPCA), reaches 13.1°.**
   - That is level with the robust methods of its generation: Wu 2010 at 13.4°
     and Ikehata 2012 at 13.7°.
   - It is behind the methods that model the reflectance itself: Goldman 12.0°,
     Shi 2012 10.3°, Ikehata 2014 10.6°.
   - It is well behind current learning-based methods.
   - Helio is a sound classical solver, not a state-of-the-art one.
2. **Solving an 8-bit sRGB stack as stored costs 2.3° on average and up to
   10.5° on a matte object** (ball: 14.0° against 3.5°). Decoding to linear
   recovers the least-squares figure almost exactly.
   - The tone curve happens to help on cow (16.5° against 25.6°): it
     compresses that metallic paint's highlights. That is luck, not a method.
   - **What changed:** the default Photo tone is now *Automatic*. It decodes
     when a white-board light field or flat areas even the lighting, and
     solves as stored otherwise. KD-052 had found that, on real captures,
     decoding doubles the false tilt that uneven lighting puts in. The result
     and the processing record name the tone used.
3. **Estimating the lamps' strengths from the photographs** (semi-calibrated)
   costs 2.2° against knowing them, and fails on harvest (43.5°). Ignoring the
   strengths costs 7°. A grey card or the white-board light field is worth
   having.
4. **The solvers labelled for shine are not reliably better on shiny
   objects.**
   - "Robust Least Squares (Shiny Metal)" (renamed Mid-Tone Weighted LS in
     November 2026) and "Shadow-Rejection LS" are worse
     than plain least squares on cow (metallic paint), reading and harvest.
   - Their thresholds are fixed pixel values, so they suit a well-exposed frame
     but miss the outliers of a strongly specular surface.
   - RPCA is the only solver that is better than least squares on almost every
     object (9 of 10). It should be the recommended solver for glossy or metal
     objects.
5. **Speed is not an issue at this size.** All six solvers together take under
   15 s for the ten objects (about 380,000 pixels × 96 lights).

## Limits of this test

- DiLiGenT uses distant, calibrated lights and an orthographic-like camera.
  It measures the normal solve only. It does not test light calibration from
  a chrome ball, near-field correction, lens correction or depth integration.
  The accuracy check on a gauge block (Measure › Accuracy check) covers those
  on a real rig.
- One run, no tuning: every solver at Helio's default settings.

## No ball: the lights found from the photographs

The no-ball solve (`helio.core.uncalibrated`, the Uncalibrated PCA solver) is given the photographs alone: no light directions, no strengths. It takes the lamps as equally strong, keeps the turn whose normals form a real surface, and settles raised against sunken from the cast shadows. Below, mean angular error in degrees. "Even" divides each frame by its light's intensity, as one lamp moved round would give; 10% and 20% scale each frame at random by that much. Where the photographs do not bear the lamps out, it falls back to the rough solve and says so. `python tools/benchmark_no_ball.py`.

| Object | Least squares, true lights | Rough (old) | No ball, even | 10% uneven | 20% uneven |
|---|---|---|---|---|---|
| ball | 4.3 | 55.0 | 4.5 | 5.8 | 9.3 |
| cat | 8.4 | 44.9 | 9.6 | 10.0 | 11.0 |
| pot1 | 9.1 | 48.1 | 8.8 | 9.0 | 10.2 |
| bear | 8.9 | 47.4 | 11.7 | 12.0 | 12.5 |
| pot2 | 14.9 | 50.7 | 14.3 | 14.1 | 14.4 |
| buddha | 15.1 | 50.0 | 14.6 | 14.5 | 14.7 |
| goblet | 18.7 | 52.9 | 16.5 | 16.4 | 16.4 |
| reading | 19.1 | 55.9 | 22.4 | 23.2 | 42.0 |
| cow | 25.6 | 34.8 | 19.3 | 19.8 | 20.4 |
| harvest | 30.5 | 43.9 | 30.8 | 31.1 | 46.4 (rough) |
| **mean** | 15.5 | 48.4 | 15.3 | 15.6 | 19.7 |

With the lamps even, finding the lights from the photographs is about as accurate as least squares given the true ones; the rough solve it replaces was 48 degrees out. With DiLiGenT's own lamps as photographed, which differ sixfold, the equal-lamp step does not hold: the error is about 30 degrees (19-32), still better than the rough solve, and the checks send three of the ten to the rough solve. A no-ball capture wants one lamp, moved round at a steady distance.
