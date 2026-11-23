# Helio on DiLiGenT-Pi

DiLiGenT-Pi (Wang et al., ICCV 2023): 30 near-flat objects with fine relief — coins, medals, carved and
cast plaques, the scale and kind of a punch or a seal — each under 100 calibrated lights, 16-bit linear
photographs, normals from a scan. Run with `python tools/benchmark_diligent_pi.py --scale=0.5` (the photographs
halved: a quarter of the pixels, about 90,000 on each object). Figures: `docs/benchmark_diligent_pi.json`.
Measured 2026-11-12, after a colleague's Baskerville punches came out shallow.

## Normals: mean angular error over the 30 objects

| Solver (16-bit linear photographs) | mean | worst object | better than least squares on |
|---|---:|---:|---:|
| Linear Least Squares | 9.27° | 19.3° | — |
| IRLS (Metallic/Specular) | 9.22° | 19.3° | 11 |
| Iterative Specular Separation | 9.54° | 19.7° | 11 |
| Shadow-Rejection LS (relative cut, from 2026-11-12) | 10.52° | 18.5° | 8 |
| Mid-Tone Weighted LS | 12.58° | 17.7° | 7 |
| Sparse-Outlier LS (RPCA) | 13.64° | 23.2° | 10 |
| Shadow-Rejection LS (fixed cut at 12, before) | 21.07° | 38.3° | 7 |

- **Helio's least squares is the textbook fit** (identical to `numpy.linalg.lstsq` on the same data), and
  every depth method gives back the true shape from the true normals (relief, detail and shape 0.95–1.00).
  The solving and the integration are working.
- **Shadow-Rejection LS was broken on glossy and dark objects, and is fixed.** Its cut was a fixed pixel
  value (12 of 255). Exposed so the highlights do not clip, a glossy object's ordinary lit frames sit below
  that (a median of 8 on Sun, 2 on Crab), so most were thrown out and the fit made from the brightest:
  25° and 30° off where least squares is 7°. The cut is now relative to each pixel's own bright frames
  (12 = a tenth of its 90th percentile): 21.1° → 10.5° mean, 38° → 18° worst.
- **The solvers made for shine are not better on these objects.** RPCA, recommended for glossy objects
  from the DiLiGenT test, is worse than least squares on 20 of 30 here. Least squares, or IRLS, is the
  sound default for coins, punches and plaques.

## Camera JPEGs: the tone

The same photographs as an 8-bit sRGB camera JPEG holds them:

| | mean error | fine detail's depth (true = 1) | broad relief (true = 1) |
|---|---:|---:|---:|
| Solved as stored (Automatic, without evened lighting) | 9.37° | **0.59** | 1.34 |
| Decoded to linear first | 9.23° | 0.97 | 2.03 |

- The direction error hardly differs, but **solved as stored, the fine detail comes out at 60% of its
  depth**: the tone curve compresses the differences between frames, so every normal leans toward the
  camera. This is the "shallow" relief: the Height slider multiplies it, but cannot put back what the
  normals lost against the broad shape.
- In the window (50 JPEGs from a ring of lights at 28–55°, like a dome): as stored, the depth spans
  33 px and the normals are 15.2° off; decoded, 59 px and 13.8°.
- Solving as stored is better by angle on 12 of 30 objects, the glossy ones: the curve compresses their
  highlights (as on DiLiGenT's cow).

## Depth

- Integrated from the solved normals, the broad shape comes out about twice as deep as it is (relief 2.0,
  shape correlation 0.67): small biases in the normals add up over the object into a bow. The fine detail
  is right (0.97). Use *Flattened* in the 3D relief (or the Shape section's Detail vs shape) to look at the
  marks without the bow.
- The depth method makes no difference on these objects; the object alone and the whole frame differ by
  under 2% here (the surround is the dataset's black, without noise).

## The score

On four objects, seven solves each (five solvers, a JPEG as stored and decoded), the score's Depth part
(the one part that changes with the solver) gave 20/20 to nearly all of them, and its ranking followed
the real error only loosely (rank correlation 0.45). It ranked the JPEG solved as stored first on every
object, the least accurate on two of them, because a flatter solve agrees with its own depth more easily.
It did mark the one failed solve (RPCA on Queen, 23° off: 4/20). It warns of a broken solve; it cannot
choose between two sound ones.
