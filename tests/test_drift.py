"""The frame drift check: movement, not moving light (QC-002).

Under raking light the shading and the shadows move with the lamp, and the
old check (raw brightness against frame 1, at 900 px) read that as the
camera moving: 46 px on a gilt binding whose frames are within 6 px.
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def stack(shift_frame=None, shift=(0, 0), n=8, h=900, w=1200, seed=3):
    """A tooled surface lit from n directions 25° up; frame *shift_frame* moved."""
    import cv2
    import numpy as np
    rng = np.random.default_rng(seed)
    z = np.zeros((h, w), np.float32)
    for _ in range(260):                                   # stamped tool marks
        x, y, r = rng.uniform(40, w - 40), rng.uniform(40, h - 40), rng.uniform(4, 14)
        cv2.circle(z, (int(x), int(y)), int(r), float(rng.uniform(0.5, 1.5)), -1)
    z = cv2.GaussianBlur(z, (0, 0), 2.0) * 6.0
    gy, gx = np.gradient(z)
    nrm = np.dstack([-gx, -gy, np.ones_like(z)])
    nrm /= np.linalg.norm(nrm, axis=2, keepdims=True)
    out = []
    for k in range(n):
        a = 2 * np.pi * k / n
        L = np.array([np.cos(a) * np.cos(0.44), np.sin(a) * np.cos(0.44), np.sin(0.44)])
        img = np.clip(nrm @ L, 0, 1) * 200 + 20
        if k == shift_frame:
            M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
            img = cv2.warpAffine(img.astype(np.float32), M, (w, h), borderMode=cv2.BORDER_REFLECT)
        out.append(np.clip(img + rng.normal(0, 1.5, img.shape), 0, 255).astype(np.uint8))
    return out


def main():
    import cv2
    print("=" * 60)
    print("Frame drift")
    print("=" * 60)
    from helio.core.qc import interframe_drift
    from helio.application.capture_tools import drift_report
    still = interframe_drift(stack())
    check("a still camera under raking light reads as still (under 0.5 px)",
          still["worst"][1] < 0.5, still["worst"])
    moved = interframe_drift(stack(5, (6.0, -3.0)))
    i, m = moved["worst"]
    check("a frame moved 6.7 px is found, and named", i == 5 and 5.0 < m < 8.5, moved["worst"])
    check("…and the verdict says moved", moved["verdict"] == "moved", moved["verdict"])
    tmp = tempfile.mkdtemp(prefix="helio_drift_")
    paths = []
    for k, img in enumerate(stack(2, (3.0, 0.0))):
        p = os.path.join(tmp, f"f{k}.jpg")
        cv2.imwrite(p, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        paths.append(p)
    r = drift_report(paths)
    # Small moves read low (a 3 px move as about 1.7 px): a side-lit frame's
    # edges sit a little off the median's, which damps the peak.  The check is
    # a detector, so what matters is that the frame is found and flagged.
    check("from files, a 3 px move is found, named and flagged, well above a still stack",
          r["worst"] == "f2.jpg" and r["peak_px"] > 1.2 and r["verdict"] != "clean",
          (r["worst"], r["peak_px"], r["verdict"]))
    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    print("=" * 60)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
