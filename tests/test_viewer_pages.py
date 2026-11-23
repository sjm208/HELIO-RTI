"""The two 3D web viewers, opened in a real browser (headless Chromium, software WebGL).

* both pages load with no JavaScript or shader errors, show the file name top-left (no 📸 counter);
* the clay relief is not blown out to white (the detail is still there when the light moves);
* the fine-detail viewer has a Colour slider: at 0 the relief is neutral grey; at 1 the photograph's colour lies
  over it (the picture really turns the albedo's colour), and in between it is in between.

Needs node with playwright (skipped when they are absent).

    python tests/test_viewer_pages.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PASS = FAIL = 0

JS = r"""
const { chromium } = require('playwright');
(async () => {
  const args = process.argv.slice(2);
  const [file, outdir, mode] = args;
  const b = await chromium.launch({ args: ['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader','--ignore-gpu-blocklist','--enable-webgl','--allow-file-access-from-files'] });
  const p = await b.newPage({ viewport: { width: 1100, height: 700 } });
  const errs = [], logs = [];
  p.on('pageerror', e => errs.push(String(e)));
  p.on('console', m => { if (['error','warning'].includes(m.type())) logs.push(m.type() + ': ' + m.text().slice(0, 300)); });
  await p.goto('file://' + file);
  await p.waitForTimeout(2500);
  const out = { title: await p.title(), panel: await p.evaluate(() => document.getElementById('panel').innerText), errs, logs };
  out.name = await p.evaluate(() => (document.getElementById('objname') || {}).textContent || ''); out.noShotCounter = await p.evaluate(() => !document.getElementById('shot-badge'));
  out.hasColour = await p.evaluate(() => !!document.getElementById('s_col'));
  if (mode === 'colour') {
    out.colourEnabled = await p.evaluate(() => !document.getElementById('s_col').disabled);
    for (const v of [0, 0.5, 1]) {
      await p.evaluate(v => { const s = document.getElementById('s_col'); s.value = v; s.dispatchEvent(new Event('input')); }, v);
      await p.waitForTimeout(1200);
      await p.screenshot({ path: outdir + '/colour_' + v + '.png', clip: { x: 0, y: 0, width: 760, height: 700 } });
    }
    out.label = await p.evaluate(() => document.getElementById('v_col').textContent);
  }
  await b.close();
  console.log('JSON:' + JSON.stringify(out));
})().catch(e => { console.log('JSON:' + JSON.stringify({ crash: String(e) })); });
"""


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def run_page(html, outdir, mode):
    npm = subprocess.run(["npm", "root", "-g"], capture_output=True, text=True).stdout.strip()
    js = os.path.join(outdir, "page.js")
    open(js, "w").write(JS)
    r = subprocess.run(["node", js, html, outdir, mode], capture_output=True, text=True, timeout=180,
                       env=dict(os.environ, NODE_PATH=npm))
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("JSON:")]
    return json.loads(line[-1][5:]) if line else {"crash": (r.stdout + r.stderr)[-500:]}


def main():
    if not shutil.which("node"):
        print("  skip node is not installed")
        return 0
    probe = subprocess.run(["node", "-e", "require('playwright')"], capture_output=True, text=True,
                           env=dict(os.environ, NODE_PATH=subprocess.run(["npm", "root", "-g"], capture_output=True, text=True).stdout.strip()))
    if probe.returncode != 0:
        print("  skip playwright is not installed for node")
        return 0
    import cv2
    from helio.application import exports as X
    from helio.io.exporters import export_3d_web_viewer, export_3d_web_viewer_hf_normals

    H, W = 120, 160
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    depth = 4.0 * np.exp(-(((xx - 80) / 30) ** 2 + ((yy - 60) / 24) ** 2)) + 0.4 * np.sin(xx / 4.0)
    gy, gx = np.gradient(depth)
    n = np.dstack([-gx, gy, np.ones_like(gx)])
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    albedo = np.zeros((H, W, 3), np.uint8)
    albedo[:] = (210, 60, 40)                                         # a strong red: the colour is unmistakable
    albedo[:, ::2] = (200, 55, 38)
    settings = X.build_export_settings(X.shared_defaults(), X.MeshOptions(), W, H, obj_id="binding test")
    settings.update(step=1)

    d = tempfile.mkdtemp(prefix="helio_pages_")
    hf = os.path.join(d, "hf"); os.makedirs(hf)
    out = export_3d_web_viewer_hf_normals(n, albedo, dict(settings), hf)
    print("The fine-detail viewer")
    r = run_page(out["html"], hf, "colour")
    check("the page loads: no JavaScript errors, no shader errors", "crash" not in r and not r["errs"] and not [x for x in r["logs"] if "error" in x.lower()], r)
    check("the Colour slider is there and, with a colour image in the model, switched on", r.get("hasColour") and r.get("colourEnabled"), r)
    imgs = {v: cv2.imread(os.path.join(hf, f"colour_{v}.png")) for v in (0, 0.5, 1)}
    check("the pictures were taken", all(i is not None for i in imgs.values()))
    if all(i is not None for i in imgs.values()):
        def rb(img):                                            # red minus blue over the lit relief (BGR order)
            sel = img[100:600, 40:700].reshape(-1, 3).astype(float)
            sel = sel[sel.sum(1) > 60]
            return float(sel[:, 2].mean() - sel[:, 0].mean()), float(sel.mean())
        c0, c5, c1 = rb(imgs[0]), rb(imgs[0.5]), rb(imgs[1])
        print(f"       red minus blue: colour 0 -> {c0[0]:.1f}, 0.5 -> {c5[0]:.1f}, 1 -> {c1[0]:.1f}  (brightness {c0[1]:.0f}, {c1[1]:.0f})")
        check("at 0 the relief is neutral grey (no colour cast)", abs(c0[0]) < 6.0, c0)
        check("at 1 the photograph's red lies over it", c1[0] > 40.0, c1)
        check("and in between it is in between", c0[0] < c5[0] < c1[0], (c0[0], c5[0], c1[0]))
        check("the relief itself is still drawn (not blank)", float(imgs[0][100:600, 40:700].std()) > 5.0)
        g = imgs[0].astype(float).mean(axis=2)
        v = g[g > 25]
        check(f"the clay is not washed out: {100 * (v >= 240).mean():.0f}% of the surface near white, p10-p90 spread {np.percentile(v, 90) - np.percentile(v, 10):.0f}",
              (v >= 240).mean() < 0.05 and np.percentile(v, 90) - np.percentile(v, 10) > 40)
    check("the slider shows its value", r.get("label") == "1.00", r.get("label"))
    check("the file name is top-left and the screenshot counter is gone", r.get("name") == "binding test" and r.get("noShotCounter"), (r.get("name"), r.get("noShotCounter")))

    print("The GLB web viewer")
    gl = os.path.join(d, "glb"); os.makedirs(gl)
    out2 = export_3d_web_viewer(depth, depth / depth.max(), n, albedo, dict(settings), gl)
    r2 = run_page(out2["html"], gl, "plain")
    check("the page loads with no errors", "crash" not in r2 and not r2["errs"] and not [x for x in r2["logs"] if "error" in x.lower()], r2)
    check("the file name is top-left and the screenshot counter is gone", r2.get("name") == "binding test" and r2.get("noShotCounter"), (r2.get("name"), r2.get("noShotCounter")))
    print(f"\nResults: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    return FAIL


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
