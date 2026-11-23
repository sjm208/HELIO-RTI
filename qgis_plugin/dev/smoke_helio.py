"""One-click HELIO import end to end.

    dev/qpy.sh dev/smoke_helio.py <helio export folder> [out_dir] [render]

Imports the folder (height maps, colour, normals into a group; tiled copies of slow files; colour aligned;
integrations compared), optionally renders the standard set, then previews the result and saves screenshots of the
dialog and of the Start here guide. Outputs go to out_dir, never into the export folder itself."""
import os, re, sys, time
S = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(S, "pkgs"))
from qgis.core import QgsApplication, QgsProject
app = QgsApplication([], True); app.initQgis()
folder = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else os.path.join(S, "out_smoke_helio")
render = len(sys.argv) > 3 and sys.argv[3] == "render"
os.makedirs(out, exist_ok=True)
import helio_relief.dialog as D
D.LOOKS_FILE = os.path.join(out, "looks.json")
dlg = D.HelioReliefDialog(None, None); dlg._run_sync = True; dlg.resize(1500, 920); dlg.show()
app.processEvents()
dlg._w["export_dir"].setFilePath(out)
t = time.perf_counter(); dlg._import_helio(folder, render=render); app.processEvents()
print(f"import{' + standard renders' if render else ''}: {time.perf_counter() - t:.0f} s")
for node in QgsProject.instance().layerTreeRoot().children():
    if hasattr(node, "findLayers"):
        print("group:", node.name())
        for l in node.findLayers():
            print(f"   {'[x]' if l.itemVisibilityChecked() else '[ ]'} {l.name()}")
    else:
        print(f"layer: {'[x]' if node.itemVisibilityChecked() else '[ ]'} {node.name()}")
w = dlg._w
name = lambda k: w[k].currentLayer().name() if w[k].currentLayer() is not None else "-"
print("height map:", name("surface"), "| colour:", name("colour"), "| normals:", name("normals"))
print("mode:", dlg._mode(), "| colour composite on:", w["r_albedo"].isChecked(), "| units:", dlg._unit_label(dlg._ls["info"]))
print("colour note:", dlg.lbl_colour.text() or "-", "| normals note:", dlg.lbl_normals.text() or "-")
if dlg.lbl_pick.text():
    print("picker:", " ".join(re.sub(r"<[^>]+>", " ", dlg.lbl_pick.text()).split())[:600])
print("files:", sorted(f for f in os.listdir(out) if not f.endswith((".json", ".aux.xml"))))
info = dlg._ls["info"]
dlg.canvas.showWindow((info.width // 2 - 367, info.height // 2 - 330, 734, 655)); dlg.canvas.zoomTo(1.0)
app.processEvents(); dlg._want_overview = True; dlg._request_previews()
for _ in range(4):
    app.processEvents()
dlg.grab().save(os.path.join(out, "dialog_helio.png"))
dlg._show_start_here(); app.processEvents()
dlg._start_here.grab().save(os.path.join(out, "start_here.png"))
dlg.shutdown(); app.exitQgis()
