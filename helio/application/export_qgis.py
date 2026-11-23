"""Send to QGIS (EXP-011): the surface opened in QGIS, with Helio Relief
(the QGIS plugin shipped in helio/vendor) when it is installed.

HELIO writes its GIS height-map package (helio.application.export_dem: the
README with honest units, the Poisson and Frankot-Chellappa heights as
32-bit float GeoTIFFs with world files, the colour, normals, uncertainty
and any metal mask), then starts QGIS with a small script.  The script
waits for QGIS to load its plugins and hands the folder to the plugin's
own HELIO import (``_import_helio``), which loads every layer in a group
and compares the two integrations.  Without the plugin it adds the layers
plainly.  The plugin's earlier name (the Heritage Photometric Stereo
Suite, 3.x) has the same import and is used when it is the one installed.

QGIS's profile is written to only when asked: install_plugin puts the plugin
Helio ships there, and the script turns it on.  Tests pass ``profiles_path`` so QGIS runs on a scratch profile.
"""
from __future__ import annotations

import glob
import os
import plistlib
import shutil
import subprocess
import sys
import tempfile
import zipfile

PLUGIN = "helio_relief"
PLUGIN_NAMES = (PLUGIN, "heritage_photometric_stereo_suite")     # the second: its name before 4.0


def find_qgis() -> list[dict]:
    """QGIS applications on this Mac, newest first: {app, exe, version, major}."""
    found = []
    roots = ["/Applications", os.path.expanduser("~/Applications")]
    for root in roots:
        for app in sorted(glob.glob(os.path.join(root, "QGIS*.app"))):
            try:
                with open(os.path.join(app, "Contents", "Info.plist"), "rb") as f:
                    info = plistlib.load(f)
            except (OSError, plistlib.InvalidFileException):
                continue
            exe = os.path.join(app, "Contents", "MacOS", info.get("CFBundleExecutable", ""))
            ver = str(info.get("CFBundleShortVersionString") or info.get("CFBundleVersion") or "0")
            if os.path.isfile(exe):
                try:
                    major = int(ver.split(".")[0])
                except ValueError:
                    major = 0
                found.append(dict(app=app, exe=exe, version=ver, major=major))
    found.sort(key=lambda q: tuple(int(p) if p.isdigit() else 0 for p in q["version"].split(".")),
               reverse=True)
    return found


def plugin_path(major: int, profiles_path: str | None = None) -> str | None:
    """Where the plugin is installed for QGIS *major* (Helio Relief, else its
    old name), or None.  Looked at, never written."""
    base = profiles_path or os.path.expanduser(
        f"~/Library/Application Support/QGIS/QGIS{major}/profiles")
    for name in PLUGIN_NAMES:
        p = os.path.join(base, "default", "python", "plugins", name)
        if os.path.isdir(p):
            return p
    return None


def plugin_version(path: str | None) -> str:
    if not path:
        return ""
    try:
        with open(os.path.join(path, "metadata.txt"), encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("version="):
                    return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return ""


def choose(qgis: list[dict], profiles_path: str | None = None) -> dict | None:
    """The QGIS to use: the newest with the plugin, else the newest."""
    qgis = sorted(qgis, key=lambda q: tuple(int(p) if p.isdigit() else 0
                                            for p in str(q["version"]).split(".")), reverse=True)
    for q in qgis:
        if plugin_path(q["major"], profiles_path):
            return q
    return qgis[0] if qgis else None


def bundled_zip() -> str | None:
    """The plugin Helio ships, newest first."""
    from helio.io.exporters import QGIS_PLUGIN_ZIP
    return QGIS_PLUGIN_ZIP if os.path.isfile(QGIS_PLUGIN_ZIP) else None


def bundled_version() -> str:
    z = bundled_zip()
    if not z:
        return ""
    with zipfile.ZipFile(z) as f:
        for line in f.read(f"{PLUGIN}/metadata.txt").decode("utf-8", "replace").splitlines():
            if line.startswith("version="):
                return line.split("=", 1)[1].strip()
    return ""


def _parts(v: str) -> tuple:
    return tuple(int(p) if p.isdigit() else 0 for p in str(v).split("."))


def plugin_state(major: int, profiles_path: str | None = None) -> dict:
    """What is installed against what Helio ships: {found, version, old_name, bundled, install}.
    *install*: Helio's copy would be an improvement (none, a lower version, or the old name)."""
    p = plugin_path(major, profiles_path)
    v, shipped = plugin_version(p), bundled_version()
    old = bool(p) and os.path.basename(p) != PLUGIN
    return dict(found=bool(p), version=v, old_name=old, bundled=shipped,
                install=bool(shipped) and (not p or old or _parts(v) < _parts(shipped)))


def install_plugin(major: int, profiles_path: str | None = None) -> str:
    """Helio's copy of the plugin into QGIS *major*'s default profile, replacing an older
    copy of Helio Relief.  The old-named plugin is left where it is.  Returns the folder."""
    z = bundled_zip()
    if not z:
        raise FileNotFoundError("This copy of Helio has no plugin to install.")
    base = profiles_path or os.path.expanduser(f"~/Library/Application Support/QGIS/QGIS{major}/profiles")
    plugins = os.path.join(base, "default", "python", "plugins")
    os.makedirs(plugins, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=".helio_relief_", dir=plugins)   # complete before it replaces anything
    try:
        with zipfile.ZipFile(z) as f:
            f.extractall(tmp)
        dest = os.path.join(plugins, PLUGIN)
        if os.path.isdir(dest):
            shutil.rmtree(dest)
        os.replace(os.path.join(tmp, PLUGIN), dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return dest


STARTUP = r'''"""Run by QGIS (--code): open a Helio package.  Written by Helio."""
import os
from qgis.PyQt.QtCore import QTimer
from qgis.core import QgsProject, QgsRasterLayer, QgsCoordinateReferenceSystem

FOLDER = {folder!r}
MM_PER_PX = {mm!r}


def _plain():
    from qgis.utils import iface
    crs = QgsCoordinateReferenceSystem.fromProj(
        "+proj=tmerc +lat_0=0 +lon_0=0 +k=1 +x_0=0 +y_0=0 +units=mm +no_defs")
    root = QgsProject.instance().layerTreeRoot()
    group = root.insertGroup(0, "Helio · " + os.path.basename(FOLDER))
    for f in sorted(os.listdir(FOLDER)):
        if f.lower().endswith((".tif", ".tiff")):
            lyr = QgsRasterLayer(os.path.join(FOLDER, f), os.path.splitext(f)[0])
            if lyr.isValid():
                if crs.isValid():
                    lyr.setCrs(crs)
                QgsProject.instance().addMapLayer(lyr, False)
                group.addLayer(lyr)
    if iface is not None:
        iface.messageBar().pushInfo("Helio", "Loaded " + os.path.basename(FOLDER)
                                    + " (Helio Relief was not found).")


def _turn_on():
    """A plugin just installed is off until it is turned on."""
    from qgis.core import QgsSettings
    from qgis.utils import available_plugins, loadPlugin, plugins, startPlugin
    for n in {names!r}:
        if n in plugins:
            return
        if n in available_plugins:
            QgsSettings().setValue("PythonPlugins/" + n, True)
            if loadPlugin(n) and startPlugin(n):
                return


def _go(tries=[0]):
    from qgis.utils import plugins, iface
    _turn_on()
    p = next((plugins[n] for n in {names!r} if n in plugins), None)
    if p is None and tries[0] < 10:          # plugins may still be loading
        tries[0] += 1
        QTimer.singleShot(1000, _go)
        return
    if p is None:
        _plain()
        return
    try:
        p.run()
        p.dialog._import_helio(FOLDER, render=False)
    except Exception as e:
        if iface is not None:
            iface.messageBar().pushWarning("Helio", "Helio Relief could not import the folder ("
                                           + str(e) + "); the layers are loaded plainly.")
        _plain()


QTimer.singleShot(1500, _go)
'''


def write_startup(folder: str, mm_per_px=None) -> str:
    path = os.path.join(folder, "open_in_qgis.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(STARTUP.format(folder=os.path.abspath(folder), mm=mm_per_px, names=PLUGIN_NAMES))
    return path


def package(D, out_dir: str, obj_name: str, *, mm_per_px=None, z_mm_per_unit=None,
            normal_engine: str = "", lights=None, provenance_app=None) -> dict:
    """The GIS package the plugin reads, plus the startup script.  Returns
    dem_package's figures with ``script`` and ``folder``."""
    from helio.application.export_dem import dem_package
    os.makedirs(out_dir, exist_ok=True)
    info = dem_package(D, out_dir, obj_name, mm_per_px=mm_per_px, z_mm_per_unit=z_mm_per_unit,
                       normal_engine=normal_engine, lights=lights, provenance_app=provenance_app)
    info = dict(info or {})
    info["folder"] = out_dir
    info["script"] = write_startup(out_dir, mm_per_px)
    return info


def clean_env(env=None) -> dict:
    """The environment for another program, without what this app's own bundle set.  Helio's
    QT_PLUGIN_PATH points at its Qt, and QGIS's Qt then aborts as it starts ("no Qt platform
    plugin could be initialized"); the rest would load Helio's libraries and Python into it."""
    src = dict(os.environ if env is None else env)
    out = {k: v for k, v in src.items()
           if not k.startswith(("QT_", "QML", "QML2_", "PYTHON", "DYLD_", "PYI_", "_PYI", "_MEIPASS"))
           and k != "LD_LIBRARY_PATH" and not k.endswith("_ORIG")}
    for k, v in src.items():                     # PyInstaller keeps what it replaced as NAME_ORIG
        if k.endswith("_ORIG"):
            out[k[:-5]] = v
    return out


def launch(qgis: dict, script: str, profiles_path: str | None = None, popen=None) -> object:
    """Start QGIS on *script*.  *popen* is for tests."""
    args = [qgis["exe"], "--code", script]
    if profiles_path:
        args += ["--profiles-path", profiles_path]
    run = popen or subprocess.Popen
    kw = {} if popen else dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               start_new_session=True, env=clean_env())
    return run(args, **kw)


if __name__ == "__main__":                                  # pragma: no cover
    for q in find_qgis():
        print(q, plugin_path(q["major"]), plugin_version(plugin_path(q["major"])), file=sys.stderr)
