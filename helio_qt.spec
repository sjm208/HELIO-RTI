# -*- mode: python ; coding: utf-8 -*-
"""Helio.app, built with PyInstaller (Apple silicon).

    ./build_helio3.sh            # the app, its self-test, then the disk image

PyInstaller, not py2app: its PySide6 hooks bring only the Qt modules the app
imports (no WebEngine, no Quick), and its multiprocessing support lets the
separate 3D window and the contact sheet's renders start child processes in
the built app.  The classic HELIO Pro keeps its py2app build (setup.py).
"""
from PyInstaller.utils.hooks import collect_all, collect_submodules

datas = [("helio/vendor/*.js", "helio/vendor"),
         ("helio/vendor/osd_images/*.png", "helio/vendor/osd_images"),
         ("helio/vendor/*.zip", "helio/vendor")]
binaries = []
hiddenimports = collect_submodules("helio_qt") + ["helio_help", "helio_build", "rvt.vis",
    # savefig picks its writer by the file's extension, by name at run time:
    # the plate and figures save PDF and SVG.  The first build's self-test
    # failed on exactly this.
    "matplotlib.backends.backend_pdf", "matplotlib.backends.backend_svg",
    "matplotlib.backends.backend_ps"]
# The Rust kernels (helio_core_rs, built by ./build_rust.sh) are an extension module
# the code imports inside try/except, so PyInstaller's analysis would not see it: named
# here.  Without it the app still runs, on its NumPy kernels, slower.
try:
    from helio_core_rs import lls_normals  # noqa: F401  (a bare import also matches the crate folder)
    hiddenimports.append("helio_core_rs")
except ImportError:
    print("helio_qt.spec: helio_core_rs is not installed - building WITHOUT the Rust kernels "
          "(run ./build_rust.sh first).")
# OpenUSD reads plugInfo.json files and its own libraries at run time; without
# them USDZ export fails with "no plugins".  scikit-image loads lazily from .pyi
# stubs.  vispy and glfw draw the contact sheet's 3D panels and the separate 3D
# window in child processes: vispy picks its GLFW backend by name and reads its
# GLSL from package data, and glfw finds libglfw.3.dylib in its own folder.
# None of the four has a PyInstaller hook, so each is collected whole.
for pkg in ("pxr", "skimage", "vispy", "glfw"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    ["helio3.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    # PyInstaller keeps only the matplotlib backends it sees named in
    # matplotlib.use() (Agg) and drops the rest, hidden imports included; the
    # plate saves PDF, the figures SVG and PostScript.
    hooksconfig={"matplotlib": {"backends": ["Agg", "pdf", "svg", "ps", "QtAgg"]}},
    runtime_hooks=[],
    excludes=[
        # The new app never loads Tk (test_qt_shell checks it); the classic does.
        "tkinter", "_tkinter", "customtkinter", "tkinterdnd2",
        "PyQt5", "PyQt6", "PySide2",
        "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
        "PySide6.Qt3DCore", "PySide6.Qt3DRender", "PySide6.QtQuick", "PySide6.QtQml",
        "PySide6.QtMultimedia", "PySide6.QtCharts", "PySide6.QtDataVisualization",
        "torch", "torchvision", "torchaudio", "tensorflow", "tensorboard", "onnxruntime",
        "IPython", "jupyter", "jupyter_client", "notebook", "ipykernel", "zmq", "tornado",
        "sympy", "pytest", "debugpy",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="Helio",
    debug=False,
    strip=False,
    upx=False,
    console=False,
    argv_emulation=False,
    target_arch="arm64",
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="Helio")
app = BUNDLE(
    coll,
    name="Helio.app",
    icon="packaging/helio_icon.icns",
    bundle_identifier="com.sjm208.helio",
    version="4.0.0",
    info_plist={
        "CFBundleName": "Helio",
        "CFBundleDisplayName": "Helio",
        "CFBundleShortVersionString": "4.0.0",
        "CFBundleVersion": "4.0.0",
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "13.0",          # Qt 6.11's own minimum
        "NSRequiresAquaSystemAppearance": False,
        # As the classic: App Nap can leave a minimised window unpainted.
        "NSAppSleepDisabled": True,
    },
)
