"""The QR code a phone can open, and a web export shared to a phone.

Offscreen, no capture needed.  An address pasted after https:// is mended,
localhost and file paths are called out, the code shows on screen and decodes
to its address, and Share serves a folder's index.html.

    python tests/test_qr_share.py
"""
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["HELIO_QT_SETTINGS"] = os.path.join(tempfile.mkdtemp(prefix="helio_qt_"), "prefs.ini")
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


def main():
    import cv2
    print("=" * 60)
    print("QR codes and sharing to a phone")
    print("=" * 60)
    from helio.application.phone_share import Share, tidy_url
    check("a doubled https is undone", tidy_url("https://https://example.org/a")
          == ("https://example.org/a", ""))
    check("…a missing one is added", tidy_url("example.org/a b")[0] == "https://example.org/a%20b")
    check("…localhost and file paths are called out", tidy_url("localhost:8000")[1] != ""
          and tidy_url("/Users/x/index.html")[1] != "")

    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    import helio_qt.window as hw
    import helio_qt.dialogs as dg
    hw.QMessageBox.warning = staticmethod(lambda *a, **k: None)
    w = hw.MainWindow(appearance="dark")
    d = dg.QrDialog(w)
    png = os.path.join(tempfile.mkdtemp(prefix="helio_qr_"), "qr.png")
    d.url.setText("https://https://example.org/helio/object-12")      # pasted after the https://
    d.save(png)
    txt = cv2.QRCodeDetector().detectAndDecode(cv2.imread(png))[0]
    check("the saved code decodes to the mended address", txt == "https://example.org/helio/object-12", txt)
    check("…and the code is on screen to scan", d.code.pixmap() is not None
          and not d.code.pixmap().isNull())
    d.url.setText("localhost:8000/index.html")
    check("localhost is called out: on a phone it is the phone", "phone" in d.status.text(),
          d.status.text())
    d.url.setText("/Users/someone/viewer/index.html")
    check("a file path makes no code", d.code.pixmap().isNull() and "file" in d.status.text())
    d.close()

    site = tempfile.mkdtemp(prefix="helio_share_")
    with open(os.path.join(site, "index.html"), "w") as fh:
        fh.write("<html><body>relief</body></html>")
    import urllib.request
    sh = Share(site, host="127.0.0.1")
    try:
        body = urllib.request.urlopen(sh.url, timeout=5).read().decode()
    except Exception as ex:
        body = str(ex)
    sh.stop()
    check("a web export is shared from this Mac, its index.html at the address",
          "relief" in body and sh.url.endswith("/index.html"), body[:80])
    w.close()
    app.processEvents()
    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    print("=" * 60)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
