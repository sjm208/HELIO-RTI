"""A link a phone can open: the address a QR code holds, tidied, and a web
export shared from this Mac over Wi-Fi.  GUI-free.

A QR code is only as good as its address.  A phone cannot open an address
typed twice over (``https://https://…``, from pasting into a box that
already said ``https://``), a bare file path, or ``localhost``, which on the
phone means the phone.  ``tidy_url`` mends what can be mended and says why
the rest will not open.  ``Share`` serves one folder (an RTI web package or
web viewer) to the phones on the same network, at this Mac's own address.
"""
from __future__ import annotations

import functools
import http.server
import os
import re
import socket
import threading

LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]")


def tidy_url(text: str) -> tuple[str, str]:
    """(address, problem): the address a phone can open, and why it may not
    open there ("" when it will).  A doubled scheme is undone, a missing
    one added (https), spaces percent-encoded."""
    u = (text or "").strip()
    if not u:
        return "", "Enter the address first."
    if u.startswith(("/", "~")) or re.match(r"^[A-Za-z]:\\", u) or u.lower().startswith("file:"):
        return u, ("That is a file on this computer: a phone cannot open it. Put the export "
                   "on a web site, or share it from this Mac below.")
    # https://https://x, http://https://x, https:// https://x: the last scheme wins
    m = re.match(r"^(?:\s*[a-z][a-z0-9+.-]*://\s*)+", u, re.I)
    if m:
        schemes = re.findall(r"([a-z][a-z0-9+.-]*)://", m.group(0), re.I)
        u = f"{schemes[-1].lower()}://" + u[m.end():]
    else:
        u = "https://" + u
    u = u.replace(" ", "%20")
    host = re.sub(r"^[a-z][a-z0-9+.-]*://", "", u, flags=re.I).split("/")[0].split(":")[0].lower()
    if not host or host == "https" or "." not in host and host not in LOCAL_HOSTS:
        return u, "That does not look like a web address (it needs a name like example.org)."
    if host in LOCAL_HOSTS:
        return u, ("localhost is this computer: on a phone it means the phone. Share it from "
                   "this Mac below instead.")
    return u, ""


def lan_address() -> str:
    """This Mac's address on its network (Wi-Fi or Ethernet), "" without one.
    Connecting a UDP socket sends nothing; it only asks which interface
    would carry the traffic."""
    for probe in ("192.168.1.1", "10.255.255.255", "8.8.8.8"):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect((probe, 1))
            ip = s.getsockname()[0]
            if ip and not ip.startswith("127."):
                return ip
        except OSError:
            continue
        finally:
            s.close()
    return ""


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):                     # no lines on the terminal per request
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")    # a re-export shows at once
        super().end_headers()


class Share:
    """*folder* served to the network at http://<this Mac>:<port>/, until
    stop().  The page opened is index.html when the folder has one."""

    PORTS = (8765, 8766, 8767, 8768, 0)

    def __init__(self, folder: str, host: str = ""):
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"No such folder: {folder}")
        self.folder = os.path.abspath(folder)
        self.host = host or lan_address()
        if not self.host:
            raise RuntimeError("This Mac is not on a network: join the same Wi-Fi as the phone.")
        handler = functools.partial(_Quiet, directory=self.folder)
        last = None
        for port in self.PORTS:
            try:
                self.server = http.server.ThreadingHTTPServer(("0.0.0.0", port), handler)
                break
            except OSError as e:
                last = e
        else:
            raise RuntimeError(f"No port to share on: {last}")
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True,
                                       name="helio-phone-share")
        self.thread.start()

    @property
    def url(self) -> str:
        page = "index.html" if os.path.exists(os.path.join(self.folder, "index.html")) else ""
        return f"http://{self.host}:{self.port}/{page}"

    def stop(self):
        if getattr(self, "server", None) is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None


def web_folder(start: str) -> str:
    """The folder to share for *start*: itself when it has an index.html,
    else the newest folder inside it (one level down) that has one, else
    *start*."""
    if not start or not os.path.isdir(start):
        return ""
    if os.path.exists(os.path.join(start, "index.html")):
        return start
    subs = [os.path.join(start, d) for d in os.listdir(start)]
    subs = [d for d in subs if os.path.isdir(d) and os.path.exists(os.path.join(d, "index.html"))]
    return max(subs, key=os.path.getmtime) if subs else start
