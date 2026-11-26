Helio 4.0
Surface imaging for objects that hide their detail
===================================================================

Photograph an object several times from a fixed camera, moving only
the light. Helio turns that stack of photographs into surface shape
you can relight, measure, model and archive.

Capture, process, inspect, measure and publish, in five steps.

New in 4.0: no chrome ball needed (Helio can find the lights from the
photographs themselves); flat objects kept flat, with the lamps'
distance read from the photographs; RAW files through the whole
program (Phase One IIQ, Sony, Canon, Nikon, Hasselblad, uncompressed
DNG and more); colour read by each file's ICC profile; a publication
sheet you can see and edit; Legend Unwrap for several seals on a page;
deposit packages with paradata, OpenEXR masters and layered USD; and
the Helio Relief plugin for QGIS.


INSTALLING  -  READ THIS FIRST
-------------------------------------------------------------------

Double-click  "Install Helio.command"  in this window. It copies
Helio to Applications, clears the flag described below, and opens
Helio. If a copy is already there it asks before replacing it.

The first time, macOS may refuse to open the installer itself,
because it was downloaded and Apple has not checked it:

  macOS 15 (Sequoia) or later:
    1  Click Done in the message.
    2  Open System Settings > Privacy & Security, scroll down to
       Security, and click "Open Anyway" beside Install Helio.
    3  Enter your password, then click Open Anyway once more.

  macOS 13 or 14:
    Right-click (or Control-click) "Install Helio.command", choose
    Open, then click Open again.

You only do this once. After the installer has run, Helio opens like
any other app.


NO WARNINGS AT ALL: INSTALL FROM TERMINAL
-------------------------------------------------------------------

Open Terminal (Applications > Utilities), paste this line and press
Return:

  curl -fsSL https://raw.githubusercontent.com/sjm208/HELIO-RTI/main/install_helio.sh | bash

It downloads the latest Helio, copies it to Applications and opens
it. A download made this way carries no quarantine flag, so macOS
shows none of the messages above. Run it again to update.


"HELIO IS DAMAGED AND CAN'T BE OPENED"
-------------------------------------------------------------------

If you see this, the app is NOT damaged, and you should NOT move it
to the Trash.

macOS marks everything a browser downloads with a hidden
"quarantine" flag, and refuses to open a quarantined application that
has not been through Apple's paid notarisation process. Helio has
not. macOS shows the same alarming wording whether an app is perfectly
sound or genuinely corrupt.

Removing the quarantine flag fixes it, and that is exactly what
"Install Helio.command" does. If you dragged Helio across by hand,
run the installer afterwards; it finds the copy in Applications and
clears it. Or paste this into Terminal and press Return:

  xattr -dr com.apple.quarantine /Applications/Helio.app

If your Mac is managed by an IT department, they may block all of the
above. In that case send them this paragraph: Helio is an
unnotarised open-source application (MIT licence) and needs either a
quarantine exemption or an allowlist entry.


GETTING STARTED WITH HELIO
-------------------------------------------------------------------

Drag a folder of photographs onto the window. The bar at the bottom
always shows the next thing to do.

  1  Capture   the photographs, and where the lights were: a chrome
               ball found for you, a light file, placed by hand, or
               no ball at all (found from the photographs)
  2  Process   solve the surface
  3  Inspect   look at it under any light, in any view, or in 3D
  4  Measure   scale, distances, profiles, roughness, colour
  5  Publish   images, 3D models, web viewers, reports, archives

Capture One: export TIFFs, or uncompressed DNGs. Its sessions cannot be
read, and DNGs written with JPEG XL compression cannot be decoded. Use 8
lights or more: with 4, the quality score stays low.

Tip for hand-held flash: place a flat, matte grey card beside the
object (a QPcard or a ColorChecker grey patch), lying flat and lit by
every flash. Helio finds it and evens out the flash strength, which
keeps flat objects flat. With fixed lamps it matters less.

Command-K finds any tool by name. Command-/ lists every keyboard
shortcut. Hover over anything to see what it does. Help > Helio
Help explains the rest.


IF SOMETHING GOES WRONG
-------------------------------------------------------------------

Helio writes a diagnostic log to:

  ~/Library/Logs/HELIO_Pro/

Help > Report a Problem in Helio gathers what is useful to send.


QGIS
-------------------------------------------------------------------

Inspect > Send to QGIS opens the surface in QGIS 4. If the Helio Relief
plugin is missing or older than the one in this Helio, a button there
installs it into QGIS's plugins folder (only when you press it).


REQUIREMENTS
-------------------------------------------------------------------

  An Apple silicon Mac (M1 or later). This image is not for Intel Macs.
  macOS 13 (Ventura) or later for Helio.
  16 GB RAM recommended; 32 GB for large stacks.
  No GPU or deep-learning runtime is needed: the solvers are NumPy
  and SciPy, with the heaviest arithmetic in built-in Rust. The 3D relief inside Helio uses OpenGL 4.1, which
  every supported Mac has.


REMOVING IT
-------------------------------------------------------------------

Drag Helio from Applications to the Trash. To
remove the logs as well, delete the ~/Library/Logs/HELIO_Pro folder.
Helio does not install anything elsewhere (apart from the QGIS plugin,
if you ask it to), does not run at startup, and nothing phones home.


-------------------------------------------------------------------
(c) Scott Maloney 2026 - MIT License
