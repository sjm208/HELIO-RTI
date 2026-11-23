"""The colour schemes: Graphite and Paper (default), Night Dome and Parchment (UX-001).

Graphite (dark) and Paper (light) are neutral greys, as photo and colour
tools use (Lightroom, Capture One, darktable's grey themes): a tinted
interface shifts how the colours of the object beside it are seen, and
ISO 12646 asks for a neutral surround when colour is being judged.  Night
Dome and Parchment, the colours of the Helio icon (dome navy, manuscript
parchment), stay as a choice: View › Appearance.

Lamplight gold is the only accent in every scheme and marks the one next
step on a screen.  Status colours always travel with an icon and a word,
never alone.  Images sit on a neutral grey well whichever scheme is on, so
the surround never tints what is being judged.  Data colour maps (viridis,
cividis) stay out of the chrome.

Every text token is checked against every surface it sits on at WCAG AA
(4.5:1) by tests/test_qt_shell.py; disabled text is exempt, as WCAG allows.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPalette

BODY_PT = 13          # body text; 14 with Larger Text
CAPTION_PT = 12       # the floor: nothing smaller, anywhere
SECTION_PT = 15
TITLE_PT = 22


@dataclass(frozen=True)
class Tokens:
    name: str
    ground: str        # window background
    panel: str         # sidebar, inspector, header, bottom bar
    raised: str        # selected row, secondary button
    well: str          # sunken track behind the workspace switch
    line: str          # dividers and outlines
    popover: str       # menus, palette, cards that float
    popover_line: str
    text: str
    text2: str         # supporting text
    text3: str         # section headings in caps, captions
    disabled: str
    accent: str        # lamplight gold: fills only in the light theme
    accent_hover: str
    on_accent: str     # text on gold
    accent_text: str   # gold used as text or a link
    image_well: str    # neutral, behind every image
    ok: str
    warn: str
    problem: str
    info: str

    def colours(self) -> dict[str, str]:
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "name"}


GRAPHITE = Tokens(
    name="dark",
    ground="#1B1C1F", panel="#232428", raised="#323439", well="#18191C",
    line="#36383D", popover="#2A2B2F", popover_line="#45474D",
    text="#ECEAE5", text2="#B4B6BC", text3="#A0A3AA", disabled="#6C6F76",
    accent="#E6A93E", accent_hover="#F0BE62", on_accent="#1C1406", accent_text="#EDB65A",
    image_well="#2B2B2B",
    ok="#4CC89C", warn="#E9C46A", problem="#FF8F6E", info="#7CC0EE",
)

PAPER = Tokens(
    name="light",
    ground="#EDEBE7", panel="#F7F6F3", raised="#E4E1DB", well="#E6E3DE",
    line="#D6D2CB", popover="#FFFFFF", popover_line="#CFCAC1",
    text="#1E1F22", text2="#45484F", text3="#585B62", disabled="#80838A",
    accent="#E6A93E", accent_hover="#D49A30", on_accent="#1C1406", accent_text="#80530A",
    image_well="#767676",
    ok="#0B6B4C", warn="#844F00", problem="#B42318", info="#1F5F8B",
)


NIGHT_DOME = Tokens(
    name="dark",
    ground="#10162A", panel="#151C33", raised="#26304F", well="#0D1224",
    line="#2C3756", popover="#1E2745", popover_line="#34405F",
    text="#ECE7DA", text2="#AEB5C8", text3="#959DB6", disabled="#6E7896",
    accent="#E9B44C", accent_hover="#F3CB78", on_accent="#1B1405", accent_text="#E9B44C",
    image_well="#0E0F12",
    ok="#3CCB9A", warn="#F0C76A", problem="#FF8A65", info="#6CC4F0",
)

PARCHMENT = Tokens(
    name="light",
    ground="#F4EFE4", panel="#FBF8F1", raised="#EDE5D3", well="#E9E1CE",
    line="#DCD2BC", popover="#FFFDF8", popover_line="#D3C8AF",
    text="#1F2433", text2="#4C5366", text3="#5F6577", disabled="#787D8A",
    accent="#E9B44C", accent_hover="#D9A136", on_accent="#1B1405", accent_text="#8A5D0C",
    image_well="#6F7176",
    ok="#0B6B4C", warn="#8A5300", problem="#B42318", info="#1F5F8B",
)

# Scheme -> {mode: tokens}.  Graphite is the default: neutral for judging colour.
FAMILIES = {"graphite": {"dark": GRAPHITE, "light": PAPER},
            "nightdome": {"dark": NIGHT_DOME, "light": PARCHMENT}}
FAMILY_NAMES = {"graphite": "Graphite and Paper (neutral)", "nightdome": "Night Dome and Parchment"}
DEFAULT_FAMILY = "graphite"
# Every palette, for the contrast checks.
THEMES = {f"{fam}-{mode}": t for fam, modes in FAMILIES.items() for mode, t in modes.items()}
APPEARANCES = ("system", "light", "dark")
DEFAULT_APPEARANCE = "dark"     # until the user picks one in View › Appearance

# Text tokens and the surfaces each must be legible on (checked by the tests).
TEXT_TOKENS = ("text", "text2", "text3", "accent_text", "ok", "warn", "problem", "info")
SURFACES = ("ground", "panel", "raised", "popover")


def _lin(c: float) -> float:
    c /= 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def luminance(hex_colour: str) -> float:
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def contrast(a: str, b: str) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def system_scheme() -> str:
    """'dark' or 'light' as macOS reports it now (Qt 6.5+)."""
    try:
        scheme = QGuiApplication.styleHints().colorScheme()
        return "dark" if scheme == Qt.ColorScheme.Dark else "light"
    except Exception:
        return "dark"


def resolve(appearance: str, family: str = DEFAULT_FAMILY) -> Tokens:
    mode = system_scheme() if appearance == "system" else appearance
    return FAMILIES.get(family, FAMILIES[DEFAULT_FAMILY])[mode]


def palette(t: Tokens) -> QPalette:
    p = QPalette()
    C = QColor
    for group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive):
        p.setColor(group, QPalette.ColorRole.Window, C(t.ground))
        p.setColor(group, QPalette.ColorRole.WindowText, C(t.text))
        p.setColor(group, QPalette.ColorRole.Base, C(t.panel))
        p.setColor(group, QPalette.ColorRole.AlternateBase, C(t.raised))
        p.setColor(group, QPalette.ColorRole.Text, C(t.text))
        p.setColor(group, QPalette.ColorRole.Button, C(t.raised))
        p.setColor(group, QPalette.ColorRole.ButtonText, C(t.text))
        p.setColor(group, QPalette.ColorRole.Highlight, C(t.raised))
        p.setColor(group, QPalette.ColorRole.HighlightedText, C(t.text))
        p.setColor(group, QPalette.ColorRole.ToolTipBase, C(t.popover))
        p.setColor(group, QPalette.ColorRole.ToolTipText, C(t.text))
        p.setColor(group, QPalette.ColorRole.PlaceholderText, C(t.text3))
        p.setColor(group, QPalette.ColorRole.Link, C(t.accent_text))
        p.setColor(group, QPalette.ColorRole.Accent, C(t.accent))
        p.setColor(group, QPalette.ColorRole.Mid, C(t.line))
        p.setColor(group, QPalette.ColorRole.Dark, C(t.well))
    dis = QPalette.ColorGroup.Disabled
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text,
                 QPalette.ColorRole.ButtonText, QPalette.ColorRole.HighlightedText):
        p.setColor(dis, role, C(t.disabled))
    p.setColor(dis, QPalette.ColorRole.Window, C(t.ground))
    p.setColor(dis, QPalette.ColorRole.Base, C(t.panel))
    p.setColor(dis, QPalette.ColorRole.Button, C(t.panel))
    return p


def _indicator_files(t: Tokens) -> tuple[str, str, str]:
    """A tick and a dot for check boxes and radio buttons, in this theme's colours.

    Qt style sheets take images only from files, so they are written once
    per theme to the temporary folder."""
    import os
    import tempfile
    d = os.path.join(tempfile.gettempdir(), "helio_qt_indicators")
    os.makedirs(d, exist_ok=True)
    tick = os.path.join(d, f"tick_{t.name}.svg")
    dot = os.path.join(d, f"dot_{t.name}.svg")
    arrow = os.path.join(d, f"arrow_{t.name}.svg")
    for path, body in (
            (arrow, f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><path d="M4 6l4 4 4-4" '
                    f'fill="none" stroke="{t.text2}" stroke-width="1.75" stroke-linecap="round" '
                    f'stroke-linejoin="round"/></svg>'),
            (tick, f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><path d="M3.5 8.5l3 3 6-7" '
                   f'fill="none" stroke="{t.on_accent}" stroke-width="2" stroke-linecap="round" '
                   f'stroke-linejoin="round"/></svg>'),
            (dot, f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">'
                  f'<circle cx="8" cy="8" r="3.5" fill="{t.on_accent}"/></svg>')):
        try:
            with open(path, "w") as f:
                f.write(body)
        except OSError:
            pass
    return tuple(f.replace(os.sep, "/") for f in (tick, dot, arrow))


def stylesheet(t: Tokens) -> str:
    """Widget styling by role.  Sizes are points; nothing is below CAPTION_PT."""
    tick, dot, arrow = _indicator_files(t)
    return f"""
* {{ font-size: {BODY_PT}pt; }}
QMainWindow, QDialog {{ background: {t.ground}; color: {t.text}; }}
QToolTip {{ background: {t.popover}; color: {t.text}; border: 1px solid {t.popover_line};
            padding: 6px 8px; border-radius: 6px; font-size: {CAPTION_PT}pt; }}

#Header, #BottomBar, #Sidebar, #Inspector {{ background: {t.panel}; }}
#Header {{ border-bottom: 1px solid {t.line}; background: {t.panel}; }}
#BottomBar {{ border-top: 1px solid {t.line}; }}
#Sidebar {{ border-right: 1px solid {t.line}; }}
#Inspector {{ border-left: 1px solid {t.line}; }}
#ImageWell {{ background: {t.image_well}; border: none; }}
#Filmstrip {{ background: {t.panel}; border: none; border-top: 1px solid {t.line}; }}
#Page {{ background: {t.ground}; }}

QLabel {{ color: {t.text}; background: transparent; }}
QLabel[role="title"] {{ font-size: {TITLE_PT}pt; font-weight: 600; }}
QLabel[role="section"] {{ font-size: {SECTION_PT}pt; font-weight: 600; }}
QLabel[role="heading"] {{ font-size: {CAPTION_PT}pt; font-weight: 600; color: {t.text3};
                          letter-spacing: 0.5px; }}
QLabel[role="secondary"] {{ color: {t.text2}; }}
QLabel[role="caption"] {{ font-size: {CAPTION_PT}pt; color: {t.text3}; }}
QLabel[role="projectTitle"] {{ font-size: 14pt; font-weight: 600; }}
QLabel[role="projectSub"] {{ font-size: {CAPTION_PT}pt; color: {t.text2}; }}
QLabel[status="ok"] {{ color: {t.ok}; }}
QLabel[status="warn"] {{ color: {t.warn}; }}
QLabel[status="problem"] {{ color: {t.problem}; }}
QLabel[status="info"] {{ color: {t.info}; }}
QLabel[status="idle"] {{ color: {t.text2}; }}
QLabel[role="reason"] {{ font-size: {CAPTION_PT}pt; color: {t.warn}; }}
QLabel[role="chip"] {{ font-size: {CAPTION_PT}pt; font-weight: 600; color: {t.accent_text};
                       background: {t.panel}; border: 1px solid {t.line}; border-radius: 9px;
                       padding: 1px 8px; letter-spacing: 0.4px; }}

#WorkspaceBar {{ background: {t.well}; border: 1px solid {t.line}; border-radius: 8px; }}
#WorkspaceBar QPushButton {{ background: transparent; color: {t.text2}; border: none;
                             border-radius: 6px; padding: 5px 12px; text-align: left; }}
#WorkspaceBar QPushButton:hover {{ color: {t.text}; }}
#WorkspaceBar QPushButton:checked {{ background: {t.raised}; color: {t.text}; font-weight: 600; }}
#WorkspaceBar QPushButton:focus {{ border: 1px solid {t.accent}; }}
#QuickWeb {{ background: {t.raised}; border: 1px solid {t.accent}; color: {t.text};
             font-weight: 600; text-align: left; padding: 7px 12px; }}
#QuickWeb:hover {{ border-color: {t.accent_hover}; }}
#ViewPicker {{ background: {t.well}; border: 1px solid {t.line}; border-radius: 6px; padding: 0;
               text-align: left; }}
#ViewPicker:hover {{ border-color: {t.text3}; }}
#ViewPicker:focus {{ border: 1px solid {t.accent}; }}
#KeyCap {{ background: {t.raised}; color: {t.text}; border: 1px solid {t.popover_line};
           border-bottom-width: 2px; border-radius: 4px; padding: 0 5px;
           font-family: "SF Mono", "Menlo", monospace; font-size: {CAPTION_PT}pt; }}
#KeyCard {{ background: {t.popover}; border: 1px solid {t.popover_line}; border-radius: 8px; }}
#StepHead {{ background: transparent; border: none; border-bottom: 1px solid {t.line};
             border-radius: 0; padding: 8px 12px; text-align: left; color: {t.text}; }}
#StepHead:hover {{ background: {t.raised}; }}
#StepHead:checked {{ background: transparent; }}
#StepHead:focus {{ border: 1px solid {t.accent}; }}
#Toolbar {{ background: {t.panel}; border-bottom: 1px solid {t.line}; }}
QFrame[floatbar="true"] {{ background: {t.popover}; border: 1px solid {t.popover_line};
                           border-radius: 9px; }}
QToolButton[tool="true"]:checked, QPushButton[tool="true"]:checked {{ background: {t.raised}; color: {t.accent_text};
                                    border: 1px solid {t.accent}; border-radius: 6px; }}
QFrame[floatbar="true"] QToolButton {{ background: transparent; border: none; border-radius: 6px;
                                       padding: 4px; color: {t.text2}; min-width: 26px; min-height: 24px; }}
QFrame[floatbar="true"] QToolButton:hover {{ background: {t.raised}; color: {t.text}; }}
QFrame[floatbar="true"] QToolButton:checked {{ background: {t.raised}; color: {t.accent_text}; }}
QFrame[floatbar="true"] QToolButton:focus {{ border: 1px solid {t.accent}; }}

QPushButton {{ background: {t.raised}; color: {t.text}; border: 1px solid {t.line};
               border-radius: 6px; padding: 5px 12px; min-height: 20px; }}
QPushButton:hover {{ border-color: {t.text3}; }}
QPushButton:focus {{ border: 2px solid {t.accent}; }}
QPushButton:disabled {{ background: {t.panel}; color: {t.disabled}; border-color: {t.line}; }}
QPushButton[kind="primary"] {{ background: {t.accent}; color: {t.on_accent}; border: none;
                               font-weight: 600; font-size: 13pt; border-radius: 7px;
                               padding: 7px 18px; }}
QPushButton[kind="primary"]:hover {{ background: {t.accent_hover}; }}
QPushButton[kind="primary"]:focus {{ border: 2px solid {t.text}; }}
QPushButton[kind="primary"]:disabled {{ background: {t.raised}; color: {t.disabled}; }}
QPushButton[kind="quiet"] {{ background: transparent; border: none; color: {t.accent_text};
                             padding: 4px 2px; text-align: left; }}
QPushButton[kind="quiet"]:disabled {{ color: {t.disabled}; }}
QPushButton[kind="outline"] {{ background: transparent; color: {t.text2}; }}
QPushButton[kind="destructive"] {{ background: transparent; color: {t.problem};
                                   border-color: {t.problem}; }}
QPushButton[kind="icon"] {{ background: transparent; border: 1px solid {t.line};
                            padding: 4px; min-width: 24px; }}
QPushButton[kind="info"] {{ background: transparent; border: 1px solid {t.line};
                            border-radius: 11px; padding: 0; min-width: 20px; max-width: 20px;
                            min-height: 20px; max-height: 20px; color: {t.text3};
                            font-size: {CAPTION_PT}pt; }}

QFrame[card="true"] {{ background: {t.panel}; border: 1px solid {t.line}; border-radius: 8px; }}
QFrame[card="true"]:hover {{ border-color: {t.text3}; background: {t.popover}; }}
QFrame[card="true"][selected="true"] {{ background: {t.popover}; border-color: {t.accent}; }}
QFrame[hud="true"] {{ background: {t.popover}; border: 1px solid {t.popover_line}; border-radius: 8px; }}
QFrame[rule="true"] {{ background: {t.line}; max-height: 1px; min-height: 1px; border: none; }}

QListWidget, QListView, QTreeView {{ background: transparent; border: none; color: {t.text};
                                     outline: 0; }}
QListWidget::item {{ padding: 6px 10px; border-radius: 6px; color: {t.text2}; }}
QListWidget::item:selected {{ background: {t.raised}; color: {t.text}; }}
QListWidget::item:disabled {{ color: {t.disabled}; }}
QTreeWidget {{ background: transparent; border: none; color: {t.text}; outline: 0; }}
QTreeWidget::item {{ padding: 5px 4px; border-radius: 6px; color: {t.text2}; }}
QTreeWidget::item:selected {{ background: {t.raised}; color: {t.text}; }}
QTreeWidget::item:disabled {{ color: {t.disabled}; }}
#Filmstrip::item {{ padding: 2px; border-radius: 6px; border: 2px solid transparent; }}
#Filmstrip::item:selected {{ border: 2px solid {t.accent}; background: transparent; }}

QComboBox {{ background: {t.well}; color: {t.text}; border: 1px solid {t.line};
             border-radius: 6px; padding: 4px 8px; min-height: 22px; }}
QComboBox:hover {{ border-color: {t.text3}; }}
QComboBox:focus {{ border: 1px solid {t.accent}; }}
QComboBox:disabled {{ color: {t.disabled}; }}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox::down-arrow {{ image: url({arrow}); width: 14px; height: 14px; }}
QComboBox QAbstractItemView {{ background: {t.popover}; color: {t.text};
                               selection-background-color: {t.raised}; border: 1px solid {t.popover_line}; }}
QLineEdit {{ background: {t.well}; color: {t.text}; border: 1px solid {t.line};
             border-radius: 6px; padding: 5px 8px; selection-background-color: {t.raised}; }}
QLineEdit:focus {{ border: 2px solid {t.accent}; }}
QCheckBox, QRadioButton {{ color: {t.text}; spacing: 8px; }}
QCheckBox:disabled, QRadioButton:disabled {{ color: {t.disabled}; }}
QCheckBox::indicator, QRadioButton::indicator {{ width: 16px; height: 16px; background: {t.well};
                                                border: 1px solid {t.text3}; }}
QCheckBox::indicator {{ border-radius: 4px; }}
QRadioButton::indicator {{ border-radius: 9px; }}
QCheckBox::indicator:checked {{ background: {t.accent}; border-color: {t.accent}; image: url({tick}); }}
QRadioButton::indicator:checked {{ background: {t.accent}; border-color: {t.accent}; image: url({dot}); }}
QCheckBox::indicator:focus, QRadioButton::indicator:focus {{ border: 2px solid {t.accent}; }}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{ border-color: {t.line}; }}
QSlider::groove:horizontal {{ height: 4px; background: {t.raised}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {t.accent}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {t.text}; width: 14px; height: 14px; margin: -5px 0;
                              border-radius: 7px; }}
QProgressBar {{ background: {t.raised}; border: none; border-radius: 3px; max-height: 6px;
                color: transparent; }}
QProgressBar::chunk {{ background: {t.accent}; border-radius: 3px; }}
QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QSplitter::handle {{ background: {t.line}; }}
QScrollBar:vertical, QScrollBar:horizontal {{ background: transparent; width: 10px; height: 10px; }}
QScrollBar::handle {{ background: {t.raised}; border-radius: 5px; min-height: 24px; min-width: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}

QToolButton[section="true"] {{ background: transparent; border: none; color: {t.text3};
                               font-size: {CAPTION_PT}pt; font-weight: 600; letter-spacing: 0.6px;
                               padding: 2px 0; }}
QToolButton[section="true"]:hover {{ color: {t.text}; }}
QToolButton[section="true"]:focus {{ color: {t.accent_text}; }}
QFrame[segmented="true"] {{ background: {t.well}; border: 1px solid {t.line}; border-radius: 7px; }}
QFrame[segmented="true"] QPushButton {{ background: transparent; border: none; border-radius: 5px;
                                        padding: 5px 4px; color: {t.text2}; min-height: 22px;
                                        font-size: {CAPTION_PT}pt; }}
QFrame[segmented="true"] QPushButton:hover {{ color: {t.text}; }}
QFrame[segmented="true"] QPushButton:checked {{ background: {t.raised}; color: {t.text};
                                                font-weight: 600; }}
QFrame[segmented="true"] QPushButton:focus {{ border: 1px solid {t.accent}; }}
QLabel[hud="true"] {{ background: {t.popover}; color: {t.text}; border: 1px solid {t.popover_line};
                      border-radius: 8px; padding: 6px 8px; font-family: "SF Mono", "Menlo", monospace;
                      font-size: {CAPTION_PT}pt; }}
QLabel[badge="true"] {{ background: {t.popover}; color: {t.warn}; border: 1px solid {t.warn};
                        border-radius: 8px; padding: 4px 8px; font-size: {CAPTION_PT}pt;
                        font-weight: 600; }}
QLabel[loupe="true"] {{ background: {t.image_well}; color: {t.text3}; border: 1px solid {t.line};
                        border-radius: 8px; font-size: {CAPTION_PT}pt; }}
QDoubleSpinBox, QSpinBox {{ background: {t.well}; color: {t.text}; border: 1px solid {t.line};
                            border-radius: 6px; padding: 2px 6px; min-height: 20px;
                            font-family: "SF Mono", "Menlo", monospace; font-size: {CAPTION_PT}pt;
                            selection-background-color: {t.raised}; }}
QDoubleSpinBox:focus, QSpinBox:focus {{ border: 2px solid {t.accent}; }}
QDoubleSpinBox:disabled, QSpinBox:disabled {{ color: {t.disabled}; }}

#Palette {{ background: {t.popover}; border: 1px solid {t.popover_line}; border-radius: 14px; }}
#Palette QLineEdit {{ background: transparent; border: none; border-bottom: 1px solid {t.popover_line};
                      border-radius: 0; font-size: 16pt; padding: 12px 14px; }}
"""


def base_font() -> QFont:
    f = QGuiApplication.font()
    f.setPointSize(BODY_PT)
    return f


def apply(app, appearance: str, family: str = DEFAULT_FAMILY) -> Tokens:
    """Apply *appearance* ('system', 'light' or 'dark') in the colour scheme
    *family* ('graphite' or 'nightdome') live, without a restart."""
    t = resolve(appearance, family)
    try:   # Qt 6.8+: native title bars and menus follow too
        app.styleHints().setColorScheme(
            Qt.ColorScheme.Unknown if appearance == "system"
            else (Qt.ColorScheme.Dark if t.name == "dark" else Qt.ColorScheme.Light))
    except Exception:
        pass
    global CURRENT
    CURRENT = t
    app.setFont(base_font())
    app.setPalette(palette(t))
    app.setStyleSheet(stylesheet(t))
    app.setProperty("helio_theme", t.name)
    return t


CURRENT: Tokens = GRAPHITE      # what custom-painted controls draw with
