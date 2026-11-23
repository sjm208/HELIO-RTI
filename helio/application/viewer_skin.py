"""The Helio look for the web viewers.  GUI-free.

The viewers Helio writes (WebRTI, the combined and normal-map RTI viewers,
the PTM, HSH and DeepZoom pages, the 3D GLB viewers, the engine sweep and
the A/B comparison) were drawn in HELIO Pro's colours: green on navy,
orange, teal, Arial at 9-10 px.  ``skin`` gives every one of them the new
interface's Graphite look instead (helio_qt/theme.py), so a file a reader
opens matches the app it came from:

* the Graphite tokens: neutral greys, a neutral image surround, lamplight
  gold as the one accent;
* the system font, nothing smaller than 12 px for words;
* the app's slider: a thin track filled in gold up to a light round handle;
* section headings in quiet capitals, values in a monospaced face;
* chosen modes marked as the app marks a chosen tool (gold edge and text),
  the one next step (save, export) as a gold button.

The rules go in last in <head>, so they win over the page's own colours
without touching its layout or script.  ``DOME`` holds the colours the
pages' scripts paint their light domes in, the same as the app's dome.
"""
from __future__ import annotations

MARK = "helio-skin"

# Graphite (helio_qt.theme.GRAPHITE): kept as literals so this stays GUI-free.
TOKENS = {
    "ground": "#1B1C1F", "panel": "rgba(35,36,40,.94)", "raised": "#323439", "well": "#18191C",
    "line": "#36383D", "popover": "rgba(42,43,47,.96)", "popover-line": "#45474D",
    "text": "#ECEAE5", "text2": "#B4B6BC", "text3": "#A0A3AA", "disabled": "#6C6F76",
    "accent": "#E6A93E", "accent-hover": "#F0BE62", "on-accent": "#1C1406",
    "accent-text": "#EDB65A", "image-well": "#2B2B2B", "problem": "#FF8F6E", "info": "#7CC0EE",
}

# The light dome as the app and the new RTI viewer draw it.
DOME = {"face": "#1A1B1E", "ring": "#3E4046", "label": "#B4B6BC",
        "light": "#E6A93E", "light_edge": "#222326"}

SANS = '-apple-system,BlinkMacSystemFont,"SF Pro Text","Segoe UI",Roboto,"Helvetica Neue",sans-serif'
MONO = 'ui-monospace,"SF Mono",Menlo,Consolas,monospace'

_VARS = ":root{" + "".join(f"--h-{k}:{v};" for k, v in TOKENS.items()) + "color-scheme:dark}"

_CSS = _VARS + """
html,body{font-family:%(sans)s!important;color:var(--h-text);-webkit-font-smoothing:antialiased}
body{background:var(--h-image-well)}

/* panels and bars */
#panel,#bar,#ui,#toolbar{background:var(--h-panel)!important;border-color:var(--h-line)!important;
  -webkit-backdrop-filter:blur(10px);backdrop-filter:blur(10px)}
#ui{border-radius:10px;box-shadow:0 8px 24px rgba(0,0,0,.35)}

/* headings: quiet capitals, as the app's section heads */
#panel h2,#panel h3,#bar h3,#ui h3,#panel>h2{color:var(--h-text3)!important;font:600 12px/1.3 %(sans)s!important;
  text-transform:uppercase;letter-spacing:.06em;border-color:var(--h-line)!important}
#bar h1{color:var(--h-text)!important;font:600 14px/1.3 %(sans)s!important}
body>h1{color:var(--h-text2)!important;font:600 14px/1.4 %(sans)s!important}

/* words and values */
label,.row,.slg,.crow,.krow{color:var(--h-text2)!important;font-size:12px!important}
.val,.slg .val,.row .val,.crow span,label span,#zlbl,#count{color:var(--h-text)!important;
  font:500 12px/1.3 %(mono)s!important;font-variant-numeric:tabular-nums}
#readout,#light-info,#lbl{color:var(--h-text2)!important;font:12px/1.4 %(mono)s!important}
.hint,.note,.sub,#hint,#hints,#status,.lgrouphdr{color:var(--h-text3)!important;font-size:12px!important}
#byline{color:var(--h-text2)!important;font-size:12px!important}
#name{color:var(--h-text)!important}
hr,.sep{border-color:var(--h-line)!important}
#bar>.sep{background:var(--h-line)!important}

/* buttons: secondary by default, a chosen mode as the app's chosen tool */
button,.mbtn,.hbtn,.mode-btn{background:var(--h-raised)!important;color:var(--h-text)!important;
  border:1px solid var(--h-line)!important;border-radius:6px!important;font:12px/1.3 %(sans)s!important;
  transition:border-color .12s,color .12s,background .12s}
button:hover,.mbtn:hover,.hbtn:hover,.mode-btn:hover{border-color:var(--h-text3)!important;color:var(--h-text)!important}
.mbtn.on,.hbtn.active,.mode-btn.active,#toolbar button.active{background:var(--h-raised)!important;
  color:var(--h-accent-text)!important;border-color:var(--h-accent)!important;font-weight:600!important}
button:focus-visible,.mbtn:focus-visible{outline:2px solid var(--h-accent);outline-offset:1px}
#shot,#expbtn,#export-btn{background:var(--h-accent)!important;color:var(--h-on-accent)!important;
  border:0!important;font-weight:600!important;border-radius:7px!important}
#shot:hover,#expbtn:hover,#export-btn:hover{background:var(--h-accent-hover)!important}
#zbar button{border-radius:14px!important;min-width:30px;background:var(--h-popover)!important;
  border-color:var(--h-popover-line)!important}
#toggle,#hp-toggle{background:var(--h-popover)!important;color:var(--h-text)!important;
  border:1px solid var(--h-popover-line)!important}

/* fields */
select{background:var(--h-well)!important;color:var(--h-text)!important;border:1px solid var(--h-line)!important;
  border-radius:6px!important;font:12px %(sans)s!important;padding:5px 8px!important}
select:focus{outline:none;border-color:var(--h-accent)!important}
kbd{background:var(--h-raised)!important;color:var(--h-text)!important;border:1px solid var(--h-popover-line)!important;
  border-bottom-width:2px!important;border-radius:4px!important;font:12px %(mono)s!important}

/* the app's slider: a thin track, gold up to a light round handle */
input[type=range]{-webkit-appearance:none;appearance:none;background:transparent!important;height:18px;
  cursor:pointer;accent-color:var(--h-accent);--fill:50%%}
input[type=range]:focus{outline:none}
.slg input[type=range],.row input[type=range],.crow input[type=range]{min-width:0}
.slg input[type=range]{width:auto}
input[type=range]::-webkit-slider-runnable-track{height:4px;border-radius:2px;
  background:linear-gradient(var(--h-accent),var(--h-accent)) 0/var(--fill) 100%% no-repeat,var(--h-raised)}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;margin-top:-5px;
  border:0;border-radius:50%%;background:var(--h-text);box-shadow:0 1px 3px rgba(0,0,0,.45)}
input[type=range]:hover::-webkit-slider-thumb{transform:scale(1.1)}
input[type=range]:focus-visible::-webkit-slider-thumb{box-shadow:0 0 0 3px var(--h-accent)}
input[type=range]::-moz-range-track{height:4px;border-radius:2px;background:var(--h-raised)}
input[type=range]::-moz-range-progress{height:4px;border-radius:2px;background:var(--h-accent)}
input[type=range]::-moz-range-thumb{width:14px;height:14px;border:0;border-radius:50%%;background:var(--h-text)}
input[type=range]:disabled{opacity:.45;cursor:default}

/* floating labels and notices */
#objname,#titlebar,#badge,.lbl{background:var(--h-popover)!important;color:var(--h-text)!important;
  border:1px solid var(--h-popover-line)!important;border-radius:8px!important;font-family:%(sans)s!important}
#dte-overlay{background:var(--h-popover)!important;border:1px solid var(--h-popover-line)!important;
  border-radius:12px!important;box-shadow:0 12px 32px rgba(0,0,0,.45)}
#dte-overlay h3{color:var(--h-text)!important}
#dte-overlay p{color:var(--h-text2)!important;font-size:12px!important}
#dome,#compass{border-color:var(--h-line)!important}
#hp-tip{background:var(--h-popover)!important;color:var(--h-text)!important;border-color:var(--h-popover-line)!important}
*::-webkit-scrollbar{width:10px;height:10px}
*::-webkit-scrollbar-thumb{background:var(--h-raised);border-radius:5px;border:2px solid transparent;background-clip:padding-box}
*::-webkit-scrollbar-track{background:transparent}
"""

# Keeps each slider's gold fill where its handle is: on input, and for values
# a page's script sets itself (the light sliders follow the dome).
_JS = """<script id="helio-skin-js">(function(){
function fill(r){var a=+r.min||0,b=+(r.max||100),v=+r.value,p=b>a?(v-a)/(b-a)*100:0;
 var s=Math.max(0,Math.min(100,p)).toFixed(1)+'%';if(r.style.getPropertyValue('--fill')!==s)r.style.setProperty('--fill',s);}
function all(){var rs=document.querySelectorAll('input[type=range]');for(var i=0;i<rs.length;i++)fill(rs[i]);}
document.addEventListener('input',function(e){if(e.target&&e.target.type==='range')fill(e.target);},true);
if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',all);else all();
setInterval(all,250);
})();</script>"""


def css(extra: str = "") -> str:
    """The skin's rules, with *extra* page rules after them."""
    return _CSS % dict(sans=SANS, mono=MONO) + extra


def skin(html: str, extra: str = "") -> str:
    """*html* in the Helio look (see the module's note).  The rules go last in
    <head>; *extra* adds rules for one page's layout.  Skinned already: as is."""
    if MARK in html:
        return html
    block = f'<style id="{MARK}">{css(extra)}</style>'
    low = html.lower()
    j = low.find("</head>")
    html = html[:j] + block + "\n" + html[j:] if j >= 0 else block + html
    low = html.lower()
    k = low.rfind("</body>")
    return html[:k] + _JS + html[k:] if k >= 0 else html + _JS
