"""The phone layer every web export carries.  GUI-free.

Without a viewport tag a phone lays a page out 980 px wide and shrinks it,
so a viewer with a fixed side panel puts its picture off the screen.
``phone_ready`` adds the tag (and a charset where a page has none), turns
the page's side panel into a sheet along the bottom that a button opens and
closes, makes the buttons big enough for a finger, and gives the picture
touch gestures:

    one finger    moves the light (or pans, on a page without one)
    two fingers   pinch to zoom, drag to pan
    double tap    resets

A page offers its light and zoom to the gestures through
``window.HelioPhone``: ``getLight() -> [az°, el°]``, ``setLight(az°, el°)``,
``getView() -> [zoom, panX, panY]``, ``setView(zoom, panX, panY)`` and
``reset()``; any of them may be missing.  Desktop browsers see no change:
everything is inside a narrow-screen media query or waits for a touch.
"""
from __future__ import annotations

import re

MARK = "helio-phone"
NARROW = 760                         # px: below this the side panel becomes a sheet

VIEWPORT = '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'

# The RTI viewers (combined, quick view, WebRTI) keep their light and view in
# init()'s closure: these lines, placed inside it, hand them to the gestures.
RTI_HOOKS = (
    "window.HelioPhone={"
    "getLight:function(){return [Math.atan2(lx,ly)*180/Math.PI,"
    "Math.asin(Math.min(1,Math.abs(lz)))*180/Math.PI];},"
    "setLight:function(a,e){setLight(a*Math.PI/180,e*Math.PI/180);},"
    "getView:function(){return [zoom,panX,panY];},"
    "setView:function(z,x,y){zoom=Math.max(.1,Math.min(20,z));panX=x;panY=y;applyT();},"
    "reset:resetView};\n")

_CSS = """
#hp-toggle,#hp-tip{display:none}
@media (max-width:%(narrow)dpx){
  :root{--hp-sheet:min(46vh,430px)}
  body.hp-closed{--hp-sheet:0px}
  %(panel)s{position:fixed!important;top:auto!important;left:0!important;right:0!important;
    bottom:0!important;width:auto!important;height:var(--hp-sheet)!important;max-height:none!important;
    border-left:0!important;border-top:1px solid rgba(140,160,180,.35)!important;
    border-radius:14px 14px 0 0;transform:none!important;z-index:200!important;
    overflow-y:auto!important;-webkit-overflow-scrolling:touch;
    padding-bottom:calc(14px + env(safe-area-inset-bottom))!important}
  body.hp-closed %(panel)s{visibility:hidden}
  %(stage)s
  %(hide)s{display:none!important}
  button,select,.mbtn,.hbtn{min-height:40px;font-size:14px!important}
  input[type=range]{min-height:32px}
  #hp-toggle{display:block;position:fixed;right:12px;z-index:210;
    bottom:calc(var(--hp-sheet) + 12px + env(safe-area-inset-bottom));
    min-height:40px;padding:8px 14px;border-radius:20px;font:600 14px system-ui,sans-serif;
    background:rgba(10,16,24,.88);color:#e8eef5;border:1px solid rgba(140,160,180,.45)}
  #hp-tip{display:block;position:fixed;left:50%%;top:calc(12px + env(safe-area-inset-top));
    transform:translateX(-50%%);z-index:220;max-width:92vw;padding:8px 14px;border-radius:18px;
    font:13px/1.35 system-ui,sans-serif;text-align:center;background:rgba(10,16,24,.9);
    color:#e8eef5;border:1px solid rgba(140,160,180,.45);pointer-events:none;
    transition:opacity .6s}
  #hp-tip.gone{opacity:0}
  %(css)s
}
"""

_JS = r"""
(function(){
var NARROW=%(narrow)d, STAGE=%(stage)s, PANEL=%(panel)s;
function narrow(){return window.innerWidth<=NARROW;}
if(PANEL){
  var b=document.createElement('button');b.id='hp-toggle';b.type='button';
  function label(){b.textContent=document.body.classList.contains('hp-closed')?'Controls ▴':'Controls ▾';}
  b.onclick=function(){document.body.classList.toggle('hp-closed');label();
    setTimeout(function(){window.dispatchEvent(new Event('resize'));},30);};
  document.body.appendChild(b);
  if(narrow()){document.body.classList.add('hp-closed');window.dispatchEvent(new Event('resize'));}
  label();
}
var tip=%(tip)s;
if(tip&&narrow()&&('ontouchstart' in window||navigator.maxTouchPoints>0)){
  var t=document.createElement('div');t.id='hp-tip';t.textContent=tip;document.body.appendChild(t);
  setTimeout(function(){t.classList.add('gone');},4500);
  setTimeout(function(){t.remove();},5200);
}
var el=STAGE&&document.querySelector(STAGE);
if(!el)return;
el.style.touchAction='none';
var one=null,two=null,lastTap=0,moved=false;
function H(){return window.HelioPhone||{};}
function mid(t){return [(t[0].clientX+t[1].clientX)/2,(t[0].clientY+t[1].clientY)/2];}
function dist(t){return Math.hypot(t[0].clientX-t[1].clientX,t[0].clientY-t[1].clientY);}
el.addEventListener('touchstart',function(e){
  var h=H(),t=e.touches;
  if(t.length===1){
    one={x:t[0].clientX,y:t[0].clientY,light:h.getLight?h.getLight():null,view:h.getView?h.getView():null};
    moved=false;
  }else if(t.length===2&&h.getView){
    one=null;var m=mid(t);two={d:dist(t),m:m,view:h.getView()};
  }
  e.preventDefault();
},{passive:false});
el.addEventListener('touchmove',function(e){
  var h=H(),t=e.touches;
  if(two&&t.length===2&&h.setView){
    var m=mid(t),v=two.view,z=v[0]*dist(t)/Math.max(1,two.d);
    h.setView(z,v[1]+(m[0]-two.m[0]),v[2]+(m[1]-two.m[1]));
  }else if(one&&t.length===1){
    var dx=t[0].clientX-one.x,dy=t[0].clientY-one.y;
    if(Math.abs(dx)+Math.abs(dy)>6)moved=true;
    if(one.light&&h.setLight){
      var az=((one.light[0]+dx*0.6)%%360+360)%%360,elv=Math.max(5,Math.min(88,one.light[1]-dy*0.3));
      h.setLight(az,elv);
    }else if(one.view&&h.setView){
      h.setView(one.view[0],one.view[1]+dx,one.view[2]+dy);
    }
  }
  e.preventDefault();
},{passive:false});
el.addEventListener('touchend',function(e){
  if(e.touches.length===0){
    if(one&&!moved){
      var now=Date.now();
      if(now-lastTap<320){if(H().reset)H().reset();lastTap=0;}else lastTap=now;
    }
    one=null;two=null;
  }else if(e.touches.length===1){two=null;one=null;}
},{passive:true});
})();
"""


def phone_ready(html: str, *, panel: str = "", stage: str = "", hide=(), css: str = "",
                tip: str = "", hooks: str = "", hooks_before: str = "",
                skin: bool = True, skin_css: str = "") -> str:
    """*html* made phone-ready (see the module's note).  *panel* is the side
    panel's selector, *stage* the picture's (the element the gestures go
    to), *hide* the selectors with nothing to say on a phone (key and mouse
    hints), *css* extra narrow-screen rules, *tip* the line shown once on a
    phone.  *hooks* is page script placed just before the last
    *hooks_before*, so it runs inside the page's own closure.  *skin* gives
    the page the Helio look (viewer_skin), *skin_css* its own layout rules
    for it; a page drawn in the Helio tokens already passes skin=False.  A
    page made ready already is returned as it is."""
    if MARK in html:
        return html
    if skin:
        from helio.application.viewer_skin import skin as _skin
        html = _skin(html, skin_css)
    if hooks and hooks_before:
        i = html.rfind(hooks_before)
        if i < 0:
            raise ValueError(f"phone_ready: {hooks_before!r} is not in the page")
        html = html[:i] + hooks + html[i:]
    head = []
    if not re.search(r"<meta[^>]+charset", html[:2000], re.I):
        head.append('<meta charset="UTF-8">')
    if not re.search(r"<meta[^>]+name=[\"']viewport", html, re.I):
        head.append(VIEWPORT)
    head.append('<meta name="theme-color" content="#1B1C1F">')
    head.append('<meta name="apple-mobile-web-app-capable" content="yes">')
    rules = _CSS % dict(narrow=NARROW, panel=panel or f"#{MARK}-none", stage=css_stage(stage),
                        hide=", ".join(hide) or f"#{MARK}-none", css=css)
    head.append(f'<style id="{MARK}">{rules}</style>')
    m = re.search(r"<head[^>]*>", html, re.I)
    if m:
        html = html[:m.end()] + "\n" + "\n".join(head) + "\n" + html[m.end():]
    else:
        html = "<head>" + "\n".join(head) + "</head>" + html
    import json
    script = "<script>" + _JS % dict(narrow=NARROW, stage=json.dumps(stage),
                                     panel=json.dumps(panel), tip=json.dumps(tip)) + "</script>"
    j = html.lower().rfind("</body>")
    return html[:j] + script + html[j:] if j >= 0 else html + script


def css_stage(stage: str) -> str:
    """The picture fills the screen above the sheet.  No width: a canvas
    given width:auto falls back to the size of its drawing buffer."""
    if not stage:
        return ""
    return f"{stage}{{left:0!important;right:0!important;bottom:var(--hp-sheet)!important}}"


# The RTI viewers' panel, a little wider for 12 px words (desktop only: the
# phone sheet's own rules win below NARROW px).
RTI_LAYOUT = (
    "@media (min-width:%dpx){#bar{width:252px!important;padding:14px 14px!important}"
    "#wrap,#zbar{right:252px!important}#hint{right:262px!important}}"
    "#bar .slg label{width:68px!important}#bar .slg .val{width:40px!important}"
    "#bar .mbtn{padding:6px 9px!important;margin:3px 0!important;text-align:left}"
    "#bar h3{margin:4px 0 8px!important}#bar .sep{margin:10px 0!important}"
    "#dome{border-radius:50%%}" % (NARROW + 1))


def rti_viewer(html: str, hooks_before: str) -> str:
    """The RTI viewers (combined, quick view, WebRTI): a 220 px panel on the
    right (#bar) beside the picture (#wrap), with zoom buttons and a hint."""
    return phone_ready(
        html, panel="#bar", stage="#wrap", hide=("#hint",), hooks=RTI_HOOKS,
        hooks_before=hooks_before, skin_css=RTI_LAYOUT,
        css="#zbar{right:0!important;bottom:calc(var(--hp-sheet) + 62px)!important}"
            "#bar .mbtn{display:inline-block;width:calc(50% - 3px);vertical-align:top}"
            "#titlebar{max-width:60vw}#cv{width:100%!important;height:100%!important;object-fit:contain}",
        tip="Drag to move the light \u00b7 pinch to zoom \u00b7 double-tap to reset")
