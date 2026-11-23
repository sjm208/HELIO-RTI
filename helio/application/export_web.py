"""Web viewers and sharing, GUI-free (WEB-001…007, HER-001, HER-005).

The classic exports' writing, moved so both apps write the same files: the
WebRTI PTM viewer and its page, the relight / OpenLIME package, the IIIF 3D
manifest, the A/B comparison page, the combined RTI viewer with its e-mail
size budget, the engine sweep, the HSH DeepZoom relight viewer, the embed
snippet and the labelled QR card.  Renders come in as arrays or a
``draw(engine)`` callable; the classic methods keep their dialogs.

The IIIF manifest placed each light at (x, z, -y): behind a model that HELIO
writes facing +Z.  Lights are now (x, y, z) in the model's own frame
(KD-036).  The provenance note gave the screen-frame azimuth formula for
+Y-up light vectors (KD-035).
"""
from __future__ import annotations

from helio.core.frame_read import read_frame

import base64
import io
import json
import os
import time
import types
from typing import Callable

import cv2
import numpy as np
from PIL import Image

from helio.application.exports import NothingToExport


# ── a provenance record from plain values ────────────────────────────────

def app_like(*, name: str, images, lights, light_kind: str, h: int, w: int,
             normal_engine: str = "", depth_engine: str = "", engine: str = "",
             engine_settings: dict | None = None, version: str = "") -> object:
    """What helio.io.provenance.build_provenance reads from an app, as a
    plain object: so the new interface writes the same record."""
    data = types.SimpleNamespace(
        plate_paths=list(images or []), current_L_mat=lights, h=h, w=w,
        normal_engine=normal_engine, depth_engine=depth_engine, render_engine=engine,
        engine_settings=engine_settings or {})
    return types.SimpleNamespace(
        data=data, object_id_var=name or "", crop_coords=[],
        loaded_lp_matrix=lights if light_kind in ("file", "measured", "sphere", "dome") else None,
        **({"APP_VERSION": version} if version else {}))


def write_webrti_html(folder_path, w, h):
    """WebRTI viewer — clean absolute-position mouse relight, no sensitivity complexity."""
    import time as _t
    ver = str(int(_t.time()))

    # Cap canvas to 4096px on longest side — GPU max renderbuffer ~16MP
    # IW/IH are the capture dims; DW/DH are the capped display dims
    html = (
        "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>"
        "<title>Helio WebRTI</title><style>"
        "*{margin:0;padding:0;box-sizing:border-box}"
        "html,body{width:100%;height:100%;background:#080c12;overflow:hidden;font-family:Arial,sans-serif;color:#ddd}"
        "#wrap{position:fixed;top:0;left:0;right:220px;bottom:0;display:flex;align-items:center;justify-content:center;overflow:hidden}"
        "#cv{display:block;max-width:100%;max-height:100%;width:auto;height:auto;transform-origin:center center;cursor:crosshair}"
        "#bar{position:fixed;top:0;right:0;bottom:0;width:220px;background:rgba(6,14,22,.96);border-left:1px solid #1a4a2a;overflow-y:auto;padding:10px 12px;z-index:100}"
        "h3{color:#4ade80;font-size:12px;margin-bottom:6px}"
        "#dome{cursor:crosshair;border-radius:50%;border:1px solid #1a4a2a;display:block;margin:0 auto}"
        "#readout{text-align:center;color:#f9e547;font-size:10px;margin:2px 0 5px}"
        ".mbtn{display:block;width:100%;padding:4px 7px;margin:2px 0;background:#0d1a12;color:#8a9;border:1px solid #1a4a2a;border-radius:3px;cursor:pointer;font-size:10px;text-align:left}"
        ".mbtn.on,.mbtn:hover{background:#1a4a2a;color:#4ade80}"
        ".sep{border-top:1px solid #1a3a2a;margin:6px 0}"
        ".slg{display:flex;align-items:center;gap:4px;margin:3px 0;font-size:10px;color:#8899aa}"
        ".slg label{width:60px;flex-shrink:0}.slg input{flex:1;accent-color:#4ade80}.slg .val{width:30px;text-align:right;color:#4ade80;font-weight:bold}"
        "#zbar{position:fixed;bottom:8px;left:0;right:220px;display:flex;align-items:center;justify-content:center;gap:6px;z-index:50;pointer-events:none}"
        "#zbar button{background:rgba(6,14,22,.85);border:1px solid #1a4a2a;color:#4ade80;font-size:14px;padding:2px 9px;border-radius:12px;cursor:pointer;pointer-events:all}"
        "#zlbl{color:#4ade80;font-size:10px;min-width:40px;text-align:center;background:rgba(6,14,22,.7);padding:1px 5px;border-radius:7px}"
        "#hint{position:fixed;bottom:36px;right:228px;font-size:9px;color:#334;pointer-events:none}"
        "</style></head><body>"
        "<div id='wrap'><canvas id='cv'></canvas></div>"
        "<div id='bar'>"
        "<h3>WebRTI Viewer</h3>"
        "<canvas id='dome' width='100' height='100'></canvas>"
        "<div id='readout'>Az: 315&deg; | El: 45&deg;</div>"
        "<div class='sep'></div>"
        "<button class='mbtn on' id='m0' onclick='setMode(0)'>Colour Relight</button>"
        "<button class='mbtn' id='m1' onclick='setMode(1)'>Greyscale RTI</button>"
        "<button class='mbtn' id='m2' onclick='setMode(2)'>Specular Enhance</button>"
        "<button class='mbtn' id='m3' onclick='setMode(3)'>Normal Map</button>"
        "<button class='mbtn' id='m4' onclick='setMode(4)'>Normals USM</button>"
        "<button class='mbtn' id='m5' onclick='setMode(5)'>Surface Gradient</button>"
        "<button class='mbtn' id='m6' onclick='setMode(6)'>Col &times; Specular</button>"
        "<div class='sep'></div>"
        "<div class='slg'><label>Bright</label><input type='range' id='sBr' min='0.2' max='3' step='0.05' value='1' oninput='upd()'><span class='val' id='vBr'>1.00</span></div>"
        "<div class='slg'><label>Contrast</label><input type='range' id='sCo' min='0.2' max='3' step='0.05' value='1' oninput='upd()'><span class='val' id='vCo'>1.00</span></div>"
        "<div class='slg'><label>Gamma</label><input type='range' id='sGa' min='0.3' max='3' step='0.05' value='1' oninput='upd()'><span class='val' id='vGa'>1.00</span></div>"
        "<div class='slg'><label>Light Elev</label><input type='range' id='sEl' min='5' max='85' step='5' value='45' oninput='setElev(this.value)'><span class='val' id='vEl'>45&deg;</span></div>"
        "<div class='slg'><label>Light Az</label><input type='range' id='sAz' min='0' max='360' step='5' value='315' oninput='setAz(this.value)'><span class='val' id='vAz'>315&deg;</span></div>"
        "</div>"
        "<div id='zbar'>"
        "<button onclick='doZoom(-1)'>&#8722;</button>"
        "<span id='zlbl'>100%</span>"
        "<button onclick='doZoom(1)'>&#43;</button>"
        "<button onclick='resetView()' style='font-size:10px'>Reset</button>"
        "</div>"
        "<div id='hint'>Move mouse over image to relight &middot; Scroll = zoom &middot; Space+drag = pan</div>"
        "<script>\n"
        "var IW=__W__,IH=__H__;\n"
        "var MAX_CV=4096,_cs=Math.min(1,MAX_CV/Math.max(IW,IH));\n"
        "var DW=Math.floor(IW*_cs),DH=Math.floor(IH*_cs);\n"
        "var cv=document.getElementById('cv');\n"
        "cv.width=DW; cv.height=DH;\n"
        "var gl=cv.getContext('webgl')||cv.getContext('experimental-webgl');\n"
        "if(!gl){document.getElementById('wrap').innerHTML='<p style=\"color:#FF8F6E;padding:40px;font:14px system-ui,sans-serif\">WebGL not available</p>';}\n"
        "else{ gl.getExtension('OES_standard_derivatives'); init(); }\n"
        "function init(){\n"
        "var VS='attribute vec2 a;varying vec2 uv;void main(){uv=a*.5+.5;uv.y=1.-uv.y;gl_Position=vec4(a,0,1);}';\n"
        "var FS='#extension GL_OES_standard_derivatives : enable\\nprecision highp float;\\n'\n"
        "+'varying vec2 uv;uniform sampler2D uA,uN,uR,uS;uniform vec3 uL;uniform float uM,uBr,uCo,uGa;\\n'\n"
        "+'vec3 fc(float t){t=clamp(t,0.,1.);if(t<.25)return mix(vec3(0,0,.5),vec3(0,.8,.8),t*4.);if(t<.5)return mix(vec3(0,.8,.8),vec3(0,.8,0),(t-.25)*4.);if(t<.75)return mix(vec3(0,.8,0),vec3(1,1,0),(t-.5)*4.);return mix(vec3(1,1,0),vec3(1,0,0),(t-.75)*4.);}\\n'\n"
        "+'void main(){\\n'\n"
        "+'  if(uv.x<0.||uv.x>1.||uv.y<0.||uv.y>1.){gl_FragColor=vec4(.06,.08,.12,1.);return;}\\n'\n"
        "+'  vec3 n=normalize(texture2D(uN,uv).rgb*2.-1.);\\n'\n"
        "+'  vec3 alb=texture2D(uA,uv).rgb;\\n'\n"
        "+'  float rgh=texture2D(uR,uv).r,spc=texture2D(uS,uv).r;\\n'\n"
        "+'  vec3 L=normalize(uL),H=normalize(L+vec3(0,0,1));\\n'\n"
        "+'  float NL=max(dot(n,L),0.),NH=max(dot(n,H),0.);\\n'\n"
        "+'  float sp=spc*pow(NH,mix(128.,2.,rgh));\\n'\n"
        "+'  float lum=dot(alb,vec3(.299,.587,.114));\\n'\n"
        "+'  vec3 col;\\n'\n"
        "+'  if(uM<.5){col=alb*(NL*.88+.12)+vec3(sp*.4);}\\n'\n"
        "+'  else if(uM<1.5){col=vec3(lum*(NL*.88+.12));}\\n'\n"
        "+'  else if(uM<2.5){col=vec3(lum*.2+sp*2.+NL*.5);}\\n'\n"
        "+'  else if(uM<3.5){col=n*.5+.5;}\\n'\n"
        "+'  else if(uM<4.5){\\n'\n"
        "+'    vec2 dx=dFdx(uv)*3.,dy=dFdy(uv)*3.;\\n'\n"
        "+'    vec3 nb=normalize((texture2D(uN,uv+dx)+texture2D(uN,uv-dx)+texture2D(uN,uv+dy)+texture2D(uN,uv-dy)+texture2D(uN,uv)*4.).rgb*.125)*2.-1.;\\n'\n"
        "+'    col=vec3(max(dot(normalize(n+(n-nb)*2.5),L),0.));}\\n'\n"
        "+'  else if(uM<5.5){float slp=length(vec2(n.x,n.y));float az=atan(n.y,n.x)/6.28318+.5;col=fc(az)*(slp*.9+.1);col=mix(col,vec3(NL*.8+.1),.25);float cx=dFdx(n.z),cy=dFdy(n.z);col=clamp(col*(1.+clamp(sqrt(cx*cx+cy*cy)*60.,0.,.6)),0.,1.);}\\n'\n"
        "+'  else{col=alb*(NL*.9+.1)+vec3(sp*2.)*mix(vec3(1.),alb,.3);}\\n'\n"
        "+'  col=clamp(col*uBr,0.,1.);col=clamp((col-.5)*uCo+.5,0.,1.);\\n'\n"
        "+'  col=pow(col,vec3(1./max(uGa,.01)));gl_FragColor=vec4(col,1.);}\\n';\n"
        "function mkSh(t,s){var sh=gl.createShader(t);gl.shaderSource(sh,s);gl.compileShader(sh);\n"
        "  if(!gl.getShaderParameter(sh,gl.COMPILE_STATUS))console.error(gl.getShaderInfoLog(sh));return sh;}\n"
        "var pg=gl.createProgram();\n"
        "gl.attachShader(pg,mkSh(gl.VERTEX_SHADER,VS));\n"
        "gl.attachShader(pg,mkSh(gl.FRAGMENT_SHADER,FS));\n"
        "gl.linkProgram(pg);\n"
        "if(!gl.getProgramParameter(pg,gl.LINK_STATUS)){console.error(gl.getProgramInfoLog(pg));return;}\n"
        "var buf=gl.createBuffer();\n"
        "gl.bindBuffer(gl.ARRAY_BUFFER,buf);\n"
        "gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,1,1]),gl.STATIC_DRAW);\n"
        "var aL=gl.getAttribLocation(pg,'a');\n"
        "gl.enableVertexAttribArray(aL); gl.vertexAttribPointer(aL,2,gl.FLOAT,false,0,0);\n"
        "var uL_=gl.getUniformLocation(pg,'uL'),uM_=gl.getUniformLocation(pg,'uM');\n"
        "var uBr_=gl.getUniformLocation(pg,'uBr'),uCo_=gl.getUniformLocation(pg,'uCo'),uGa_=gl.getUniformLocation(pg,'uGa');\n"
        "function mkTex(src,unit,name){\n"
        "  var tex=gl.createTexture();\n"
        "  gl.activeTexture(gl.TEXTURE0+unit); gl.bindTexture(gl.TEXTURE_2D,tex);\n"
        "  gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,1,1,0,gl.RGBA,gl.UNSIGNED_BYTE,new Uint8Array([100,100,100,255]));\n"
        "  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);\n"
        "  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);\n"
        "  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);\n"
        "  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);\n"
        "  gl.useProgram(pg); gl.uniform1i(gl.getUniformLocation(pg,name),unit);\n"
        "  var img=new Image();\n"
        "  img.onload=function(){\n"
        "    gl.activeTexture(gl.TEXTURE0+unit); gl.bindTexture(gl.TEXTURE_2D,tex);\n"
        "    gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,img);\n"
        "    rdr();};\n"
        "  img.onerror=function(){console.warn('Failed:',src); rdr();};\n"
        "  img.src=src;}\n"
        "mkTex('albedo.jpg',0,'uA'); mkTex('normals.png',1,'uN');\n"
        "mkTex('roughness.png',2,'uR'); mkTex('specular.png',3,'uS');\n"
        "var lx=0,ly=0,lz=1;\n"
        "var mode=0,zoom=1,panX=0,panY=0;\n"
        "var spaceDown=false,panDrag=false,dX=0,dY=0,pX0=0,pY0=0;\n"
        "// ── Dome ──────────────────────────────────────────────────────\n"
        "var dome=document.getElementById('dome');\n"
        "var dc=dome.getContext('2d'),DR=dome.width/2;\n"
        "var dDrag=false;\n"
        "function drawDome(dlx,dly){\n"
        "  dc.clearRect(0,0,dome.width,dome.height);\n"
        "  var g=dc.createRadialGradient(DR,DR,0,DR,DR,DR);\n"
        "  g.addColorStop(0,'#1A1B1E');g.addColorStop(1,'#1A1B1E');\n"
        "  dc.fillStyle=g;dc.beginPath();dc.arc(DR,DR,DR,0,Math.PI*2);dc.fill();\n"
        "  dc.strokeStyle='#3E4046';dc.lineWidth=1;\n"
        "  [.33,.67,1].forEach(function(r){dc.beginPath();dc.arc(DR,DR,r*DR,0,Math.PI*2);dc.stroke();});\n"
        "  for(var i=0;i<8;i++){var a=i*Math.PI/4;dc.beginPath();dc.moveTo(DR,DR);dc.lineTo(DR+Math.cos(a)*DR,DR+Math.sin(a)*DR);dc.stroke();}\n"
        "  var m=Math.sqrt(dlx*dlx+dly*dly),sc=m>0.95?0.95/m:1;\n"
        "  var px=DR+dlx*DR*sc,py=DR-dly*DR*sc;\n"
        "  dc.beginPath();dc.arc(px,py,7,0,Math.PI*2);\n"
        "  dc.fillStyle='#E6A93E';dc.fill();dc.strokeStyle='#222326';dc.lineWidth=2;dc.stroke();}\n"
        "function setLight(az,el){\n"
        "  var ce=Math.cos(el),se=Math.sin(el);\n"
        "  lx=ce*Math.sin(az);ly=ce*Math.cos(az);lz=se;\n"
        "  var len=Math.sqrt(lx*lx+ly*ly+lz*lz);if(len>0){lx/=len;ly/=len;lz/=len;}\n"
        "  var azD=Math.round(((az*180/Math.PI)%360+360)%360);\n"
        "  var elD=Math.round(el*180/Math.PI);\n"
        "  document.getElementById('readout').textContent='Az: '+azD+'\\u00b0 | El: '+elD+'\\u00b0';\n"
        "  var s=document.getElementById('sEl');if(s)s.value=elD;\n"
        "  var v=document.getElementById('vEl');if(v)v.textContent=elD+'\\u00b0';\n"
        "  var s2=document.getElementById('sAz');if(s2)s2.value=azD;\n"
        "  var v2=document.getElementById('vAz');if(v2)v2.textContent=azD+'\\u00b0';\n"
        "  drawDome(lx,ly);\n"
        "  rdr();}\n"
        "function domeXY(px,py){\n"
        "  var dlx=(px-DR)/DR,dly=-(py-DR)/DR;\n"
        "  var r=Math.min(1,Math.sqrt(dlx*dlx+dly*dly));\n"
        "  var el=Math.max(3*Math.PI/180,Math.min(87*Math.PI/180,Math.acos(r)));\n"
        "  var az=Math.atan2(dlx,dly);\n"
        "  setLight(az,el);}\n"
        "dome.addEventListener('mousedown',function(e){e.preventDefault();e.stopPropagation();dDrag=true;domeXY(e.offsetX,e.offsetY);});\n"
        "window.addEventListener('mousemove',function(e){if(!dDrag)return;var r=dome.getBoundingClientRect();domeXY(e.clientX-r.left,e.clientY-r.top);});\n"
        "window.addEventListener('mouseup',function(){dDrag=false;});\n"
        "dome.addEventListener('touchstart',function(e){e.preventDefault();e.stopPropagation();dDrag=true;var r=dome.getBoundingClientRect(),t=e.touches[0];domeXY(t.clientX-r.left,t.clientY-r.top);},{passive:false});\n"
        "dome.addEventListener('touchmove',function(e){e.preventDefault();var r=dome.getBoundingClientRect(),t=e.touches[0];domeXY(t.clientX-r.left,t.clientY-r.top);},{passive:false});\n"
        "dome.addEventListener('touchend',function(){dDrag=false;});\n"
        "// ── Zoom / Pan ────────────────────────────────────────────────\n"
        "function setZL(){document.getElementById('zlbl').textContent=Math.round(zoom*100)+'%';}\n"
        "function applyT(){setZL();cv.style.transform='scale('+zoom+') translate('+panX/zoom+'px,'+panY/zoom+'px)';}\n"
        "document.addEventListener('keydown',function(e){if(e.code==='Space'){spaceDown=true;e.preventDefault();}});\n"
        "document.addEventListener('keyup',function(e){if(e.code==='Space'){spaceDown=false;panDrag=false;}});\n"
        "document.getElementById('wrap').addEventListener('wheel',function(e){e.preventDefault();zoom=Math.max(.1,Math.min(20,zoom*(e.deltaY<0?1.12:1/1.12)));applyT();},{passive:false});\n"
        "cv.addEventListener('mousedown',function(e){if(dDrag)return;if(e.button===1||spaceDown){panDrag=true;dX=e.clientX;dY=e.clientY;pX0=panX;pY0=panY;e.preventDefault();}});\n"
        "window.addEventListener('mousemove',function(e){if(panDrag){panX=pX0+(e.clientX-dX);panY=pY0+(e.clientY-dY);applyT();}});\n"
        "window.addEventListener('mouseup',function(e){if(e.button===1||spaceDown)panDrag=false;});\n"
        "cv.addEventListener('dblclick',resetView);\n"
        "// ── Render ────────────────────────────────────────────────────\n"
        "function rdr(){\n"
        "  gl.viewport(0,0,DW,DH); gl.useProgram(pg);\n"
        "  gl.uniform3f(uL_,lx,ly,lz); gl.uniform1f(uM_,mode);\n"
        "  gl.uniform1f(uBr_,parseFloat(document.getElementById('sBr').value));\n"
        "  gl.uniform1f(uCo_,parseFloat(document.getElementById('sCo').value));\n"
        "  gl.uniform1f(uGa_,parseFloat(document.getElementById('sGa').value));\n"
        "  gl.drawArrays(gl.TRIANGLE_STRIP,0,4);}\n"
        "function upd(){\n"
        "  document.getElementById('vBr').textContent=parseFloat(document.getElementById('sBr').value).toFixed(2);\n"
        "  document.getElementById('vCo').textContent=parseFloat(document.getElementById('sCo').value).toFixed(2);\n"
        "  document.getElementById('vGa').textContent=parseFloat(document.getElementById('sGa').value).toFixed(2);\n"
        "  rdr();}\n"
        "function setMode(m){mode=m;for(var i=0;i<7;i++){var b=document.getElementById('m'+i);if(b)b.className='mbtn'+(i===m?' on':'');}rdr();}\n"
        "function setElev(v){setLight(curAz(),parseFloat(v)*Math.PI/180);}\n"
        "function setAz(v){setLight(parseFloat(v)*Math.PI/180,curEl());}\n"
        "function curAz(){return Math.atan2(lx,ly);}\n"
        "function curEl(){return Math.asin(Math.min(1,Math.abs(lz)));}\n"
        "function doZoom(d){zoom=Math.max(.1,Math.min(20,zoom*Math.pow(1.25,d)));applyT();}\n"
        "function resetView(){zoom=1;panX=0;panY=0;setLight(315*Math.PI/180,45*Math.PI/180);applyT();}\n"
        "window.setMode=setMode;window.setElev=setElev;window.setAz=setAz;\n"
        "window.doZoom=doZoom;window.resetView=resetView;window.upd=upd;\n"
        "// ── Init ──────────────────────────────────────────────────────\n"
        "setLight(315*Math.PI/180,45*Math.PI/180);\n"
        "}  // end init()\n"
        "</script></body></html>"
    ).replace('__W__', str(w)).replace('__H__', str(h))
    from helio.application.phone_web import rti_viewer
    html = rti_viewer(html, "}  // end init()")

    with open(os.path.join(folder_path, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)


# ── WebRTI (PTM) ─────────────────────────────────────────────────────────

def webrti(D, out_dir: str, hybrid_rgb=None, *, obj_id: str = "", mm_per_px=None,
           z_mm_per_unit=None, normal_engine: str = "", depth_engine: str = "",
           progress: Callable | None = None) -> list[str]:
    """The six PTM planes as JPEG, albedo, normals, roughness, specular, a
    depth heatmap, a hybrid render, info.json and the viewer page."""
    from helio.application.export_images import sharable_launchers
    if getattr(D, "ptm_deferred", False):
        D.ensure_ptm_hsh(progress)
    if getattr(D, "ptm_coeffs", None) is None or getattr(D, "normals_full", None) is None:
        raise NothingToExport("Process images first.")
    upd = progress or (lambda *_a: None)
    os.makedirs(out_dir, exist_ok=True)
    h, w = D.h, D.w
    files = []

    def put(name, arr, params=None):
        p = os.path.join(out_dir, name)
        cv2.imwrite(p, arr, params) if params else cv2.imwrite(p, arr)
        files.append(p)
    upd("Exporting PTM coefficients (6 channels)…")
    for i in range(6):
        put(f"ptm_c{i}.jpg", cv2.normalize(D.ptm_coeffs[i].reshape(h, w), None, 0, 255,
                                           cv2.NORM_MINMAX).astype(np.uint8),
            [int(cv2.IMWRITE_JPEG_QUALITY), 97])
    upd("Saving albedo…")
    put("albedo.jpg", cv2.cvtColor(D.albedo_rgb_full, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), 97])
    upd("Saving normal map…")
    put("normals.png", cv2.cvtColor((((D.normals_full + 1.0) / 2.0) * 255).astype(np.uint8),
                                    cv2.COLOR_RGB2BGR))
    upd("Saving roughness and specular…")
    r = getattr(D, "roughness_full", None)
    if r is not None:
        r8 = (np.clip(r, 0, 1) * 255).astype(np.uint8) if r.dtype != np.uint8 else r
    else:
        r8 = np.full((h, w), 128, np.uint8)
    put("roughness.png", r8)
    spec = D.spec_var if getattr(D, "spec_var", None) is not None else np.zeros((h, w), np.float32)
    put("specular.png", cv2.normalize(spec, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    upd("Saving depth heatmap…")
    if getattr(D, "depth_vis_p", None) is not None:
        put("depth.png", cv2.applyColorMap((np.clip(D.depth_vis_p, 0, 1) * 255).astype(np.uint8),
                                           cv2.COLORMAP_TURBO))
    else:
        put("depth.png", np.zeros((h, w, 3), np.uint8))
    upd("Computing hybrid render (proxy)…")
    hyb = hybrid_rgb if hybrid_rgb is not None else D.albedo_rgb_full
    hh, hw = hyb.shape[:2]
    if max(hh, hw) > 4096:
        sc = 4096 / max(hh, hw)
        hyb = cv2.resize(hyb, (int(hw * sc), int(hh * sc)), cv2.INTER_LANCZOS4)
    put("hybrid.jpg", cv2.cvtColor(hyb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    upd("Writing info.json…")
    mm, zmm = mm_per_px, z_mm_per_unit
    calib = {
        "xy_resolution_mm_per_px": float(mm) if mm else None,
        "xy_resolution_ppi": float(25.4 / mm) if mm else None,
        "z_mm_per_unit": float(zmm) if zmm else None,
        "calibration_note": (f"XY: {float(mm) * 1000:.2f} µm/px, Z: {float(zmm):.6f} mm per pixel of height"
                             if (mm and zmm) else
                             f"XY: {float(mm) * 1000:.2f} µm/px" if mm else "uncalibrated"),
        "helio_normal_engine": normal_engine, "helio_depth_engine": depth_engine,
        "helio_object_id": obj_id}
    ip = os.path.join(out_dir, "info.json")
    with open(ip, "w") as j:
        json.dump({"format": "ptm", "width": w, "height": h,
                   "layers": [f"ptm_c{i}.jpg" for i in range(6)],
                   "albedo": "albedo.jpg", "normals": "normals.png", "calibration": calib}, j, indent=4)
    files.append(ip)
    upd("Writing HTML viewer…")
    write_webrti_html(out_dir, w, h)
    files.append(os.path.join(out_dir, "index.html"))
    return files + sharable_launchers(out_dir)


# ── relight / OpenLIME ───────────────────────────────────────────────────

def source_frame(path: str, region=None):
    """One photograph as Relight reads it: 8-bit sRGB RGB, cut to *region*
    (fractions x0, y0, x1, y1, pipeline.region_fraction).  None if unreadable."""
    from helio.io.colour_space import to_srgb
    img = read_frame(path, cv2.IMREAD_COLOR, sharp=True)     # a RAW decoded whole, not half
    if img is None:
        return None
    if region is not None:
        ih, iw = img.shape[:2]
        img = img[int(round(region[1] * ih)):int(round(region[3] * ih)),
                  int(round(region[0] * iw)):int(round(region[2] * iw))]
    rgb = to_srgb(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), path)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb)

def relight_package(out_dir: str, ptm_coeffs, albedo, lights, h: int, w: int, images=(),
                    provenance_app=None, region=None) -> dict:
    """relight_rti/ (info.json + coefficient planes, what RelightLab and the
    relight.js / OpenLIME viewers read) and relight_source/ (the frames as
    .jpg with a lights.lp, ready to re-fit), plus the provenance record.
    *region* (fractions, pipeline.region_fraction): a region solve's frames
    are cut to it, so they match the coefficients."""
    from helio.io.relight_export import RELIGHT_IMAGE_EXT, export_relight_package
    if ptm_coeffs is None:
        raise NothingToExport("Process a capture first — this needs PTM coefficients.")
    export_relight_package(os.path.join(out_dir, "relight_rti"), ptm_coeffs, albedo, lights,
                           int(h), int(w))
    made = 0
    paths = list(images or [])
    if paths and lights is not None and len(lights) == len(paths):
        src_dir = os.path.join(out_dir, "relight_source")
        os.makedirs(src_dir, exist_ok=True)
        rows = []
        for k, p_in in enumerate(paths):
            img = source_frame(p_in, region)
            if img is None:
                continue
            nm = os.path.splitext(os.path.basename(p_in))[0] + RELIGHT_IMAGE_EXT
            cv2.imwrite(os.path.join(src_dir, nm), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            rows.append((nm, lights[k]))           # each frame keeps its own light
            made += 1
        if rows:
            with open(os.path.join(src_dir, "lights.lp"), "w") as f:
                f.write(f"{len(rows)}\n")
                for nm, v in rows:
                    n = float(np.linalg.norm(v)) or 1.0
                    f.write(f"{nm} {v[0] / n:.6f} {v[1] / n:.6f} {max(0.001, v[2] / n):.6f}\n")
    if provenance_app is not None:
        from helio.io.provenance import write_provenance
        write_provenance(provenance_app, out_dir)
    return {"dir": out_dir, "frames": made}


def openlime_hsh_package(out_dir: str, images, lights, *, normals=None, region=None,
                         layout: str = "tarzoom", pixel_mm=None, result_w: int | None = None,
                         title: str = "", provenance_app=None, progress=None,
                         white_balance=None) -> dict:
    """RGB HSH (27 coefficients) for OpenLIME and RelightLab, fitted from the
    photographs themselves at full size, as relight-cli -b hsh -p 27 does;
    tarzoom tiles by default, with HELIO's normals as the normals plane.
    *pixel_mm* is per pixel of the solve (*result_w* wide); *white_balance*
    (R, G, B linear gains, helio.core.white_balance) is applied to each frame."""
    from helio.core.white_balance import apply as _wb
    from helio.io import openlime_hsh
    paths = list(images or [])
    if lights is None or not paths or len(lights) != len(paths):
        raise NothingToExport("This needs the photographs and one light for each.")
    if len(paths) < 9:
        raise NothingToExport("HSH needs at least nine photographs.")
    first = source_frame(paths[0], region)
    if first is None:
        raise NothingToExport(f"Could not read {paths[0]}.")
    h, w = first.shape[:2]
    L = np.asarray(lights, np.float64)

    def read(k):
        img = first if k == 0 else source_frame(paths[k], region)
        if img is None:
            raise OSError(f"Could not read {paths[k]}")
        if img.shape[:2] != (h, w):
            raise OSError(f"{os.path.basename(paths[k])} is {img.shape[1]} x {img.shape[0]}, "
                          f"the others {w} x {h}")
        try:                                   # the colour only; a bad gain never stops the export
            return _wb(img, white_balance)
        except Exception:
            return img

    px = None
    if pixel_mm and result_w:
        px = float(pixel_mm) * float(result_w) / float(w)
    res = openlime_hsh.export(out_dir, read, L, h, w, normals=normals, layout=layout,
                              pixel_mm=px, title=title, progress=progress)
    if provenance_app is not None:
        from helio.io.provenance import write_provenance
        write_provenance(provenance_app, out_dir)
    res["frames"] = len(paths)
    return res


# ── IIIF 3D manifest ─────────────────────────────────────────────────────

def iiif_manifest(path: str, *, obj_id: str, w: int, h: int, n_images: int, lights=None,
                  normal_engine: str = "", depth_engine: str = "", mm_per_px=None,
                  z_mm_per_unit=None) -> dict:
    """A IIIF Presentation v3 manifest with the 3D TSG draft's Scene (M1),
    camera (M2), the capture's lights (M3) and the physical scale (M4).
    Replace https://example.org with the hosting domain before publishing."""
    obj_id = obj_id or "HELIO_OBJECT"
    base_url = f"https://example.org/iiif/{obj_id}"
    glb_url = f"https://example.org/models/{obj_id}.glb"
    alb_url = f"https://example.org/textures/{obj_id}_albedo.jpg"
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if mm_per_px and h and w:
        scale_note = f"{w * mm_per_px:.1f} × {h * mm_per_px:.1f} mm"
        scale_xyz = [round(mm_per_px, 6)] * 3
    else:
        scale_note = "uncalibrated (draw scale bar to add physical dimensions)"
        scale_xyz = None
    lights_ann = []
    if lights is not None:
        for idx, lv in enumerate(lights):
            lx, ly, lz = float(lv[0]), float(lv[1]), float(lv[2])
            lights_ann.append({
                "id": f"{base_url}/annotations/light/{idx}", "type": "Annotation",
                "motivation": "painting",
                "body": {"type": "PointLight", "color": "#ffffff",
                         "intensity": round(lz * 0.9 + 0.1, 3),
                         # The model faces +Z with north up (+Y), as HELIO writes
                         # it, so a light keeps its own coordinates (KD-036).
                         "position": {"x": round(lx, 4), "y": round(ly, 4), "z": round(lz, 4)}},
                "target": f"{base_url}/scene"})
    elev, azim = 30.0, 0.0
    dist = max(h, w) * 0.8 if (h and w) else 1.0
    camera = {"id": f"{base_url}/annotations/camera/default", "type": "Annotation",
              "motivation": "painting",
              "body": {"type": "Camera",
                       "perspective": {"fieldOfView": 35, "nearClip": 0.1, "farClip": 10000},
                       "position": {"spherical": {"elevation": elev, "azimuth": azim,
                                                  "radius": round(dist, 2)}},
                       "target": {"x": 0, "y": 0, "z": 0}},
              "target": f"{base_url}/scene"}
    transforms = []
    if scale_xyz:
        transforms.append({
            "id": f"{base_url}/annotations/transform/scale", "type": "Annotation",
            "motivation": "painting",
            "body": {"type": "Transform",
                     "transform": {"type": "Scale", "scale": {"x": scale_xyz[0], "y": scale_xyz[1],
                                                              "z": scale_xyz[2]}},
                     "comment": f"Physical dimensions: {scale_note}. 1 IIIF unit = {mm_per_px:.4f} mm."},
            "target": f"{base_url}/scene"})
    scene = {"id": f"{base_url}/scene", "type": "Scene", "label": {"en": [obj_id]},
             "items": [{"id": f"{base_url}/scene/page", "type": "AnnotationPage",
                        "items": [{"id": f"{base_url}/annotations/model", "type": "Annotation",
                                   "motivation": "painting",
                                   "body": {"id": glb_url, "type": "Model",
                                            "format": "model/gltf-binary",
                                            "label": {"en": [f"{obj_id} — PS Mesh"]},
                                            "thumbnail": [{"id": alb_url, "type": "Image",
                                                           "format": "image/jpeg"}]},
                                   "target": f"{base_url}/scene"}]}],
             "annotations": [{"id": f"{base_url}/scene/annotations", "type": "AnnotationPage",
                              "items": [camera] + transforms + lights_ann}]}
    canvas = {"id": f"{base_url}/canvas/2d", "type": "Canvas", "label": {"en": ["Albedo (2D)"]},
              "width": w or 1, "height": h or 1,
              "items": [{"id": f"{base_url}/canvas/2d/page", "type": "AnnotationPage",
                         "items": [{"id": f"{base_url}/annotations/albedo", "type": "Annotation",
                                    "motivation": "painting",
                                    "body": {"id": alb_url, "type": "Image", "format": "image/jpeg",
                                             "width": w or 1, "height": h or 1},
                                    "target": f"{base_url}/canvas/2d"}]}]}
    manifest = {
        "@context": ["https://iiif.io/api/presentation/3/context.json",
                     "https://iiif.github.io/3d/context/iiif-3d.json"],
        "id": f"{base_url}/manifest", "type": "Manifest", "label": {"en": [obj_id]},
        "summary": {"en": [f"Photometric Stereo 3D scan produced by Helio. Normal solver: "
                           f"{normal_engine}. Depth engine: {depth_engine}. {n_images} capture "
                           f"images. {scale_note}."]},
        "metadata": [
            {"label": {"en": ["Object ID"]}, "value": {"en": [obj_id]}},
            {"label": {"en": ["Technique"]}, "value": {"en": ["Photometric Stereo / RTI"]}},
            {"label": {"en": ["Software"]}, "value": {"en": ["Helio"]}},
            {"label": {"en": ["Date Created"]}, "value": {"en": [now]}},
            {"label": {"en": ["Image Count"]}, "value": {"en": [str(n_images)]}},
            {"label": {"en": ["Normal Solver"]}, "value": {"en": [normal_engine]}},
            {"label": {"en": ["Depth Engine"]}, "value": {"en": [depth_engine]}},
            {"label": {"en": ["Pixel Dimensions"]}, "value": {"en": [f"{w}×{h} px"]}},
            {"label": {"en": ["Physical Size"]}, "value": {"en": [scale_note]}},
            {"label": {"en": ["IIIF 3D TSG"]}, "value": {"en": ["Milestone Groups M1–M4"]}}],
        "requiredStatement": {"label": {"en": ["Attribution"]},
                              "value": {"en": ["Produced with Helio. Replace base URL "
                                               "'https://example.org' with your institution's "
                                               "hosting domain before publishing."]}},
        "seeAlso": [{"id": "https://iiif.github.io/3d/", "type": "Text",
                     "label": {"en": ["IIIF 3D TSG"]}, "format": "text/html"}],
        "items": [scene, canvas]}
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return {"path": path, "lights": len(lights_ann), "scaled": bool(scale_xyz),
            "scale_note": scale_note, "elev": elev, "azim": azim}


# ── A/B comparison page ──────────────────────────────────────────────────

def _jpg_b64(arr) -> str:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(arr)).save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def comparison_page(img_a, img_b, label_a: str, label_b: str, out_dir: str) -> str:
    """comparison.html: two renders, a draggable divider, both embedded so the
    file works on its own from file://."""
    a64, b64 = _jpg_b64(img_a), _jpg_b64(img_b)
    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>Helio A/B Comparison — {label_a} vs {label_b}</title>
<style>
  body{{margin:0;background:#111;font-family:Arial,sans-serif;color:#eee}}
  #wrap{{position:relative;width:100%;overflow:hidden;user-select:none;touch-action:pan-y}}
  #imgA,#imgB{{position:absolute;top:0;left:0;width:100%;pointer-events:none}}
  #imgB{{clip-path:inset(0 0 0 50%)}}
  #divider{{position:absolute;top:0;width:3px;background:#fff;cursor:col-resize;height:100%;box-shadow:0 0 8px rgba(255,255,255,0.8)}}
  #handle{{position:absolute;top:50%;transform:translate(-50%,-50%);width:36px;height:36px;background:#fff;border-radius:50%;display:flex;align-items:center;justify-content:center;color:#000;font-size:16px;box-shadow:0 2px 8px rgba(0,0,0,0.5)}}
  #labels{{position:absolute;top:10px;width:100%;display:flex;justify-content:space-between;pointer-events:none;padding:0 16px;box-sizing:border-box}}
  .lbl{{background:rgba(0,0,0,0.55);padding:4px 10px;border-radius:4px;font-size:13px}}
  h1{{text-align:center;padding:10px 0 4px;font-size:15px;color:#aaa;margin:0}}
</style></head><body>
<h1>Helio — {label_a} &nbsp;⟷&nbsp; {label_b}</h1>
<div id="wrap">
  <img id="imgA" src="data:image/jpeg;base64,{a64}">
  <img id="imgB" src="data:image/jpeg;base64,{b64}">
  <div id="divider"><div id="handle">⟷</div></div>
  <div id="labels"><span class="lbl">{label_a}</span><span class="lbl">{label_b}</span></div>
</div>
<script>
const wrap=document.getElementById('wrap'),div=document.getElementById('divider'),imgB=document.getElementById('imgB');
const imgA=document.getElementById('imgA');
// A data-URI image is often decoded before this runs, so size it now as well as on load.
function fitWrap(){{if(imgA.offsetHeight)wrap.style.height=imgA.offsetHeight+'px';}}
imgA.onload=fitWrap;
if(imgA.complete)fitWrap();
window.addEventListener('load',fitWrap);
window.onresize=fitWrap;
let pos=50,drag=false;
function setPos(x){{const r=wrap.getBoundingClientRect();pos=Math.max(1,Math.min(99,(x-r.left)/r.width*100));div.style.left=pos+'%';imgB.style.clipPath=`inset(0 0 0 ${{pos}}%)`;}}
div.addEventListener('mousedown',e=>{{drag=true;e.preventDefault()}});
document.addEventListener('mouseup',()=>drag=false);
document.addEventListener('mousemove',e=>{{if(drag)setPos(e.clientX)}});
// On a phone the whole picture is the handle: a finger anywhere moves the divider.
wrap.addEventListener('touchstart',e=>{{drag=true;setPos(e.touches[0].clientX)}},{{passive:true}});
document.addEventListener('touchend',()=>drag=false);
document.addEventListener('touchmove',e=>{{if(drag)setPos(e.touches[0].clientX)}},{{passive:true}});
setPos(wrap.getBoundingClientRect().left+wrap.offsetWidth/2);
</script></body></html>"""
    from helio.application.phone_web import phone_ready
    html = phone_ready(html, css="h1{font-size:13px!important}#handle{width:44px;height:44px}")
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "comparison.html")
    with open(p, "w", encoding="utf-8") as f:
        f.write(html)
    return p


# ── engine frames for the viewers ────────────────────────────────────────

def baked_frames(draw: Callable, engines, min_std: float) -> tuple[list, list]:
    """(engine, uint8 RGB) for each engine whose render is not flat; the rest
    are returned as skipped."""
    frames, skipped = [], []
    for eng in engines:
        try:
            a = np.asarray(draw(eng), np.float32)
            if float(np.nanmax(a)) <= 1.0 + 1e-6:
                a = a * 255.0
            a = np.clip(np.nan_to_num(a), 0, 255).astype(np.uint8)
            if float(a.std()) < min_std:
                skipped.append(eng)
                continue
            frames.append((eng, a))
        except Exception:
            skipped.append(eng)
    return frames, skipped


COMBINED_BAKE = ("PCA 1", "PCA 2", "PCA 3")


def combined_rti_viewer(out: str, normals, albedo, roughness, spec_var, h: int, w: int,
                        crop_box, obj_id: str, extra_layers=(), budget_mb: float | None = None
                        ) -> tuple[float, int]:
    """The one-file combined RTI viewer.  With *budget_mb* (an e-mail
    attachment, 25 MB) the textures step down 4096 → 768 px until it fits.
    Returns (MB, texture size used)."""
    from helio.io.exporters import export_combined_rti_viewer_html

    def write(mt):
        export_combined_rti_viewer_html(out, normals, albedo, roughness, spec_var, h, w, crop_box,
                                        obj_id, extra_layers=list(extra_layers), max_tex=mt)
        return os.path.getsize(out) / 1e6
    steps = [4096, 3072, 2048, 1536, 1024, 768]
    mb, used = write(steps[0]), steps[0]
    if budget_mb is not None:
        for mt in steps[1:]:
            if mb <= budget_mb:
                break
            mb, used = write(mt), mt
    return mb, used


RTI_3D_MAX_VERTS = 500_000     # the 3D tab's mesh at its finest
RTI_3D_TEX = 4096              # its colour and normal maps, long side, at its finest
RTI_3D_RTI_MB = 15.0           # the Relight tab's budget (combined_rti_viewer steps down to it)
RTI_3D_MB = 48.0               # the whole file: under 50 MB
# The 3D tab's (mesh points, texture side, JPEG quality), finest first: the first
# whose file fits RTI_3D_MB is kept.  JPEG textures are a third of PNG's size,
# which pays for the finer mesh and the larger maps.
RTI_3D_TIERS = ((500_000, 4096, 92), (350_000, 4096, 90), (250_000, 3072, 90),
                (100_000, 2048, 90))
# The 3D tab's starting light.  The stand-alone viewer's own levels add up to about
# four times the photograph's brightness (its centre near white, 220 of 255, on a
# book whose colour is 118); the exposure brings it back near the photograph's, the
# lights left as they are so the relief keeps its modelling.  The sliders go as high.
RTI_3D_LOOK = {"s_exp": 0.30, "s_key": 2.0, "s_fill": 0.9, "s_amb": 1.1, "s_rake": 2.0}
RTI_3D_TILT = 35.0             # the 3D tab opens looking 35° from overhead, from the object's front


def rti_3d_step(w: int, h: int, max_verts: int = RTI_3D_MAX_VERTS) -> int:
    """The grid step that keeps a *w* x *h* surface under *max_verts* vertices."""
    import math
    return max(1, int(math.ceil(math.sqrt(max(1, w) * max(1, h) / float(max_verts)))))


def _page_literal(page: str) -> str:
    """*page* as a JavaScript string that is safe inside a <script>: every '<'
    escaped, so no '</script' or '<!--' in it can end or upset the element."""
    return json.dumps(page, ensure_ascii=False).replace("<", "\\u003c")


def rti_3d_viewer(out: str, rti_page: str, model_page: str, obj_id: str) -> str:
    """One HTML file holding two whole viewers behind tabs: Relight (the combined
    RTI viewer, *rti_page*) and 3D model (the GLB web viewer, *model_page*).

    Each viewer is an existing self-contained page, carried as a string and put
    into its own iframe (srcdoc), so neither is changed and their scripts and
    styles cannot meet.  The 3D page is only built the first time its tab is
    opened, and both stay alive (stacked, the hidden one invisible but still its
    full size), so switching back leaves each view where it was."""
    import html as _html
    title = _html.escape(obj_id or "Object")
    mb = (len(rti_page.encode("utf-8")) + len(model_page.encode("utf-8"))) / 1e6
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} · Relight and 3D</title>
<style>
:root{{--bg:#111214;--bar:#18191C;--line:#2A2C31;--fg:#ECEAE5;--dim:#9A978F;--on:#E2B860}}
html,body{{margin:0;height:100%;background:var(--bg);color:var(--fg);
  font:14px/1.3 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}}
body{{display:flex;flex-direction:column}}
header{{display:flex;align-items:center;gap:16px;padding:8px 14px;background:var(--bar);
  border-bottom:1px solid var(--line);flex-wrap:wrap}}
h1{{font-size:15px;font-weight:600;margin:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:40vw}}
nav{{display:flex;gap:4px;background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:3px}}
nav button{{background:none;border:0;color:var(--dim);font:inherit;padding:6px 14px;border-radius:6px;cursor:pointer}}
nav button[aria-selected=true]{{background:#2A2C31;color:var(--fg);box-shadow:inset 0 -2px 0 var(--on)}}
nav button:focus-visible{{outline:2px solid var(--on);outline-offset:1px}}
.hint{{color:var(--dim);font-size:12px;margin-left:auto}}
main{{position:relative;flex:1;min-height:0}}
iframe{{position:absolute;inset:0;width:100%;height:100%;border:0;background:var(--bg)}}
iframe[hidden]{{display:block;visibility:hidden;pointer-events:none}}
.wait{{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:var(--dim);
  pointer-events:none}}
.wait[hidden]{{display:none}}
@media (max-width:600px){{.hint{{display:none}} h1{{max-width:50vw}}}}
</style></head>
<body>
<header>
  <h1 title="{title}">{title}</h1>
  <nav role="tablist" aria-label="Viewer">
    <button role="tab" id="t-rti" aria-selected="true" aria-controls="f-rti">Relight</button>
    <button role="tab" id="t-3d" aria-selected="false" aria-controls="f-3d">3D model</button>
  </nav>
  <span class="hint" id="hint">Move the light with the mouse · press 1 or 2 to switch</span>
</header>
<main>
  <iframe id="f-rti" title="Relightable image (RTI)" allow="fullscreen"></iframe>
  <iframe id="f-3d" title="3D model" allow="fullscreen" hidden></iframe>
  <div class="wait" id="wait" hidden>Opening the 3D model…</div>
</main>
<script>
var PAGES={{"rti":{_page_literal(rti_page)},
"3d":{_page_literal(model_page)}}};
var HINTS={{rti:"Move the light with the mouse · press 1 or 2 to switch",
"3d":"Drag to turn, scroll to zoom · press 1 or 2 to switch"}};
var built={{}};
function show(k){{
  var f=document.getElementById('f-'+k);
  if(!built[k]){{
    built[k]=true;
    if(k==='3d'){{document.getElementById('wait').hidden=false;
      f.addEventListener('load',function(){{document.getElementById('wait').hidden=true;}},{{once:true}});}}
    f.srcdoc=PAGES[k]; PAGES[k]=null;          /* the string is no longer needed once loaded */
  }}
  ['rti','3d'].forEach(function(o){{
    document.getElementById('f-'+o).hidden=(o!==k);
    document.getElementById('t-'+o).setAttribute('aria-selected',String(o===k));
  }});
  document.getElementById('hint').textContent=HINTS[k];
  try{{f.contentWindow.dispatchEvent(new Event('resize'));}}catch(e){{}}
  try{{f.contentWindow.focus();}}catch(e){{}}
}}
document.getElementById('t-rti').onclick=function(){{show('rti');}};
document.getElementById('t-3d').onclick=function(){{show('3d');}};
function key(e){{if(e.key==='1')show('rti');else if(e.key==='2')show('3d');}}
document.addEventListener('keydown',key);
['f-rti','f-3d'].forEach(function(id){{document.getElementById(id).addEventListener('load',function(){{
  try{{this.contentWindow.document.addEventListener('keydown',key);}}catch(e){{}}}});}});
show('rti');
</script>
<!-- Helio: Relight and 3D, {mb:.1f} MB of viewers -->
</body></html>
"""
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(doc)
    return out


def rti_deepzoom_relight(out_dir: str, normals, albedo, hsh, h: int, w: int, obj_id: str) -> str:
    from helio.io.exporters import export_hsh_deepzoom_relight_html
    return export_hsh_deepzoom_relight_html(out_dir, normals, albedo, hsh, h, w, obj_id or "Object")


# ── sharing ──────────────────────────────────────────────────────────────

def embed_snippet(url: str, width: str = "100%", height: str = "600px", obj: str = "") -> str:
    obj = obj or "Helio export"
    return (f'<iframe src="{url}" width="{width}" height="{height}" title="{obj} — Helio 3D '
            f'Viewer" frameborder="0" allowfullscreen loading="lazy"></iframe>')


def qr_card(qr_img, url: str, *, obj: str = "", institution: str = "", scale: str = ""):
    """The QR on a card that says what it points at: object, institution,
    scale, the URL and the HELIO build, so a code pinned beside an object
    still identifies it after the link has moved."""
    from PIL import ImageDraw, ImageFont
    try:
        import helio_build
        build = helio_build.stamp()
    except Exception:
        build = ""
    rows = []
    if obj:
        rows.append(("", obj, True))
    if institution:
        rows.append(("", institution, False))
    if scale:
        scale = str(scale).strip()
        rows.append(("Scale", scale if "mm" in scale else f"{scale} mm/pixel", False))
    rows.append(("", url if len(url) <= 52 else url[:49] + "…", False))
    if build:
        rows.append(("", f"Helio · {build}", False))
    qw, qh = qr_img.size
    pad, line_h = 28, 26
    W = max(qw + pad * 2, 520)
    H = qh + len(rows) * line_h + 14 + pad * 2
    card = Image.new("RGB", (W, H), "white")
    card.paste(qr_img.convert("RGB"), ((W - qw) // 2, pad))
    d = ImageDraw.Draw(card)

    def font(sz, bold=False):
        for nm in (("Helvetica-Bold" if bold else "Helvetica"), ("Arial Bold" if bold else "Arial")):
            try:
                return ImageFont.truetype(nm, sz)
            except Exception:
                continue
        return ImageFont.load_default()
    y = pad + qh + 10
    for lbl, val, big in rows:
        f = font(19 if big else 14, bold=big)
        txt = f"{lbl}: {val}" if lbl else val
        try:
            tw = d.textlength(txt, font=f)
        except Exception:
            tw = len(txt) * (10 if big else 7)
        d.text(((W - tw) / 2, y), txt, fill=(17, 17, 17) if big else (85, 85, 85), font=f)
        y += line_h
    d.rectangle([(0, 0), (W - 1, H - 1)], outline=(200, 200, 200))
    return card


def qr_png(url: str, path: str, *, label: bool = True, obj: str = "", institution: str = "",
           scale: str = "") -> str:
    """A QR code PNG, labelled unless *label* is False.  Refuses (RuntimeError)
    without the qrcode library rather than writing a picture of the URL."""
    try:
        import qrcode
    except ImportError:
        raise RuntimeError("Cannot make a QR code: the 'qrcode' library is missing from this "
                           "build. Nothing was written — a picture of a URL is not a QR code.")
    if not (url or "").strip():
        raise ValueError("Nothing to encode.")
    from helio.application.phone_share import tidy_url
    url = tidy_url(url)[0]                     # https://https://… and a missing https mended
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(url.strip())
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    if label:
        img = qr_card(img, url.strip(), obj=obj, institution=institution, scale=scale)
    img.save(path)
    return path


# ── Sketchfab upload ─────────────────────────────────────────────────────

SKETCHFAB_SLOTS = (("Albedo / Diffuse", "_baseColor.png", "your colour/photo texture"),
                   ("Normal Map", "_normal.png", "Y-flipped + gamma-encoded"),
                   ("Roughness", "_roughness.png", ""),
                   ("Specular", "_specular.png", ""),
                   ("Ambient Occlusion", "_ao.png", "if exported"),
                   ("Displacement", "_displacement.png", "strength ~0.5–2, if exported"))


def sketchfab_request(glb_path: str, token: str, name: str):
    """The multipart POST Sketchfab's v3 API expects (no network)."""
    import urllib.request
    import uuid
    if not token.strip():
        raise ValueError("Enter your Sketchfab API token (sketchfab.com › Settings › Password & API).")
    boundary = f"HelioUpload{uuid.uuid4().hex}"
    with open(glb_path, "rb") as fh:
        glb = fh.read()
    safe_name = (name or "Helio export").replace("\r", " ").replace("\n", " ")
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"modelFile\"; filename=\"model.glb\"\r\n"
            f"Content-Type: model/gltf-binary\r\n\r\n").encode() + glb
    body += (f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"name\"\r\n\r\n{safe_name}\r\n"
             f"--{boundary}--\r\n").encode()
    return urllib.request.Request("https://api.sketchfab.com/v3/models", data=body,
                                  headers={"Authorization": f"Token {token.strip()}",
                                           "Content-Type": f"multipart/form-data; boundary={boundary}"})


def sketchfab_upload(glb_path: str, token: str, name: str, *, opener=None, timeout: float = 300.0) -> str:
    """Upload a GLB; returns the model's page.  A certificate failure stops
    with the fix: HELIO 2 retried without verification, which would hand
    the token to anyone intercepting the connection (KD-048)."""
    import json as _json
    from helio.application.ai import _open
    resp = _open(sketchfab_request(glb_path, token, name), timeout, opener)
    uid = _json.loads(resp.read().decode()).get("uid", "")
    if not uid:
        raise RuntimeError("Sketchfab accepted the upload but returned no model id.")
    return f"https://sketchfab.com/models/{uid}"
