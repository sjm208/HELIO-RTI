"""Real figures for the manuscript explainer, from the test capture."""
import os, sys, glob, math, multiprocessing
import numpy as np, cv2
HERE=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0,HERE); os.chdir(HERE)
OUT=os.path.join(os.environ.get("HELIO_EXPLAINER_DIR", os.path.dirname(os.path.abspath(__file__))), "figs")

def save(name, img, q=86, maxw=1100):
    a=np.asarray(img)
    if a.dtype!=np.uint8:
        a=np.clip(a*255 if a.max()<=1.001 else a,0,255).astype(np.uint8)
    if a.ndim==2: a=np.dstack([a]*3)
    if a.shape[1]>maxw:
        s=maxw/a.shape[1]; a=cv2.resize(a,(maxw,int(a.shape[0]*s)),interpolation=cv2.INTER_AREA)
    cv2.imwrite(f"{OUT}/{name}", a[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY,q])
    print(f"  {name}  {a.shape[1]}x{a.shape[0]}  {os.path.getsize(OUT+'/'+name)//1024} KB", flush=True)

def main():
    import Helio_Pro
    from helio.core.processing import auto_detect_sphere, detect_sphere_candidates
    class _B:
        def __getattr__(self,n): return lambda *a,**k:"ok"
    Helio_Pro.messagebox=_B()
    paths=sorted(glob.glob("test images/*.jpeg"))

    # ── 1. the source frames ────────────────────────────────────────────
    fr=[cv2.imread(p) for p in paths]
    h,w=fr[0].shape[:2]
    th=260; tw=int(w*th/h)
    strip=np.hstack([cv2.resize(f,(tw,th)) for f in fr[:8]])
    save("frames.jpg", strip[:, :, ::-1], maxw=1400)

    # ── 2. the sphere, highlight moving ─────────────────────────────────
    cands=detect_sphere_candidates(paths, h, w, want_two=False)
    if cands:
        cx,cy,r=cands[0]; r=int(r*1.5)
        x1,y1=max(0,int(cx-r)),max(0,int(cy-r)); x2,y2=min(w,int(cx+r)),min(h,int(cy+r))
        tiles=[cv2.resize(f[y1:y2, x1:x2],(200,200)) for f in fr[:4]]
        save("sphere.jpg", np.hstack(tiles)[:, :, ::-1], maxw=800)

    # ── solve, calibrated ───────────────────────────────────────────────
    L=auto_detect_sphere(paths,None)
    app=Helio_Pro.RTIViewerPro(); app.update()
    app.data.plate_paths=list(paths)
    app.loaded_lp_matrix=L; app.data.current_L_mat=L
    app.data.normal_engine.set("Linear Least Squares")
    app.data.depth_engine.set("Fast Poisson (FFT)")
    st={}
    def wait(t=0):
        th_=getattr(app,"_proc_thread_obj",None)
        if th_ is not None and not th_.is_alive() and getattr(app.data,"normals_full",None) is not None:
            st["ok"]=True; app.quit()
        elif t>2400: st["to"]=True; app.quit()
        else: app.after(200, lambda: wait(t+1))
    app.after(200, lambda:(app.run_processing(), wait()))
    app.mainloop(); assert st.get("ok"), st
    print("  solved", app.data.normals_full.shape, flush=True)

    H,W=app.data.h, app.data.w
    # a text-block crop used for every comparison figure
    cx1,cy1,cx2,cy2 = int(W*0.16),int(H*0.30),int(W*0.74),int(H*0.52)
    OW=1100; OH=int(OW*(cy2-cy1)/(cx2-cx1))
    def render(eng, el=None, az=None):
        a=np.asarray(app.get_shading(OH, OW, False,
                                     crop_coords=(cx1,cy1,cx2,cy2),
                                     engine_override=eng), np.float32)
        if a.max()<=1.001: a=a*255
        return np.clip(np.nan_to_num(a),0,255).astype(np.uint8)

    save("albedo.jpg",  render("Albedo"))
    save("normals.jpg", render("Normals RGB"))
    save("relief.jpg",  render("Normals Unsharp Masking"))
    save("fusion.jpg",  render("Multi-Light Fusion"))

    # depth as a heat map over the same crop
    d=cv2.resize(np.asarray(app.depth_poisson,np.float32),(W,H))[cy1:cy2, cx1:cx2]
    d=cv2.resize(d,(OW,OH))
    dn=(d-np.percentile(d,1))/(np.ptp(np.percentile(d,[1,99]))+1e-9)
    save("depth.jpg", cv2.applyColorMap((np.clip(dn,0,1)*255).astype(np.uint8),
                                        cv2.COLORMAP_VIRIDIS)[:, :, ::-1])

    # ── the profile: a slice across the page's own relief ────────────────
    prof=d[d.shape[0]//2]
    prof=prof-np.median(prof)
    ph,pw=300,1100
    plot=np.full((ph,pw,3),18,np.uint8)
    x=np.linspace(0,pw-1,len(prof)).astype(int)
    sc=(ph*0.36)/(np.percentile(np.abs(prof),99)+1e-9)
    y=(ph//2 - prof*sc).astype(int)
    for gx in range(0,pw,pw//8): plot[:,gx]=(34,42,40)
    plot[ph//2,:]=(58,72,68)
    for i in range(len(x)-1):
        cv2.line(plot,(x[i],int(np.clip(y[i],0,ph-1))),
                      (x[i+1],int(np.clip(y[i+1],0,ph-1))),(95,211,178),2)
    save("profile.jpg", plot[:, :, ::-1], maxw=1100)
    try: app.destroy()
    except Exception: pass

if __name__=="__main__":
    multiprocessing.set_start_method("spawn", force=True); main()
