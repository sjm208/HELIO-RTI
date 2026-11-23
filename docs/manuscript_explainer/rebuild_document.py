import base64, os
SP=os.environ.get("HELIO_EXPLAINER_DIR", os.path.dirname(os.path.abspath(__file__)))
def img(name):
    with open(f"{SP}/figs/{name}","rb") as fh:
        return "data:image/jpeg;base64," + base64.b64encode(fh.read()).decode()

F = {n[:-4]: img(n) for n in os.listdir(f"{SP}/figs") if n.endswith(".jpg")}

HTML = """<title>Reading the Surface</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Spectral:ital,wght@0,400;0,500;1,400&display=swap">
<style>
:root{
  --ground:#F1F4F3; --surface:#FFFFFF; --ink:#141C1B; --muted:#546462;
  --faint:#7E918E; --rule:#D5DCDA; --accent:#1B6B5A; --accent-soft:#E2EEEA;
  --flag:#8A5A12; --flag-soft:#F6EDDF;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#0E1413; --surface:#151D1C; --ink:#DFE7E5; --muted:#9AACA9;
  --faint:#6D807D; --rule:#26312F; --accent:#5FD3B2; --accent-soft:#16302A;
  --flag:#D6A15C; --flag-soft:#2C2417;
}}
:root[data-theme="dark"]{
  --ground:#0E1413; --surface:#151D1C; --ink:#DFE7E5; --muted:#9AACA9;
  --faint:#6D807D; --rule:#26312F; --accent:#5FD3B2; --accent-soft:#16302A;
  --flag:#D6A15C; --flag-soft:#2C2417;
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
  font-family:"Spectral",Georgia,serif;font-size:17.5px;line-height:1.68;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:760px;margin:0 auto;padding:0 26px 110px}
h1,h2,h3,.eyebrow,.meta,.cap,th,.note b,.flag b,.lede{font-family:"IBM Plex Sans",system-ui,sans-serif}
.eyebrow{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11.5px;
  letter-spacing:.15em;text-transform:uppercase;color:var(--accent);margin:0 0 20px}
header{padding:74px 0 34px}
h1{font-weight:600;font-size:clamp(32px,5.4vw,52px);line-height:1.06;
  letter-spacing:-.022em;margin:0;text-wrap:balance}
.lede{margin:22px 0 0;font-size:20px;line-height:1.5;color:var(--muted);font-weight:400}
.meta{display:flex;flex-wrap:wrap;gap:8px 24px;margin:30px 0 0;padding-top:22px;
  border-top:1px solid var(--rule);
  font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11.5px;color:var(--faint)}
.meta b{color:var(--muted);font-weight:500}
h2{font-weight:600;font-size:26px;line-height:1.2;letter-spacing:-.012em;
  margin:64px 0 0;text-wrap:balance;scroll-margin-top:20px}
h2 .n{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12px;
  color:var(--accent);letter-spacing:.1em;display:block;margin-bottom:9px;font-weight:500}
h2::after{content:"";display:block;width:56px;height:2px;margin-top:13px;
  background:linear-gradient(100deg,var(--accent),transparent)}
h3{font-weight:600;font-size:17.5px;margin:34px 0 -6px;letter-spacing:-.005em}
p{margin:16px 0}
ul{margin:16px 0;padding-left:21px}li{margin:9px 0}li::marker{color:var(--accent)}
strong{font-weight:500}
figure{margin:32px 0;background:var(--surface);border:1px solid var(--rule);
  border-radius:3px;overflow:hidden}
figure img{display:block;width:100%;height:auto}
figcaption{padding:12px 16px;font-family:"IBM Plex Sans",system-ui,sans-serif;
  font-size:13.5px;line-height:1.5;color:var(--muted);border-top:1px solid var(--rule)}
figcaption b{color:var(--ink);font-weight:500}
.note{margin:24px 0;padding:16px 20px;background:var(--accent-soft);border-radius:3px}
.note b{display:block;font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:10.5px;
  letter-spacing:.11em;text-transform:uppercase;color:var(--accent);margin-bottom:6px;font-weight:500}
.note p{margin:0}
.flag{margin:24px 0;padding:16px 20px;background:var(--flag-soft);border-radius:3px}
.flag b{display:block;font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:10.5px;
  letter-spacing:.11em;text-transform:uppercase;color:var(--flag);margin-bottom:6px;font-weight:500}
.flag p{margin:0}
.tw{overflow-x:auto;margin:24px 0;border:1px solid var(--rule);border-radius:3px}
table{border-collapse:collapse;width:100%;background:var(--surface);font-size:15.5px}
th,td{text-align:left;padding:11px 15px;border-bottom:1px solid var(--rule);vertical-align:top}
th{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:10.5px;letter-spacing:.1em;
  text-transform:uppercase;color:var(--muted);font-weight:500;white-space:nowrap}
tr:last-child td{border-bottom:none}
code{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:14px;color:var(--accent)}
footer{margin-top:72px;padding-top:26px;border-top:1px solid var(--rule);
  color:var(--faint);font-size:14.5px}
@media(max-width:600px){body{font-size:16.5px}.wrap{padding:0 18px 80px}}
</style>
<div class="wrap">

<header>
<p class="eyebrow">HELIO Pro &middot; for rare books and manuscripts</p>
<h1>Reading the Surface</h1>
<p class="lede">What photometric stereo records about a manuscript page, how it does it, and what it cannot tell you.</p>
<div class="meta">
  <span><b>Software</b> HELIO Pro 2.6.0</span>
  <span><b>Figures</b> from an eight-frame capture</span>
  <span><b>Audience</b> curators, conservators, digitisation</span>
</div>
</header>

<h2><span class="n">&sect; 01</span>What is hard to see</h2>
<p>Every codicologist has done this: tilted a leaf toward a window, or run a torch across it at a low angle, and watched the page change. Under flat light a parchment folio is writing on a ground. Under light that skims the surface almost edge-on, it becomes a landscape &mdash; ruling and pricking, the furrow of a nib, a scraped correction, the ghost of writing pressed through from the facing leaf.</p>
<p>Raking light works because relief that has no colour still has shape, and shape casts shadows when the light is low enough. Nothing in this document improves on that observation. What follows is about three limitations of how it is usually done.</p>
<p><strong>It is not repeatable.</strong> The angle lives in the hand holding the torch. Two scholars describing the same feature cannot be certain they saw it under the same light, and neither can describe the angle precisely enough for a third to reproduce it.</p>
<p><strong>It is not shareable.</strong> The observation happens in the reading room, in front of the object. A photograph fixes one angle and discards every other.</p>
<p><strong>It requires the object.</strong> Each re-examination is another consultation, another handling, of something fragile.</p>
<div class="note"><b>The turn this document makes</b><p>These are not problems with raking light. They are problems with raking light being a <em>gesture</em> rather than a <em>record</em>.</p></div>

<h2><span class="n">&sect; 02</span>What HELIO Pro produces</h2>
<p>HELIO does not enhance a photograph. It photographs a page several times with the camera fixed and the light moved between exposures, and from the differences in brightness it computes which way every point on the surface faces. From those orientations it reconstructs the relief.</p>
<p>What it stores, then, is not a picture of the page under one light. It is a measurement of the page's surface, from which any lighting can be generated afterwards &mdash; in a browser, by a colleague on another continent, years later, with the manuscript back in its box.</p>
<div class="note"><b>The sentence this document rests on</b><p>Raking light stops being something you <em>do</em> to a manuscript and becomes something you <em>store</em> about it.</p></div>

<h2><span class="n">&sect; 03</span>How it works</h2>
<p>A matte surface lit by a lamp is brightest where it faces the lamp squarely, and darkens progressively as it turns away. That relationship is simple and predictable. If the same point is photographed under several lights coming from known directions, its several brightnesses together pin down the direction that point faces.</p>
<figure>
  <img src="__frames__" alt="Eight photographs of the same manuscript opening, each lit from a different direction">
  <figcaption><b>Figure 1.</b> The raw material: eight exposures of one opening. The camera has not moved; only the lamp has. The differences between these frames &mdash; subtle here, and largely invisible in the shadows and gutter &mdash; are the entire input to the reconstruction.</figcaption>
</figure>
<p>Do this for every pixel and the result is a <strong>normal map</strong>: a record of surface orientation across the whole page. Integrate those orientations and you get a <strong>height map</strong> &mdash; the relief itself, in relative units until a scale is set.</p>
<p>Two constraints follow directly, and they govern how a capture must be made:</p>
<ul>
  <li><strong>The camera must not move between frames.</strong> Everything depends on a given pixel meaning the same point of the page in every exposure.</li>
  <li><strong>The light directions must be known.</strong> This is the whole of the next section, and it is where accuracy is decided.</li>
</ul>

<h2><span class="n">&sect; 04</span>Where the lights were</h2>
<p>The reconstruction can only be as good as the light directions given to it. HELIO offers three routes, and they are not equivalent.</p>
<div class="tw"><table>
<thead><tr><th>Route</th><th>Use when</th><th>Accuracy</th></tr></thead>
<tbody>
<tr><td><code>1A</code> External lights</td><td>A dome of known geometry, or an existing <code>.lp</code> file</td><td>As good as the rig that produced it</td></tr>
<tr><td><code>1B</code> Sphere calibration</td><td>A reflective sphere is in shot</td><td><strong>Best.</strong> Measured from your own frames</td></tr>
<tr><td><code>1C</code> Virtual dome</td><td>Neither &mdash; a legacy or field capture</td><td>Approximate; positions placed by hand</td></tr>
</tbody></table></div>
<p>The middle route deserves the emphasis. A mirrored sphere placed in the frame reflects each lamp as a small highlight, and the position of that highlight on the sphere is a direct read-out of the lamp's direction. Nothing has to be measured on site, remembered, or carried over from a previous session: the calibration is recorded in the same frames as the object.</p>
<figure>
  <img src="__sphere__" alt="The same chrome sphere in four frames, the highlight in a different place each time">
  <figcaption><b>Figure 2.</b> The calibration sphere in four of the eight frames. The highlight moves across it as the lamp moves; its position gives the lamp's direction. This is why a sphere in shot is worth more than any amount of care taken over rig measurements.</figcaption>
</figure>
<p>The practical recommendation is simply this: include a sphere in every capture as a matter of course, exactly as a colour target is included. It costs nothing at capture time and it is the difference between a measurement and an estimate.</p>
<div class="flag"><b>Stated plainly</b><p>HELIO's automatic sphere detection is experimental, and the software says so on the panel. It places a circle for you to check and adjust; it is not a hands-off step, and a capture should never be processed without looking at where it put the circle.</p></div>

<h2><span class="n">&sect; 05</span>What it finds on a page</h2>
<p>Photometric stereo records topography. That makes it suited to a specific family of manuscript evidence &mdash; everything that has <em>shape</em>, whether or not it has colour.</p>
<h3>Preparation of the page</h3>
<p>Ruling and pricking, including blind ruling that left no pigment at all; bounding lines; the pattern of prickings along an edge; the impression of a stylus. These are precisely the features that are invisible under flat light and obvious under raking light, which is to say they are what the method is best at.</p>
<h3>Writing, and its removal</h3>
<p>The furrow a nib leaves in the surface; scraping and erasure, which thins or flattens the substrate; corrections written over a prepared area. An erasure has a topography even when it has no remaining ink.</p>
<h3>Structure and handling</h3>
<p>Sewing stations, cockling and distortion, set-off from a facing leaf, losses, tears, insect grazing, the impression left by a former binding.</p>
<h3>Impressions and applied material</h3>
<p>Seal impressions and their tooling, binding stamps and rolls, and &mdash; with the caveats in &sect;09 &mdash; gilding and its ground.</p>
<div class="flag"><b>The boundary of this list</b><p>These are the categories the method is <em>suited</em> to, not a promise about any particular manuscript. Whether a given feature resolves depends on whether its relief is deep enough to measure at the resolution captured. &sect;09 sets that expectation honestly; this section should not be read without it.</p></div>

<h2><span class="n">&sect; 06</span>Choosing what to look at</h2>
<p>After a solve, HELIO holds a number of different views of the same surface. Treating them as a menu of filters is the commonest way to waste the tool. Each answers a different question.</p>
<figure>
  <img src="__albedo__" alt="The manuscript text block with lighting removed">
  <figcaption><b>Figure 3a. Albedo</b> &mdash; the page's own colour with the lighting divided out, as if lit perfectly evenly. This is what the parchment and ink actually are, separated from how they were lit.</figcaption>
</figure>
<figure>
  <img src="__normals__" alt="The same text block rendered as a normal map">
  <figcaption><b>Figure 3b. Normal map</b> &mdash; surface orientation as colour. The lilac ground is the flat of the page; every departure from it is a change in the direction the surface faces. This is the primary measurement, before any interpretation.</figcaption>
</figure>
<figure>
  <img src="__relief__" alt="The same text block with relief exaggerated">
  <figcaption><b>Figure 3c. Relief</b> &mdash; the orientation data sharpened so that shallow structure reads. The writing is visible here as <em>shape</em> rather than as ink: what you are seeing is the surface, not the pigment.</figcaption>
</figure>
<figure>
  <img src="__fusion__" alt="The same text block combining all eight light directions">
  <figcaption><b>Figure 3d. Multi-light fusion</b> &mdash; all eight exposures combined into a single legibility-optimised image. Unlike the others this does not respond to the light control, because it is not a lighting: it is a summary of the whole capture.</figcaption>
</figure>
<p>Two practical points repay their space in any training. The first is that <strong>a low light is the useful one</strong>: most disappointment with this method comes from lighting a page from overhead and concluding there is nothing to see. The second is that the stack-derived views deliberately ignore the light control, which regularly reads as a fault and is not.</p>

<h2><span class="n">&sect; 07</span>Measurement, not illustration</h2>
<p>This is the section that separates HELIO from image enhancement, and the one that matters most for conservation. The reconstruction is a height field: once a scale is set from a target in frame, it carries real units.</p>
<figure>
  <img src="__depth__" alt="The text block rendered as a depth map in false colour">
  <figcaption><b>Figure 4.</b> The same area as height rather than as an image. Warm is proud, cool is recessed. The broad gradient across the leaf is the page's own cockling &mdash; the substrate, not the writing.</figcaption>
</figure>
<figure>
  <img src="__profile__" alt="A cross-section through the height map">
  <figcaption><b>Figure 5.</b> A cross-section through that height field. This is the form a measurement takes: a profile that can be re-measured on the same leaf in ten years and compared. <b>In this capture the vertical axis is relative</b> &mdash; no scale target was set, so it shows form, not millimetres.</figcaption>
</figure>
<p>With a scale established, HELIO will report:</p>
<ul>
  <li><strong>Depth profiles</strong> across a groove, a score, or a loss.</li>
  <li><strong>Volume and true three-dimensional surface area</strong> over a defined region &mdash; the latter differing from the flat area by an amount that is itself a measure of distortion.</li>
  <li><strong>Change against a previous session</strong>, by loading an earlier depth map as a reference. This is the basis for monitoring a condition over years rather than describing it in adjectives.</li>
  <li><strong>Reconstruction quality</strong>, so that a reader can tell which parts of the surface the solve is confident about.</li>
</ul>
<div class="note"><b>The argument for conservation use</b><p>A condition report that says &ldquo;slight cockling, stable&rdquo; is a judgement, and a perfectly reasonable one. A report that carries a dated depth profile is a measurement that someone else can repeat.</p></div>

<h2><span class="n">&sect; 08</span>Sharing, publishing, depositing</h2>
<p>A record that cannot leave the workstation is of limited use. HELIO's outputs are grouped here by where they are going rather than by file format.</p>
<div class="tw"><table>
<thead><tr><th>Destination</th><th>What to export</th></tr></thead>
<tbody>
<tr><td>A colleague, by email</td><td>A single self-contained HTML viewer &mdash; they relight the page in a browser with nothing installed</td></tr>
<tr><td>A IIIF viewer</td><td>Manifest and tiles</td></tr>
<tr><td>A collections system</td><td>LIDO XML</td></tr>
<tr><td>A citable repository</td><td>Zenodo package</td></tr>
<tr><td>Long-term storage</td><td>Heritage archive bundle</td></tr>
<tr><td>A publication</td><td>Figure sheets and 16-bit TIFFs</td></tr>
<tr><td>A 3D context</td><td>GLB model, or the PBR map suite</td></tr>
</tbody></table></div>
<p>The first row is the one to demonstrate. A curator who can send a relightable page to a colleague, and have it open in a browser with no software and no server, has understood the whole proposition in about four seconds.</p>

<h2><span class="n">&sect; 09</span>What it does not do</h2>
<p>The credibility of &sect;05 depends entirely on this section being honest.</p>
<ul>
  <li><strong>It is not multispectral imaging.</strong> HELIO records shape, not chemistry. An erasure appears as a change in the surface; recovering the text that was removed is a different instrument answering a different question. The two are complementary and are often worth doing together.</li>
  <li><strong>Depth is relative until it is calibrated.</strong> Without a scale target in frame, comparisons within a single capture are valid but absolute figures are not. Figure 5 in this document is exactly that case.</li>
  <li><strong>Shiny and translucent surfaces break the assumption.</strong> The method assumes light scatters evenly off a matte surface. Gilding, glaze, heavy burnish and some inks do not behave that way, and the reconstruction is correspondingly less reliable there. HELIO can show where this is happening rather than conceal it, which is the right response &mdash; but it cannot repair it.</li>
  <li><strong>Very shallow relief may not resolve at all.</strong> The method finds what is physically there to find, at the resolution it was captured. A feature below that threshold will not appear, and no processing choice will conjure it.</li>
  <li><strong>It does not replace examining the object.</strong> It is a record that reduces how often the object must be brought out, and that lets a question be revisited without another consultation.</li>
</ul>
<div class="flag"><b>On the AI-assisted features</b><p>HELIO includes AI tools for description and detection. In cataloguing use these are drafting aids that require verification, and should be described as such. Presenting their output as findings would undermine &sect;05, which is the section that has to be believed.</p></div>

<h2><span class="n">&sect; 10</span>What it takes in practice</h2>
<p>The commonest barrier to trying this is an assumption that it needs a dome. It does not.</p>
<ul>
  <li>A camera on a copy stand, held rigid for the duration of a capture.</li>
  <li>One movable light.</li>
  <li>A reflective sphere in frame.</li>
  <li>A scale target, if measurements are wanted.</li>
</ul>
<p>Three exposures is the arithmetic minimum; eight is a comfortable working number and is what produced every figure here. More frames improve the solve and cost only capture time.</p>
<div class="flag"><b>To be completed from your own studio</b><p>Frames per opening, minutes to capture, minutes to process, and gigabytes stored per opening at working and archival resolution. These should be measured on your equipment rather than estimated &mdash; a digitisation lead will reasonably disbelieve round numbers.</p></div>

<h2><span class="n">&sect; 11</span>Glossary</h2>
<div class="tw"><table>
<tbody>
<tr><td><strong>Photometric stereo</strong></td><td>Recovering surface shape from several photographs lit from different directions.</td></tr>
<tr><td><strong>RTI</strong></td><td>Reflectance Transformation Imaging &mdash; a capture that can be relit interactively afterwards.</td></tr>
<tr><td><strong>Normal map</strong></td><td>A record of which way each point of the surface faces, stored as colour.</td></tr>
<tr><td><strong>Albedo</strong></td><td>The surface's own colour with the lighting divided out.</td></tr>
<tr><td><strong>Height map</strong></td><td>The relief, derived from the normals.</td></tr>
<tr><td><strong>Raking light</strong></td><td>A lamp close to the plane of the surface, almost side-on.</td></tr>
<tr><td><strong>Specular</strong></td><td>Mirror-like reflection; the shine.</td></tr>
<tr><td><strong>Lambertian</strong></td><td>A surface that scatters light evenly in all directions. The assumption the method rests on.</td></tr>
</tbody></table></div>

<footer>
<p><strong>About the figures.</strong> Every image in this document was produced in HELIO Pro 2.6.0 from a single eight-frame capture of one manuscript opening, calibrated from the sphere visible in Figure 2. They illustrate the views the software produces; they are not presented as a discovery about this particular manuscript.</p>
<p style="margin-top:14px"><strong>Two things this draft still needs from you:</strong> the studio figures in &sect;10, and a worked example in which a real feature was found on a real page &mdash; ideally a modest one, since a recovered ruling pattern convinces a sceptical reader more reliably than a spectacular result they suspect was chosen for effect.</p>
</footer>

</div>
"""
for k,v in F.items():
    HTML = HTML.replace(f"__{k}__", v)
out=f"{SP}/doc/reading-the-surface-full.html"
open(out,"w").write(HTML)
print(f"  written: {os.path.getsize(out)//1024} KB")
missing=[k for k in ("frames","sphere","albedo","normals","relief","fusion","depth","profile") if f"__{k}__" in HTML]
print("  unresolved image slots:", missing or "none")
