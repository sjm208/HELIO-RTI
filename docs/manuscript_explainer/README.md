# Reading the Surface — manuscript explainer

Explains what HELIO Pro records about a manuscript page, how it does it, and
what it cannot tell you. Written for curators, conservators and digitisation
leads.

| File | What it is |
|---|---|
| `Reading_the_Surface.html` | **The document.** Self-contained — every figure is embedded, so it opens in any browser and can be emailed as one file. |
| `Reading_the_Surface_blueprint.html` | The section-by-section plan it was written from, including notes on what each section argues and the traps to avoid. |
| `figures/` | The figures as separate JPEGs, for reuse in slides or a paper. |
| `regenerate_figures.py` | Re-renders every figure from the capture in `test images/`. |
| `rebuild_document.py` | Re-embeds `figures/` into the HTML. Run after regenerating. |

## The figures are real

All seven were produced in HELIO Pro 2.6.0 from the eight-frame capture in
`test images/`, calibrated from the chrome sphere visible in Figure 2. They
illustrate the views the software produces; they are **not** presented as a
discovery about that manuscript, and the document says so.

## Two gaps left deliberately

* **§10 needs your studio's real numbers** — frames per opening, minutes to
  capture and process, gigabytes stored. Measured, not estimated: a
  digitisation lead will reasonably disbelieve round figures.
* **There is no worked example.** The blueprint argues the document is only as
  persuasive as one sequence in which a real feature was found on a real page.
  That has to come from your own material, and a modest finding convinces a
  sceptic more reliably than a spectacular one.

## Regenerating

```bash
cd ~/Desktop/Helio_NoTorch
python3 docs/manuscript_explainer/regenerate_figures.py   # ~2 min, solves the capture
python3 docs/manuscript_explainer/rebuild_document.py
```

Both scripts still point at the session scratchpad they were written in; edit
the paths at the top before relying on them.
