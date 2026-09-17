# Finding: generated figures asserted things the images did not show

Date: 2026-09-16
Raised by: user observation on the v5 illustrated workflow document, quantified
and remediated by Claude
Status: remediated 2026-09-16, pending independent review
Affects: `Saturn_V5.7.1_Illustrated_Technical_Workflow_v5.docx` and
`scripts/build_v571_workflow_v5_figures.py`. No pipeline measurement changed.

## What was observed

The user reported that in figure 6.2 of the v5 document one of the four nuclei
had its dashed link line drawn through the whole slab but none of its coloured
pixels were visible on any plane, and separately that the image planes in that
panel were too dark to see where the nuclei were.

Both were real. The caption for that figure reads "the coloured pixels are the
nucleus found on each slice and the dashed line links the ones judged to be the
same nucleus", so the panel asserted something the image did not support.

## Root causes

**The invisible nucleus was an occlusion, not a missing artist.** Instrumenting
the selection confirmed the object was painted onto all five planes, at 65, 69,
71, 73 and 11 pixels for slices 33 to 37. At the box aspect then in use,
`set_box_aspect((1.0, 1.0, 0.80))` with `elev=17`, the projected vertical extent
of one plane was 0.399 unit-cube units against a gap to the next plane of 0.191,
so each plane concealed the rear corner of the one below and that nucleus sat in
the concealed corner.

**The darkness came from stretching a mostly empty field.** The display stretch
ran from the 2nd to the 99.6th percentile of the whole five-plane crop, which in
a field that is largely background maps the nuclei into a narrow bright tail.

**A third defect surfaced while fixing those two, and it was a wrong number in a
deliverable.** The caption claimed the slice spacing was drawn eleven times
larger than life. `set_box_aspect` renormalises the z axis, so the
`exaggeration = 11.0` factor the code multiplied into the z data never reached
the screen at all. The displayed factor was `0.80 * 37.84 / 1.384 = 21.9`.

## Remediation

`scripts/build_v571_workflow_v5_figures.py`, `fig_joining_through_depth`:

- The box aspect is now computed from the elevation, azimuth and plane span so
  that a plane's vertical screen extent is smaller than the gap to the next
  plane, with a 15 percent margin. Occlusion is prevented by construction rather
  than avoided by choosing a lucky viewpoint.
- Track selection requires every detection in a shown track to exceed thirty
  pixels, so a track that dwindles to a speck on one slice cannot leave its link
  line drawn over nothing.
- The z data now carries true microns and the caption states a factor derived
  from the rendered geometry, so the stated exaggeration cannot drift from the
  figure again.
- Display stretch moved to the 25th to 99.3rd percentile with a 0.65 gamma,
  commented as display-only at the point of use.

## Four more mismatches found by auditing every figure against its caption

1. `fig_merge_correction` said recognising clumps "evens out" the correction.
   Its own bars show plus 12.7 percent in KJ against plus 7.1 percent in WT, so
   the correction is not even. The honest statement, now used, is that the old
   under-counting fell unevenly and recognising clumps removes that shortfall.
   Its middle panel was titled "Most clumps are two nuclei", but it plots a
   length histogram of all 559 objects, not of clumps.
2. `fig_availability_bias` used internal reason codes as axis labels.
3. The document caption for the hero figure claimed the neighbouring nuclei sit
   "barely a micron apart". Measured edge to edge with a distance transform, the
   closest pair are 1.89 um apart.
4. Section 6.3 attributed every withheld width to a close neighbour. That is
   76.9 percent of them; 16.2 percent are centrelines too short to sample.

## How to verify

Regenerate and read the images:

```powershell
python scripts/build_v571_workflow_v5_figures.py --output docs/v5_7_illustrated_workflow/figures_v5
python scripts/build_v571_workflow_v5_document.py
```

In `v5_fig21_joining_depth.png`, each of the four numbered nuclei must appear as
coloured pixels on all five planes, and no plane may overlap the one below. The
stated exaggeration in the caption must equal
`box_z * (nx * um) / ((n_planes - 1) * um_z)` for the values the script computes.

`docs/v5_7_illustrated_workflow/README.md` gives the general procedure, and
`figure_manifest.json` records a SHA-256 for every figure together with where
each plotted number comes from.

## Why this is recorded as a finding

No measurement changed, so no claim in `audits/claims_registry.json` is affected.
It is recorded because the defect class matters for review: a figure script that
exits zero proves only that a file was written. It says nothing about whether an
artist produced visible output, whether a legend entry corresponds to anything on
the canvas, or whether a number quoted in a caption matches what the figure
computed. Every figure in a generated document has to be opened and read against
its caption before the document ships.
