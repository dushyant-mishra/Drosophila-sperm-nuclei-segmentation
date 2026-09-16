# v5 illustrated workflow document — what changed and why (2026-09-15/16)

`Saturn_V5.7.1_Illustrated_Technical_Workflow_v5.docx` replaces the v4 document.
It is written for a biologist reading the output, uses the KJ and WT study data
throughout, and is built reproducibly by two scripts rather than edited by hand:

- `scripts/build_v571_workflow_v5_figures.py` produces every figure into
  `docs/v5_7_illustrated_workflow/figures_v5/` (git-ignored, as all
  microscopy-derived imagery in this repository is) plus a `figure_manifest.json`
  recording provenance.
- `scripts/build_v571_workflow_v5_document.py` assembles the tracked `.docx`
  from those figures. Re-running both regenerates the deliverable from source.

## Figures rebuilt in this pass

**`v5_fig20_how_measured.png` — how length and width are computed.** An earlier
version selected the object with the greatest length, which returned a 31.77 um
object against an 8.04 um median; long objects at this density are usually
several nuclei stuck together rather than one long one. Selection now prefers an
unbranched, non-merge-flagged object with a finite width, some visible curvature,
and a length near the median. The panel now also prints the numbers it is
illustrating: centre-line length, straight tip-to-tip distance and their ratio;
the number of cuts taken and the width reported. The profile shown is the cut
whose width is closest to the median rather than whichever cut fell at the
midpoint, so the panel and the reported value agree.

**`v5_fig_hero_neighbourhood.png` — measurement in a crowded field.** Rebuilt
from scratch. The previous version cut one horizontal line across a crowded field
and plotted the brightness along it, which was wrong in two ways. A horizontal
cut through nuclei that lie at an angle returns an oblique chord rather than the
width the pipeline reports, so the bar drawn on the graph did not correspond to
the value labelling it. Separately, prominent peaks in the transect carried no
label at all, because the bright signal at those positions belonged to nuclei
whose mask did not reach that exact image row. The replacement profiles each
neighbour along its own perpendicular cut, exactly as the pipeline does, so every
curve in the panel belongs to one named nucleus and nothing is left unexplained.

A helper, `_centerline_profiles`, mirrors `measure_intensity_profile_width`: it
reads background from the far tails of the same profile, restricts the profile to
the object being measured, and refuses a cut whose half-maximum crossing is not
observed on both sides inside that object's own mask.

**`v5_fig21_joining_depth.png` — joining through depth.** This replaces a
two-panel scatter and histogram, and takes its cue from v4's
`workflow_figure_09_joined_image_planes.png`, which showed exploded image planes
but at a magnification where individual nuclei were not visible. The new left
panel is a close-up: five consecutive slices of KJ-01, segmented and linked by
`track_across_slices` itself, with each tracked nucleus painted into the image
plane in its own colour and a dashed line joining the detections judged to be the
same nucleus. Painting the nucleus into the surface's face colours is deliberate;
drawn as a separate three-dimensional artist it is hidden behind the plane at
almost every viewing angle, which is what the first attempt did. The right-hand
panels come from the recorded production run for the whole 88-slice stack, and
show that movement between neighbouring slices sits far inside the 4.30 um limit
and that joining turns 26,651 detections into 5,766 counted nuclei.

**`v5_fig00_clean_examples.png`** now draws the perpendicular cuts the width is
actually taken on, rather than only the centre line, and applies a display
contrast stretch so the fainter WT panels are legible.

**`v5_fig03_sampling_limit.png`, `v5_fig05_merge_correction.png` and
`v5_fig06_availability_bias.png`** were re-worded. The measurements are
unchanged; the titles and captions now read as acquisition guidance and as
statements of what the pipeline does, which is what a biologist-facing document
needs, rather than as a list of shortcomings.

## Document sections added

- **5.1 and 5.2** explain how length and width are computed, including why length
  follows the centre line rather than the straight tip-to-tip distance, and why
  the outline the network draws scopes the measurement without setting its value.
- **6.1** explains joining through depth: a global assignment rather than greedy
  nearest-neighbour matching, one missing slice may be bridged, and size and
  shape carry zero weight in comparative mode so that a mutant's phenotype cannot
  change how readily its nuclei are joined.
- **7. Every Number Saturn Reports, and How to Use It** is the parameter
  reference. Every measure is classed Absolute, Comparison or Diagnostic, in four
  tables covering size and shape, orientation and neighbourhood, counts and
  densities, and quality flags. The rule of thumb given is that lengths, angles,
  ratios and counts can be quoted, while anything involving width, depth or
  density is for comparing groups imaged the same way.

Sections 7 through 12 of the previous draft were renumbered to 8 through 13.

## Numbers verified against the code and data before they were written

Claims in the document were checked rather than assumed:

- Width is withheld for 21.4 percent of detections across slices 33-37 of KJ-01,
  which supports "about one object in five".
- The number of cuts per nucleus has a median of 16 and a floor of 5
  (`BODY_WIDTH_MIN_SAMPLES`), so the text says "typically sixteen, never fewer
  than five" rather than the "ten or more" first drafted.
- `ASSIGNMENT_LENGTH_WEIGHT`, `ASSIGNMENT_WIDTH_WEIGHT` and
  `ASSIGNMENT_AREA_WEIGHT` are all zero in the production profile because
  `ANALYSIS_MODE` is comparative, so the first draft's claim that area similarity
  contributes to joining was wrong and was removed.
- The representative plane is chosen by
  `largest_filled_mask_area_then_unet_support_then_lowest_z`, so the text says
  the slice on which the nucleus appeared largest rather than the one where it
  was most in focus.
- Taper is the largest slice area divided by the smallest, not a narrowing
  towards the ends.
- At the current 0.378 um pixel a 0.65 um nucleus spans under two pixels, not
  one, so the zoom guidance was corrected.

## Deferred or blocked, unchanged

- The 35-specimen cohort run is deferred until the user has reviewed how the
  pipeline runs.
- Acceptance audits must be run by Codex. The implementer cannot be the sole
  validator, and the codex CLI is not present in this environment.
