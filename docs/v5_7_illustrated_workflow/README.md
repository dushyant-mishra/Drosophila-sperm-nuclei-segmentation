# How the v5 illustrated workflow document is built

`Saturn_V5.7.1_Illustrated_Technical_Workflow_v5.docx` at the repository root is
a generated artifact. It is never edited by hand. Two scripts produce it, and
re-running them reproduces the deliverable from source.

## The two files that generate everything

| File | What it does |
|---|---|
| `scripts/build_v571_workflow_v5_figures.py` | Renders every figure into `docs/v5_7_illustrated_workflow/figures_v5/` and writes `figure_manifest.json` beside them. |
| `scripts/build_v571_workflow_v5_document.py` | Assembles the `.docx` from those figures plus the section text, which lives inline in that script. |

Build order matters: the document builder raises `FileNotFoundError` if a figure
it references is absent, so figures come first.

```powershell
python scripts/build_v571_workflow_v5_figures.py --output docs/v5_7_illustrated_workflow/figures_v5
python scripts/build_v571_workflow_v5_document.py
```

The figure build takes roughly six minutes on CPU because it segments twelve
optical planes with the production network. The document build takes seconds.
Word holds an exclusive lock on the `.docx` while it is open, so close the
document before rebuilding or the save fails with `PermissionError`.

## What the figures are built from

Inputs, all resolved by the scripts themselves:

- **Specimens**: the study manifest at
  `scratch/aborted_v57_full_20260730_0630/study_manifest.csv`, from which the
  first included KJ and the first included WT specimen are used. Every figure in
  the document is drawn from KJ-01, except the six-panel gallery which shows
  three nuclei from each group.
- **Network and settings**:
  `production_profiles/saturn_v5_7_1_model_c_epoch003.json`, loaded through
  `sperm_segmentation_saturnv5.7.1.py`.
- **Recorded production run**, for the whole-stack counts in the joining figure:
  `audits/evidence/v571_rc6_candidate/provenance/tracking_replay_inputs_outputs.zip`.
- **Archived measurements**, for the figures that plot numbers rather than
  images: the findings and evidence directories named per figure in
  `figure_manifest.json`.

Figure loading is shared with the audit evidence generator: the figure script
imports `scripts/generate_v571_intensity_width_evidence.py` for `load_pipeline`
and `segment_plane`, so figures and audit evidence segment the same way.

## Auditing the figures

The figures are git-ignored, along with the rest of the microscopy-derived
imagery in this repository, so there is nothing to diff in a pull request. The
embedded copies inside the tracked `.docx` are the shipped artifact. To audit:

1. Regenerate with the command above.
2. Compare each digest against `figures[].sha256` in `figure_manifest.json`.
   Rendering is byte-reproducible: re-running a figure in the same environment
   reproduces the recorded digest exactly, and this was confirmed on both an
   archived figure and a live-segmented one, so network inference on CPU is
   deterministic here too. The manifest also records the git commit and the
   SHA-256 of the pipeline and the production profile, so a mismatch separates
   "the code or data changed" from "the environment differs"; a different
   BLAS or torch build can move the last bits of an inference result even when
   nothing in this repository has changed.
3. For every figure, read the rendered image against the caption in
   `scripts/build_v571_workflow_v5_document.py` and against the paragraph that
   introduces it. A caption clause naming a count, a colour, a line or a marker
   must have a visible counterpart in the panel. This check has already caught
   real defects; see
   `audits/findings/2026-09-16-v5-document-figure-caption-audit.md`.
4. For plotted numbers, follow `figures[].inputs` and `figures[].numbers` in the
   manifest to the record they came from. Entries marked `live` are recomputed
   from the named specimen and plane on every build; entries marked `archived`
   are transcribed from an `audits/` record and can be checked against it.

## What the figures deliberately do and do not do

- Contrast stretches and display gammas are applied to rendered images so a
  biologist can see the nuclei. They never touch a measured value; every
  measurement is read from the unstretched linear image. Each such adjustment
  is commented at the point of use.
- No morphological dilation appears anywhere in this code, in keeping with the
  project rule that dilation is for visualisation only. Where the figures need
  the width measurement, `_centerline_profiles` mirrors
  `measure_intensity_profile_width`, including reading background from the far
  tails of the same profile.
- Object selection is by explicit, stated rules rather than by eye. The
  length-and-width figure takes an unbranched, non-merge-flagged object nearest
  the median length; the joining figure requires every detection in a shown
  track to exceed thirty pixels so no link line is drawn over nothing. Each rule
  is commented where it is applied and stated in the figure caption where it
  affects interpretation.
- Where a rendering choice could hide something, the geometry is constrained
  rather than tuned. The three-dimensional joining panel computes the plane
  separation it needs from the viewing angle so that no plane can occlude the
  one below it, and derives the stated depth exaggeration from that same
  geometry so the caption cannot drift from the picture.

## Related records

- `docs/plans/2026-09-15-v5-workflow-document-progress.md` — what changed from
  v4 to v5, figure by figure, and the numbers that were verified against the
  code before they were written into the text.
- `docs/plans/2026-09-14-v571-handover-state.md` — the overall v5.7.1 handover
  state, including what still needs an independent audit.
- `audits/findings/2026-09-16-v5-document-figure-caption-audit.md` — the
  figure-versus-caption defects found and fixed.
- `Saturn_V5.7_Illustrated_Technical_Workflow_v4.docx` — the superseded
  document, kept for comparison.
