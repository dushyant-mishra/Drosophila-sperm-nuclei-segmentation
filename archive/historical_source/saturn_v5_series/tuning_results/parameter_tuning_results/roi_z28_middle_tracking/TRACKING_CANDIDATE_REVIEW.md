# Tracking Candidate Review

Tracking tuning was run on ROI `roi_z28.1.npy` with the middle-slice
segmentation settings from:

`parameter_tuning_results/roi_z28_middle_segmentation/best_segmentation_params_v5_2_001_run_sperm_images.json`

Tuning slices:

`28-44`

## Result

The raw optimizer winner is:

`best_tracking_params_v5_2_001.json`

Do **not** treat this as the recommended tracking JSON yet.

It improved the optimizer score slightly, but it shifted the tuning block toward
single-slice tracks:

- Raw optimizer winner median Z-span: `0.0 um`
- Current/default tracking median Z-span on same block: `1.04 um`
- Raw optimizer winner single-slice fraction: `0.512`
- Current/default tracking single-slice fraction: `0.495`

When requiring median Z-span to remain at least one slice, the current/default
tracking settings were the best candidate from this short tuning pass.

## Recommendation

Keep the current tracking settings for `batch_output_7`.

The tracking scoring function has been updated to penalize median Z-span
collapse, so future tracking tuning should be less likely to select this kind
of candidate.

## Follow-up Fixes Applied

The V5.2 tracking tuner now also scores the post-detection stop diagnostics
written by the main pipeline. Candidates are penalized when too many tracks
stop after `overlap_but_0_stable`, which is a fragmentation signal: the tracker
found a plausible overlap but rejected it as unstable.

Future tracking tuning runs will write both:

- `best_tracking_params_v5_2_###.json`: raw optimizer winner
- `best_safe_tracking_params_v5_2_###.json`: best candidate that keeps
  `zspan_median_um >= 0.5` and `single_frac <= 0.52`

Prefer the safe file for first review/use when it exists.

## Updated Quick Retune

After adding the stop-reason and safe-candidate scoring, a quick consecutive
tracking tune was run on:

`z28-32`

with the same ROI and middle-slice segmentation parameters.

New output:

`best_safe_tracking_params_v5_2_003.json`

This candidate passed the continuity guardrails:

- Median Z-span: `1.04 um`
- Single-slice fraction: `0.371`
- Multi-slice tracks: `669 / 1063`

However, it also shifts the population longer:

- Median 3D length: `12.129 um`
- Mean 3D length: `13.424 um`
- Long outliers: `317 / 1063`

Interpretation: this is a better tracking-candidate than the old raw
`best_tracking_params_v5_2_001.json`, but it should still be reviewed on
overlays/full-batch QC before replacing the current/default tracking settings.

Ignore `best_tracking_params_v5_2_002.json`; it came from a deliberately fast
non-consecutive slice test and produced all single-slice tracks.

## Constrained Quick Retune

The tracking search space was then tightened to reduce over-linking:

- `TRACK_BBOX_PADDING_PX` limited to `1-4`
- `TRACK_MAX_DIST_UM` limited to `4.0-8.5`
- `CONSERVATIVE_MAX_CENTROID_JUMP_UM` limited to `5.0-10.0`
- length/area jump tolerances were pulled back
- scoring now penalizes median/mean 3D length inflation more strongly

The same consecutive block was retuned:

`z28-32`

New output:

`best_safe_tracking_params_v5_2_004.json`

Metrics:

- Median Z-span: `1.04 um`
- Single-slice fraction: `0.468`
- Multi-slice tracks: `658 / 1236`
- Median 3D length: `11.300 um`
- Mean 3D length: `12.599 um`
- Long outliers: `306 / 1236`
- Stop fraction: `0.466`
- `overlap_but_0_stable` fraction: `0.270`

Interpretation: `004` is a better compromise than `003`. It preserves the
one-slice median Z continuity while pulling the length distribution back toward
the current/default run. It is still experimental and should be checked on the
full ROI batch before replacing default tracking, but it is the best tracking
candidate from the tuning folder so far.
