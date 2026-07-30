# Saturn v5.7 U-Net-primary smoke test

## Purpose

`SEGMENTATION_ENGINE = "unet_primary"` treats the 2.5D U-Net probability map
as the authoritative 2D foreground evidence. It is intended to test whether the
model-supported population can be measured without the classical Saturn shape
filters silently vetoing short, wide, curved, or low-ratio nuclei.

The first real-image validation is deliberately limited to a few named Z
slices. It does not run tracking, parameter tuning, a full stack, or a
WT-versus-mutant comparison.

## Processing contract

The U-Net-primary path is:

1. Load or infer one full-size U-Net probability map.
2. Zero probability outside the ROI and inside the exclusion mask.
3. Apply true low/high hysteresis using
   `UNET_CANDIDATE_THRESHOLD` and `UNET_SEED_THRESHOLD`.
4. Remove only components below `UNET_PRIMARY_MIN_COMPONENT_PX`.
5. Split connected foreground with connected high-confidence seed regions.
6. Preserve watershed integer labels.
7. Skeletonize each filled instance independently.
8. Reduce a branched skeleton to a deterministic longest geodesic centerline.
9. Measure each filled mask and its mapped centerline.
10. Report unusual morphology as warnings, not rejection reasons.

The initial smoke values are `0.05` for candidate support and `0.30` for
high-confidence seeds. They are validation starting points, not finalized
biological parameters.

## Technical exclusions

U-Net-primary objects are excluded only for technical failures such as tiny
isolated noise, no high-confidence seed, an invalid or empty mask, no valid
centerline, invalid nonfinite geometry, a duplicate instance, or an unresolved
multi-instance merge. A long object is retained and flagged for merge review.

`UNET_PRIMARY_CLASSICAL_ADDITIONS_ENABLE` defaults to `False` for the first
smoke test. When enabled later, classical detections may only add objects in
residual space. They cannot remove, shorten, overwrite, or relabel accepted
U-Net-primary instances.

## Command

Run from the repository root after replacing the paths and Z values:

```powershell
.\.venv\Scripts\python.exe `
  .\scratch\run_v57_unet_primary_smoke.py `
  --input-dir "C:\path\to\stack" `
  --unet-model "C:\path\to\epoch_003.pt" `
  --base-params "C:\path\to\base_parameters.json" `
  --roi-mask "C:\path\to\roi.npy" `
  --exclusion-mask "C:\path\to\exclusion.npy" `
  --z-values "5,35,60" `
  --outdir ".\scratch\v5_7_unet_primary_smoke" `
  --engines "hybrid,unet_primary" `
  --repeat 2
```

Omit `--exclusion-mask` when none is used. The runner refuses more than six
target slices unless `--allow-large-run` is supplied explicitly.

## Outputs

The output directory contains:

- `smoke_summary_v5_7.json`
- `smoke_summary_v5_7.csv`
- `instance_audit_v5_7.csv`
- `technical_failures_v5_7.csv`
- `review_panels/`
- `probability_maps/`

Each review panel labels the distinction between filled U-Net masks and thin
measured centerlines. Cyan is U-Net-primary, green is an optional Saturn-only
addition, and red is a hard technical failure. Overlay dilation is display-only
and never changes counts, lengths, widths, or instance labels.

## Acceptance gates

The smoke run requires:

- exactly the requested target Z values;
- no output outside the ROI;
- no output inside the exclusion mask;
- a high-confidence seed in every accepted U-Net instance;
- unique instance IDs;
- identical label hashes across repeated runs;
- no silent U-Net failure or classical fallback.

Counts and morphology are observations, not pass/fail targets.

## Completed KJ-01 smoke

The first real-image smoke used the previously reviewed KJ-01 stack and its
saved ROI. The targets were `z17`, `z35`, and `z70`, representing lower,
middle, and upper stack regions already included in the mixed WT/KJ tuning
manifest. The shared `0.05/0.30` evidence preset and `epoch_003.pt` checkpoint
were used. Tracking and classical additions were disabled.

| Z | Hybrid measured | U-Net-primary measured | U-Net-primary warnings |
|---|---:|---:|---:|
| 17 | 363 | 232 | 209 |
| 35 | 397 | 270 | 231 |
| 70 | 434 | 299 | 265 |

All structural quality gates passed in two repeated runs:

- identical label hashes;
- zero outside-ROI pixels;
- zero exclusion-mask overlap;
- zero accepted instances without a high-confidence seed;
- zero duplicate instance IDs;
- zero filled-instance/measurement mapping mismatches;
- zero centerline failures;
- zero unresolved merges.

Every technically sized, seed-supported U-Net child instance was measured.
Excluded low-threshold components were limited to seedless support and tiny
isolated pixels. Visual review showed coherent filled masks and centerlines.
The larger hybrid counts came mainly from additional short classical skeleton
fragments and should not be interpreted automatically as better recall.

The complete local artifacts are under:

```text
scratch/v5_7_unet_primary_smoke_kj01_z17_z35_z70_run2
```
