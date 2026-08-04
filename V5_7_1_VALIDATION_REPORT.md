# Saturn v5.7.1 Validation Report

Date: 2026-08-04

## Production candidate

- Pipeline: `sperm_segmentation_saturnv5.7.1.py`
- Tuner: `utils/tune_parameters_Saturnv5_7_1.py`
- Profile: `production_profiles/saturn_v5_7_1_model_c_epoch003.json`
- Checkpoint: `model_checkpoints/v571_model_c_dual_head_epoch003.pt`
- Checkpoint SHA-256: `7d49031dbcce31f0600c44146d9b5282b0df6e28bb1fc1bdde6ef2146ed15d25`
- Segmentation: dual-head U-Net primary, foreground threshold 0.60, core threshold 0.50
- Tracking: morphology-neutral global assignment with a 4.295 um displacement cap
- Calibration: resolved independently from each specimen's Leica XML metadata

The checkpoint and dual-head thresholds are frozen for this validation. The
tuner searches tracking behavior only; it does not optimize morphology toward
a wild-type target.

## Mixed tuning

The same candidate set was evaluated on four strata using each specimen's own
ROI and metadata calibration:

| Stratum | Slices | Best individual score |
| --- | ---: | ---: |
| KJ-01 | z33-z37 | 25.55 |
| KJ-13 | z48-z52 | 9.13 |
| WT-01 | z38-z42 | 9.44 |
| WT-13 | z24-z28 | 38.01 |

The shared `reviewed_base` candidate was retained instead of the numerically
highest aggregate candidate. Across all four strata it recovered at least
97.5% of reciprocal overlapping links and had lower nonreciprocal and
long-distance link fractions. The shared tuning artifacts are under
`parameter_tuning_results_v5_7_1/mixed_tracking/shared`.

## Tracking replay

Relative to the historical morphology-restrictive tracker, the selected
morphology-neutral tracker reduced premature fragmentation while preserving
technical integrity:

| Specimen | Selected tracks | Legacy tracks | Selected single-slice | Legacy single-slice | Selected median slices |
| --- | ---: | ---: | ---: | ---: | ---: |
| KJ-01 | 4,377 | 7,086 | 29.91% | 50.51% | 6 |
| WT-01 | 2,734 | 4,564 | 30.61% | 52.96% | 5 |

No selected track contains duplicate observations from one Z plane. The
maximum accepted displacement is approximately 4.29 um. Proposed joins that
would reconstruct an object above 20 um are rejected without deleting their
original 2D detections.

## Full-specimen pilot

Fresh full-stack runs were completed under
`scratch/v571_final_production_pilot`.

| Measurement | KJ-01 | WT-01 |
| --- | ---: | ---: |
| Source slices | 88 | 67 |
| Estimated unique technical-valid nuclei | 4,238 | 2,655 |
| Median slices per track | 6 | 6 |
| Single-slice fraction | 27.61% | 28.55% |
| Median 3D length | 10.67 um | 9.40 um |
| Median maximum 2D length | 10.31 um | 9.04 um |
| Median apparent body-mask width | 1.55 um | 1.69 um |
| Median body-width P90 | 1.70 um | 1.93 um |
| Median length/body-width ratio | 7.17 | 5.85 |
| Technical-valid tracks above 20 um | 0 | 0 |

These are descriptive pilot values from one specimen per group and are not a
WT-versus-mutant biological inference.

The biologist-facing PDF first pages match `analysis_summary.csv` exactly. A
fresh render confirmed that both reports are complete and unclipped. Middle
slice overlays show dense U-Net-primary coverage of visible elongated nuclei.

Report paths:

- `scratch/v571_final_production_pilot/KJ-01/batch_report_v5.7.1-body-width.pdf`
- `scratch/v571_final_production_pilot/WT-01/batch_report_v5.7.1-body-width.pdf`

## Remaining sensitivity item

The technical-valid population includes 831 KJ-01 and 560 WT-01 tracks below
2 um. Many are single-plane, strongly supported U-Net fragments. They remain
visible because single-plane and unusual morphology are not automatic failures.
They should be reviewed as a sensitivity stratum in the full study rather than
silently removed or used to retune toward expected WT morphology.

## Automated validation

- Python compilation: passed
- Full test suite: `179 passed in 23.45s`
- Tuner self-check: passed
- `git diff --check`: passed; only line-ending conversion warnings were emitted

## Decision

The v5.7.1 segmentation, calibration, body-width measurement, and shared
morphology-neutral tracking profile are ready for a controlled full-study run.
The primary analysis population remains `technical_valid`; morphology warnings
are annotations, not exclusions. The below-2-um sensitivity stratum must be
reported separately during study-level interpretation.
