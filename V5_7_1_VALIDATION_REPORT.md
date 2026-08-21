# Saturn v5.7.1 Validation Report

Date: 2026-08-21

## Production candidate

- Pipeline: `sperm_segmentation_saturnv5.7.1.py`
- Tuner: `utils/tune_parameters_Saturnv5_7_1.py`
- Profile: `production_profiles/saturn_v5_7_1_model_c_epoch003.json`
- Checkpoint: `model_checkpoints/v571_model_c_dual_head_epoch003.pt`
- Checkpoint SHA-256: `7d49031dbcce31f0600c44146d9b5282b0df6e28bb1fc1bdde6ef2146ed15d25`
- Segmentation: dual-head U-Net primary; foreground 0.60; core 0.50
- Instance repair: foreground-preserving learned-core watershed; 20 um review
  trigger, 0.05 core-peak prominence, and 4.0 um minimum peak spacing
- Tracking: morphology-neutral global assignment; 4.295 um displacement cap
- Gap recovery: one missing Z plane may be bridged
- Calibration: resolved independently from each specimen's Leica XML

The checkpoint and U-Net thresholds are frozen. Comparative tuning does not
reward similarity to wild-type length, width, shape, or count.

## Population rules

The biologist-facing population is `technical_valid`. Short, long, wide,
curved, irregular, and single-slice nuclei remain measurable morphology. A
smooth object above 20 um remains visible with a review warning. An object
above 20 um with a branched connected centerline is a technical multi-object
merge and is not counted as one nucleus.

Sub-2-um observations remain eligible for cross-slice joining. Tracks may span
one missing plane by a straight calibrated centroid segment. Volume is summed
from observed filled masks only; no missing mask area is invented.

## Matched segmentation comparison

The same KJ-01 and WT-01 specimens were compared under three configurations.

| Configuration | KJ valid | KJ >20 um | KJ median slices | WT valid | WT >20 um | WT median slices |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No overlong split | 5,492 | 1,100 | 2 | 3,194 | 410 | 3 |
| Historical 18 um split and hidden merge veto | 4,238 | 0 visible | 6 | 2,655 | 0 visible | 6 |
| Final 20 um split and explicit merge test | 4,359 | 8 | 6 | 2,721 | 2 | 5 |

The historical row contained 43 KJ and 22 WT components above 20 um, but they
were hidden as technical failures based on length alone. The final logic does
not repeat that mistake. Relative to the unsplit diagnostic, it removes the
large connected-chain failure while preserving supported morphology.

On matched z35, the final 20 um trigger retained every sub-2-um observation and
every 15-20 um object. Lowering the trigger to 18 um split three additional KJ
and four additional WT objects in the 15-20 um review band, so 20 um was kept.

Evidence:

- `audits/evidence/v571_final_trigger20_smoke`
- `audits/evidence/v571_overlong_split_trigger18_smoke`
- `audits/evidence/v571_remediation_tracking`

## Final post-audit full-specimen replay

Fresh outputs are under `scratch/v571_post_audit_remediation_pilot_run3`.
This replay replaces length-only marker placement with objective learned-core
evidence. A component above 20 um is split only when the core head supplies
multiple disconnected regions or multiple longitudinal peaks separated by a
probability valley. The filled foreground mask is partitioned without erosion.

| Measurement | KJ-01 | WT-01 |
| --- | ---: | ---: |
| Source slices | 88 | 67 |
| Estimated unique technical-valid nuclei | 4,624 | 2,816 |
| Technical failures | 282 | 132 |
| Clear multi-object connected-component failures | 183 | 75 |
| Single-slice fraction | 33.35% | 31.57% |
| Gap-linked tracks | 617 | 356 |
| Median projection + Z extent | 10.92 um | 9.29 um |
| Median maximum 2D length | 10.52 um | 8.95 um |
| Median apparent body-mask width | 1.56 um | 1.69 um |
| Median length/body-width ratio | 7.40 | 5.93 |
| Technical-valid tracks below 2 um | 886 | 571 |
| Technical-valid tracks from 15-20 um | 805 | 428 |
| Smooth technical-valid tracks above 20 um | 229 | 65 |

These values describe one specimen per group and are not a genotype inference.
`analysis_summary.csv` matches the technical-valid track table exactly for
count, median projection + Z extent, and median body width. The projection + Z
extent is an orientation-sensitive hypotenuse, not an integrated 3D
centerline. Objects above 20 um remain
visible as technical-review morphology when the learned core does not provide
independent split evidence; they are not silently deleted or forced toward a
WT reference length.

Reports:

- `scratch/v571_post_audit_remediation_pilot_run3/samples/kj_sv_40xx0.75-1/attempt_001/batch_report_v5.7.1-body-width.pdf`
- `scratch/v571_post_audit_remediation_pilot_run3/samples/w1118_sv_feb_40xx0.75-1/attempt_001/batch_report_v5.7.1-body-width.pdf`
- `scratch/v571_post_audit_remediation_pilot_run3/between_sample_analysis/01_biological_results/Biological_Comparison_Report.pdf`
- `scratch/v571_post_audit_remediation_pilot_run3/between_sample_analysis/02_quality_control/Quality_Control_Report.pdf`

The biological comparison is descriptive only because this pilot has one
specimen per group. Inferential statistics are unavailable until each group
contains at least three independent specimens.

## Width plateau validation

The primary track width is the subpixel perpendicular contour-chord width from
the representative observed Z plane, not the quantized distance-transform
median.

| Check | KJ-01 | WT-01 |
| --- | ---: | ---: |
| Distinct primary widths at 4 decimals | 2,362 | 1,615 |
| Primary-width modal fraction | 3.07% | 3.84% |
| Distinct legacy widths at 4 decimals | 52 | 52 |
| Legacy-width modal fraction | 41.55% | 31.90% |
| Spearman correlation with area/length width | 0.815 | 0.816 |

The legacy field retains its pixel-grid plateau for reproducibility. It is not
the primary biological width. The new field is continuous and agrees strongly
with an independent filled-area/length estimate.

## Report and study behavior

- Biologist summaries contain completed specimens only.
- Excluded, missing, or unrun specimens remain in the exclusion ledger and
  technical run-state table.
- Below-2-um sensitivity is automatic and stored under `technical_qc`; it does
  not create a competing biological count or a routine manual-review queue.
- The normal PDF first page presents one technical-valid biological population.

## Automated validation

- Python compilation: passed
- Full test suite: `205 passed in 25.33s`
- Focused gap-recovery and merge-classification tests: passed
- Tuner self-check: passed
- `git diff --check`: passed; line-ending conversion warnings only

## Decision

The corrected v5.7.1 segmentation, learned-core instance separation,
calibration, body-width measurement, morphology-neutral tracking, gap recovery,
and report population logic pass the two-specimen production replay. A fresh
independent seven-role audit is still required on a clean pushed commit before
release tagging or a full biological rerun.
