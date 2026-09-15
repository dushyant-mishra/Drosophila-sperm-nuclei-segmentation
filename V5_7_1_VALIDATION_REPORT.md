# Saturn v5.7.1 Validation Report

Original date: 2026-08-21
Superseded in part: 2026-09-15

## Status: partly superseded, do not cite the measurements below as current

This report records the 2026-08-21 two-specimen replay. Several measurement
definitions have changed since, so the numbers in the body describe behaviour the
pipeline no longer produces. Sections that are superseded say so where they
appear. What remains valid is the description of the population rules, the
calibration and provenance requirements, and the conclusion that an independent
audit is still required.

The production gate is closed. `production_audit_gate_state` reports four
unaccepted claims:

| claim | status |
|---|---|
| `MEAS-BODY-WIDTH-001` | implemented, not accepted |
| `MEAS-INTENSITY-WIDTH-001` | implemented, not audited |
| `REPORT-BIOLOGIST-CONCISE-001` | implemented, not audited |
| `WORKFLOW-GUI-PRIMARY-001` | implemented, not audited |

Changes since this report that invalidate parts of it:

- Biological width is now the background-corrected raw-signal FWHM, not the
  mask contour chord. The mask boundary follows the training annotation
  convention and is roughly 2.4 times the optical width of a nucleus.
- Area and the observed-slice footprint are derived from the profile width, with
  the filled-mask slab retained separately as a technical diagnostic.
- Merges are flagged and split on objective branch or profile evidence rather
  than behind a 20 um length gate, which changes `estimated_unique_nuclei`.
- The study design accepts one reference group and several comparison groups.

A superseding audit run is required against `PIPELINE-V571-PRODUCTION-001`,
which is accepted but no longer describes current behaviour.

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

> **Superseded.** These counts predate evidence-based merge splitting. On plane
> 35 that change raised instance counts by 12.7 percent in KJ-01 and 7.1 percent
> in WT-01, and the widths below are mask-derived. Treat the table as a record of
> the 2026-08-21 behaviour, not as current values.


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

## Width measurement: superseded

> **Superseded.** This section described the subpixel mask contour chord as the
> primary width. It is now an explicitly named technical diagnostic and does not
> drive biological reports.

Primary comparative width is the background-corrected full width at half maximum
of the raw intensity profile, sampled perpendicular to the centerline and
restricted to the object's own instance mask. The reason for the change is that
the mask boundary reproduces the training annotation convention rather than the
nucleus: annotations in this project have a median width of 1.606 um against an
optical nucleus width near 0.643 um, and the resulting bias is not constant
between specimens.

Measured on planes 34 to 36 after profile isolation was hardened:

| Measurement | KJ-01 | WT-01 |
| --- | ---: | ---: |
| Signal-profile FWHM width, median | 0.701 um | 0.730 um |
| Mask contour-chord width, median | 1.514 um | 1.629 um |
| Mask to signal ratio | 2.12x | 2.29x |

That the ratio differs between specimens is why the mask width cannot carry a
genotype comparison. These figures are from three sampled planes, not a full
stack replay, and are not a substitute for a fresh audited run.

Absolute nucleus diameter is not established and must not be reported. At
0.378 um per pixel against a 0.23 um point spread function the image is about
3.3 times below Nyquist laterally, and a deconvolved estimate saturates near
0.6 um, so PSF correction is disabled by default. Comparison between groups is
supported; an absolute width claim is not.

Integrated profile signal is recorded as technical quality control only. It is
far more sensitive to real width than the half maximum but scales directly with
staining brightness, and this study has no independent staining control.

## Report and study behavior

- Biologist summaries contain completed specimens only.
- Excluded, missing, or unrun specimens remain in the exclusion ledger and
  technical run-state table.
- Below-2-um sensitivity is automatic and stored under `technical_qc`; it does
  not create a competing biological count or a routine manual-review queue.
- The normal PDF first page presents one technical-valid biological population.

## Automated validation

As of 2026-09-15:

- Python compilation: passed
- Full test suite: `448 passed`
- Frozen v5.7 pipeline source: unchanged, line-ending differences only

Note for anyone reproducing this: a bare `python -m pytest -q` on the
development machine reports 122 spurious errors because pytest cannot create its
temporary directory. Pass `--basetemp` to a writable path to get a true result.

The 2026-08-21 run recorded `205 passed in 25.33s`, which is retained here only
as the historical figure.

## Decision

**2026-08-21 decision, retained for the record.** The corrected v5.7.1
segmentation, learned-core instance separation, calibration, body-width
measurement, morphology-neutral tracking, gap recovery, and report population
logic passed the two-specimen production replay, with a fresh independent
seven-role audit still required before release tagging or a full biological
rerun.

**2026-09-15 status.** That audit has still not been run, and more now requires
it. Biological width, area, volume, merge handling and the study design have all
changed since, so the replay above no longer describes the pipeline. The
production gate is closed on four unaccepted claims, and
`PIPELINE-V571-PRODUCTION-001` needs a superseding run because its accepted state
predates the merge-splitting change to `estimated_unique_nuclei`.

No biological conclusion may rest on the numbers in this report. The current
state of the work, including what is verified and what remains, is recorded in
`docs/plans/2026-09-14-v571-handover-state.md`.
