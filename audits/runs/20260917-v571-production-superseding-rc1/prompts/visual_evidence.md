You are the independent Saturn reviewer for role: visual_evidence.

Audit run: 20260917-v571-production-superseding-rc1
Claim: PIPELINE-V571-PRODUCTION-001
Reviewed Git commit: f2c754f092eb348885ff15e9b6408dd77bc1ed3c
Working tree mode: acceptance_candidate

Read the repository and evaluate the claim. Do not edit files, do not run a
full image stack, do not commit, and do not push. You may run focused read-only
inspection and tests. Do not read other reviewers' outputs. Treat missing
evidence as missing; do not infer that another agent checked it.

Use C:\Users\dmishra\Desktop\sperm_project\.venv\Scripts\python.exe for Python tests; system Python is not
the validated project environment. Prefer the commit-bound evidence and
validation receipt under audits over rerunning expensive image inference.

# Visual Evidence Reviewer

Review generated images as evidence, not decoration.

Required checks:

- Compare raw images, probability maps, filled masks, centerlines, instance
  boundaries, tracks, and final overlays at identical framing.
- Confirm colors, symbols, legends, and line thicknesses do not alter or obscure
  the measured geometry.
- Look for missed nuclei, merged neighbors, false splits, boundary leakage,
  cropped panels, unreadable labels, and unrepresentative examples.
- Confirm the same track keeps the same identity across planes and figures.
- Require examples covering faint, bright, curved, short, long, touching, and
  irregular objects.
- Do not infer correctness from a summary chart alone.

Return only JSON conforming to `audits/review_schema.json`.


CLAIM SNAPSHOT:
{
    "claim_id":  "PIPELINE-V571-PRODUCTION-001",
    "title":  "Saturn v5.7.1 is a technically defensible production candidate",
    "status":  "accepted",
    "risk":  "high",
    "implementation_owner":  "primary_implementation_agent",
    "statement":  "The v5.7.1 dual-head U-Net-primary pipeline, calibrated measurement path, morphology-neutral global tracking, and biologist-facing reporting operate consistently across GUI, batch, tuner, and study workflows without using WT morphology as a technical acceptance target.",
    "biological_meaning":  "Technical-valid nuclei and tracks remain measurable across plausible WT and mutant morphology, with provenance sufficient for specimen-level comparisons.",
    "population":  "technical_valid detections and reconstructed nuclei",
    "units":  [
                  "um",
                  "um2",
                  "um3",
                  "count"
              ],
    "calibration_dependencies":  [
                                     "Leica XY pixel size",
                                     "Leica Z spacing"
                                 ],
    "source_dependencies":  [
                                "sperm_segmentation_saturnv5.7.1.py",
                                "utils/tune_parameters_Saturnv5_7_1.py",
                                "utils/saturn_unet25d_bridge.py",
                                "production_profiles/saturn_v5_7_1_model_c_epoch003.json",
                                "model_checkpoints/v571_model_c_dual_head_epoch003.pt"
                            ],
    "implementation_evidence":  [
                                    "tests/test_saturn_v571_dual_head.py",
                                    "tests/test_saturn_v571_body_width.py",
                                    "V5_7_1_VALIDATION_REPORT.md",
                                    "audits/V5_7_1_DESIGN_DECISIONS.md",
                                    "audits/evidence/v571_rc6_candidate/stages/visual_evidence_manifest.json",
                                    "audits/evidence/v571_rc6_candidate/tracking/tracking_replay_manifest.json",
                                    "audits/evidence/v571_rc6_candidate/end_to_end/end_to_end_visual_evidence_manifest.json",
                                    "audits/evidence/v571_rc6_candidate/provenance/acceptance_provenance_manifest.json",
                                    "audits/evidence/v571_rc6_candidate/report/report_source_binding.json",
                                    "audits/evidence/v571_rc6_candidate/report/01_biological_results/data/report_consistency_validation.json",
                                    "audits/evidence/v571_rc6_candidate/report/02_quality_control/data/specimen_sensitivity_artifact.json",
                                    "scripts/rebuild_v571_acceptance_report_from_replay.py",
                                    "tests/test_v571_acceptance_report_replay.py",
                                    "audits/validation/v571_6f9e73b_validation.md"
                                ],
    "acceptance_criteria":  [
                                "Calibration is resolved before physical thresholds and measurements",
                                "Dual-head checkpoint identity and SHA-256 are enforced",
                                "Morphology warnings do not become comparative technical vetoes",
                                "Tracking has no duplicate Z observations and rejects impossible joins without deleting source detections",
                                "Primary reports agree with source tables and use specimen-level biological terminology",
                                "All supported execution entry points use the same production semantics",
                                "Automated and adversarial validation covers measurement, tracking, reporting, and failure paths"
                            ],
    "required_roles":  [
                           "measurement_geometry",
                           "biological_validity",
                           "calibration_provenance",
                           "software_reproducibility",
                           "statistics_reporting",
                           "visual_evidence",
                           "repository_release"
                       ],
    "known_limitations":  [
                              "Below-2-um reconstructed tracks can include genuine short nuclei, optical tips, split fragments, or noise; their specimen-level influence is reported automatically without excluding them from the primary population",
                              "Body width is apparent mask width, not PSF-corrected physical chromatin width",
                              "Existing repeated 2D ROIs cannot establish anatomical SV volume"
                          ],
    "non_claims":  [
                       "The pipeline does not establish a biological KJ-versus-WT effect from one pilot specimen per group",
                       "The pipeline does not currently measure anatomical SV volume from a 3D organ mask"
                   ],
    "supersedes":  [

                   ],
    "latest_audit":  {
                         "run_id":  "20260827-v571-remediation-acceptance-rc7",
                         "gate_passed":  true,
                         "decision":  "accepted"
                     }
}

PROJECT DECISION CONTEXT:
# Saturn v5.7.1 Design Decisions

This ledger records the intended production semantics that reviewers must test.
It explains why a behavior exists; it does not override contradictory evidence
or excuse a defect.

## Comparative biological population

- The primary WT-versus-mutant population is `technical_valid`.
- Short, long, wide, thin, curved, tortuous, irregular, and single-slice
  morphology remains measurable and may receive a morphology warning.
- WT-like length, width, ratio, count, or shape must not be an optimization
  target or a technical acceptance rule.
- A 15-20 um object is retained with a review annotation. Length above 20 um is
  not sufficient evidence to delete or split an object. Objective fusion or
  merge evidence is required before a technical intervention.

## U-Net-primary segmentation

- The dual-head U-Net is the primary segmentation source. Classical morphology
  may annotate supported instances but must not veto them for unusual shape.
- Foreground probability defines supported mask extent; learned core components
  provide independent separation evidence for touching objects.
- Instance splitting must not be driven solely by a desired biological length.
- The original filled parent mask, parent identity, and split evidence must
  remain auditable whenever an objective split is applied.

## Cross-slice tracking and gaps

- A short 2D observation may be the optical tip of a valid multi-plane nucleus
  and remains eligible for tracking.
- One missing Z plane may be bridged when calibrated position, motion, and
  overlap/support evidence are compatible. This handles a faint or missed
  intermediate optical section.
- Gap linking does not invent a 2D detection, mask, area, or width on the
  missing plane. Observed-mask volume sums observed masks only.
- Single-slice tracks remain valid because specimen orientation and Z spacing
  can make a complete nucleus visible primarily in one plane.
- A proposed impossible join is rejected without deleting its original 2D
  detections.

## Signal-profile width and technical area/volume diagnostics

- The mask boundary reproduces the training annotation convention rather than the
  nucleus. Annotations in this project have a median width of 1.606 um while the
  optical width of a nucleus is about 0.643 um, so the learned mask is roughly
  2.4 times too wide. The bias is not constant: it was 2.32x in KJ-01 against
  2.51x in WT-01 and rises with brightness, which is larger than the 0.169 um
  width difference between those specimens. Mask width therefore cannot carry a
  genotype comparison.
- Primary width is the half-maximum extent of the background-corrected intensity
  profile along centerline normals. It is independent of where a boundary was
  drawn: inflating a mask from 1.9 to 6.4 um leaves it unchanged at 0.797 um.
- Object-owned integrated profile signal is recorded in arbitrary units as
  technical QC only. There is no independent staining reference in this study;
  same-genotype specimens are biological replicates, not staining controls.
  Integrated signal therefore cannot be a primary endpoint, acceptance gate, or
  chromatin-content measurement.
- Centerline length times signal FWHM is retained as an explicitly named
  signal-profile footprint proxy. It is not a filled-mask area and is not an
  anatomical volume. Missing profile widths remain missing and never fall back
  to mask area under that name.
- Filled-mask area and the corresponding observed-slice mask slab sum retain
  their historical definitions and explicit names for reproducibility. They are
  technical, segmentation-sensitive diagnostics rather than primary biological
  morphometry.
- Dilation is never used in a measurement path. Profile background is read from
  the far tails of the same profile, taking the quieter side, because a dilated
  ring would let the chosen radius set the background, the half maximum and
  therefore the width. Dilation remains acceptable only for display overlays.
- Merges are detected from the profile: a cross-section through two filaments is
  bimodal. Peak counting is restricted to pixels inside the instance mask so a
  neighbouring nucleus is not mistaken for a merge.
- Absolute nucleus diameter is not established and must never be reported. At
  0.378 um per pixel against a 0.23 um point spread function the image is 3.3
  times below Nyquist, and the deconvolved estimate saturates near 0.6 um.
  Comparison between groups is supported; an absolute width claim is not. This
  caveat travels with the metric wherever it is calculated or plotted.

## Measurements

- Primary length follows the final instance-mask centerline and remains separate
  from centroid trajectory and legacy fields.
- Primary comparative width uses background-corrected raw-signal FWHM along
  centerline normals. It is an apparent optical signal width, not an absolute or
  PSF-corrected molecular diameter.
- A reconstructed track receives its representative FWHM width and paired
  centerline length from the technically valid observed plane with the largest
  filled-mask area. Missing width remains unavailable; it is not fabricated
  from a gap or replaced by a mask width.
- Subpixel mask-contour chord width and distance-transform width remain
  explicitly named technical diagnostics. They do not drive biological reports.

## Merge evidence and multiple comparison groups

- Objective evidence that one mask holds several objects is branching of the
  instance centerline, or a bimodal intensity profile across it. Length is not
  merge evidence and never triggers the flag on its own.
- Three or more branch nodes is the threshold, so a single skeletonisation spur
  does not reclassify a valid nucleus. The previously documented case of any
  branching above the twenty micron review length is retained, so the rule only
  ever adds evidence and never unflags a case that was flagged before.
- The two evidence sources are complementary rather than redundant. Branching
  catches filaments joined end to end; profile bimodality catches nuclei lying
  side by side, which are unbranched yet still two objects.
- A study carries one reference group and one or more comparison groups. Every
  comparison is contrasted against the single reference, and direction comes only
  from manifest roles, never from group names.
- Two multiple-testing families are reported because neither subsumes the other.
  The headline q-value corrects across the metrics tested within one contrast,
  which is the historical family and is unchanged for a two-group study. A second
  q-value corrects across the comparison groups tested for one metric, and
  becomes meaningful when a study gains a mutant or a rescue line. Reporting only
  the latter would have silently removed the across-metric correction in a
  two-group study and made every q-value smaller.

## Calibration and ROI

- Leica calibration must be resolved before any physical threshold or
  measurement is applied.
- Organized filenames may differ from Leica source names, so the retained
  manifest-to-XML mapping is authoritative and must be hashed and archived.
- Every run archives the exact ordered source files, applied ROI/exclusion mask,
  profile, checkpoint identity, and resolved calibration.
- A repeated 2D ROI supports sampled-area normalization. It does not establish
  anatomical seminal-vesicle volume; that requires slice-specific 3D organ
  masks.

## Reporting and review burden

- Biological reports show one primary technical-valid population and
  specimen-level outcomes. Internal rescue lanes and detailed audit categories
  belong in technical QC, not the main PDF.
- Individual nuclei are nested observations. Biological inference uses
  specimens as replicates and is unavailable when group sample size is
  insufficient.
- Detailed visual evidence exists for software validation and adversarial audit;
  it must not become a routine manual-review queue for the biologist.


Treat the decision context as intended behavior to verify, not as proof that
the implementation is correct. Report any mismatch between intent and code.

Your final response must be JSON matching audits/review_schema.json. Use the
exact audit_run_id, claim_id, role, and reviewed_commit above. Every pass/fail
check and every finding must cite concrete evidence such as path:line, a test
name, a command result, or a generated artifact path. A conditional verdict
does not pass the acceptance gate.
