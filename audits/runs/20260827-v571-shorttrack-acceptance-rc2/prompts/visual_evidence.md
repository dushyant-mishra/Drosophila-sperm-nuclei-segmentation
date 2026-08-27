You are the independent Saturn reviewer for role: visual_evidence.

Audit run: 20260827-v571-shorttrack-acceptance-rc2
Claim: POP-SHORTTRACK-001
Reviewed Git commit: 22d29621230d31e2cf812ec28bec680d984ad2de
Working tree mode: acceptance_candidate

Read the repository and evaluate the claim. Do not edit files, do not run a
full image stack, do not commit, and do not push. You may run focused read-only
inspection and tests. Do not read other reviewers' outputs. Treat missing
evidence as missing; do not infer that another agent checked it.

Use C:\tmp\saturn-v571-focused-22d2962\.venv\Scripts\python.exe for Python tests; system Python is not
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
    "claim_id":  "POP-SHORTTRACK-001",
    "title":  "Below-2-um tracks remain measurable with automated sensitivity analysis",
    "status":  "implemented",
    "risk":  "high",
    "implementation_owner":  "primary_implementation_agent",
    "statement":  "Below-2-um technical-valid tracks remain in the primary biological population; an automated specimen-level sensitivity calculation reports how counts and summary morphometry would change if they were omitted, without creating a second accepted population or a routine manual-review queue.",
    "biological_meaning":  "Preserves genuine short nuclei and terminal optical sections while making potential fragment sensitivity transparent and reproducible.",
    "population":  "technical-valid reconstructed tracks below 2 um",
    "units":  [
                  "um",
                  "count",
                  "fraction"
              ],
    "calibration_dependencies":  [
                                     "Leica XY pixel size",
                                     "Leica Z spacing"
                                 ],
    "source_dependencies":  [
                                "raw Z context",
                                "U-Net heads",
                                "instance masks",
                                "tracking audit"
                            ],
    "implementation_evidence":  [
                                    "sperm_segmentation_saturnv5.7.1.py",
                                    "tests/test_saturn_v571_dual_head.py",
                                    "scripts/generate_specimen_sensitivity_artifact.py",
                                    "tests/test_specimen_sensitivity_artifact.py",
                                    "scripts/rebuild_v571_acceptance_report_from_replay.py",
                                    "tests/test_v571_acceptance_report_replay.py",
                                    "audits/evidence/v571_rc6_candidate/report/02_quality_control/data/below_2_um_specimen_sensitivity.csv",
                                    "audits/evidence/v571_rc6_candidate/report/02_quality_control/data/below_2_um_specimen_sensitivity.json",
                                    "audits/evidence/v571_rc6_candidate/report/02_quality_control/data/specimen_sensitivity_artifact.json",
                                    "audits/validation/v571_6f9e73b_validation.md"
                                ],
    "acceptance_criteria":  [
                                "Primary technical-valid population is preserved",
                                "Short 2D observations remain eligible for morphology-neutral cross-slice linking",
                                "Specimen-level count and morphometry sensitivity is computed automatically",
                                "The sensitivity calculation is stored under technical QC and does not create a second user-facing biological population"
                            ],
    "required_roles":  [
                           "measurement_geometry",
                           "biological_validity",
                           "software_reproducibility",
                           "statistics_reporting",
                           "visual_evidence"
                       ],
    "known_limitations":  [
                              "Length alone cannot distinguish a genuine short nucleus from an unresolved fragment"
                          ],
    "non_claims":  [
                       "Length below 2 um is not automatically technical noise"
                   ],
    "supersedes":  [

                   ]
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

## Measurements

- Primary length follows the final instance-mask centerline and remains separate
  from centroid trajectory and legacy fields.
- Primary apparent body width uses subpixel perpendicular contour chords after
  endpoint trimming. Legacy distance-transform width remains explicitly
  labelled and must not drive biological reports.
- A reconstructed track receives representative width from the technically
  valid observed plane with the largest filled-mask area. Missing width remains
  unavailable; it is not fabricated from a gap.
- Width is apparent mask width and is sensitive to segmentation boundary,
  annotation thickness, focus, and lateral PSF. It is not a deconvolved
  molecular diameter.

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
