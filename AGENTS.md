# Saturn Agent Review Rules

These rules apply to every coding or review agent working in this repository.

## Scientific invariants

- Treat the biological specimen, not each detected nucleus, as the independent
  unit for between-group inference.
- Resolve microscope calibration before applying any threshold or formula in
  physical units.
- Separate technical validity from morphology annotation. Unusual length,
  width, curvature, tortuosity, or slice span is not automatically invalid.
- Do not optimize WT and mutant results toward a shared expected morphology,
  count, or distribution.
- Preserve prior measurements as explicitly named legacy fields when a
  measurement definition changes.
- Never describe a repeated 2D ROI as anatomical 3D organ volume.

## Evidence rules

- Every scientific or measurement claim must have an entry in
  `audits/claims_registry.json`.
- Findings must cite concrete repository paths and line numbers, generated
  artifacts, test names, or reproducible commands.
- An implementing agent cannot be the sole validator of its own high-risk
  claim.
- High-risk claims require independent measurement, biological-validity, and
  software-reproducibility reviews. Claims affecting figures or group
  comparisons also require statistical-reporting review.
- A failed required review blocks acceptance. The coordinator records the
  decision but cannot overrule a blocking finding without new evidence and a
  new audit run.
- Generated audit records are append-only. Corrections create a superseding
  run; they do not rewrite an earlier verdict.

## Change discipline

- Do not modify frozen v5.7 behavior while implementing v5.7.1 or later work.
- Keep raw evidence, derived measurements, reports, and biological conclusions
  distinguishable.
- Tests must cover formulas, units, calibration, failure behavior, report field
  selection, and at least one adversarial or synthetic example.
- Repository publication is handled by the repository/release steward only
  after the audit gate passes and explicit user authorization is available.

See `audits/README.md` for the operating workflow.
