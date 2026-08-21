# v5.7.1 Independent Production Diagnostic

Claim: `PIPELINE-V571-PRODUCTION-001`

Reviewed commit: `d8c498ca3c76b17ce76c02dc85bed5d9ac542d0b`

Mode: pre-commit diagnostic

Decision: **blocked pending remediation**

## Verdicts

| Reviewer | Verdict | Confidence |
| --- | --- | --- |
| Measurement and geometry | Fail | High |
| Biological validity | Fail | High |
| Calibration and provenance | Fail | High |
| Software and reproducibility | Fail | High |
| Statistics and reporting | Conditional | High |
| Visual evidence | Fail | High |
| Repository and release | Fail | High |

## Confirmed strengths

- The subpixel central-body chord-width implementation is traceable,
  rotation-tested, separated from legacy EDT width, and returns unavailable
  rather than inventing widths when geometry is insufficient.
- The production checkpoint SHA-256 matches the profile and is enforced.
- Primary between-group statistics use specimens rather than pooled nuclei and
  include effect sizes, uncertainty, and multiplicity correction.
- Most short, wide, thin, curved, tortuous, and single-slice morphologies are
  retained as warnings rather than rejected for being non-WT-like.
- Pilot report values agree with their source tables within serialization
  precision.

## Blocking themes

1. **3D geometry:** reported 3D tortuosity does not use one consistently
   reconstructed 3D path, and synthetic known-geometry validation is missing.
2. **Biological preservation:** length above 20 um currently becomes a
   technical rejection based on length alone; it should trigger evidence-based
   merge/fusion review without automatically deleting plausible morphology.
3. **Calibration provenance:** production permits fallback calibration and does
   not hash/copy all source metadata, image, and ROI identities needed to
   reproduce the result.
4. **Entry-point consistency:** the study launcher and GUI tuner still route to
   v5.7 paths, while unprofiled v5.7.1 launches default to classical behavior.
5. **Visual validation:** retained examples do not show identical-frame raw,
   foreground/core probabilities, masks, centerlines, boundaries, and tracks
   across representative easy and difficult objects.
6. **Release evidence:** the reviewed test claims and ignored pilot artifacts
   are not yet bound to a clean, immutable release commit.

## Important non-blocking gaps

- The below-2-um sensitivity population remains unclassified.
- Primary body-width outcomes should be present in the principal specimen
  summary table, not only the technical-QC table.
- The group summary should expose metric-specific non-missing specimen counts.
- Model caching should include checkpoint content identity, not path alone.
- Import-time logging should not require a writable repository or home folder.

The complete evidence and recommendations are in the seven JSON files under
`reviews/`. No production claim was accepted by this diagnostic.
