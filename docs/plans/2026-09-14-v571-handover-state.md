# Saturn v5.7.1 handover state (2026-09-14)

Peer agents: Claude and Codex alternate implementation and independent audit.
This file records what is done, what is verified, and what remains, so either
agent can take over without replaying the conversation.

## Completed and verified since the Codex audit

- **Codex's profile-isolation fix independently reproduced.** The pre-fix defect
  inflated FWHM from 2.90 px to 7.08 px when a brighter neighbour sat outside the
  instance mask. Re-running that case now yields
  `unavailable_boundary_clipped_profiles` for a neighbour at x=42.5..44 and
  2.909 px at x=46 against a 2.904 px clean baseline. The fix is correct: the
  half-maximum crossing walk and the integration are both gated on `inside`,
  where previously only the peak search and merge count were.
- **Real-data re-measurement with the hardened code** (planes 34-36):

  | | KJ-01 | WT-01 |
  |---|---:|---:|
  | detections | 870 | 784 |
  | signal FWHM width, median | 0.701 um | 0.730 um |
  | mask chord width, median | 1.514 um | 1.629 um |
  | mask / signal ratio | 2.12x | 2.29x |
  | profile merge flag | 3.4% | 5.4% |
  | width unavailable | 22% | 22% |

  Merge flags concentrate where merges are: every mask wider than 3 um is
  flagged in both specimens.
- **`MEAS-INTENSITY-WIDTH-001` registered** as `implemented`, `latest_audit:
  null`, high risk, five required roles, with acceptance criteria covering mask
  independence, neighbour isolation, the no-dilation rule, same-plane pairing,
  integrated signal staying QC-only, caveat propagation, and group balance of
  the unavailable fraction.
- **Production gate now requires that claim.** `PRODUCTION_REQUIRED_CLAIM_IDS`
  previously listed only the superseded mask-width claim, so the gate could have
  passed while the primary biological width was unaudited.
- **Interpretation caveat propagated** into the metric definitions sheet, the
  Excel README, the package README, `report_metadata.json`, and a new
  `metric_interpretation_limits.csv` sidecar, with tests enforcing it.

## Corrections to earlier Claude statements

- Claude reported a ledger/code mismatch, claiming mask volume was still a
  biological metric. That was wrong. `BIOLOGICAL_METRICS` is overridden at
  runtime by `--metric-profile`, and `scripts/generate_v571_biological_comparison.py`
  forces `concise_v571`, whose biological set excludes both mask width and mask
  volume. No fix was needed.
- Claude's earlier "26% -> 3.1% merge rate" was measured before the isolation fix
  and should not be cited. The current figures are 3.4% and 5.4% above.

## Open items, in order

1. Generate current synthetic and adversarial evidence artifacts for
   `MEAS-INTENSITY-WIDTH-001` and bind them into the claim's
   `validation_evidence`. The claim currently has none.
2. Check that the ~22% width-unavailable fraction does not differ by group at
   cohort scale. If boundary clipping correlates with packing density and packing
   differs by genotype, the measured subset is a biased sample. This is an
   acceptance criterion on the new claim.
3. Regenerate the stratified visual evidence, which currently predates the
   isolation fix.
4. Phase 3 of the approved plan: generalize the study design from one reference
   plus one comparison to one reference plus N comparisons, with
   Benjamini-Hochberg applied per metric across comparison groups.
5. Gate and GUI-services hardening (`WORKFLOW-GUI-PRIMARY-001`,
   `REPORT-BIOLOGIST-CONCISE-001`).
6. Independent acceptance audits on a clean commit for every gate claim.
7. Only then the 35-specimen cohort run: 18 KJ and 17 WT, excluding
   `w1118 sv feb 40xx0.75-15` which has no slices. Roughly 8 to 11 hours on CPU.

## Standing constraints

- No morphological dilation in any measurement path, including background
  estimation. Display overlays only.
- Width is comparative, never an absolute nucleus diameter, and the caveat must
  travel with the value into every calculation, table, figure label, and report.
- Integrated profile signal is technical QC. There is no staining control in this
  study; same-genotype specimens are biological replicates, not a calibration.
- No genotype name may be hard-coded. Group identity comes from the manifest.
- Nothing is pushed, merged, or tagged until the audit gate genuinely passes.

## Verification commands

```powershell
python -m pytest -q --basetemp=<writable-dir>   # 388 passing; bare pytest gives
                                                # 122 spurious WinError 5 errors
python scripts/validate_v571_body_width.py      # must exit 0
python -c "import sys,pathlib; sys.path.insert(0,'utils'); import saturn_v571_gui_services as s; print(s.production_audit_gate_state(pathlib.Path('.')))"
```
