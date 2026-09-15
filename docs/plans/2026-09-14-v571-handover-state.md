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

## Availability bias check: PASSED (2026-09-14)

`scripts/validate_v571_width_availability_bias.py` over all 35 specimens, three
sampled planes each, 22381 detections. Evidence is bound into the claim under
`audits/evidence/v571_width_availability_bias_20260914/`.

| | KJ (n=18) | WT (n=17) | Welch p |
|---|---:|---:|---:|
| width-unavailable fraction | 0.2121 | 0.2211 | 0.325 |
| mask-width selection bias | -0.540 um | -0.563 um | 0.305 |

- Overall unavailable fraction 21.4%, of which 76.9% is boundary clipping,
  16.2% short centerline, 6.9% insufficient profiles.
- The selection bias is negative in every one of the 35 specimens, so the dropped
  objects are consistently narrower than the measured ones and the measured
  subset skews wide, by a similar amount in both groups.
- Availability correlates only weakly and negatively with crowding
  (Spearman -0.26) and density (-0.36), so denser specimens do not lose more.
- A shared and equal bias does not distort a between-group comparison, so the
  acceptance criterion on this claim is satisfied on this evidence.

**Disclosure for audit.** That run also produced a specimen-level signal-width
group contrast, so a group difference was seen before the gate passed. It is a
technical readout on three sampled planes without tracking, where one nucleus
spanning several planes is counted more than once, so it is not a biological
result. No parameter, threshold, or gate has been changed since it was seen, and
none may be. A reviewer should treat any later biological result as independent
of it and should check that nothing was tuned in between.

## Visual evidence for the new width: DONE (2026-09-14)

`scripts/generate_v571_intensity_width_evidence.py`, twelve panels in
`audits/evidence/v571_intensity_width_visual_20260914/`, bound into the claim.
One exemplar per behaviour including refusals and a merge. The widest-mask
exemplar shows a 3.904 um mask chord against a 0.668 um signal width with three
peaks in one mask. Categories that cannot render a panel are recorded in the
manifest rather than dropped.

## Phase 3 pipeline side: DONE (2026-09-14), report side OUTSTANDING

`_study_group_design` returns a reference plus a list of comparison groups.
`_study_one_metric_contrast` was extracted so the fan-out needs no extra nesting.
The group-count guard accepts two or more groups.

Verified bit-identical to the previous implementation on a synthetic two-group
study across all nineteen shared columns, including permutation p-values. This
required keeping the permutation seed base so the first comparison group
reproduces the pairwise random stream; `seed_offset` is
`comparison_index * 10000 + metric_index`. Adding a third group leaves the
existing contrast's p-values and within-contrast q-values unchanged.

**Two BH families are reported and the primary one is not yet decided.**
`bh_fdr_q_value` corrects across the metrics within one contrast, which is the
historical family and is unchanged for a two-group study.
`bh_fdr_q_value_across_comparisons` corrects across the comparison groups within
one metric, which is the family requested for multi-group designs. Reporting only
the latter would silently remove the existing across-metric correction in a
two-group study and make every q-value smaller. The report generator must present
one of these as primary, so this needs an explicit decision before the report side
is written.

## Merges counted as single nuclei: REMEDIATED (2026-09-14), pending audit

`audits/findings/2026-09-14-merge-flag-length-gate.md`. The user spotted this on
the visual evidence panels.

`suspected_multi_object_merge` requires `geodesic_um > 20.0 AND branch_count > 0`.
Median instance length is 7.91 um, so the length gate almost never opens. On 559
instances, 7.33% are branched but only 0.18% are flagged, leaving 7.16%
objectively joined structures counted as one nucleus each. Three of the four
instances with eight or more branch nodes are missed.

Remediated with owner approval. Flagging now uses three or more branch nodes,
or profile bimodality, or the previously documented overlong-branched case,
which makes the rule purely additive. Splitting is triggered by objective
evidence rather than length, with branch-topology markers supplying the
evidence the learned core head cannot. Counts rose 12.7 percent in KJ-01 and
7.1 percent in WT-01 on plane 35, so the correction is not balanced between
groups and the old under-counting was differentially suppressing KJ.
`estimated_unique_nuclei` is a primary biological metric, so this still needs a
superseding run against the accepted `PIPELINE-V571-PRODUCTION-001`.

Group rates are close, 7.19% KJ against 7.12% WT, so a count comparison is less
distorted than the absolute count, but that balance is from one plane of one
specimen per group and must not be assumed cohort-wide.

## Phase 3 report side: DONE (2026-09-14)

`--comparison-group` takes several values. Rather than refactoring the figure
builders to merge contrasts, which would have put the validated single-contrast
report at risk, the driver renders that same report once per comparison into
`contrast_<group>/` and then writes `cross_contrast_statistical_tests.csv` plus
`CROSS_CONTRAST_README.md` at the root. Each contrast stays independently
reviewable.

Verified: for a two-group study, all 26 deterministic outputs are byte-identical
to the pre-change script. The only differing files are the PDF and Excel hashes,
and a control rerun of the *unchanged* script shows those are not byte
reproducible between any two runs, so the difference is inherent.

The cross-contrast table carries `*_bh_fdr_q_across_comparisons` for all three
test families the report already corrects: permutation, Mann-Whitney and Welch.
On a synthetic three-group study, KJ length moves from a within-contrast q of
0.0266 to an across-comparison q of 0.0390, as expected for two comparisons.

BH decision settled by the owner: the headline q-value stays the across-metric
family inside each contrast; the across-comparison family is reported alongside.

## Gate and GUI hardening: DONE (2026-09-15)

- `production_audit_gate_state` read `registry.get("claims")` outside the try
  guarding parsing and assumed `latest_audit` was a mapping, so a registry that
  parsed to a list, null, string or number raised `AttributeError` instead of
  returning a verdict. A gate that raises is not fail-closed. Both are
  type-checked now; 8 of 20 new tests failed before the fix.
- `MEAS-INTENSITY-WIDTH-001` added to `PRODUCTION_REQUIRED_CLAIM_IDS`, which
  previously named only the superseded mask-width claim, so the gate could have
  opened while the width actually presented biologically was unaudited.
- Overlay read-only is asserted behaviourally now, not by grepping a warning
  string: `on_click` is driven in review, view and ROI modes with a control
  proving the check can detect a mutation, plus an AST check that no GUI method
  calls a correction entry point.
- `correction_label_state_sha256` folded the array dtype into the digest, so the
  same labels as uint8 and int32 hashed differently and a correction could look
  like it changed state it did not. Canonicalised to int64; nothing stored
  depended on the old values.
- First failure coverage for `reduce_study_progress`.

## Validation report: corrected (2026-09-15)

`V5_7_1_VALIDATION_REPORT.md` still presented the mask chord as the primary
width, carried counts predating merge splitting, and reported 205 passing tests.
Superseded sections are marked in place rather than deleted, the width section is
rewritten, and the live gate state is stated at the top.
`tests/test_v571_validation_report_currency.py` guards it against going stale by
checking it against the repository's own records rather than fixed figures; 6 of
its 7 checks fail against the previous report.

## Open items, in order

1. The stratified body-width evidence cannot be refreshed by re-running its
   generator. See `audits/findings/2026-09-15-stratified-evidence-is-archive-bound.md`.
   It takes every number from a frozen replay archive last changed 2026-08-27 and
   re-segments only to draw masks, so regenerating on commit `48977a3` produced
   byte-identical output. The regenerated directory was deleted rather than
   committed, because a folder stamped with a current commit but carrying August
   numbers is worse than none. The durable fix is to have that generator measure
   from fresh segmentation the way the intensity-width generator already does.
   `MEAS-INTENSITY-WIDTH-001` evidence is unaffected and is current.
2. Independent acceptance audits on a clean commit. **Claude cannot run these.**
   The launcher requires the `codex` CLI, which is not installed on this machine,
   and more fundamentally `AGENTS.md` forbids an implementing agent from being the
   sole validator of its own high-risk claim. Claude implemented all of these, so
   the audits must be run by Codex or another independent reviewer. Readiness was
   verified instead: all five claims have required roles, all seven charters
   exist, every evidence path resolves, and zero are dead. The working tree must
   be clean at the time of the run or the launcher records `pre_commit` mode,
   which cannot pass the gate. Claims to audit: `MEAS-INTENSITY-WIDTH-001`,
   `MEAS-BODY-WIDTH-001`, `REPORT-BIOLOGIST-CONCISE-001`,
   `WORKFLOW-GUI-PRIMARY-001`, and a superseding run for the accepted
   `PIPELINE-V571-PRODUCTION-001` whose behaviour has changed.
3. Only then the 35-specimen cohort run: 18 KJ and 17 WT, excluding
   `w1118 sv feb 40xx0.75-15` which has no slices. Roughly 8 to 11 hours on CPU.

## Acquisition guidance given to the owner (2026-09-15)

For any future imaging, the Z step of 0.346 um is already correct against a
0.729 um axial FWHM. The limitation is lateral: 0.378 um per pixel against a
0.228 um lateral FWHM is 3.3 times below Nyquist, and a 0.5 um nucleus spans
1.3 pixels, which is why width saturates. Zoom 2.5 to 3.0 would reach Nyquist.
Also recommended: 12-bit rather than 8-bit, line averaging 2 to 4, pinhole held
at 1.0 Airy, and identical laser, gain and zoom across every specimen. A stable
second channel or a sub-resolution bead would lift the two limits software
cannot: integrated signal has no staining reference, and PSF correction is
unvalidated and therefore disabled.

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
python -m pytest -q --basetemp=<writable-dir>   # 466 passing; bare pytest gives
                                                # 122 spurious WinError 5 errors
python scripts/validate_v571_body_width.py      # must exit 0
python -c "import sys,pathlib; sys.path.insert(0,'utils'); import saturn_v571_gui_services as s; print(s.production_audit_gate_state(pathlib.Path('.')))"
```
