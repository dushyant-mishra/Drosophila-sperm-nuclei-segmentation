# Saturn v5.7.1 handover state (2026-09-14, extended through 2026-09-17)

Peer agents: Claude and Codex alternate implementation and independent audit.
This file records what is done, what is verified, and what remains, so either
agent can take over without replaying the conversation. Everything below was
implemented by Claude, so under the `AGENTS.md` rule that an implementing agent
cannot be the sole validator of its own high-risk claim, all of it needs an
independent reviewer.

## Where to start, in order of risk

| What | Section | Why it needs a look |
|---|---|---|
| Merge flagging and splitting | Merges counted as single nuclei | Changes `estimated_unique_nuclei`, a primary biological metric, against an already accepted claim. Needs a superseding run. |
| Area and volume from profile width | Area and volume derived from the profile width | Replaces mask-derived values outright rather than keeping them as legacy fields, which departs from `AGENTS.md` with owner authorisation. |
| Intensity-profile width | Completed and verified since the Codex audit | The primary width now presented biologically. Registered as `MEAS-INTENSITY-WIDTH-001`, never audited. |
| Classical body width by chord | Classical body width measured by chord | Removes a silent legacy fallback from unqualified width fields. |
| Multi-group study design | Phase 3 pipeline side, then Phase 3 report side | Changes a group-comparison estimand and the BH family. |
| Production gate hardening | Gate and GUI hardening | A gate that raised instead of returning a verdict was not fail-closed. |
| The v5 illustrated document | Illustrated workflow document v5 | Needs editorial and provenance review, not a measurement audit. |

Findings raised and remediated, each with its own record under
`audits/findings/`: the merge-flag length gate (2026-09-14), the stratified
evidence being archive-bound (2026-09-15), and figures asserting what their
captions claimed (2026-09-16).

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

## Classical body width measured by chord: DONE (2026-09-08), validated

Commit `982f038`, with geometry validation in `646af0d`.

`rows_from_results` previously wrote the quantized distance-transform median
into the unqualified `width_px`, `width_um` and `length_width_ratio` whenever a
classical detection had no contour chord, so a consumer reading "width" could
receive either definition without being told which. The subpixel contour-chord
measurement now covers classical detections too, and the fallback is gone:

```python
primary_width_px = body_width_px if body_width_available else np.nan
```

A filled mask holding more than one pruned centerline, a degenerate mask, or a
disabled measurement now report the width as unavailable with a stated reason
rather than substituting the legacy value. The slender-area estimate follows the
same primary width, so it can no longer mix a chord length with a
distance-transform width, and the legacy back-fill no longer relabels a chord
value as distance-transform legacy.

Verified against known geometry rather than only against the existing suite.
`scripts/validate_v571_body_width.py` now drives `measure_spermatids` over
rotated rectangles of known width through the classical path, not just the chord
kernel on bare masks. The classical path clears the bar the U-Net path already
met: maximum absolute error 0.435 px and maximum rotation spread 0.441 px across
widths of 5, 9 and 13 px at five orientations. On a synthetic seven-pixel rod the
chord recovers 7.0 px where the legacy median quantizes to 8.0 px. A filled
component holding two centerlines is asserted to refuse a merged width.

## Signal-profile footprint kept separate from mask volume: CURRENT STATE

Corrected 2026-09-17 after Codex challenged this section in `AGENT_CHANNEL.md`
message [002]. The previous text described an intermediate implementation that
no longer exists and is preserved as superseded below.

**What the code does now.** Mask-derived and profile-derived quantities are
separate, explicitly named, and neither falls back to the other:

- `sperm_segmentation_saturnv5.7.1.py:6677` computes
  `observed_slice_mask_volume_um3` from the filled-mask pixel sum.
- `:6680-6685` computes `observed_slice_profile_footprint_proxy_um3` from
  centerline length times signal width, with
  `profile_footprint_method = "sum_length_times_signal_fwhm_observed_slices_no_fallback"`.
- `:6895-6897` keeps `volume_um3` as an alias of the observed-slice **mask**
  volume, with
  `volume_method = "sum_filled_mask_area_observed_slices_no_interpolation"`.
- `tests/test_saturn_v571_intensity_width_contract.py:115-145` asserts the
  separation and the absence of a fallback.
- `audits/V5_7_1_DESIGN_DECISIONS.md:59-65` states this contract correctly and
  always did.

`rows_from_results` also emits `profile_area_px`, centerline length times profile
width, which the measurement stage had been computing and the row builder had
been dropping.

**Nothing here departs from the preservation rule in `AGENTS.md`.** Filled-mask
area and the observed-slice mask slab sum retain their historical definitions
under explicit names as technical diagnostics. Which quantity reaches a
biological table is decided by the metric profile, not by deletion: the v5.7.1
entrypoint forces `concise_v571`, whose biological set excludes both mask width
and mask volume.

### Superseded intermediate state, commit `a667696` only

For the record, because the earlier version of this section described it as
current and a reviewer may meet it in the history. `a667696` did replace
`volume_area_px` with a profile-derived area that fell back to the mask pixel
count where no profile width existed, and its commit message says so. The very
next commit, `7160c32`, restructured that into the separated, no-fallback
contract above while addressing Codex's profile-isolation audit. No released or
audited state ever carried the fallback.

The measurement that motivated the change stands and is still worth knowing:
summing mask pixels inflates volume by 2.10x on KJ-01 planes 34 to 36, because
the mask boundary follows the training annotation convention rather than the
nucleus. That is why mask volume is a diagnostic rather than biological
morphometry, not why it was removed; it was not removed.

**How this error happened,** since it bears on how the rest of this document
should be read: the section was written from the commit message of `a667696`
rather than from the code as it stands. Commit messages describe a moment. Any
remaining statement in this handover should be checked against current source
before it is relied on.

## Stratified visual evidence for body width: DONE (2026-09-08)

Commit `b50347f`, closing the third blocker from
`20260828-v571-body-width-acceptance-rc2`.

The generator rendered a single cleanest track per specimen, chosen by excluding
branched centerlines, morphology warnings, suspected merges and anything outside
a narrow area and ratio band. Evidence curated to well-behaved objects cannot
falsify a measurement. Selection is now one exemplar per category: clean,
morphology warning, branched centerline, suspected merge, width unavailable,
short track, and the narrowest and widest measured widths. A category with no
eligible track is recorded as absent rather than quietly skipped, and a track
that fresh segmentation does not reproduce is recorded rather than aborting the
run.

Coverage went from two tracks and four panels to sixteen tracks and thirty-two
panels across both specimens, spanning widths from 0.379 to 6.118 um.

The widest exemplar is itself a finding for review: WT-01 track 2372 at 6.118 um
is several overlapping filaments in one filled mask and was **not** flagged as a
suspected merge, so the upper tail of the width distribution can contain
unflagged merges. That observation is what later led to the merge-flag finding
below.

Note the limitation recorded separately in
`audits/findings/2026-09-15-stratified-evidence-is-archive-bound.md`: this
generator takes its numbers from a frozen replay archive and re-segments only to
draw masks, so it cannot be refreshed by re-running it.

## Unattended reporting, and the PDF error the owner kept seeing: FIXED (2026-09-08)

Also commit `982f038`. The owner reported a recurring "PDF Report failed to
generate completely" dialog during runs.

Two distinct defects sat behind it, both in paths reachable from batch and
multi-sample study runs:

- **Modal dialogs in unattended runs.** The Excel, PDF and PowerPoint generators
  raised `messagebox` warnings unconditionally. In an overnight cohort run that
  blocks until somebody dismisses it, and in a test run it pushes dialogs onto
  the operator's screen. `notify_report_warning` now prints the warning always
  and shows a modal only when a Tk root already exists.
- **A viewer lock discarding a completed report.** On Windows an open PDF viewer
  holds an exclusive lock, and matplotlib opens the file lazily on the first
  `savefig`, so the `PermissionError` surfaced deep inside rendering and threw
  away the report of an analysis that had already finished.
  `resolve_writable_pdf_path` probes the target first and falls back to a
  timestamped sibling. A missing parent directory is deliberately not treated as
  a lock; it is left to raise with its own context.

## Audit evidence images were being dropped by .gitignore: FIXED (2026-09-14)

Commit `0a0011c`. `.gitignore` excluded `*.png` with only a
`docs/readme_assets` exception, so every figure written into `audits/evidence`
was dropped by a plain `git add` and only the manifests describing them were
committed. Fifty-four evidence images from earlier work are tracked, so they must
have been force-added individually, which is why the omission went unnoticed.

This was not cosmetic. `visual_evidence` is a required review role for both width
claims, the manifests reference each figure by SHA-256, and
`scripts/validate_v571_evidence_provenance.py` checks those hashes against git
blobs, so an auditor cloning the repository would have found manifests describing
figures that were not there. Negations for `audits/evidence/**/*.png` and
`audits/runs/**/*.png` prevent a recurrence, and the sixty missing figures were
committed. Verified before committing that all 104 manifest-referenced artifacts
resolve and every recorded SHA-256 matches the file on disk.

## Test coverage added alongside this work

The suite is 466 passing tests. Ten files were added, 88 test functions, each
covering a path that previously had none:

| File | Tests | Covers |
|---|---:|---|
| `test_saturn_v571_classical_body_width.py` | 9 | chord routing on the classical path, refusal instead of legacy fallback |
| `test_saturn_v571_intensity_width_contract.py` | 5 | the emitted intensity-width fields and their unavailable reasons |
| `test_v571_width_availability_bias.py` | 9 | the cohort availability-bias validator |
| `test_saturn_v571_merge_evidence.py` | 13 | branch-node and bimodal merge evidence, and the additive rule |
| `test_saturn_v571_multigroup_design.py` | 14 | one reference and N comparisons, and both BH families |
| `test_saturn_v571_gate_hardening.py` | 16 | the production gate against every malformed registry shape |
| `test_saturn_v571_overlay_read_only.py` | 6 | overlay review cannot mutate state, asserted behaviourally |
| `test_saturn_v571_pdf_report_locking.py` | 5 | the viewer-lock fallback |
| `test_saturn_v571_report_notifications.py` | 4 | no modal dialogs without an interactive session |
| `test_v571_validation_report_currency.py` | 7 | the validation report against the repository's own records |

Where a test guards a fix, it was checked by reintroducing the defect and
confirming the test fails: 8 of the 20 gate-hardening assertions and 6 of the 7
validation-report checks fail against the pre-fix code.

## Availability bias: NO DIFFERENCE DETECTED, equivalence not established

Corrected 2026-09-17 after Codex challenged this section in `AGENT_CHANNEL.md`
message [002]. It previously read `PASSED` and concluded the bias was "shared and
equal". Two non-significant Welch tests cannot support that: absence of a
detected difference is not evidence of equivalence, and at n=18 against n=17 this
design has limited power.

`scripts/validate_v571_width_availability_bias.py` over all 35 specimens, three
sampled planes each, 22381 detections. Evidence is bound into the claim under
`audits/evidence/v571_width_availability_bias_20260914/`.

Intervals recomputed from the per-specimen CSV on 2026-09-17, because the
p-value alone does not say what the data exclude:

| | KJ (n=18) | WT (n=17) | difference | 95% CI on the difference | Welch p |
|---|---:|---:|---:|---|---:|
| width-unavailable fraction | 0.2121 | 0.2211 | -0.90 pp | -2.73 to +0.93 pp | 0.325 |
| mask-width selection bias | -0.540 um | -0.563 um | +0.024 um | -0.023 to +0.070 um | 0.305 |

What can honestly be said:

- No differential availability was detected between the groups. The data are
  consistent with anything from KJ being 2.7 percentage points lower to 0.9
  points higher.
- Equivalence is **not** established. This design can only detect a difference
  of roughly 2.5 percentage points at 80 percent power, so a smaller real
  imbalance would not have shown up here.
- What the interval does support: any materiality margin of about plus or minus
  3 percentage points or wider is consistent with this evidence. A tighter
  margin is not.
- Overall unavailable fraction 21.4%, of which 76.9% is boundary clipping,
  16.2% short centerline, 6.9% insufficient profiles.
- The selection bias is negative in every one of the 35 specimens, so the dropped
  objects are consistently narrower than the measured ones and the measured
  subset skews wide in both groups. How closely the two groups match is bounded
  by the interval above, not established as equal.
- Availability correlates only weakly and negatively with crowding
  (Spearman -0.26) and density (-0.36), so denser specimens do not lose more.

**The acceptance criterion was undefined and has now been narrowed.** Criterion
10 of `MEAS-INTENSITY-WIDTH-001` read "The fraction of objects with unavailable
width does not differ materially between compared groups". Nobody had defined
"materially", so it could not be judged either way. The owner narrowed it on
2026-09-17 rather than declare an equivalence margin. It now requires that **no
differential availability is detected**, with the confidence interval on the
difference reported alongside the test so the detectable effect size is visible.

That is a weaker guarantee than equivalence and is recorded as such in the
claim's `known_limitations`: a real imbalance smaller than about 2.5 percentage
points would not have been caught by this design. The evidence above satisfies
the criterion as it now reads. If a future study needs equivalence rather than
absence of detection, it needs a declared margin and more sampling, not a
reinterpretation of this evidence.

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

## Illustrated workflow document v5: DONE (2026-09-16), reviewable

`Saturn_V5.7.1_Illustrated_Technical_Workflow_v5.docx` replaces the v4 document.
It is generated, never hand-edited, by exactly two files:

- `scripts/build_v571_workflow_v5_figures.py` renders every figure into
  `docs/v5_7_illustrated_workflow/figures_v5/` and writes `figure_manifest.json`.
- `scripts/build_v571_workflow_v5_document.py` assembles the tracked `.docx`;
  the section text lives inline in that script.

Figures must be built first, because the document builder raises
`FileNotFoundError` on a missing figure. The figure build takes about six
minutes on CPU; close the document in Word first or the save fails with
`PermissionError`.

`docs/v5_7_illustrated_workflow/README.md` is the audit entry point. It names the
inputs, the build commands, and the four checks a reviewer should run. The
figures stay git-ignored with the rest of the microscopy-derived imagery, so a
reviewer regenerates them and compares the SHA-256 digests the manifest records,
alongside the git commit and the digests of the pipeline and production profile
the figures were built from. The manifest also records, per figure, whether its
numbers are recomputed live from a named specimen and plane or transcribed from a
named record under `audits/`.

`docs/plans/2026-09-15-v5-workflow-document-progress.md` records what changed
from v4 figure by figure, and which claims in the text were checked against the
code before they were written. Several first drafts were wrong and are recorded
there, notably that area similarity contributes to joining, which it does not
because `ASSIGNMENT_LENGTH_WEIGHT`, `ASSIGNMENT_WIDTH_WEIGHT` and
`ASSIGNMENT_AREA_WEIGHT` are zero whenever `ANALYSIS_MODE` is comparative.

`audits/findings/2026-09-16-v5-document-figure-caption-audit.md` records a defect
class worth carrying forward: a figure script exiting zero proves only that a
file was written, not that the picture supports its caption. One nucleus was
being painted onto all five planes of the joining figure and then hidden behind
the plane above, and the caption's stated depth exaggeration was wrong by a
factor of two because `set_box_aspect` renormalises the z axis. Both are fixed by
construction rather than by tuning.

No pipeline measurement changed for any of this, so no entry in
`audits/claims_registry.json` is affected and no claim state moved. What the
document needs is an editorial and provenance review rather than a measurement
audit: that every figure supports its caption, that every plotted number traces
to the record the manifest names, and that the absolute-versus-comparison
classification in section 7 matches what the code computes.

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

## Registry and gate state as of 2026-09-17

Verified live, not transcribed from memory:

```
PIPELINE-V571-PRODUCTION-001     accepted      latest_audit accepted
MEAS-BODY-WIDTH-001              implemented   latest_audit not_accepted
MEAS-INTENSITY-WIDTH-001         implemented   never audited
REPORT-BIOLOGIST-CONCISE-001     implemented   never audited
WORKFLOW-GUI-PRIMARY-001         implemented   never audited
POP-SHORTTRACK-001               implemented   never audited
WORKFLOW-MANUAL-CORRECTION-001   proposed      never audited, disabled by design
VOL-3DROI-001                    proposed      never audited, unimplemented
```

`production_audit_gate_state(Path('.'))` returns:

```
(False, 'Required scientific claims are not accepted: MEAS-BODY-WIDTH-001:
implemented (not_accepted); MEAS-INTENSITY-WIDTH-001: implemented (not audited);
REPORT-BIOLOGIST-CONCISE-001: implemented (not audited); WORKFLOW-GUI-PRIMARY-001:
implemented (not audited)')
```

The gate is correctly closed. `PIPELINE-V571-PRODUCTION-001` still reads
`accepted`, but the behaviour behind it has changed since that acceptance:
merge flagging and splitting now alter `estimated_unique_nuclei`, and area and
volume are derived from the profile width rather than mask pixels. That claim
therefore needs a superseding run rather than resting on the existing verdict.

## Verification commands

```powershell
python -m pytest -q --basetemp=<writable-dir>   # 466 passing; bare pytest gives
                                                # 122 spurious WinError 5 errors
python scripts/validate_v571_body_width.py      # must exit 0
python -c "import sys,pathlib; sys.path.insert(0,'utils'); import saturn_v571_gui_services as s; print(s.production_audit_gate_state(pathlib.Path('.')))"
```
