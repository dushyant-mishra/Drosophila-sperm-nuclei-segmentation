# Saturn v5.7.1 RC6 validation receipt

- Production code commit: `6f9e73b170db121460ab82f28542d1ed1e8f96bb`
- Validation date: 2026-08-27
- Profile: `production_profiles/saturn_v5_7_1_model_c_epoch003.json`
- Profile Git-blob SHA-256: `1fcc10c2199295d57210283255ead78cbe9ab51071121552a3668e5d3887d902`
- Pipeline Git-blob SHA-256: `1f56404b2bd683065bb229e3aa5b3579fbf3470d58563967c9db0efc12f572a1`
- Checkpoint SHA-256: `7d49031dbcce31f0600c44146d9b5282b0df6e28bb1fc1bdde6ef2146ed15d25`

Git-blob hashes are the canonical tracked-text identity. Working-copy hashes
are retained separately in the evidence manifests to document exact execution
bytes on Windows.

## Automated validation

```text
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .\scratch\pytest_full_<timestamp>
229 passed in 32.32s

.\.venv\Scripts\python.exe .\utils\tune_parameters_Saturnv5_7_1.py --self-check
Saturn v5.7.1 tuner self-check passed

.\.venv\Scripts\python.exe .\scripts\validate_v571_evidence_provenance.py <RC6 manifests/profile>
PASS: evidence artifacts remain content-identical from generation commit 6f9e73b through reviewed commit cfd0f3f

git diff --check
PASS for tracked RC6 production sources (pre-existing v5.7 line-ending notices remain outside v5.7.1 scope)
```

The first pytest attempt used an inaccessible stale Windows pytest temporary
directory and produced setup errors only. Re-running with the explicit clean
`--basetemp` above exercised the complete suite successfully.

## Retained evidence

| Evidence | SHA-256 |
|---|---|
| `audits/evidence/v571_rc6_candidate/stages/visual_evidence_manifest.json` | `be9104879b116d88d80b3f634865d8cee8026a88446e7df5a67423f034b0fef8` |
| `audits/evidence/v571_rc6_candidate/tracking/tracking_replay_manifest.json` | `35820bf364756a727060383b68adfb50c01d75a936847dd7781623b374181497` |
| `audits/evidence/v571_rc6_candidate/tracking/tracking_evidence_manifest.json` | `fd4810e877977e8b6263c1e9b207169ae0cb9c6e9774116997d945aa8d0de25d` |
| `audits/evidence/v571_rc6_candidate/end_to_end/end_to_end_visual_evidence_manifest.json` | `a1af7bfa9e4f725620bf012c4507bfd7ef9b253014a72a19cfd9aeb4989ddc0e` |
| `audits/evidence/v571_rc6_candidate/end_to_end/v571_end_to_end_visual_evidence.pdf` | `72f0f86cbeeca6efc9478057449d9898bf773a1ab2dcd6f9a1a0e85a1e528689` |
| `audits/evidence/v571_rc6_candidate/provenance/acceptance_provenance_manifest.json` | `63463808567c92e7e933b8d53b093daa1e4c4a084ed170b299b042cbb9b9da5b` |
| `audits/evidence/v571_rc6_candidate/provenance/tracking_replay_inputs_outputs.zip` | `11580cb7dd187830c1e50fcb51317bda4f75599adbf7bf9e0875b74db61e08b9` |
| `audits/evidence/v571_rc6_candidate/report/report_source_binding.json` | `4ccece01281f6ceb9f7f1d69cea22d62c6b01f5880c06748300c75f6989affb2` |
| `audits/evidence/v571_rc6_candidate/report/01_biological_results/Biological_Comparison_Report.pdf` | `b1937d51bdc4e9d533a7cccfa869667d47fa5956466a369917166a09ab01886f` |
| `audits/evidence/v571_rc6_candidate/report/01_biological_results/data/report_consistency_validation.json` | `208bcfa20c8a5822fe4bf70fc7da37fd71055e2602132a5765786ce3bfa28625` |
| `audits/evidence/v571_rc6_candidate/report/02_quality_control/Quality_Control_Report.pdf` | `de8200015664a180fe8721bdd3cd0ad895f1dc2bd7524e518fe00c3ac3a6d343` |
| `audits/evidence/v571_rc6_candidate/report/02_quality_control/data/specimen_sensitivity_artifact.json` | `4b9d41283d9f694bf2a7df39dd6a7c75f1dbe2b225ef73c35cbee25d49901088` |

## Provenance remediation

- KJ-01 resolves 88 unique source images to channel `0`.
- WT-01 resolves 67 unique source images to channel `0`.
- The resolved source manifests state the filename parsing rule and reject
  unresolved or nonzero channels.
- Evidence is bound to the generation commit by Git-blob hashes for the
  pipeline, profile, and generator. Validation permits later audit-only commits
  only when those blobs remain identical.
- The report package was rebuilt from the frozen 2D detections and corrected
  tracking replay after the RC6 audit identified stale RC5 report paths and a
  KJ count mismatch. `report_source_binding.json` verifies every retained
  source, tracked-detection, and track-summary CSV against the durable replay
  archive before report generation.
- Report metadata now points only to the RC6 report package. KJ counts reconcile
  to 5,766 reconstructed / 5,517 technical-valid tracks; WT counts reconcile to
  3,541 / 3,423 in the replay, biological report, and sensitivity artifact.

## Representative results and regression check

| Specimen | 2D detections | Tracks | Technical-valid tracks | Single-slice fraction | Median slices/track | Duplicate-Z tracks |
|---|---:|---:|---:|---:|---:|---:|
| KJ-01 | 26,651 | 5,766 | 5,517 | 0.358 | 3.0 | 0 |
| WT-01 | 16,421 | 3,541 | 3,423 | 0.350 | 3.0 | 0 |

RC6 stage and tracking summaries match RC5 exactly. The remediation changes
evidence binding, channel provenance, and automated sensitivity reporting; it
does not change segmentation or tracking behavior. The retained report
consistency validator passed source CSV, Excel, and PDF numerical agreement.

The compact sensitivity artifact uses the specimen as the unit, retains
excluded specimens in its ledger, separates technical-valid from all
reconstructed tracks, reports width missingness, and distinguishes projected
ROI area from sampled ROI slab volume. It does not perform group inference or
claim anatomical organ volume.

These two representative specimens validate execution and provenance only;
they do not establish a biological group effect.

## Audit history

The independent RC6 audit is retained at
`audits/runs/20260827-v571-remediation-acceptance-rc6`. Its gate failed because
the original report package referenced RC5 paths and disagreed with the
corrected KJ tracking replay. This receipt documents the remediation; acceptance
requires a new superseding audit and does not rewrite the RC6 verdict.
