# Saturn v5.7.1 RC5 validation receipt

- Production code commit: `35396b0e42f9d1afb9a2991c41bf31330a4c89aa`
- Validation date: 2026-08-27
- Profile: `production_profiles/saturn_v5_7_1_model_c_epoch003.json`
- Profile Git-blob SHA-256: `1fcc10c2199295d57210283255ead78cbe9ab51071121552a3668e5d3887d902`
- Pipeline Git-blob SHA-256: `b1183da136146a6e48dc9ffafcde11244768c5fe565bf4a2ba7c6486fec2b808`
- Checkpoint SHA-256: `7d49031dbcce31f0600c44146d9b5282b0df6e28bb1fc1bdde6ef2146ed15d25`

Git-blob hashes are the canonical tracked-text identity. Working-copy hashes
are also retained in manifests to document exact execution bytes on Windows;
they may differ because of CRLF checkout conversion without changing source
semantics.

## Automated validation

```text
.\.venv\Scripts\python.exe -m py_compile <v5.7.1 pipeline, tuner, and evidence scripts>
PASS

.\.venv\Scripts\python.exe -m pytest -q
219 passed in 29.02s

.\.venv\Scripts\python.exe .\utils\tune_parameters_Saturnv5_7_1.py --self-check
Saturn v5.7.1 tuner self-check passed

.\.venv\Scripts\python.exe .\scripts\validate_v571_evidence_provenance.py <RC5 manifests/profile>
PASS: evidence provenance matches commit 35396b0e42f9d1afb9a2991c41bf31330a4c89aa

git diff --check
PASS (line-ending notices only)
```

## Retained evidence

| Evidence | SHA-256 |
|---|---|
| `audits/evidence/v571_rc5_candidate/stages/visual_evidence_manifest.json` | `8d56732fe549e9fb7394ffff2c5029ec3664583a754e13730b3222565364328d` |
| `audits/evidence/v571_rc5_candidate/tracking/tracking_evidence_manifest.json` | `ee2fb783dffadf2285a41a68427d780262efc285b646978c302faba11dee7a92` |
| `audits/evidence/v571_rc5_candidate/tracking/tracking_replay_manifest.json` | `c3b6754ff3d792412ec265b514322fabea57003cfb6e6e191b42753886a9afe0` |
| `audits/evidence/v571_rc5_candidate/end_to_end/end_to_end_visual_evidence_manifest.json` | `04dcc492599a9e7b2735b5f641b80cda29575780b1a84bd5f74e9aa603bec5c3` |
| `audits/evidence/v571_rc5_candidate/end_to_end/v571_end_to_end_visual_evidence.pdf` | `aa169058ad50d2ecd56aece9cea80a33409483447306c9c6b09937dd018b5679` |
| `audits/evidence/v571_rc5_candidate/provenance/acceptance_provenance_manifest.json` | `2115516565c23174347d0ee12bd77f80c3b9dc36c6bb390ac464bd3f3c7ecb40` |
| `audits/evidence/v571_rc5_candidate/provenance/tracking_replay_inputs_outputs.zip` | `11580cb7dd187830c1e50fcb51317bda4f75599adbf7bf9e0875b74db61e08b9` |
| `audits/evidence/v571_rc5_candidate/report/01_biological_results/data/report_consistency_validation.json` | `16a2b195f625ad78ab1ae97c3dd5703c387dcbf18ccd21b383c544d5f661bdbb` |

The retained component PNGs are named by repository-relative path and hash in
the manifests. Compact specimen settings include the ROI, resolved calibration,
validated Leica XML, ordered source-image hashes, runtime environment, and
loaded profile. The deterministic ZIP retains complete 2D replay inputs and
tracked outputs.

## Representative results and regression check

| Specimen | 2D detections | Tracks | Technical-valid tracks | Single-slice fraction | Median slices/track | Duplicate-Z tracks |
|---|---:|---:|---:|---:|---:|---:|
| KJ-01 | 26,651 | 5,766 | 5,517 | 0.358 | 3.0 | 0 |
| WT-01 | 16,421 | 3,541 | 3,423 | 0.350 | 3.0 | 0 |

RC5 tracking summaries and full tracked/track-table SHA-256 values match RC4
exactly. The remediation changed terminology and evidence/provenance, not
segmentation or tracking behavior. The report validator passed source CSV,
Excel, and PDF numerical agreement.

These two representative specimens validate execution and provenance only;
they do not establish a biological group effect.
