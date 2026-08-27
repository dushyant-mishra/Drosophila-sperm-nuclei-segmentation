# Saturn v5.7.1 RC4 validation receipt

- Production code commit: `3a62520b1cfd92a66c060e92df683eb1ab933442`
- Validation date: 2026-08-27
- Profile: `production_profiles/saturn_v5_7_1_model_c_epoch003.json`
- Profile SHA-256: `3620551cd6df82a42aaa53ca234061bb699a9eb7cf43cb67c21371f0c1cc3c82`
- Checkpoint SHA-256: `7d49031dbcce31f0600c44146d9b5282b0df6e28bb1fc1bdde6ef2146ed15d25`

The evidence commit necessarily follows the production-code commit because it
adds generated artifacts and this receipt. The stage and tracking manifests
bind the evaluated production source by Git commit and pipeline SHA-256.

## Automated validation

```text
.\.venv\Scripts\python.exe -m py_compile <v5.7.1 pipeline, tuner, bridge, and evidence scripts>
PASS

.\.venv\Scripts\python.exe -m pytest -q
214 passed in 21.00s

.\.venv\Scripts\python.exe .\utils\tune_parameters_Saturnv5_7_1.py --self-check
Saturn v5.7.1 tuner self-check passed

git diff --check
PASS (line-ending notices only)
```

## Commit-bound evidence

| Evidence | SHA-256 |
|---|---|
| `audits/evidence/v571_rc4_candidate/stages/visual_evidence_manifest.json` | `3da1f928ca4ede713d6270c64a65b0c2cbd6a267d66438f756e89680c5a3d74d` |
| `audits/evidence/v571_rc4_candidate/tracking/tracking_replay_manifest.json` | `fde31a1a07626ba83b15a319d9ebac9a6704e4e0933a7eafd185bb29c540fb74` |
| `audits/evidence/v571_rc4_candidate/tracking/tracking_replay_summary.csv` | `f09eb5ce74c3ddc0a18f64db8f83717db9478a2cfb40f872b39379233cef0e15` |
| `audits/evidence/v571_rc4_candidate/end_to_end/end_to_end_visual_evidence_manifest.json` | `52e0eef4b739fd6aa6762d5a9e81de5475ff12e0d4574cf33880808e4d3022b8` |
| `audits/evidence/v571_rc4_candidate/end_to_end/v571_end_to_end_visual_evidence.pdf` | `6f9ec0b4c4d70a88d65d013c9274f8af8c0fea8d4f6ad2f378259c3681814658` |
| `audits/evidence/v571_rc4_candidate/report/01_biological_results/data/report_consistency_validation.json` | `134e46aeec741215811ae509f5aa2187663f28bfdf5fccdde4cb07ca6e65c46a` |

## Representative results

The two-slice segmentation evidence used KJ-01 and WT-01 at Z35. It resolved
the Leica calibration independently for each specimen, enforced the exact
epoch-003 checkpoint identity, and recorded source image and ROI hashes.

The deterministic downstream replay used frozen, hashed 2D detections:

| Specimen | 2D detections | Tracks | Single-slice fraction | Median slices/track | Duplicate-Z tracks |
|---|---:|---:|---:|---:|---:|
| KJ-01 | 26,651 | 5,766 | 0.358 | 3.0 | 0 |
| WT-01 | 16,421 | 3,541 | 0.350 | 3.0 | 0 |

The report consistency validator passed source CSV, Excel, and PDF agreement.
These representative results validate execution and provenance only; they do
not establish a biological group effect.
