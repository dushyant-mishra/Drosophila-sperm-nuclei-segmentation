# Saturn v5.6 Validation Report

Branch: `feature/saturn-v5.6-roi-adaptive`

Starting commit SHA: `1b81d73700b089e26637fc9632417348573bf0c7`

## Created Files

- `sperm_segmentation_saturnv5.6.py`
- `utils/tune_parameters_Saturnv5_6.py`
- `tests/test_saturn_v56_preprocessing.py`
- `scratch/run_v56_smoke_test.py`
- `V5_6_PIPELINE_IMPLEMENTATION.md`
- `V5_6_TUNER_IMPLEMENTATION.md`
- `V5_6_VALIDATION_REPORT.md`
- `codex_prompts/v56_full_implementation.md`

## Modified Files

- `sperm_segmentation_saturnv5.6.py`
  - Added behavior-preserving `remove_objects_smaller_than()` wrapper for the scikit-image `remove_small_objects(min_size=...)` deprecation.
  - Added numeric ridge thresholds to returned `preprocess_debug` metadata for smoke-test auditability.
- `V5_6_VALIDATION_REPORT.md`

No Saturn v5.5 source file was edited.

## Validation Commands Run

- `.\.venv\Scripts\python.exe -m py_compile .\sperm_segmentation_saturnv5.6.py .\utils\tune_parameters_Saturnv5_6.py`
- `.\.venv\Scripts\python.exe -m pytest -q`
- `.\.venv\Scripts\python.exe .\utils\tune_parameters_Saturnv5_6.py --self-check`
- `git diff --check`
- `git status --short --branch`
- `certutil -hashfile .\sperm_segmentation_saturnv5.5.py SHA256`
- `certutil -hashfile .\utils\tune_parameters_Saturnv5_5.py SHA256`
- `.\.venv\Scripts\python.exe .\scratch\run_v56_smoke_test.py`

## Compile Results

Passed:

- `sperm_segmentation_saturnv5.6.py`
- `utils/tune_parameters_Saturnv5_6.py`

## Test Results

After installing `pytest` into `.venv`, the synthetic suite passed:

```text
12 passed in 3.14s
```

The earlier seven scikit-image deprecation warnings were addressed in v5.6 by replacing the deprecated `min_size` argument with a compatibility wrapper using `max_size = min_size - 1`.

## Self-Check Result

Passed:

```text
Saturn v5.6 tuner self-check passed
```

## Real-Image Smoke Test

Full microscopy batch: not run.

Input image directory:

```text
C:\Users\dmishra\Desktop\sperm images
```

ROI path:

```text
C:\Users\dmishra\Desktop\sperm images\roi_z28.1.npy
```

Smoke-test output directory:

```text
scratch\v5_6_smoke_test
```

Requested slices:

```text
z05, z06, z12, z35, z60, z87
```

Selected slices:

```text
z05, z06, z12, z35, z60, z87
```

Substitutions: none.

Stack preprocessing context:

- Profile: `standard`
- CLAHE clip: `0.025`
- Context sample z indices: `0, 8, 16, 24, 32, 40, 47, 55, 63, 71, 79, 87`
- Exclusion mask: none supplied

## Smoke Comparison Table

| Z | v5.5 count | v5.6 count | v5.5 median length | v5.6 median length | v5.6 median width | v5.6 median L/W | hyst occ | clean occ | skel before | skel after | bridge infl | ROI-edge frac | outside ROI px | exclusion overlap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 817 | 227 | 7.752 | 12.293 | 2.141 | 4.874 | 0.1550 | 0.1438 | 11567 | 11567 | 0.0000 | 0.0000 | 0 | 0 |
| 6 | 809 | 245 | 8.065 | 12.033 | 3.027 | 4.874 | 0.1545 | 0.1452 | 11720 | 11720 | 0.0000 | 0.0000 | 0 | 0 |
| 12 | 859 | 238 | 7.309 | 12.828 | 3.027 | 5.112 | 0.1516 | 0.1397 | 12174 | 12174 | 0.0000 | 0.0000 | 0 | 0 |
| 35 | 852 | 241 | 7.309 | 12.736 | 3.027 | 4.846 | 0.1500 | 0.1330 | 11777 | 11777 | 0.0000 | 0.0041 | 0 | 0 |
| 60 | 872 | 260 | 7.309 | 13.077 | 3.027 | 5.019 | 0.1497 | 0.1405 | 11777 | 11777 | 0.0000 | 0.0000 | 0 | 0 |
| 87 | 774 | 201 | 7.687 | 13.807 | 3.027 | 5.018 | 0.1516 | 0.1405 | 11509 | 11509 | 0.0000 | 0.0000 | 0 | 0 |

CSV and JSON tables:

- `scratch/v5_6_smoke_test/v5_5_vs_v5_6_smoke_comparison.csv`
- `scratch/v5_6_smoke_test/v5_5_vs_v5_6_smoke_comparison.json`

## Debug Output Paths

Montages:

- `scratch/v5_6_smoke_test/z05_debug_montage.png`
- `scratch/v5_6_smoke_test/z06_debug_montage.png`
- `scratch/v5_6_smoke_test/z12_debug_montage.png`
- `scratch/v5_6_smoke_test/z35_debug_montage.png`
- `scratch/v5_6_smoke_test/z60_debug_montage.png`
- `scratch/v5_6_smoke_test/z87_debug_montage.png`

Per-slice final overlays:

- `scratch/v5_6_smoke_test/z05_v5_6_final_overlay.tif`
- `scratch/v5_6_smoke_test/z06_v5_6_final_overlay.tif`
- `scratch/v5_6_smoke_test/z12_v5_6_final_overlay.tif`
- `scratch/v5_6_smoke_test/z35_v5_6_final_overlay.tif`
- `scratch/v5_6_smoke_test/z60_v5_6_final_overlay.tif`
- `scratch/v5_6_smoke_test/z87_v5_6_final_overlay.tif`

Stage images and per-slice debug JSON:

- `scratch/v5_6_smoke_test/debug/z##_01_raw_robust_normalized.png`
- `scratch/v5_6_smoke_test/debug/z##_02_denoised.png`
- `scratch/v5_6_smoke_test/debug/z##_03_clahe.png`
- `scratch/v5_6_smoke_test/debug/z##_04_background.png`
- `scratch/v5_6_smoke_test/debug/z##_05_foreground.png`
- `scratch/v5_6_smoke_test/debug/z##_06_ridge.png`
- `scratch/v5_6_smoke_test/debug/z##_07_hysteresis.png`
- `scratch/v5_6_smoke_test/debug/z##_08_clean.png`
- `scratch/v5_6_smoke_test/debug/z##_09_skeleton_clean.png`
- `scratch/v5_6_smoke_test/debug/z##_10_skeleton_bridged.png`
- `scratch/v5_6_smoke_test/debug/z##_11_skeleton_pruned.png`
- `scratch/v5_6_smoke_test/debug/z##_12_final_detections.png`
- `scratch/v5_6_smoke_test/debug/z##_debug_record.json`

Additional smoke records:

- `scratch/v5_6_smoke_test/selected_slices.json`
- `scratch/v5_6_smoke_test/off_roi_bright_threshold_checks.json`
- `scratch/v5_6_smoke_test/stack_preprocessing_qc.json`
- `scratch/v5_6_smoke_test/stack_preprocessing_qc.csv`

## Explicit Smoke-Test Checks

1. No mask, skeleton, label, or bridge pixels outside ROI:
   - Passed. Outside-ROI pixel count was `0` for all tested slices.
2. No pixels inside exclusion mask:
   - Not applicable. No exclusion mask was supplied.
3. Off-ROI bright tissues do not influence ROI thresholds:
   - Passed in the synthetic perturbation check. Numeric high/low ridge thresholds were unchanged after adding a bright off-ROI object to every tested slice.
4. ROI boundary does not create an artificial ridge:
   - No obvious strong boundary band in inspected montages. ROI boundary ridge p95 ratio was about `1.00-1.42`; this should remain a watched metric during profile tuning.
5. True elongated nuclei remain visible:
   - Passed visually. Elongated nuclei remain visible in the final overlays.
6. Round puncta and broad tissue edges are reduced relative to v5.5:
   - Partially passed. Final accepted v5.6 overlays are much more selective than v5.5 counts, but bright round puncta remain visible in foreground/hysteresis stages and are mostly removed by final shape filters.
7. Bridging does not substantially join neighboring nuclei:
   - Passed. Bridge inflation fraction was `0.0` for all tested slices.
8. Foreground occupancy is not excessive:
   - Borderline. Hysteresis occupancy was stable at about `0.150-0.155`, and clean-mask occupancy was about `0.133-0.145`.
9. Detection counts remain biologically plausible:
   - Caution. v5.6 counts are much lower than v5.5 counts on the same ROI (`201-260` vs `774-872`), suggesting high specificity but reduced sensitivity.
10. Median 2D lengths remain near expected 9-10 um:
   - Failed. v5.6 median lengths were `12.0-13.8 um`, likely because current defaults preferentially retain longer/stronger detections.

## Visual Observations

- ROI containment is strong: outside-ROI tissue is visible only in final overlay background, not in accepted masks/skeletons.
- The selected stack profile was stable and off-ROI bright perturbations did not move ROI thresholds.
- v5.6 suppresses broad off-ROI tissue and avoids bridge inflation.
- The current default v5.6 segmentation is too conservative for counting. It appears to keep a cleaner, longer subset rather than the full plausible biological population.
- The foreground and hysteresis stages still contain many small bright puncta, but final topology/shape filtering removes most of them.

## Readiness Assessment

- Ready for profile tuning: yes.
- Ready for representative-slice segmentation tuning: yes, after profile comparison.
- Ready for tracking tuning: not yet; segmentation parameters should be tuned first.
- Complete batch recommended: no, not with current v5.6 defaults.

Recommended next parameter direction:

- Run profile comparison first, including `low_signal` and `standard`.
- In segmentation tuning, reduce the high-specificity bias by allowing slightly lower thresholds and/or lower `MIN_SKEL_LEN_UM`.
- Keep bridge constraints strict; bridge inflation was already `0.0`, so there is no need to loosen bridging before segmentation sensitivity is tuned.
- Use count as a guardrail, not a reward. The smoke test suggests current defaults are under-counting.

## v5.5 Hash Baseline And Final Verification

Baseline and final hashes matched:

- `sperm_segmentation_saturnv5.5.py`: `bb687703d0a3ef004a36d9685417f7ceabb01da14debae1bf65231d7c973e529`
- `utils/tune_parameters_Saturnv5_5.py`: `2ac57472c5dc6185f831004897132f7983725ad2e245b6b99693550be2144344`

`git status --short -- sperm_segmentation_saturnv5.5.py utils\tune_parameters_Saturnv5_5.py` returned no modifications.

## Phase 2 Profile And Segmentation Tuning

Production source files were not edited during this phase. The validation runner was kept under:

```text
scratch\run_v56_profile_and_segmentation_tuning.py
```

Exact commands run:

```text
.\.venv\Scripts\python.exe -m py_compile .\scratch\run_v56_profile_and_segmentation_tuning.py
.\.venv\Scripts\python.exe .\scratch\run_v56_profile_and_segmentation_tuning.py
```

Input image directory:

```text
C:\Users\dmishra\Desktop\sperm images
```

ROI path:

```text
C:\Users\dmishra\Desktop\sperm images\roi_z28.1.npy
```

Fixed random seed: `560123`.

Requested and selected slices:

```text
z05, z06, z12, z35, z60, z87
```

Substitutions: none.

The runner evaluated the original v5.6 default baseline, all five preprocessing profiles, and exactly 80 deterministic segmentation candidates. The baseline was recorded separately and was not counted as one of the 80 sampled candidates. Loops, recursive local reanalysis, and permissive bridging were disabled for the tuning candidates. One ROI-aware stack preprocessing context was reused within each profile/candidate evaluation.

During the first corrected rerun, the scratch runner exposed a runner-only scoring bug: it was reading old topology field names (`endpoint_count`, `branch_nodes`) instead of v5.6 result fields (`n_endpoints`, `n_branch_nodes`). This was fixed in the scratch runner only and the run was repeated. The final tables below use the corrected topology metrics.

### Profile Results

| Profile | Score | Median count | Median length um | Median width um | Median L/W | Hyst occ | Clean occ | Bridge infl | Outside ROI | Excl overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| low_signal | 25.748 | 239.5 | 12.742 | 2.880 | 4.959 | 0.152 | 0.140 | 0.000 | 0 | 0 |
| standard | 25.844 | 239.5 | 12.796 | 2.880 | 4.957 | 0.152 | 0.140 | 0.000 | 0 | 0 |
| auto | 25.844 | 239.5 | 12.796 | 2.880 | 4.957 | 0.152 | 0.140 | 0.000 | 0 | 0 |
| high_contrast | 27.971 | 231.0 | 13.016 | 3.027 | 4.766 | 0.151 | 0.147 | 0.000 | 0 | 0 |
| no_clahe | 28.308 | 230.5 | 13.097 | 3.027 | 4.557 | 0.150 | 0.150 | 0.000 | 0 | 0 |

Selected preprocessing profile for limited segmentation tuning: `low_signal` (`CLAHE_CLIP = 0.035`).

### Top 10 Segmentation Candidates

| Candidate | Score | Median count | Median length um | Median width um | Median L/W | Short frac | Long frac | Clean occ | Bridge infl | Outside ROI | HI | LO | Min skel um | Max width um | Min L/W | Max tort |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cand_003 | 13.289 | 294.0 | 10.340 | 3.027 | 4.266 | 0.092 | 0.248 | 0.194 | 0.000 | 0 | 85.162 | 78.644 | 4.967 | 3.313 | 1.853 | 3.140 |
| cand_019 | 14.048 | 313.0 | 10.776 | 3.027 | 4.216 | 0.112 | 0.258 | 0.160 | 0.000 | 0 | 87.052 | 80.660 | 4.527 | 4.470 | 1.897 | 2.717 |
| cand_049 | 14.210 | 291.5 | 10.911 | 3.027 | 4.253 | 0.074 | 0.263 | 0.176 | 0.000 | 0 | 91.586 | 77.995 | 4.802 | 3.417 | 2.110 | 2.785 |
| cand_078 | 14.247 | 323.0 | 10.649 | 3.027 | 4.168 | 0.026 | 0.243 | 0.207 | 0.000 | 0 | 86.603 | 75.975 | 5.684 | 4.478 | 1.861 | 2.903 |
| cand_002 | 14.529 | 375.5 | 9.925 | 3.027 | 3.837 | 0.124 | 0.194 | 0.238 | 0.000 | 0 | 84.500 | 74.000 | 4.500 | 4.400 | 2.000 | 2.800 |
| cand_029 | 14.747 | 307.5 | 11.041 | 3.027 | 4.294 | 0.067 | 0.263 | 0.185 | 0.000 | 0 | 88.190 | 78.950 | 4.942 | 4.769 | 2.096 | 2.537 |
| cand_009 | 14.829 | 314.0 | 10.960 | 3.027 | 4.292 | 0.092 | 0.268 | 0.173 | 0.000 | 0 | 90.210 | 79.896 | 4.636 | 3.811 | 1.822 | 2.118 |
| cand_046 | 15.261 | 274.5 | 10.619 | 3.027 | 4.353 | 0.084 | 0.258 | 0.188 | 0.000 | 0 | 90.160 | 78.691 | 5.136 | 3.100 | 1.903 | 2.138 |
| cand_054 | 15.333 | 305.0 | 11.171 | 3.027 | 4.364 | 0.106 | 0.273 | 0.171 | 0.000 | 0 | 84.255 | 80.255 | 4.344 | 4.847 | 2.238 | 2.243 |
| cand_074 | 16.000 | 376.0 | 9.705 | 3.027 | 3.814 | 0.126 | 0.185 | 0.246 | 0.000 | 0 | 83.058 | 73.375 | 4.705 | 4.275 | 1.833 | 2.670 |

The numerically highest-scoring candidate is therefore labeled only as the first candidate for visual inspection:

```json
{
  "candidate_id": "cand_003",
  "THRESHOLD_HI": 85.1621,
  "THRESHOLD_LO": 78.6443,
  "MIN_OBJ_PX": 4,
  "MIN_SKEL_LEN_UM": 4.9666,
  "MAX_WIDTH_UM": 3.3127,
  "MIN_LENGTH_WIDTH_RATIO": 1.8526,
  "MAX_TORTUOSITY": 3.1402,
  "CLAHE_MODE": "low_signal",
  "CLAHE_CLIP": 0.035
}
```

### Top-Three Visual Comparison Summary

| Candidate | Median count | Mean count | Mean median length um | Mean median width um | Mean median L/W | Hyst occ | Clean occ | Bridge infl | Outside ROI | Excl overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default_v5_6 | 239.5 | 235.3 | 12.796 | 2.880 | 4.957 | 0.152 | 0.140 | 0.000 | 0 | 0 |
| cand_003 | 294.0 | 293.5 | 10.340 | 3.027 | 4.266 | 0.203 | 0.194 | 0.000 | 0 | 0 |
| cand_019 | 313.0 | 311.7 | 10.776 | 3.027 | 4.216 | 0.182 | 0.160 | 0.000 | 0 | 0 |
| cand_049 | 291.5 | 295.0 | 10.911 | 3.027 | 4.253 | 0.187 | 0.176 | 0.000 | 0 | 0 |

Visual observation from the z35 side-by-side comparison: all three tuned candidates recover more elongated nuclei than the default while keeping the same ROI containment. `cand_003` is visibly denser than default but not wildly overfilled; `cand_019` and `cand_049` look similar, with slightly different sensitivity. `cand_002` is not in the top-three visual panel but is worth secondary inspection because it brings the median length closest to the expected 9-10 um range at the cost of higher mask occupancy and higher count.

### Phase 2 Artifacts

Profile comparison:

- `scratch\v5_6_profile_comparison\profile_comparison_v5_6.csv`
- `scratch\v5_6_profile_comparison\profile_slice_metrics_v5_6.csv`
- `scratch\v5_6_profile_comparison\profile_parameter_dictionaries_v5_6.json`
- `scratch\v5_6_profile_comparison\profile_review_v5_6_run2.pdf`
- `scratch\v5_6_profile_comparison\profile_debug\...\z##_final_overlay.tif`

Segmentation tuning:

- `scratch\v5_6_segmentation_tuning\all_segmentation_results_v5_6.csv`
- `scratch\v5_6_segmentation_tuning\all_segmentation_slice_metrics_v5_6.csv`
- `scratch\v5_6_segmentation_tuning\baseline_v5_6_default_metrics.csv`
- `scratch\v5_6_segmentation_tuning\baseline_v5_6_default_params.json`
- `scratch\v5_6_segmentation_tuning\best_segmentation_params_v5_6.json`
- `scratch\v5_6_segmentation_tuning\top_candidate_review_v5_6_run2.pdf`

Top-three comparison:

- `scratch\v5_6_top_candidate_comparison\top_candidate_side_by_side_v5_6_run2.pdf`
- `scratch\v5_6_top_candidate_comparison\top_three_candidate_comparison_v5_6.csv`
- `scratch\v5_6_top_candidate_comparison\top_three_candidate_params_v5_6.json`
- `scratch\v5_6_top_candidate_comparison\z05_top_candidate_comparison.png`
- `scratch\v5_6_top_candidate_comparison\z06_top_candidate_comparison.png`
- `scratch\v5_6_top_candidate_comparison\z12_top_candidate_comparison.png`
- `scratch\v5_6_top_candidate_comparison\z35_top_candidate_comparison.png`
- `scratch\v5_6_top_candidate_comparison\z60_top_candidate_comparison.png`
- `scratch\v5_6_top_candidate_comparison\z87_top_candidate_comparison.png`

Summary:

- `scratch\v5_6_phase2_summary.json`

### Phase 2 Readiness

- Ready for profile tuning: yes; `low_signal` should be the first profile to inspect.
- Ready for segmentation tuning: yes; the limited candidate search improved the default's long-object survivor bias.
- Ready for a complete batch: not yet. A human visual pass should compare `cand_003`, `cand_019`, `cand_049`, and secondarily `cand_002` on the generated panels first.
- Ready for tracking tuning: not yet; choose the 2D segmentation candidate first.

## Phase 3 Focused Visual Candidate Audit

Production source files were not edited during this phase. No tracking tuning, complete Z-stack batch, commit, or push was run.

Scratch runner:

```text
scratch\run_v56_visual_candidate_audit.py
```

Exact command run:

```text
.\.venv\Scripts\python.exe .\scratch\run_v56_visual_candidate_audit.py
```

Input image directory:

```text
C:\Users\dmishra\Desktop\sperm images
```

ROI path:

```text
C:\Users\dmishra\Desktop\sperm images\roi_z28.1.npy
```

Fixed random seed: `560123`.

Requested and selected slices:

```text
z05, z06, z12, z35, z60, z87
```

Substitutions: none.

Candidates audited:

- `default_v5_6`: original v5.6 default baseline
- `cand_003`: best corrected numerical score
- `cand_002`: median length closest to expected 9-10 um, but higher count/occupancy
- `cand_019`: higher count with median length around 10.8 um
- `cand_049`: cleaner/lower-count alternative with median length around 10.9 um

### Visual Audit Pooled Metrics

| Candidate | Median detections | Median length um | Mean length um | Median width um | Median L/W | <6 um | 6-8 um | 8-11 um | 11-15 um | >15 um | Clean occ | Hyst occ | Bridge infl | Outside ROI | Broad/puncta crop detections |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default_v5_6 | 239.5 | 12.796 | 12.946 | 2.880 | 4.957 | 0.000 | 0.112 | 0.265 | 0.271 | 0.353 | 0.140 | 0.152 | 0.000 | 0 | 184 |
| cand_003 | 294.0 | 10.340 | 11.273 | 3.027 | 4.266 | 0.092 | 0.217 | 0.228 | 0.215 | 0.248 | 0.194 | 0.203 | 0.000 | 0 | 226 |
| cand_002 | 375.5 | 9.925 | 10.708 | 3.027 | 3.837 | 0.124 | 0.207 | 0.256 | 0.219 | 0.194 | 0.238 | 0.244 | 0.000 | 0 | 304 |
| cand_019 | 313.0 | 10.776 | 11.372 | 3.027 | 4.216 | 0.112 | 0.189 | 0.218 | 0.224 | 0.258 | 0.160 | 0.182 | 0.000 | 0 | 246 |
| cand_049 | 291.5 | 10.911 | 11.606 | 3.027 | 4.253 | 0.074 | 0.181 | 0.251 | 0.232 | 0.263 | 0.176 | 0.187 | 0.000 | 0 | 231 |

Overlap summary versus `cand_003` across the six audited slices:

| Candidate | Unique to cand_003 | Unique to comparison | Approximately matched |
|---|---:|---:|---:|
| default_v5_6 | 767 | 418 | 994 |
| cand_002 | 480 | 978 | 1281 |
| cand_019 | 363 | 472 | 1398 |
| cand_049 | 270 | 279 | 1491 |

Interpretation:

- `cand_003` remains the first candidate to inspect. It increases recovery versus default and reduces the long-object survivor bias without ROI leakage or bridge inflation.
- `cand_002` is plausible but more permissive. It is closest to the 9-10 um median length target, but also has the highest clean-mask occupancy and the most detections in broad-tissue/puncta review crops.
- `cand_019` is plausible but more permissive than `cand_049`, with a higher median count and moderate occupancy.
- `cand_049` is plausible but more conservative than `cand_019`/`cand_002`, with fewer extra detections relative to `cand_003`.
- `default_v5_6` is conservative but likely under-detects and enriches for longer survivors.

No final production candidate was selected. The numerical score is not ground truth; the generated panels are intended for human visual review.

Recommended visual inspection order:

1. `cand_003`
2. `cand_002`
3. `cand_019`
4. `cand_049`

### Visual Audit Artifacts

Output directory:

```text
scratch\v5_6_visual_candidate_audit
```

Full-slice panels:

- `scratch\v5_6_visual_candidate_audit\z05_full_candidate_audit.png`
- `scratch\v5_6_visual_candidate_audit\z06_full_candidate_audit.png`
- `scratch\v5_6_visual_candidate_audit\z12_full_candidate_audit.png`
- `scratch\v5_6_visual_candidate_audit\z35_full_candidate_audit.png`
- `scratch\v5_6_visual_candidate_audit\z60_full_candidate_audit.png`
- `scratch\v5_6_visual_candidate_audit\z87_full_candidate_audit.png`

Crop panels:

- `scratch\v5_6_visual_candidate_audit\crop_panels\crop_01_round_puncta_z05.png`
- `scratch\v5_6_visual_candidate_audit\crop_panels\crop_02_faint_elongated_nuclei_z05.png`
- `scratch\v5_6_visual_candidate_audit\crop_panels\crop_03_bright_transverse_non_nuclear_structure_z06.png`
- `scratch\v5_6_visual_candidate_audit\crop_panels\crop_04_dense_parallel_nuclei_z06.png`
- `scratch\v5_6_visual_candidate_audit\crop_panels\crop_05_round_puncta_z12.png`
- `scratch\v5_6_visual_candidate_audit\crop_panels\crop_06_crossing_nuclei_z12.png`
- `scratch\v5_6_visual_candidate_audit\crop_panels\crop_07_dense_parallel_nuclei_z35.png`
- `scratch\v5_6_visual_candidate_audit\crop_panels\crop_08_broad_elongated_non_sperm_tissue_z35.png`
- `scratch\v5_6_visual_candidate_audit\crop_panels\crop_09_crossing_nuclei_z60.png`
- `scratch\v5_6_visual_candidate_audit\crop_panels\crop_10_faint_elongated_nuclei_z60.png`
- `scratch\v5_6_visual_candidate_audit\crop_panels\crop_11_roi_boundary_z87.png`
- `scratch\v5_6_visual_candidate_audit\crop_panels\crop_12_broad_elongated_non_sperm_tissue_z87.png`

Tables and review files:

- `scratch\v5_6_visual_candidate_audit\visual_candidate_audit_report_v5_6.pdf`
- `scratch\v5_6_visual_candidate_audit\visual_candidate_review_sheet_v5_6.xlsx`
- `scratch\v5_6_visual_candidate_audit\candidate_parameter_provenance_v5_6.json`
- `scratch\v5_6_visual_candidate_audit\visual_candidate_pooled_metrics_v5_6.csv`
- `scratch\v5_6_visual_candidate_audit\visual_candidate_object_metrics_v5_6.csv`
- `scratch\v5_6_visual_candidate_audit\visual_candidate_overlap_vs_cand003_v5_6.csv`
- `scratch\v5_6_visual_candidate_audit\visual_audit_crop_coordinates_v5_6.csv`
- `scratch\v5_6_visual_candidate_audit\visual_candidate_audit_summary_v5_6.json`

## Phase 4 Comparative-Study Safeguards

Purpose: prevent genuine WT-versus-mutant differences in length, width, taper, tortuosity, count, volume, Z-span, pitch, or related morphology from being removed or forced toward WT-like values.

No v5.5 file was edited. No complete microscopy batch, tracking tuning, commit, or push was run. No final production parameters were selected.

Files modified:

- `sperm_segmentation_saturnv5.6.py`
- `utils\tune_parameters_Saturnv5_6.py`
- `V5_6_VALIDATION_REPORT.md`

Files created:

- `V5_6_COMPARATIVE_ANALYSIS.md`
- `tests\test_saturn_v56_comparative.py`
- `comparative_presets\comparative_conservative_v5_6.json`
- `comparative_presets\comparative_selected_v5_6.json`
- `comparative_presets\comparative_intermediate_v5_6.json`
- `comparative_presets\comparative_permissive_v5_6.json`

### Comparative Mode Behavior

Added:

```text
ANALYSIS_MODE = "comparative"
```

Supported modes:

- `comparative`
- `reference_morphology`
- `legacy`

In comparative mode, the audit asks whether an independently resolved object is technically valid. It does not remove objects merely because they do not resemble expected WT morphology.

New output columns on the full track summary:

- `technical_valid`
- `technical_failure_reasons`
- `morphology_warning`
- `morphology_warning_reasons`
- `reference_morphology_pass`
- `segmentation_parameter_set`
- `preprocessing_profile`
- `analysis_mode`

New output populations:

- `track_summary_all_v5.6-roi-adaptive.csv`
- `track_summary_technical_valid_v5.6-roi-adaptive.csv`
- `track_summary_reference_morphology_v5.6-roi-adaptive.csv`
- `track_summary_morphology_warning_v5.6-roi-adaptive.csv`
- `track_summary_technical_failures_v5.6-roi-adaptive.csv`

The primary WT-versus-mutant analysis population is `track_summary_technical_valid_v5.6-roi-adaptive.csv`. The reference-morphology subset is diagnostic only.

Required report wording was added to output notes:

```text
Morphology warnings are retained in the comparative population because they may represent genuine genotype-dependent phenotypes.
```

### Technical Failures

Comparative technical failure rules now cover processing failures such as:

- invalid coordinates
- zero or invalid length
- outside-ROI detections when such leakage columns are present
- exclusion-mask overlap when such columns are present
- segmentation leakage flags
- gross branched tissue network flags
- clear multi-object connected component flags
- unrecoverable tracking/label inconsistency flags
- extreme component length incompatible with one resolved object

### Morphology Warnings

Morphology warnings are retained in the technical-valid population. They include:

- long
- short
- wide
- thin
- high tortuosity
- high taper
- low taper
- low length-to-width ratio
- unusual pitch
- unusual volume
- unusual Z-span
- unusual nearest-neighbor distance

### Tuner Safeguard

In comparative mode, the v5.6 tuner no longer rewards a candidate for approaching a WT-like median length, WT-like width, WT-like count, or WT-like length-to-width ratio. Those morphology-prior terms are still reported as `morphology_prior_score_reported_not_optimized`, but `score` uses technical criteria only.

Technical criteria include count stability, controlled occupancy, bridge inflation, and exclusion-mask leakage. Count remains a stability/guardrail signal, not a target count.

### Sensitivity Presets

Created versioned comparative presets:

- `comparative_presets\comparative_conservative_v5_6.json` based on `cand_049`
- `comparative_presets\comparative_selected_v5_6.json` based on `cand_003`
- `comparative_presets\comparative_intermediate_v5_6.json` based on `cand_019`
- `comparative_presets\comparative_permissive_v5_6.json` based on `cand_002`

These are sensitivity-analysis presets, not ground truth or final production choices.

### Blinded Validation Workflow

Added helper functions:

- `assign_blinded_dataset_ids()`
- `make_blinded_review_sheet()`

The blinded workflow assigns anonymized dataset IDs, hides genotype labels from segmentation/review, preserves a separate reveal table, and leaves manual review columns for true detection, missed nucleus, split nucleus, merged nuclei, tissue-edge false positives, puncta/ring false positives, and uncertain cases.

Documentation for preparing blinded manifests was added in:

- `V5_6_COMPARATIVE_ANALYSIS.md`

### Differential-Error Checks

Added `differential_error_indicators()` to compare anonymized groups for:

- technical-failure fraction
- morphology-warning fraction
- short-fragment fraction
- suspected-merge fraction
- branch-network fraction
- ROI-edge fraction
- permissive-only detection fraction
- conservative-loss fraction

The function reports differences and warnings. It does not correct distributions to make groups agree.

### Comparative Tests

Added synthetic tests demonstrating that:

- longer mutant-like nuclei remain technical valid
- wider, more tapered, more tortuous objects remain technical valid as morphology warnings
- lower-count input remains lower count and is not compensated toward a target
- WT and mutant use identical morphology/audit thresholds while allowing stack-specific photometric normalization
- reference-morphology filtering does not change the technical-valid table
- conservative, selected, intermediate, and permissive presets produce sensitivity outputs
- genotype labels are hidden during blinded segmentation/review
- technical artifacts are still removed
- existing v5.6 ROI/exclusion/invariance tests continue to pass

## Phase 5 Blinded Representative-Image Validation Workflow

Purpose: validate segmentation fairness and technical error rates without using genotype labels during segmentation, crop selection, preset comparison, report generation, or scoring.

No v5.5 file was edited. No complete stacks, tracking tuning, genotype-specific tuning, unblinding analysis, commit, or push was run in this phase.

Files created:

- `scratch\run_v56_blinded_validation.py`
- `scratch\run_v56_unblind_validation.py`
- `tests\test_saturn_v56_blinded_validation.py`
- `scratch\v5_6_blinded_validation\manifests\source_manifest_v5_6.csv`

Files modified:

- `V5_6_COMPARATIVE_ANALYSIS.md`
- `V5_6_VALIDATION_REPORT.md`

### Manifest And Blinding

The blinded runner accepts a user-facing source manifest with columns:

```text
dataset_path, roi_path, exclusion_mask_path, dataset_label, sample_id, acquisition_class, genotype, slice_override
```

The runner writes:

```text
scratch\v5_6_blinded_validation\manifests\blinded_dataset_manifest_v5_6.csv
scratch\v5_6_unblinding_key\unblinding_key_v5_6.csv
```

The unblinding key is outside the blinded-analysis output directory by design. No segmentation, scoring, crop selection, preset comparison, or blinded report function reads the unblinding key.

The current local repository did not contain a filled source manifest. To avoid guessing genotype from folder names, only a template was generated:

```text
scratch\v5_6_blinded_validation\manifests\source_manifest_v5_6.csv
```

The real blinded image pass was not executed because genotype labels must be supplied explicitly by the user in the manifest.

### Representative Slices

For each manifest dataset, the runner selects six representative slices:

- first usable slice
- approximately 20%
- approximately 40%
- approximately 60%
- approximately 80%
- last usable slice

Users can override with the `slice_override` column. Selected Z indices are recorded in `blinded_validation_provenance_v5_6.json` after a real run.

### Shared Presets

Primary preset:

```text
comparative_presets\comparative_selected_v5_6.json
```

Sensitivity presets:

```text
comparative_presets\comparative_conservative_v5_6.json
comparative_presets\comparative_intermediate_v5_6.json
comparative_presets\comparative_permissive_v5_6.json
```

All blinded datasets use the same preset logic and morphology rules. Stack-specific photometric normalization is permitted. Genotype-specific morphology thresholds and independent WT/mutant retuning are prohibited.

### Planned Blinded Outputs

After a filled manifest is provided, the runner saves under:

```text
scratch\v5_6_blinded_validation
```

Required blinded artifacts:

- `manifests\blinded_dataset_manifest_v5_6.csv`
- `blinded_validation_metrics_v5_6.csv`
- `blinded_validation_slice_metrics_v5_6.csv`
- `preset_object_matching_v5_6.csv`
- `photometric_robustness\photometric_robustness_v5_6.csv`
- `review_workbook\blinded_manual_review_v5_6.xlsx`
- `reports\blinded_validation_report_v5_6.pdf`
- `blinded_validation_provenance_v5_6.json`

No genotype labels are written into blinded output filenames, report titles, crop files, or review workbook fields.

### Photometric Robustness

For the selected preset, each representative slice is rerun under:

- original
- intensity multiplied by 0.85
- intensity multiplied by 1.15
- moderate additive offset
- moderate contrast reduction

The runner records matched-object fraction, detection-count change, median-length change, technical-valid classification change, and morphology-warning classification change. This reports instability as technical QC; it does not force counts to remain equal.

### Review Crops And Manual Gate

The runner creates at least 12 crops per dataset when enough detections are available:

- random crops
- preset-disagreement crops
- warning or technical-risk crops

The manual review workbook contains blank columns for true detection, missed nucleus, split nucleus, merged nuclei, tissue-edge false positive, puncta/ring false positive, ROI-edge artifact, uncertain, and reviewer notes.

The workflow stops before unblinding and prints:

```text
Blinded review outputs are complete. Complete the manual review workbook before running the unblinding analysis.
```

Unblinding is a separate explicit command:

```powershell
.\.venv\Scripts\python.exe .\scratch\run_v56_unblind_validation.py `
  --review-workbook .\scratch\v5_6_blinded_validation\review_workbook\blinded_manual_review_v5_6.xlsx `
  --unblinding-key .\scratch\v5_6_unblinding_key\unblinding_key_v5_6.csv
```

The unblinding utility refuses incomplete review workbooks.

### Blinded Validation Tests

Added tests verifying:

- genotype labels are absent from the blinded manifest
- genotype labels are not passed to segmentation manifests
- genotype labels do not appear in standard output names
- all presets share resolved morphology settings
- stack-specific normalization may differ while morphology rules remain shared
- conservative, selected, intermediate, and permissive presets are applied identically across blinded groups
- the unblinding utility refuses incomplete review workbooks
- the unblinding utility accepts a completed synthetic review workbook
- longer synthetic nuclei remain longer after technical-valid filtering
- wider, more tapered, and more tortuous synthetic objects remain technical valid
- lower-count synthetic input remains lower count
- technical artifacts are rejected
- existing ROI, exclusion-mask, bit-depth, and brightness-invariance tests continue to pass

## Phase 6 Blinded Workflow Privacy Hardening

Purpose: harden the representative-image blinded validation workflow before any real WT/mutant source manifest, real microscopy stack, unblinding key, or genotype-bearing review output is generated.

No v5.5 file was edited. No production v5.6 pipeline or tuner file was edited in this phase. No real source manifest was processed, no real microscopy images were staged, no tracking was run, no complete stack was processed, no unblinding analysis was run, and no commit or push was run.

### Private Manifest Separation

The source manifest template now lives at:

```text
templates\source_manifest_v5_6.template.csv
```

Filled manifests must stay outside the repository. Recommended private location:

```text
C:\Users\dmishra\Desktop\sperm_validation_private\source_manifest_v5_6.csv
```

The blinded runner refuses to process a source manifest without `--private-output-dir`. The private output directory must not equal, or be inside, the blinded validation output or review-workbook output directory.

### Opaque Staging

Before segmentation, the runner stages only selected representative slices under:

```text
scratch\v5_6_blinded_inputs\B001\images\B001_z000.tif
scratch\v5_6_blinded_inputs\B001\roi\B001_roi.npy
```

The private Z-index and source-path mapping is written only to:

```text
<private-output-dir>\private_staged_input_mapping_v5_6.csv
```

The blinded manifest no longer carries original paths, sample IDs, dataset labels, or genotype labels. It carries only opaque handles and acquisition-class codes.

### Validate-Only Gate

Use this before any real validation run:

```powershell
.\.venv\Scripts\python.exe .\scratch\run_v56_blinded_validation.py `
  --manifest C:\Users\dmishra\Desktop\sperm_validation_private\source_manifest_v5_6.csv `
  --private-output-dir C:\Users\dmishra\Desktop\sperm_validation_private\v5_6_private_outputs `
  --validate-manifest-only
```

This validates manifest columns, input paths, requested/selected slices, planned blinded IDs, and planned opaque staged filenames. It creates no blinded outputs, copies no images, runs no segmentation, and writes no unblinding key.

### Leak Scanner

The blinded runner scans reviewer-facing CSV, JSON, XLSX, PDF metadata/text, filenames, and directories for private source terms before completing the blinded package. The scanner records only leak category and file path, not the leaked private value.

### Ignore Rules

The repository now ignores private/generated blinded material:

```text
scratch/v5_6_unblinding_key/
scratch/v5_6_blinded_inputs/
scratch/v5_6_blinded_validation/
*unblinding_key*.csv
*unblinding_key*.json
*source_manifest_v5_6.csv
*completed_manual_review*.xlsx
*completed_manual_review*.csv
```

The neutral template `templates\source_manifest_v5_6.template.csv` is intentionally not ignored.

### Added Privacy Tests

Synthetic-only tests now verify:

- private source fields are absent from the blinded manifest
- filled source manifests are not copied into blinded outputs
- overlapping private/blinded output directories are refused
- validate-manifest-only creates no blinded outputs or unblinding key
- opaque staged filenames preserve the original Z-index mapping privately
- only representative images are staged
- the leak scanner detects genotype text without reporting the private value
- `.gitignore` protects private outputs while leaving the template trackable

### Hardening Validation Results

Commands run:

```powershell
.\.venv\Scripts\python.exe -m py_compile .\scratch\run_v56_blinded_validation.py .\scratch\run_v56_unblind_validation.py
.\.venv\Scripts\python.exe -m py_compile .\sperm_segmentation_saturnv5.6.py .\utils\tune_parameters_Saturnv5_6.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe .\utils\tune_parameters_Saturnv5_6.py --self-check
git diff --check
git status --short
```

Results:

- runner py_compile passed
- production v5.6 pipeline/tuner py_compile passed
- pytest passed: `40 passed`
- tuner self-check passed
- `git diff --check` reported no whitespace errors, only CRLF normalization warnings
- no v5.5 file has a diff

## Unresolved Assumptions

- No exclusion mask was provided for this smoke test.
- v5.5 comparison used default v5.5 segmentation followed by ROI filtering, matching the known v5.5 behavior rather than a tuned full-batch parameter combination.
- Smoke-test visual observations are representative-slice checks only, not full-stack validation.

## Phase 7 N2V2 Archive, ROI Boundary Fix, And Ilastik Pilot Prep

The controlled N2V2 diagnostic trained a real CAREamics N2V2 model with changed weights and no fallback/mock model. Structure preservation failed: high-confidence raw nuclei lost ridge support, the ROI perimeter remained dominant, and the N2V2 branch did not produce usable interior candidates. N2V2 is therefore recorded as an unsuccessful experimental branch for the current sperm nucleus images.

Production status:

- `AI_PREPROCESSING_MODE` defaults to `off`
- N2V2 is disabled for production
- raw Saturn remains the active path
- further N2V2 development is paused
- no N2V2 training was run in this phase

### ROI Boundary Ridge Correction

The v5.6 ridge path now keeps the ROI exterior filled through denoising, CLAHE, background subtraction, foreground normalization, and ridge filtering. The exact biological ROI mask is applied after ridge calculation. Threshold estimation uses interior ROI pixels when possible, without silently discarding valid boundary detections.

Validation output:

```text
scratch\v5_6_roi_boundary_fix
```

On z05, z35, and z87, the mean 8-pixel boundary/interior ridge ratio changed from `0.07937` to `0.04255`; outside-ROI leakage remained `0`.

### Ilastik Pilot Preparation

Created neutral ilastik Pixel Classification inputs under:

```text
scratch\v5_6_ilastik_pilot
```

Training slices:

```text
z18, z25, z43, z50, z70, z78
```

Held-out evaluation slices:

```text
z05, z06, z12, z35, z60, z87
```

Training slices are distinct from evaluation slices and their immediate neighbor buffer. Exported images are boundary-safe robust-normalized images with no Saturn detections burned in. Raw references, ROI masks, and metadata JSON files are exported separately.

Class definition:

```text
scratch\v5_6_ilastik_pilot\metadata\ilastik_class_definition_v5_6.json
```

Guide:

```text
V5_6_ILASTIK_PIXEL_CLASSIFICATION_GUIDE.md
```

Prepared but not executed:

- `scratch\generate_v56_ilastik_headless_command.py`
- `scratch\validate_v56_ilastik_probability_maps.py`
- `scratch\run_v56_ilastik_probability_pilot.py`

No ilastik classifier, probability map, or ilastik-Saturn result was fabricated.
