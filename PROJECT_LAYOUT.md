# Sperm Project Layout

This folder contains the Saturn sperm nucleus segmentation pipeline plus tuning
outputs and scratch review artifacts.

## Main source files

- `sperm_segmentation_saturnv5.2.py` - current working Saturn pipeline.
- `sperm_segmentation_saturnv5.1.py` - preserved previous version.
- `utils/tune_parameters_Saturnv5_2.py` - current segmentation/tracking tuner.
- `utils/audit_sperm_outliers.py` - standalone post-run outlier audit.

## Parameter files

- `parameter_presets/` - hand-kept parameter presets worth reusing.
- `parameter_tuning_results/` - tuner outputs, review notes, and candidate JSONs.
  Keep reviewed `best_*params*.json` files and review markdown notes; generated
  review panels and full search histories are ignored by Git.

## Generated or local-only folders

- `.venv/` - current local Python environment.
- `scratch/` - temporary probes, visual comparisons, and smoke-test outputs.
- `sperm_results_saturnv*/` - generated batch results.
- `batch_output*/` - generated batch outputs.
- `build/` and `dist/` - packaged app artifacts.

## Current recommendation

For the ROI `roi_z28.1.npy` work, use the v5.2 segmentation-tuned parameters as
the current best segmentation candidate. The tracking candidate
`best_safe_tracking_params_v5_2_003.json` should be treated as experimental
because it improves continuity but shifts 3D lengths upward.
