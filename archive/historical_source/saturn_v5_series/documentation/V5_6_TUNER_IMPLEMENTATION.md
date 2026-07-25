# Saturn v5.6 Tuner Implementation

Source file created: `utils/tune_parameters_Saturnv5_6.py`.

The tuner imports `sperm_segmentation_saturnv5.6.py` by file path with module alias `sperm_segmentation_saturnv5_6`. Normal v5.6 execution does not import v5.5.

Supported modes:
- `--mode profile`
- `--mode segmentation`
- `--mode tracking`

Profile mode compares `no_clahe`, `high_contrast`, `standard`, `low_signal`, and `auto` using the same ROI, exclusion mask, representative slices, and stack context. Outputs use `*_v5_6_###` filenames.

Segmentation mode uses the requested v5.6 search space and keeps CLAHE fixed from profile/base/auto preprocessing. It records auditable objective subcomponents such as count CV, median length, width, length-width ratio, occupancy, bridge inflation, and exclusion overlap.

Tracking mode requires consecutive slices, segments once, caches segmentation summary, and tunes only tracking parameters afterward.

Representative slice selection supports `--slices auto` with default `--auto-slice-count 6`, approximating first, 20%, 40%, 60%, 80%, and last slices without duplicates.

Repeated base parameters are supported with `--base-params` using append semantics. Files merge in supplied order; later files override earlier files.

Every tuner segmentation call passes `roi_mask_global`, `preprocess_context_global`, and `exclusion_mask_global`.

Example commands:

```bash
python utils/tune_parameters_Saturnv5_6.py --mode profile --dir "path/to/images" --slices auto --roi-mask roi.tif
python utils/tune_parameters_Saturnv5_6.py --mode segmentation --dir "path/to/images" --slices auto --roi-mask roi.tif --profile auto --maxiter 12
python utils/tune_parameters_Saturnv5_6.py --mode tracking --dir "path/to/images" --slices 20-27 --roi-mask roi.tif --base-params preprocessing.json --base-params segmentation.json
python utils/tune_parameters_Saturnv5_6.py --self-check
```
