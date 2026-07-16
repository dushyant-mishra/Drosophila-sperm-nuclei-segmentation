# Saturn V5.3 Tracking Prototype

This prototype was created so tracking experiments do not touch the working
Saturn V5.2 files.

## New Files

- `sperm_segmentation_saturnv5.3.py`
- `utils/tune_parameters_Saturnv5_3.py`
- `parameter_presets/saturnv5.3_global_assignment_prototype.json`

## What Changed

V5.3 adds a tracking backend switch:

```json
{
  "TRACKING_BACKEND": "legacy"
}
```

Supported values:

- `legacy` - the V5.2 greedy overlap/centroid tracker.
- `global_assignment` - a new prototype tracker using SciPy linear assignment.

The global-assignment tracker scores candidate links with a weighted cost:

```text
cost =
  distance_weight * centroid_distance
+ overlap_weight  * non_overlap_penalty
+ length_weight   * relative_length_change
+ width_weight    * relative_width_change
+ area_weight     * relative_area_change
+ angle_weight    * orientation_change
```

It then uses `scipy.optimize.linear_sum_assignment` to choose the best
track-to-detection links for each slice transition.

## How To Try It

In the V5.3 GUI:

1. Load the current segmentation-tuned JSON.
2. Load `parameter_presets/saturnv5.3_global_assignment_prototype.json`.
3. Run a small ROI batch first.
4. Compare post-detection QC against the V5.2 default-tracking run.

For tuning:

```powershell
.\.venv\Scripts\python.exe .\utils\tune_parameters_Saturnv5_3.py --mode tracking --dir "C:\Users\dmishra\Desktop\sperm images" --slices 28-32 --roi-mask "C:\Users\dmishra\Desktop\sperm images\roi_z28.1.npy" --params .\parameter_tuning_results\roi_z28_middle_segmentation\best_segmentation_params_v5_2_001_run_sperm_images.json --maxiter 0 --popsize 1 --no-polish --outdir .\parameter_tuning_results\roi_z28_middle_tracking_v5_3
```

## Current Status

Experimental. V5.2 remains the production baseline. Use V5.3 to compare whether
global assignment reduces fragmentation without inflating 3D length.
