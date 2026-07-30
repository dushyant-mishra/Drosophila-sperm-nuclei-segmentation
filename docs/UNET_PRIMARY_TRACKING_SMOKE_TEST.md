# U-Net Primary Tracking Smoke Test

## Objective
To extend the existing Saturn v5.7 U-Net-primary smoke workflow from independent 2D slices to a consecutive-Z tracking smoke test without mutating the core production tracker.

## Validation Targets
The runner strictly evaluates tracking on a small contiguous sequence of Z-slices (default: `z33, z34, z35, z36, z37`).
It evaluates tracking determinism, boundary vs interior track behavior, and verifies that U-Net specific fields flow unchanged through tracking.

## U-Net Morphological Fields
- `area_px` is conditionally set to `instance_mask_area_px` for U-Net primary measurements.
- Classical and Hybrid `area_px` remain derived from geodesic length and median width.
- `estimated_slender_area_px` stores the historical length*width proxy for all instances.
- U-Net metrics such as `unet_mean_probability`, `source_instance_key`, `morphology_warning` are passed verbatim through the measurement pipeline.

## Diagnostics Missing from Production Tracker
- The production `track_across_slices` does not return `assignment cost` in the output dataframe.
- Link types or repair costs are not preserved per-observation.
- Area and width changes across the track are not natively stored in the `df_tracked` table. 

## Integrity Checks
- **Pre-tracking:** Validates 1:1 row per instance constraint, non-empty centroids, unique keys.
- **Post-tracking:** Validates successful linkage, uniqueness of source instance keys, determinism across identical reruns, and properly classifies tracks that span out of the boundary slice window.
