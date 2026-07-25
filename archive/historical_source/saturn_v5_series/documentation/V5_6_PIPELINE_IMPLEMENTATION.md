# Saturn v5.6 Pipeline Implementation

Source file created: `sperm_segmentation_saturnv5.6.py`.

Major changed functions:
- `resolve_pixel_parameters`
- `build_stack_preprocess_context`
- `save_stack_preprocess_context`
- `segment_slice`
- `bridge_skeleton_endpoints`
- `process_one_image`
- `process_batch`
- GUI preview and GUI batch segmentation paths

v5.6 uses ROI-adaptive preprocessing. When an ROI is supplied, the pipeline crops a padded ROI bounding box, median-fills pixels outside the ROI polygon and inside the crop, normalizes from ROI-minus-exclusion pixels, thresholds ridge response from ROI-minus-exclusion pixels, and reapplies the valid mask after hysteresis, cleanup, skeletonization, bridging, pruning, and junction breaking. Returned arrays are mapped back to full-image coordinates.

`StackPreprocessContext` stores stack normalization limits, selected stack-wide CLAHE profile, sampled Z indices, ROI percentiles, saturation and brightness statistics, source dtype/bit depth, resolved physical parameters, provenance, image shape, ROI area, and excluded area. QC is saved as `stack_preprocessing_qc.json` and `stack_preprocessing_qc.csv`.

New configuration keys include `PREPROCESS_MODE`, `LEGACY_TWO_PASS_ROI`, stack normalization percentiles, stack CLAHE profile controls, ROI crop padding, exclusion-mask path, physical denoise/background/ridge scales, physical morphology thresholds, bridge angle limits, loop/branch defaults, and ROI-edge QC distance.

Physical-unit keys take precedence over legacy pixel keys. `resolve_pixel_parameters(cfg)` converts physical values with `UM_PER_PX_XY` and returns physical values, resolved pixel values, and source provenance.

Exclusion masks are supported in stack context building and `segment_slice`; valid pixels are `ROI AND NOT exclusion`.

Endpoint bridging uses physical distance converted to pixels, tangent compatibility, and a valid-mask path check. Bridge stats are logged and included in debug JSON.

Debug outputs save numbered stages from raw normalized image through final detections plus a per-slice debug JSON. The current montage implementation is minimal and should be expanded during real-image review.

Backward compatibility is preserved for CSV/Excel/PDF/PPT naming patterns, tracking tables, audit outputs, GUI workflow, and legacy pixel-valued parameter JSON keys.

Known limitations:
- Real-image review panels are intentionally lightweight until representative-slice validation is run.
- `LEGACY_TWO_PASS_ROI` is retained as provenance but the default v5.6 path is single-pass ROI-adaptive.
- The automated tests use synthetic images, not the full microscopy stack.
