# Saturn v5.7 U-Net Integration Plan

Saturn v5.7 is a non-breaking copy of v5.6 prepared for optional 2.5D U-Net support. The default segmentation engine remains classical Saturn.

## Runtime Design

COCO annotations are training-only. The Saturn GUI and batch pipeline should not read COCO files during inference.

At runtime, U-Net inference should use:

- a trained `.pt` checkpoint from `unet25d/train_unet25d.py`
- the raw image stack
- a 3-plane context for each slice: `[z-1, z, z+1]`
- ROI and optional exclusion mask from Saturn

The model should produce a 2D probability map for the center slice. Saturn can then threshold that probability map into candidates and apply the same ROI-aware QC, measurement, overlay, and reporting machinery.

## Proposed Modes

- `classical_saturn`: current v5.6-style segmentation. This is the v5.7 default.
- `unet_assisted`: U-Net probability map proposes candidate nuclei, then Saturn measures and audits them.
- `hybrid`: U-Net proposes high-confidence nuclei; classical Saturn recovers likely missed elongated nuclei.

## Reporting

Main reports should stay simple:

- `unet_detected`
- `saturn_recovered`
- `final_accepted`
- `excluded_from_measurement`

Detailed implementation labels such as bridge completion, extension completion, merged candidates, and borderline QC should go into debug CSV/JSON files, not the first-page summary.

## Current Scaffold

`utils/saturn_unet25d_bridge.py` provides lazy PyTorch loading and checkpoint-based probability prediction. This keeps classical Saturn usable on machines without PyTorch installed.

`sperm_segmentation_saturnv5.7.py` now includes configuration placeholders for model path, segmentation engine, threshold, ROI-tiled inference, and model-accounting labels. No model inference is enabled by default.

The 3D linker is now U-Net-aware when `SEGMENTATION_ENGINE` is `hybrid` or `unet_assisted` and detection rows contain optional probability columns such as `unet_mean_probability`. The global-assignment tracker and hybrid fragment-repair pass add small penalties for weak U-Net support and abrupt probability changes across z. Classical Saturn mode ignores these fields and should behave like v5.6.

Track summaries now retain optional per-track U-Net support columns, for example `track_mean_unet_mean_probability` and `track_max_unet_mean_probability`, so reports can later separate:

- final measurable nuclei
- U-Net-supported detections
- Saturn-recovered detections
- excluded candidates
