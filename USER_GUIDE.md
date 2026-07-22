# Sperm Nuclei Segmentation macOS Tool
## User Walkthrough and Parameter Notes

### Validated Dataset
- **Saturn image datasets**: [Open link](https://example.com) (Update with actual link)

### Tool Download
- **Standalone macOS tool**: [Open link](https://example.com) (Update with actual link)

### Tuned Parameters JSON
- **Best parameters file**: [Open link](https://example.com) (Update with actual link)

> [!IMPORTANT]
> **Recommended default**: Use the built-in Saturn V5 tuned parameters and leave settings unchanged unless clearly necessary.
> This release was tested and optimized on the Saturn WT-style image set. For Saturn-like datasets, the safest workflow is to run the built-in tuned defaults before changing any settings.

---

## What This App Does
- **Detects** sperm nuclei in each z-slice.
- **Links** detections across slices into 3D tracks.
- **Applies** a post-tracking quality audit.
- **Exports** CSV, Excel, PDF, and PowerPoint summaries.

---

## Important Interpretation Notes
- **Single-slice nuclei** can still be biologically valid in this acquisition regime and should not be treated as automatically incorrect.
- **Volume, effective thickness, taper ratio**, and other width/area-derived outputs are PSF- and voxel-sensitive. Use them mainly for relative comparison between matched datasets rather than as literal physical dimensions.
- **Primary biological readouts** are typically geodesic length, tortuosity, Z-extent, pitch angle, and track continuity.

---

## Recommended Workflow
1. Open the app.
2. Select the image folder containing one z-stack series.
3. Keep the built-in tuned parameters unchanged unless the results clearly show a problem.
4. Optionally load a newer or experiment-specific tuned JSON if one has been provided.
5. Draw or load the ROI.
6. Run the batch analysis.
7. Review the PDF report, Excel workbook, and overlay images.

---

## How to Draw and Edit the ROI

### Drawing a New ROI
- **Left-click** to place points around the region you want analyzed.
- Continue clicking to build the polygon outline.
- **Press Enter** to finalize the ROI after placing at least three points.

### Erasing ROI Points
- **Right-click** to remove the most recently placed point.
- Use this whenever you mis-click while tracing the region.

### Reusing an Old ROI
- If the build supports ROI reuse, load a saved ROI and visually confirm the overlay before proceeding.
- Reject the ROI and draw a new one if it does not match the current stack or tubule region.

---

## Running a Multi-Sample Study in V5.7

1. Save `analysis_roi_v5_7.npy` inside every sample's image folder.
2. Open **Multi-Sample Study** in the sidebar and select **Open Study Manager**.
3. Select **Discover Root** and choose the parent folder containing all specimens.
4. Review the table. Double-click a sample ID, biological group, XY calibration, or Z spacing to edit it. Toggle **Include** for specimens that should not run.
5. Select an output folder outside the source study folder, then select **Validate**.
6. Resolve every invalid row before selecting **Run / Resume Study**.

The manager reads each specimen's Leica metadata when available, validates its ROI against every source image, and runs specimens independently. Completed specimens resume without rerunning when the same parameters are used. Failed specimens remain isolated and do not stop the remaining samples.

Study-level outputs include:

- `study_manifest.csv`: exact specimen, group, ROI, calibration, and source settings.
- `study_run_state.json`: per-specimen status and resume information.
- `runtime_parameters.json`: the shared V5.7 parameters used for the study.
- `specimen_summary.csv`: one raw summary row per biological specimen.
- `study_track_records.csv`: pooled track records with globally unique `study_track_id` values.
- `samples/<sample_id>/attempt_NNN/`: the complete standard V5.7 output for one specimen.

ROI-area and ROI-volume normalized biological readouts are not yet added to the aggregate table. The current `specimen_summary.csv` intentionally reports raw counts and morphology so normalization can be introduced and audited separately.

---

## Where Results Are Saved
All outputs are saved inside the selected input image folder in an auto-created subfolder such as `batch_output`, `batch_output_1`, and so on.

```text
MyExperiment/
|-- z0.tif
|-- z1.tif
|-- z2.tif
`-- batch_output/
    |-- overlays/
    |-- spermatid_measurements.csv
    |-- track_summary.csv
    |-- track_summary_quality.csv
    |-- batch_analysis_results_v5.xlsx
    |-- batch_report_v5.pdf
    `-- batch_analysis_results_v11.pptx
```

---

## Parameter Notes
These notes are for interpretation. For this release, the built-in Saturn V5 tuned parameters should be considered the default for Saturn WT-style data.

| Section | Parameter | What it controls | When to change it |
| :--- | :--- | :--- | :--- |
| **Calibration** | `UM_PER_PX_XY` / `UM_PER_SLICE_Z` | Convert pixels and slice spacing into physical units. They affect 3D length, Z-extent, pitch, and volume calculations. | Only if the microscope calibration or z-step differs from the Saturn acquisition. |
| **Segmentation** | `CLAHE_CLIP`, `CLAHE_KERNEL`, `BG_SIGMA`, `RIDGE_SIGMAS` | Control contrast enhancement, background subtraction, and ridge detection in the raw 2D images. | Only if the raw detections themselves are visibly wrong. |
| **Segmentation** | `THRESHOLD_HI` / `THRESHOLD_LO` | Upper and lower hysteresis thresholds used to convert the ridge image into a binary mask. | Only if the pipeline is clearly too permissive or too strict at the 2D detection stage. |
| **Cleanup** | `MAX_BRIDGE_PX`, `MAX_BRANCH_LEN_PX`, `BREAK_JUNCTIONS` | Control how nearby skeleton fragments are reconnected or pruned after segmentation. | Change only if true nuclei are clearly being broken apart or dense webs are not being separated enough. |
| **Morphology** | `MIN_SKEL_LEN_PX`, `MAX_GEODESIC_LEN_PX`, `MAX_WIDTH_PX`, `MIN_LENGTH_WIDTH_RATIO` | Reject short debris, implausibly long chains, broad objects, or non-elongated detections. | Adjust only if many obvious valid nuclei are filtered out or many obvious merged objects are surviving. |
| **Tracking** | `TRACK_MAX_DIST_UM`, `TRACK_MAX_GAP_SLICES`, `TRACK_BBOX_PADDING_PX` | Control how detections are linked across adjacent z-slices. | Change only when tracks are obviously too fragmented or obviously fusing neighboring nuclei. |
| **Tracking** | `OVERLAP_*` and `CONSERVATIVE_MAX_*` | Control overlap continuation and allowable slice-to-slice jumps in width, length, area, orientation, and centroid position. | These were optimized on Saturn WT data by another algorithm; do not change unless necessary. |
| **Audit** | `AUDIT_MAX_LENGTH_UM`, `AUDIT_MAX_TORTUOSITY`, `AUDIT_MAX_THICKNESS_UM`, `AUDIT_MAX_TAPER_RATIO`, `AUDIT_MIN_SLICES` | Post-tracking rules that flag suspicious tracks in the quality subset. Audit does not change raw detection or linking. | Use audit changes first when the overlays look good but the summaries feel too strict or too lenient. |
| **Output / Debug** | `SAVE_*` and `SHOW_*` options | Control overlays, masks, labels, preview windows, and debug images. | Change only if you need more visual QC or lighter output. |

---

## Plain-Language Parameter Guide

- **Audit parameters**: These are post-tracking only. They do not change raw segmentation or linking. They only determine which completed tracks are flagged as suspicious in the quality population.
- **Tracking parameters**: These control how 2D detections are linked into 3D tracks. Change them only if nuclei are clearly too fragmented or neighboring nuclei are clearly being fused.
- **Segmentation parameters**: These affect raw 2D detection. Change them only if the raw overlays themselves look wrong.
- **PSF-sensitive outputs**: Volume, effective thickness, taper ratio, and other width/area-derived values are broadened by microscope PSF and voxel sampling. Use them mainly for relative comparison between matched datasets.

---

## Common Mistakes to Avoid
- Loading an experiment-specific JSON without confirming it matches the current dataset.
- Changing parameters before running the built-in tuned defaults once.
- Treating PSF-sensitive outputs as literal physical dimensions.
- Accepting the wrong ROI without visually confirming it.
- Changing segmentation settings when the real problem is tracking or audit.

---

## Distribution Note
**Saturn-V5-macOS-Tool** is a standalone macOS application for confocal z-stack analysis of Drosophila sperm nuclei. It detects nuclei in 2D, links them into 3D tracks, applies a biologically informed audit, and exports CSV, Excel, PDF, and PowerPoint summaries. This release was tested and optimized on the Saturn image datasets.
