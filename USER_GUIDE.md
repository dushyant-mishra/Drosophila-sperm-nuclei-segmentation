# Saturn v5.7 User Guide

Saturn v5.7 analyzes Drosophila sperm nuclei in fluorescence Z-stacks. It
combines ROI-aware classical segmentation, optional 2.5D U-Net probability
support, cross-slice tracking, morphology measurement, quality-control
populations, and multi-sample study management.

The current application is a Python GUI. Historical packaged applications and
older Saturn versions are preserved under `archive/` and are not the active
workflow.

## 1. Install and Start

From PowerShell:

```powershell
Set-Location "C:\Users\dmishra\Desktop\sperm_project"
.\.venv\Scripts\Activate.ps1
python .\sperm_segmentation_saturnv5.7.py --gui
```

Launching without `--gui` also opens the application:

```powershell
python .\sperm_segmentation_saturnv5.7.py
```

To create the environment again:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The optional U-Net workflow also requires a compatible PyTorch environment and
a locally stored v5.7-compatible checkpoint. Model checkpoints and raw
microscopy data are intentionally not stored in Git.

## 2. Before Analyzing Data

For each biological specimen, confirm:

1. The folder contains one intended Z-stack series.
2. Slice names sort into the correct Z order.
3. XY pixel size and Z spacing match the microscope metadata.
4. A specimen-specific ROI is available or can be drawn.
5. The selected parameter JSON and U-Net checkpoint were validated for the
   acquisition conditions.
6. The output folder is separate from source images when running a
   multi-sample study.

Do not tune parameters toward a desired genotype count, group difference, or
morphology. Parameter selection should be based on blinded visual quality and
technical performance.

## 3. Analyze One Stack

1. Select **Load Directory** and choose any image in the stack.
2. Use the Z slider to inspect the full stack, especially the middle planes
   where nuclei are most visible.
3. Draw a new ROI or select **Load ROI**.
4. Confirm that the red ROI outline follows the specimen on several Z planes.
5. Select **Load Tuned Params** when using a reviewed v5.7 parameter JSON.
6. In **Configure Parameters**, confirm calibration, segmentation engine, and
   U-Net checkpoint settings.
7. Use **Run Slice** for a representative visual check. Its count is labeled
   **2D preview candidates** and must not be interpreted as unique nuclei.
8. Use **Run Batch (All Slices + 3D Track)** only after the slice check looks
   reasonable.
9. Review overlays and measurement tables before interpreting summary plots.

Each batch is written to a new `batch_output`, `batch_output_1`, and so on.

The top of **Configure Parameters** reports the active segmentation engine and
whether the selected U-Net checkpoint exists. The **2.5D U-Net Integration**
section appears directly below calibration and provides dropdowns for modes,
checkboxes for enable/disable settings, and **Browse** for the checkpoint.
Existing batches are not overwritten.

## 4. Draw, Save, and Reuse an ROI

### Draw an ROI

- Select the ROI drawing view.
- Left-click to place polygon points around the specimen.
- Right-click to undo the most recent point.
- Press **Enter** after placing at least three points.
- Select **Save ROI** to save the binary mask as a NumPy `.npy` file.

### Load an ROI

1. Load an image stack first.
2. Select **Load ROI** and choose the `.npy` mask.
3. Confirm that the mask dimensions match the images.
4. Inspect the red outline on top, middle, and bottom planes.

Never reuse an ROI merely because two images have the same dimensions. Every
ROI must match the specimen position and anatomy in its own stack.

For multi-sample discovery, save the reviewed ROI in each specimen folder as:

```text
analysis_roi_v5_7.npy
```

## 5. ROI-Aware Normalization

Saturn calculates preprocessing statistics from valid pixels inside the ROI.
Bright tissue outside the specimen therefore does not set the ROI thresholds.
The stack context samples representative Z planes and reuses one normalization
context across candidate evaluations.

This normalization improves technical comparability, but it does not transform
counts from specimens with different sampled areas or depths into directly
equivalent raw counts. The study manager separately reports exposure-normalized
rates based on ROI area and sampled depth.

Review the normalization warnings when:

- ROI area or sampled volume is invalid.
- Detections reach an acquisition Z boundary.
- More than 20% of tracks touch a Z boundary.
- A stack appears truncated relative to another specimen.

## 6. Classical, U-Net, and Hybrid Segmentation

`SEGMENTATION_ENGINE` controls the evidence used during segmentation:

- `classical_saturn`: ROI-aware classical ridge segmentation only.
- `hybrid`: classical detections plus a U-Net rescue lane.
- `unet_assisted`: enables U-Net evidence within the v5.7 integration path.

The 2.5D U-Net receives the previous, center, and next Z planes as three input
channels. Tiled inference runs on ROI-aware crops and stitches probabilities
back into full-frame coordinates. Stitching probabilities does not resize the
source images or directly alter length and width measurements.

The U-Net produces continuous probability evidence. Saturn then:

1. Finds U-Net-supported regions not already represented by classical
   detections.
2. Splits connected regions into putative instances.
3. Builds centerlines and measures the underlying geometry.
4. Accepts plausible rescued nuclei and records their source.
5. Retains rejected U-Net-positive candidates in review overlays.

The U-Net does not use COCO files during inference. COCO annotations are
training data only; runtime inference reads raw image planes and a model
checkpoint.

![Saturn v5.7 2.5D U-Net inputs, probability evidence, and integrated result](docs/readme_assets/saturn_v57_unet_integration.png)

## 7. Overlay Color Cues

In U-Net rescue review overlays:

- **Green:** accepted Saturn classical detection.
- **Cyan:** accepted U-Net rescue detection.
- **Magenta:** U-Net-positive candidate rejected as a short fragment.
- **Orange:** U-Net-positive candidate rejected for width or low
  length-to-width ratio.
- **Red:** U-Net-positive candidate rejected for long, branched, looped, or
  tortuous topology.
- **Red specimen outline:** the analysis ROI.

Overlay line thickness is display-only. Counts, lengths, widths, and tracking
use the underlying masks and centerlines, not the rendered colored strokes.

Red, orange, or magenta detections are not automatically biological negatives.
They are candidates that did not pass the configured rescue gates and should
be inspected when tuning for a new acquisition type.

## 8. Cross-Slice Tracking

Tracking links compatible 2D detections across adjacent Z planes using
calibrated distance, overlap, orientation, morphology continuity, and optional
U-Net support. One 3D track can therefore contain several 2D observations of
the same nucleus.

Do not calculate single-plane nuclei as:

```text
2D detections minus 3D tracks
```

Those quantities count different things. Use the per-track `n_slices` field to
identify tracks observed in exactly one plane.

Tracking errors usually appear as:

- **Fragmentation:** one nucleus becomes several short tracks.
- **False fusion:** neighboring nuclei are joined into one track.
- **Boundary truncation:** tracks begin or end at the first or last acquired
  plane.

Review track overlays and source tables before changing tracking thresholds.

## 9. Analysis Populations

Saturn exports several deliberately separate populations:

- **Raw 2D detections:** all accepted per-slice measurements.
- **Reconstructed tracks:** the complete cross-slice tracking output retained
  in the technical audit table.
- **Estimated unique nuclei:** technical-valid reconstructed tracks; this is
  the one biological analysis population.
- **Technical-valid tracks:** tracks that pass acquisition and tracking
  integrity checks. This is the primary table for WT-versus-mutant
  comparisons.
- **Reference morphology subset:** technical-valid tracks also compatible with
  reference morphology limits.
- **Morphology warnings:** technically valid tracks outside the reference
  shape limits. These should not be silently discarded from mutant studies.
- **Quality subset:** a stricter reporting population retained for sensitivity
  review, not necessarily the sole biological population.

For genotype comparisons, treat biological specimens as replicates. Individual
nuclei are measurements nested within specimens, not independent biological
replicates.

## 10. Run a Multi-Sample Study

1. Place `analysis_roi_v5_7.npy` in every specimen folder.
2. Open **Multi-Sample Study** and select **Open Study Manager**.
3. Select **Discover Root** and choose the parent study folder.
4. Review sample ID, slices, Z range, ROI, XY calibration, and Z spacing.
5. Select one or more rows and use **Assign Group** to explicitly label them
   `WT`, `KJ`, `Mutant`, or another biological group.
6. Double-click editable fields to correct individual metadata.
7. Toggle **Include** for specimens that should not run.
8. Optionally select **Organize Dataset Copy** and choose a separate empty
   folder. The reviewed group labels define the canonical group folders.
9. Select a separate analysis-output folder outside both the original and
   organized source trees.
10. Select **Validate** and resolve every invalid row.
11. Select **Run / Resume Study**.

The progress bar reports completed specimens, not individual Z slices. While a
study is running, select **Stop After Current Sample** to request a controlled
pause. The current specimen is allowed to finish so its tables, report, and
completion marker remain consistent; no new specimen is then started. Select
**Run / Resume Study** later to skip completed specimens and continue with the
remaining rows when the parameter fingerprint is unchanged.

### Supported Source-Image Names

The study manager discovers top-level TIFF planes using conservative filename
families. Matching is case-insensitive:

- Leica exports, including `Project_Series002_z00_ch00.tif` and
  `Project001_Series002_z00_ch00.tif`.
- Names with an explicit Z token and optional channel-zero token, such as
  `SampleA_z000.tif`, `SampleA-Z000-C0.TIF`, or
  `SampleA_z000_ch00.tiff`.
- Channel-free trailing numeric indices, such as `SampleA_0001.tif`.

Only channel `0` is accepted when a `c` or `ch` token is present. Generated
output directories such as `overlays`, `masks`, `debug`, and `batch_output*`
are excluded. Every discovered stack must still have unique, contiguous Z
indices, consistent dimensions, and a matching ROI. Names without an explicit
Z token or trailing numeric index should be renamed before study discovery.

### Canonical Dataset Organization

**Organize Dataset Copy** never renames, moves, or deletes original microscopy
files. It creates a separate structure:

```text
OrganizedStudy/
|-- WT/
|   `-- WT_01/
|       |-- WT_01_z0000_ch00.tif
|       |-- WT_01_z0001_ch00.tif
|       |-- analysis_roi_v5_7.npy
|       `-- MetaData/
|-- Mutant/
|   `-- Mutant_01/
|       `-- ...
|-- organized_study_manifest.csv
|-- source_file_mapping.csv
`-- organization_summary.json
```

The source mapping records every original and canonical path. Existing ROIs are
copied only to their corresponding specimen. Missing ROIs remain missing and
must be drawn for that specimen; the organizer never borrows an ROI from
another sample. Calibration and acquisition labels are retained in the
organized manifest.

The manager runs specimens independently. A failed specimen does not stop the
remaining samples, and completed specimens can resume without rerunning when
the same parameter fingerprint is used.

### Study Outputs

- `study_manifest.csv`: exact specimen, group, source, ROI, and calibration.
- `study_run_state.json`: per-specimen state and resume information.
- `runtime_parameters.json`: shared v5.7 parameters used for the study.
- `specimen_summary.csv`: one raw and normalized summary row per specimen.
- `group_summary.csv`: group summaries calculated from specimen rows.
- `normalization_qc.json`: exposure and Z-boundary warnings.
- `specimen_group_comparisons.csv`: exploratory two-group comparisons in which
  each biological specimen is one replicate. It reports group medians, the
  comparison-minus-reference median difference, percent difference, Cliff's
  delta, a bootstrap interval, a permutation p-value, and an FDR-adjusted
  q-value.
- `specimen_group_comparison.pdf`: dot plots showing every specimen and the
  group medians. A clearly named `WT`, `wild type`, or `w1118` group is used as
  the reference when present.
- `specimen_group_comparison_qc.json`: sample sizes, random seed, comparison
  direction, and small-sample warnings.
- `study_track_records.csv`: pooled tracks with unique `study_track_id`. This
  file is for descriptive summaries and audit; its nuclei are nested within
  specimens and must not be treated as independent biological replicates.
- `samples/<sample_id>/attempt_NNN/`: complete output for one specimen.

The specimen morphology fields are calculated from the technical-valid track
population used for `estimated_unique_nuclei`. A technical-failure outlier
therefore cannot alter the reported specimen median length, tortuosity,
thickness, volume, or Z span. The two-group comparison is intentionally marked
exploratory, especially when either group has fewer than five specimens.

## 11. Count Normalization

Normalization does not add, remove, or resize detections. It adds denominators
that make sampling exposure explicit:

```text
ROI area (um2) = ROI pixels x XY pixel size x XY pixel size
Sampled depth (um) = included slices x Z spacing
Sampled ROI volume (um3) = ROI area x sampled depth
```

The study table reports, among other fields:

- Raw 2D detections per `1,000 um2` per slice.
- 3D tracks per `1,000 um2`.
- 3D tracks per `100,000 um3`.
- Biological and quality-track rates using the same denominators.
- Stack span, positive Z range, and Z-boundary track fraction.

Use raw counts for traceability and normalized rates for exposure-aware
comparison. Neither corrects biological sampling bias, incomplete stacks, poor
ROIs, or acquisition differences.

## 12. Standard Batch Outputs

A typical output directory contains:

```text
batch_output/
|-- overlays/
|-- quality_overlays/
|-- plots/
|-- biologist_results/
|   |-- sample_summary.csv
|   |-- nuclei_for_analysis.csv
|   `-- README.txt
|-- analysis_summary.csv
|-- analysis_summary.json
|-- spermatid_measurements.csv
|-- track_summary.csv
|-- batch_analysis_results_v5.7.xlsx
`-- batch_report_v5.7.pdf
```

Exact optional files depend on configuration and whether U-Net inference,
quality overlays, and presentation export are enabled.

Use `biologist_results/sample_summary.csv` for sample comparisons and
`biologist_results/nuclei_for_analysis.csv` for nucleus-level analysis.
`analysis_summary.csv` and `analysis_summary.json` provide the same concise
primary result contract for GUI batch, command-line batch, and Study Manager
specimen runs. For a full tracked stack, they report the estimated unique
nuclei and morphology medians from technical-valid tracks. For **Run Slice**,
they explicitly report a 2D preview population and leave
`estimated_unique_nuclei` unavailable.
`track_summary.csv` is the complete technical audit table. The legacy
`is_biological_candidate` column is identical to `technical_valid`; it is
retained for compatibility and does not represent another population.

## 13. Measurement Nomenclature

- **2D geodesic length:** centerline path length within one image plane.
- **Width:** local mask width estimated from the unbridged clean distance map.
- **Length-to-width ratio:** elongation measure based on length and width.
- **3D length:** calibrated path length through linked observations.
- **Z-span:** endpoint-to-endpoint Z displacement:
  `(max_z - min_z) x Z spacing`.
- **Z-covered:** sampled slab thickness:
  `(max_z - min_z + 1) x Z spacing`.
- **Z-extent:** number or range of planes represented by a track, depending on
  the output field.
- **Tortuosity:** path length divided by endpoint displacement; values near
  one are straighter.
- **Pitch angle:** orientation of the reconstructed path relative to the image
  plane.
- **Approximate volume:** voxel- and PSF-sensitive integrated area estimate.
- **Reference morphology subset:** technical-valid tracks that also satisfy
  the reference shape limits. It is not synonymous with all real nuclei.

Volume, width, effective thickness, taper, and related measurements are
especially sensitive to microscope PSF and voxel sampling. Use them mainly for
relative comparison among specimens acquired and processed under matched
conditions.

## 14. Parameter Groups

| Group | Main controls | Change only when |
| :--- | :--- | :--- |
| Calibration | `UM_PER_PX_XY`, `UM_PER_SLICE_Z` | Microscope metadata differs. |
| Preprocessing | `CLAHE_*`, `BG_SIGMA`, normalization settings | ROI images have systematically different contrast or background behavior. |
| Classical segmentation | `RIDGE_SIGMAS`, `THRESHOLD_HI`, `THRESHOLD_LO` | Classical overlays are visibly too strict or permissive. |
| Cleanup | `MAX_BRIDGE_PX`, branch and junction controls | True nuclei fragment or unrelated structures connect. |
| Morphology | minimum length, maximum width, ratio, topology limits | Review overlays show plausible nuclei receiving technical-failure flags. |
| U-Net evidence | candidate, seed, rescue, split, and centerline thresholds | Probability maps are useful but rescue acceptance is too strict or permissive. |
| Tracking | distance, gap, overlap, assignment, and continuity controls | Tracks fragment or falsely fuse. |
| Audit/reporting | technical, reference-shape, and quality limits | Population labels need adjustment without changing detection or tracking. |

Load a reviewed parameter JSON rather than editing many values manually. Save
the exact runtime parameters with every study.

## 15. Adapting to New Image Conditions

For a new microscope, magnification, fluorophore, genotype, or preparation:

1. Confirm calibration and bit depth.
2. Draw specimen-specific ROIs.
3. Inspect representative top, middle, and bottom planes.
4. Run the tuner on informative planes, not empty early slices.
5. Compare probability maps separately from accepted rescue detections.
6. Review rejected U-Net candidates by reason.
7. Validate tracking on consecutive planes.
8. Run a small blinded smoke test before a complete study.
9. Keep one shared parameter set for the comparative study unless a
   predeclared technical reason requires otherwise.
10. Document every parameter and model checkpoint used.

Fine-tuning the U-Net on new data should use reviewed annotations, held-out
specimens, partial-label-aware loss where annotations are incomplete, and
probability-map review before Saturn rescue tuning.

## 16. Common Problems

### The GUI detects unrelated TIFF files

Keep one intended source series per specimen folder. Move unrelated channels,
exports, masks, and overlays outside the source folder before analysis.

### The ROI does not align

Confirm image dimensions and specimen identity. Draw a new ROI rather than
resizing an old mask.

### Many visible nuclei appear red, orange, or magenta

Inspect the raw U-Net probability map. The model may have found the nuclei
while Saturn's rescue gates rejected them. Tune rescue splitting and
centerline recovery before loosening every biological limit.

### Cyan overlays look thicker

Rendered outlines can look different because they come from distinct masks.
The standardized overlay linewidth is visual only; geometry is measured before
rendering.

### Counts differ greatly between specimens

Check ROI area, sampled depth, Z-boundary warnings, calibration, raw image
quality, and U-Net rescue fraction before interpreting a biological
difference.

### Inference is slow

Use ROI-tiled inference, a compatible GPU, suitable tile batching, and cached
probability maps during repeated tuning. Do not recompute U-Net probabilities
for every threshold candidate.

## 17. Privacy and Reproducibility

The GitHub repository is private. Raw microscopy data, ROI masks, model
checkpoints, active tuning outputs, generated reports, API keys, and local
paths remain excluded from version control.

For every reported experiment, retain:

- Source-image manifest and microscope metadata.
- ROI file and ROI QC.
- Runtime parameter JSON.
- Model checkpoint identity or checksum.
- Software commit ID.
- Per-specimen outputs and normalization warnings.
- Manual-review notes and any exclusions.

The figures in the repository README and this guide are illustrative
development examples. They are not a genotype result or an accuracy benchmark.
