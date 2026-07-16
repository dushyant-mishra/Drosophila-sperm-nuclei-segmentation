# Drosophila Sperm Nuclei Segmentation (Saturn V5)

A professional-grade pipeline for automated 2D/3D segmentation and analysis of Drosophila sperm nuclei from confocal microscopy Z-stacks.

## Standalone macOS Tool
For the most stable experience on macOS, use the standalone application built in a clean environment.

1. **Download**: Go to the [GitHub Actions](https://github.com/dushyant-mishra/Drosophila-sperm-nuclei-segmentation/actions) page.
2. **Select Build**: Click the most recent successful "Build macOS Standalone App" run.
3. **Artifacts**: Scroll down to the **Artifacts** section and download `Saturn-V5-macOS-Tool`.
4. **Launch**: Unzip the file and right-click `SpermAnalysisTool.app` -> **Open**.

---

## Developer Setup (Python)
If you prefer to run the tool from source:

1. **Requirements**: Ensure Python 3.10+ is installed.
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run GUI**:
   ```bash
   python sperm_segmentation_saturnv5.1.py --gui
   ```

---

## Recommended Workflow
1. **Open the App**: Launch the GUI using one of the methods above.
2. **Select Folder**: Click **Load Directory** and select the folder containing your Z-stack images.
3. **Confirm Parameters**: Saturn V5 tuned tracking parameters are built in by default. Use **Load Tuned Params** only if you need to apply a newer or experiment-specific JSON.
   > [!IMPORTANT]
   > For Saturn-like datasets, the safest workflow is to run the built-in tuned defaults before changing any settings.
4. **Define ROI**: Select the **Draw ROI (Polygon)** tool and trace the region of interest.
5. **Run Batch**: Click **Run Batch (All Slices + 3D Track)**.
6. **Review Results**: Automated reports (PDF, Excel, PPTX) will open once complete.

---

## ROI Management
- **Drawing**: Left-click to place points. Press **Enter** to finalize the polygon.
- **Erasing**: Right-click to remove the last point if you mis-click.
- **Reusing**: Load a saved `.npy` mask via **Load ROI Mask** to apply the same region across different stacks.

---

## Data Outputs & Interpretation
All results are saved in an auto-created subfolder, such as `batch_output/`, within your input directory.

| File | Description |
| :--- | :--- |
| `batch_report_v5.pdf` | High-resolution graphical report with 3D distributions and quality audits. |
| `batch_results_v5.xlsx` | Multi-tab Excel audit with raw data and population summaries. |
| `track_summary.csv` | Full table of every 3D track detected. |
| `track_summary_quality.csv` | The clean population that passed the biologically informed audit. |
| `overlays/` | Side-by-side PNG panels for every Z-slice for visual verification. |

### Biological Notes
- **Audit vs. Segmentation**: The quality audit flags suspicious tracks but does not change the raw segmentation. Use audit changes first if results feel too strict.
- **PSF Sensitivity**: Volume, thickness, and area-derived metrics are sensitive to the microscope point spread function. Use these for relative comparison rather than absolute physical dimensions.

---

## Parameter Guide
| Section | Parameter | When to change? |
| :--- | :--- | :--- |
| **Calibration** | `UM_PER_PX_XY` / `UM_PER_SLICE_Z` | Only if microscope calibration differs from Saturn defaults. |
| **Segmentation** | `THRESHOLD_HI` / `THRESHOLD_LO` | If the pipeline is clearly too permissive or strict at the 2D stage. |
| **Cleanup** | `MAX_BRIDGE_PX` | If true nuclei are being broken apart. |
| **Morphology** | `MIN_SKEL_LEN_PX` | To filter out short debris or surviving merged objects. |
| **Tracking** | `TRACK_MAX_DIST_UM` | If tracks are fragmented or fusing neighboring nuclei. |
| **Audit** | `AUDIT_MAX_LENGTH_UM` | To refine the quality population subset. |

---

## Common Mistakes to Avoid
1. Changing parameters before running the built-in tuned defaults once.
2. Changing segmentation settings when the real problem is tracking or audit-related.
3. Treating PSF-sensitive outputs, such as volume, as literal physical dimensions.
4. Accepting the wrong ROI without visually confirming it on the overlay.
