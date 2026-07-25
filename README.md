# Saturn v5.7 Sperm Nuclei Segmentation

Saturn v5.7 is the active Drosophila sperm-nucleus analysis pipeline. It
combines ROI-aware classical segmentation, optional 2.5D U-Net probability
support, cross-slice tracking, morphology measurements, quality-control
populations, and multi-sample study management.

## Run the GUI

From PowerShell:

```powershell
Set-Location "C:\Users\dmishra\Desktop\sperm_project"
.\.venv\Scripts\Activate.ps1
python .\sperm_segmentation_saturnv5.7.py --gui
```

Launching without arguments also opens the GUI:

```powershell
python .\sperm_segmentation_saturnv5.7.py
```

## Active Components

- `sperm_segmentation_saturnv5.7.py`: GUI, segmentation, tracking, reporting,
  ROI normalization, and multi-sample study manager.
- `utils/tune_parameters_Saturnv5_7.py`: v5.7 segmentation and U-Net rescue
  tuner.
- `utils/saturn_unet25d_bridge.py`: lazy checkpoint loading and tiled U-Net
  probability inference.
- `unet25d/`: dataset preparation, model training, inference, and threshold
  review tools.
- `parameter_tuning_results_v5_7/`: reviewed v5.7 tuning outputs.
- `docs/v5_7_illustrated_workflow/`: illustrated workflow source assets.
- `Saturn_V5.7_Illustrated_Analysis_Workflow_FINAL.docx`: readable workflow
  report.

## Recommended Workflow

1. Load one image stack or open the multi-sample study manager.
2. Draw or load a specimen-specific ROI and confirm its alignment.
3. Confirm XY and Z calibration from microscope metadata.
4. Load the reviewed v5.7 parameters and U-Net checkpoint when using hybrid
   inference.
5. Run segmentation and cross-slice tracking.
6. Review equal-thickness overlays, QC populations, normalization warnings,
   and specimen-level outputs before biological comparison.

Do not tune parameters toward a desired genotype count or morphology. Tune and
validate without using biological-group outcomes.

## Development

Install the core environment with:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the active tests with:

```powershell
python -m pytest -q
```

Historical Saturn versions, old tuning runs, build outputs, and experimental
AI pilots are preserved under `archive/`. See `PROJECT_LAYOUT.md` and
`archive/README.md`.
