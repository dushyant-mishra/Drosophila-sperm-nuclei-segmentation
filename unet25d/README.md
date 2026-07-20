# 2.5D U-Net Sperm Nucleus Pilot

This folder is a small, independent pilot for testing whether a compact 2.5D U-Net can learn sperm nucleus masks from the existing COCO annotations.

The model input is three adjacent z slices as channels:

```text
[z-1, z, z+1] -> mask for center slice z
```

Available architectures:

- `unet_small`: current lightweight baseline, compatible with the existing `best.pt` checkpoint.
- `residual_attention_unet`: experimental residual + attention-gated U-Net for noisy/background-heavy slices.

## Quick Start

From the `unet25d` folder:

```powershell
python prepare_dataset.py --config configs/pilot_unet25d.yaml
python train_unet25d.py --config configs/pilot_unet25d.yaml
python infer_unet25d.py --config configs/pilot_unet25d.yaml --checkpoint outputs/checkpoints/best.pt
python review_overlays.py --config configs/pilot_unet25d.yaml
```

For Google Colab/GPU training through GitHub, use [RUNBOOK_COLAB.md](RUNBOOK_COLAB.md).

To continue training from an existing compatible checkpoint:

```powershell
python train_unet25d.py --config configs/pilot_unet25d.yaml --warm-start outputs/checkpoints/best.pt
```

To test a pseudo-label cleanup loop from previously generated masks:

```powershell
python prepare_from_masks.py --config configs/pilot_resatt_round2_pseudo_downloaded.yaml
python train_unet25d.py --config configs/pilot_resatt_round2_pseudo_downloaded.yaml --warm-start C:\path\to\best.pt
```

Use this only as a workflow test unless the masks have been manually corrected.

For partial-label training where unlabeled bright nuclei should not be treated
as definite background, use `configs/pilot_resatt_partial_labels_tight_colab.yaml`.
It ignores only the brightest unlabeled candidate pixels (`97th` percentile)
and uses less aggressive positive patch sampling than the first partial-label
experiment.

To compare inference thresholds after training:

```powershell
python sweep_thresholds.py --config configs/pilot_resatt_partial_labels_tight_colab.yaml --checkpoint outputs_resatt_partial_tight/checkpoints/best.pt --thresholds 0.5 0.6 0.7
```

To run ROI-aware tiled soft inference for Saturn v5.7:

```powershell
python infer_tiled_unet25d.py `
  --config configs/pilot_resatt_partial_labels_tight_colab.yaml `
  --checkpoint outputs_resatt_partial_tight/checkpoints/best.pt `
  --roi /content/drive/MyDrive/unet25d_input/roi_z28.1.npy `
  --output-dir /content/drive/MyDrive/unet25d_output/tiled_soft_inference
```

This writes continuous probability maps plus two review masks:

- candidate mask: permissive field, default probability >= `0.05`
- seed mask: high-confidence cores, default probability >= `0.30`

For Saturn integration, COCO stays training-only. Runtime inference should use
the trained checkpoint and raw image stack, then pass probability maps into
Saturn for ROI-aware candidate repair, measurement, and QC.

## Notes

- COCO polygons are rasterized into binary center-slice masks.
- Context slices come from the full TIFF z stack in `images/`.
- The first experiment is intentionally tiny: 9 train images and 2 technical validation images.
- This is a proof-of-approach, not yet a production model.
