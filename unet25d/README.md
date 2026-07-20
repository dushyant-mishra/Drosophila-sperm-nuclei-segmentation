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

## Notes

- COCO polygons are rasterized into binary center-slice masks.
- Context slices come from the full TIFF z stack in `images/`.
- The first experiment is intentionally tiny: 9 train images and 2 technical validation images.
- This is a proof-of-approach, not yet a production model.
