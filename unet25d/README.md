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

This config also uses training-only forgiveness for imperfect hand masks:

- partial-label-aware loss through `supervision_mask`
- ignored unlabeled bright pixels
- slight positive-mask dilation through `train_mask_dilate_px`
- ROI/crop-friendly positive patch sampling
- a mild positive-pixel loss weight through `positive_loss_weight`

These settings affect only the training targets/loss. They do not dilate or
resize inference outputs. Tiled inference writes stitched full-frame probability
maps first, then saves thresholded candidate/seed masks only as review aids.
Saturn should still compute final length, width, count, tracking, and QC from
its own ROI-aware candidate measurement logic.

To compare inference thresholds after training:

```powershell
python sweep_thresholds.py --config configs/pilot_resatt_partial_labels_tight_colab.yaml --checkpoint outputs_resatt_partial_tight/checkpoints/best.pt --thresholds 0.5 0.6 0.7
```

For the expanded Sreeni annotated-2 export, upload `annotated-2.zip` into the
Colab data folder and train with:

```bash
cd /content/unet25d_workspace/repo/unet25d

python prepare_dataset.py \
  --config configs/pilot_resatt_partial_labels_annotated2_colab.yaml

python train_unet25d.py \
  --config configs/pilot_resatt_partial_labels_annotated2_colab.yaml \
  --warm-start /content/drive/MyDrive/unet25d_output/checkpoints_resatt_partial_tight/best.pt
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

## KJ/WT Replay Fine-Tuning

The current v5.7 fine-tuning workflow combines newly annotated KJ/WT images
with the previous Sreeni annotations as replay data. The builder:

- excludes annotations that cross or fall outside each ROI
- remaps replay images to collision-free synthetic Z indices
- carries the previous, center, and next TIFF planes into the package
- keeps the four new validation specimens out of training
- repeats the eight new training images twice for an approximately balanced
  new-versus-replay epoch
- applies conservative training-only gain, gamma, and noise augmentation

Build the package locally with:

```powershell
python build_kj_wt_replay_finetune_package.py `
  --new-package ..\training_packages\v5_7_kj_wt_tiny_finetune `
  --replay-coco C:\path\to\previous\_annotations.coco.json `
  --replay-stack C:\path\to\previous\raw_tiff_stack `
  --replay-roi C:\path\to\previous\roi.npy `
  --output ..\training_packages\v5_7_kj_wt_replay_finetune
```

After training, compare the warm-start checkpoint with the saved epoch
snapshots using:

```powershell
python evaluate_brightness_recall.py `
  --config ..\training_packages\v5_7_kj_wt_replay_finetune\kaggle_finetune.yaml `
  --checkpoint baseline=C:\path\to\warm_start.pt `
  --checkpoint epoch_003=C:\path\to\epoch_003.pt `
  --checkpoint epoch_006=C:\path\to\epoch_006.pt `
  --checkpoint epoch_009=C:\path\to\epoch_009.pt `
  --checkpoint epoch_012=C:\path\to\epoch_012.pt `
  --output C:\path\to\brightness_validation
```

The evaluator preserves continuous probability maps, shows identical image
versions across thresholds, and reports probability-support recall separately
for faint, intermediate, and bright manually annotated nuclei. This is a
model-recall diagnostic; it does not replace Saturn instance splitting or
biological QC.

## Notes

- COCO polygons are rasterized into binary center-slice masks.
- Context slices come from the full TIFF z stack in `images/`.
- The first experiment is intentionally tiny: 9 train images and 2 technical validation images.
- This is a proof-of-approach, not yet a production model.
