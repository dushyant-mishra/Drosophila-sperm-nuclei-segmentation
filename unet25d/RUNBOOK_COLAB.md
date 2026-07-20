# Colab/GPU Runbook Via GitHub

Use this to train the 2.5D U-Net on a Colab GPU while keeping the private microscopy data in Google Drive.

## 1. Put Private Inputs In Drive

Create or reuse this Drive folder:

```text
/content/drive/MyDrive/unet25d
```

It should contain:

```text
sam3_pilot.zip
sam3_inference_images_full_stack.zip
unet25d_best_local.pt
```

Do not put raw images or checkpoints in GitHub.

`unet25d_best_local.pt` should be copied from the local pilot checkpoint:

```text
C:\Users\dmishra\Desktop\sperm_validation_private\unet25d\outputs\checkpoints\best.pt
```

## 2. Fresh Colab Runtime

Use a GPU runtime, then mount Drive.

```python
from google.colab import drive
drive.mount("/content/drive")
```

## 3. Clone The GitHub Repo

```python
from pathlib import Path
import os
import shutil
import zipfile

WORK = Path("/content/unet25d_workspace")
REPO_DIR = WORK / "repo"
DRIVE_ROOT = Path("/content/drive/MyDrive/unet25d")

WORK.mkdir(parents=True, exist_ok=True)

if not REPO_DIR.exists():
    !git clone https://github.com/dushyant-mishra/Drosophila-sperm-nuclei-segmentation.git {REPO_DIR}

%cd {REPO_DIR}
!git checkout feature/saturn-v5.6-roi-adaptive
```

## 4. Install Minimal Dependencies

```python
!pip install -q -r unet25d/requirements.txt
```

Check that Colab sees the GPU:

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
```

## 5. Unzip Private Data

```python
WORK = Path("/content/unet25d_workspace")
DRIVE_ROOT = Path("/content/drive/MyDrive/unet25d")
DATA_ROOT = WORK / "data"
DATA_ROOT.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(DRIVE_ROOT / "sam3_pilot.zip", "r") as z:
    z.extractall(DATA_ROOT / "sam3_pilot")

with zipfile.ZipFile(DRIVE_ROOT / "sam3_inference_images_full_stack.zip", "r") as z:
    z.extractall(DATA_ROOT / "images")
```

The expected paths after extraction are:

```python
print((DATA_ROOT / "sam3_pilot" / "train" / "_annotations.coco.json").exists())
print((DATA_ROOT / "sam3_pilot" / "valid_technical_only" / "_annotations.coco.json").exists())
print(len(list((DATA_ROOT / "images").glob("Project001_Series002_z*_ch00.tif"))))
```

If those checks fail because the zip extracted with an extra nested folder, move the files so these paths exist:

```text
/content/unet25d_workspace/data/sam3_pilot/train/_annotations.coco.json
/content/unet25d_workspace/data/sam3_pilot/valid_technical_only/_annotations.coco.json
/content/unet25d_workspace/data/images/Project001_Series002_z35_ch00.tif
```

## 6. Run Training And Review

```python
%cd /content/unet25d_workspace/repo/unet25d
!python prepare_dataset.py --config configs/pilot_unet25d_colab.yaml
!python train_unet25d.py \
  --config configs/pilot_unet25d_colab.yaml \
  --warm-start /content/drive/MyDrive/unet25d/unet25d_best_local.pt
!python infer_unet25d.py --config configs/pilot_unet25d_colab.yaml --checkpoint outputs/checkpoints/best.pt
!python review_overlays.py --config configs/pilot_unet25d_colab.yaml
```

That command continues the existing `unet_small` model from your local `best.pt`.

To test the more robust residual-attention U-Net architecture, use the experimental config. This architecture is not strictly compatible with the old checkpoint, so either train it from scratch:

```python
!python train_unet25d.py --config configs/pilot_resatt_unet25d_colab.yaml
```

or do a partial warm start, which copies only matching tensors and leaves new residual/attention layers freshly initialized:

```python
!python train_unet25d.py \
  --config configs/pilot_resatt_unet25d_colab.yaml \
  --warm-start /content/drive/MyDrive/unet25d/unet25d_best_local.pt \
  --allow-partial-warm-start
```

For the next real run, prefer strict continuation of `unet_small` if you are mainly adding more annotations. Use `residual_attention_unet` as a comparison experiment.

## 7. Save Outputs Back To Drive

```python
import shutil
from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive/unet25d")
shutil.make_archive("/content/unet25d_outputs", "zip", "/content/unet25d_workspace/repo/unet25d/outputs")
shutil.copy2("/content/unet25d_outputs.zip", DRIVE_ROOT / "unet25d_outputs.zip")
```

The most important files to inspect are:

```text
unet25d/outputs/checkpoints/best.pt
unet25d/outputs/train_history.csv
unet25d/outputs/inference/overlays/unet25d_overlay_review.pdf
```

During training, checkpoints are also mirrored after every epoch to:

```text
/content/drive/MyDrive/unet25d/checkpoints
```

So if the Colab runtime disconnects, recover from:

```text
/content/drive/MyDrive/unet25d/checkpoints/best.pt
/content/drive/MyDrive/unet25d/checkpoints/last.pt
/content/drive/MyDrive/unet25d/checkpoints/train_history.csv
```

To continue from the most recent mirrored checkpoint after a restart, use:

```python
!python train_unet25d.py \
  --config configs/pilot_unet25d_colab.yaml \
  --warm-start /content/drive/MyDrive/unet25d/checkpoints/last.pt
```
