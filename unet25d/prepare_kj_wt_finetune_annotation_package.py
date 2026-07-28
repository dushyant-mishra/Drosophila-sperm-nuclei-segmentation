"""Build a balanced, anonymized KJ/WT annotation package for U-Net fine-tuning."""

import argparse
import csv
import hashlib
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile
import yaml
from PIL import Image


DEFAULT_SOURCE_ROOT = Path(
    r"C:\Users\dmishra\Desktop\KJ Images and mutant-Grace"
)
DEFAULT_CHECKPOINT = Path(
    r"C:\Users\dmishra\Downloads\unet25d_output_kaggle_resume"
    r"\checkpoints_resatt_partial_annotated2_resume\best.pt"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "training_packages"
    / "v5_7_kj_wt_tiny_finetune"
)

# Validation specimens are never used for training. One center plane is chosen
# from each specimen so neighboring planes cannot leak between splits.
SPECIMENS = [
    ("train", "KJ", "kj sv feb", "kj sv 40xx0.75-1"),
    ("train", "KJ", "kj sv feb", "kj sv 40xx0.75-4"),
    ("train", "KJ", "kj sv feb", "kj sv 40xx0.75-9"),
    ("train", "KJ", "kj sv feb", "kj sv 40xx0.75-13"),
    ("train", "WT", "w1118 sv feb", "w1118 sv feb 40xx0.75-1"),
    ("train", "WT", "w1118 sv feb", "w1118 sv feb 40xx0.75-4"),
    ("train", "WT", "w1118 sv feb", "w1118 sv feb 40xx0.75-9"),
    ("train", "WT", "w1118 sv feb", "w1118 sv feb 40xx0.75-13"),
    ("valid", "KJ", "kj sv feb", "kj sv 40xx0.75-6"),
    ("valid", "KJ", "kj sv feb", "kj sv 40xx0.75-15"),
    ("valid", "WT", "w1118 sv feb", "w1118 sv feb 40xx0.75-6"),
    ("valid", "WT", "w1118 sv feb", "w1118 sv feb 40xx0.75-16"),
]

Z_RE = re.compile(r"_z(\d+)_ch00\.tiff?$", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stack_files(specimen_dir):
    by_z = {}
    for path in specimen_dir.iterdir():
        if not path.is_file():
            continue
        match = Z_RE.search(path.name)
        if match:
            z = int(match.group(1))
            if z in by_z:
                raise ValueError(f"Duplicate z={z} in {specimen_dir}")
            by_z[z] = path
    if len(by_z) < 3:
        raise ValueError(f"Need at least three source planes: {specimen_dir}")
    return by_z


def load_roi(specimen_dir, shape):
    preferred = specimen_dir / "analysis_roi_v5_7.npy"
    candidates = [preferred] if preferred.exists() else sorted(specimen_dir.glob("*roi*.npy"))
    if not candidates:
        raise FileNotFoundError(f"No ROI mask in {specimen_dir}")
    roi = np.asarray(np.load(candidates[0]), dtype=bool)
    if roi.shape != shape:
        raise ValueError(f"ROI {roi.shape} does not match image {shape}: {candidates[0]}")
    return roi, candidates[0]


def read_2d(path):
    image = np.asarray(tifffile.imread(path))
    if image.ndim > 2:
        image = image[..., 0]
    if image.ndim != 2:
        raise ValueError(f"Expected 2D TIFF: {path}")
    return image


def focus_score(image, roi):
    values = image[roi].astype(np.float32)
    lo, hi = np.percentile(values, [1.0, 99.5])
    normalized = np.clip((image.astype(np.float32) - lo) / max(hi - lo, 1.0), 0, 1)
    gy, gx = np.gradient(normalized)
    gradient = np.hypot(gx, gy)[roi]
    return float(np.percentile(values, 99.0) - np.median(values)) * (
        1.0 + float(np.percentile(gradient, 90.0))
    )


def choose_center_plane(by_z, roi):
    valid = sorted(z for z in by_z if z - 1 in by_z and z + 1 in by_z)
    lo = valid[int(round((len(valid) - 1) * 0.35))]
    hi = valid[int(round((len(valid) - 1) * 0.70))]
    candidates = [z for z in valid if lo <= z <= hi]
    scored = [
        (focus_score(read_2d(by_z[z]), roi), z)
        for z in candidates
    ]
    return max(scored)[1], sorted(scored, reverse=True)[:5]


def display_image(image, roi):
    values = image[roi].astype(np.float32)
    lo, hi = np.percentile(values, [1.0, 99.5])
    normalized = np.clip((image.astype(np.float32) - lo) / max(hi - lo, 1.0), 0, 1)
    return np.round(normalized * 255).astype(np.uint8)


def roi_guide(image_u8, roi):
    edge = roi & ~(
        np.roll(roi, 1, axis=0)
        & np.roll(roi, -1, axis=0)
        & np.roll(roi, 1, axis=1)
        & np.roll(roi, -1, axis=1)
    )
    rgb = np.repeat(image_u8[..., None], 3, axis=2)
    rgb[edge] = (0, 255, 80)
    return rgb


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def blank_iap(images_dir, image_records):
    now = datetime.now().isoformat()
    images = []
    image_paths = {}
    for record in image_records:
        file_name = record["annotation_file"]
        path = (images_dir / file_name).resolve()
        with Image.open(path) as image:
            width, height = image.size
        images.append(
            {
                "file_name": file_name,
                "width": width,
                "height": height,
                "is_multi_slice": True,
                "slices": [
                    {
                        "name": Path(file_name).stem,
                        "annotations": {"sperm_nucleus": []},
                    }
                ],
                "dimensions": ["H", "W"],
                "shape": [height, width],
            }
        )
        image_paths[file_name] = str(path).replace("\\", "/")
    return {
        "classes": [{"name": "sperm_nucleus", "color": "#1f77b4"}],
        "images": images,
        "image_paths": image_paths,
        "notes": "Balanced anonymized KJ/WT fine-tuning set. Annotate every nucleus inside the ROI.",
        "creation_date": now,
        "last_modified": now,
        "dino_config": {
            "phrases": {"sperm_nucleus": ["sperm nucleus"]},
            "thresholds": {
                "sperm_nucleus": {"box": 0.25, "txt": 0.25, "nms": 0.5}
            },
        },
    }


def kaggle_config(valid_z):
    work_root = "/kaggle/working/v5_7_kj_wt_tiny_finetune"
    return {
        "project_root": work_root,
        "stack_image_dir": f"{work_root}/raw_tiffs",
        "annotation_manifest": (
            f"{work_root}/annotations/"
            "_annotations.coco.json"
        ),
        "valid_z_indices": list(valid_z),
        "output_dir": "/kaggle/working/v57_kj_wt_finetune_outputs",
        "checkpoint_mirror_dir": (
            "/kaggle/working/v57_kj_wt_finetune_outputs/checkpoints"
        ),
        "z_regex": r"Project001_Series002_z(\d+)_ch00",
        "image_pattern": "Project001_Series002_z{z:04d}_ch00.tif",
        "roi_mask_dir": f"{work_root}/roi_masks",
        "roi_mask_pattern": "Project001_Series002_z{z:04d}_ch00.npy",
        "architecture": "residual_attention_unet",
        "partial_labels": True,
        "ignore_unlabeled_intensity_percentile": 97.0,
        "ignore_labeled_dilate_px": 4,
        "ignore_candidate_dilate_px": 1,
        "positive_patch_probability": 0.8,
        "train_mask_dilate_px": 1,
        "positive_loss_weight": 1.5,
        "patch_size": 256,
        "patches_per_image": 64,
        "epochs": 12,
        "snapshot_epochs": [3, 6, 9, 12],
        "batch_size": 8,
        "base_channels": 24,
        "learning_rate": 0.00005,
        "weight_decay": 0.00001,
        "seed": 793,
        "threshold": 0.5,
        "unet_inference_mode": "roi_tiled",
        "unet_tile_size": 256,
        "unet_tile_overlap": 64,
        "unet_roi_padding_px": 32,
        "unet_outside_roi_zero": True,
        "unet_candidate_threshold": 0.05,
        "unet_seed_threshold": 0.30,
        "infer_z_indices": list(valid_z),
    }


def main():
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        if not args.force:
            raise FileExistsError(f"Output already exists; use --force: {output}")
        shutil.rmtree(output)
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)

    annotation_dir = output / "annotation_images"
    guide_dir = output / "roi_guides"
    raw_dir = output / "raw_tiffs"
    roi_dir = output / "roi_masks"
    annotations_dir = output / "annotations"
    for path in (annotation_dir, guide_dir, raw_dir, roi_dir, annotations_dir):
        path.mkdir(parents=True, exist_ok=True)
    code_dir = output / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    source_code_dir = Path(__file__).resolve().parent
    for name in ("prepare_dataset.py", "train_unet25d.py", "torch_device.py", "requirements.txt"):
        shutil.copy2(source_code_dir / name, code_dir / name)

    records = []
    selection_rows = []
    for index, (split, group, parent, specimen) in enumerate(SPECIMENS, start=1):
        specimen_dir = args.source_root / parent / specimen
        by_z = stack_files(specimen_dir)
        first = read_2d(next(iter(by_z.values())))
        roi, roi_path = load_roi(specimen_dir, first.shape)
        source_z, top_scores = choose_center_plane(by_z, roi)
        synthetic_z = 100 + index * 10
        annotation_name = f"Project001_Series002_z{synthetic_z:04d}_ch00.png"

        center_image = read_2d(by_z[source_z])
        display = display_image(center_image, roi)
        Image.fromarray(display).save(annotation_dir / annotation_name)
        # Sreeni resolves project images beside the .iap file when reopening it.
        Image.fromarray(display).save(output / annotation_name)
        Image.fromarray(roi_guide(display, roi)).save(
            guide_dir / annotation_name.replace(".png", "_roi_guide.png")
        )

        for offset in (-1, 0, 1):
            destination = (
                raw_dir
                / f"Project001_Series002_z{synthetic_z + offset:04d}_ch00.tif"
            )
            shutil.copy2(by_z[source_z + offset], destination)
        np.save(
            roi_dir / f"Project001_Series002_z{synthetic_z:04d}_ch00.npy",
            roi.astype(bool),
        )

        records.append(
            {
                "annotation_file": annotation_name,
                "synthetic_z": synthetic_z,
                "split": split,
            }
        )
        selection_rows.append(
            {
                "annotation_file": annotation_name,
                "split": split,
                "group": group,
                "specimen": specimen,
                "source_center_z": source_z,
                "synthetic_center_z": synthetic_z,
                "source_directory": str(specimen_dir),
                "source_roi": str(roi_path),
                "roi_pixels": int(roi.sum()),
                "focus_score": top_scores[0][0],
            }
        )

    write_csv(
        output / "annotation_key_private.csv",
        selection_rows,
        list(selection_rows[0]),
    )
    write_csv(
        output / "split_manifest.csv",
        [
            {
                "annotation_file": row["annotation_file"],
                "synthetic_z": row["synthetic_z"],
                "split": row["split"],
            }
            for row in records
        ],
        ["annotation_file", "synthetic_z", "split"],
    )
    project = blank_iap(annotation_dir, records)
    with open(output / "sreeni_kj_wt_tiny_blank.iap", "w", encoding="utf-8") as handle:
        json.dump(project, handle, indent=2)
        handle.write("\n")

    valid_z = [row["synthetic_z"] for row in records if row["split"] == "valid"]
    with open(output / "kaggle_finetune.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(kaggle_config(valid_z), handle, sort_keys=False)

    provenance = {
        "checkpoint_path_local": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_size_bytes": args.checkpoint.stat().st_size,
        "architecture": "residual_attention_unet",
        "base_channels": 24,
        "annotation_images": len(records),
        "train_images": sum(row["split"] == "train" for row in records),
        "validation_images": sum(row["split"] == "valid" for row in records),
    }
    with open(output / "checkpoint_provenance.json", "w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)
        handle.write("\n")

    instructions = """# KJ/WT Tiny Fine-Tuning Annotation Package

1. Open `sreeni_kj_wt_tiny_blank.iap` in Sreeni.
2. Use `roi_guides/` only to see the ROI boundary; annotate on the clean images.
3. Annotate every visible sperm nucleus inside the ROI, including faint, short,
   partial, and unusual nuclei. Do not annotate tissue or objects outside the ROI.
4. Keep touching nuclei as separate instances whenever their boundaries can be
   distinguished. Do not join nuclei because they are close.
5. Export COCO to `annotations/_annotations.coco.json`.
6. Do not move images between train and validation. The validation specimens are
   independent and must remain held out.
7. Zip this directory and upload it as a Kaggle dataset. Upload the compatible
   `best.pt` checkpoint separately; its SHA-256 is recorded in
   `checkpoint_provenance.json`.

The PNG files are display copies for annotation. Training uses the original TIFF
center planes plus their adjacent z context. ROI masks remove off-ROI pixels from
the training loss. The genotype mapping is kept only in
`annotation_key_private.csv`; genotype is never a model target.
"""
    (output / "ANNOTATION_AND_KAGGLE_INSTRUCTIONS.md").write_text(
        instructions, encoding="utf-8"
    )
    kaggle_cells = r"""# Kaggle notebook cells

## Cell 1: locate and extract the annotated package
```python
from pathlib import Path
import shutil

package_zip = next(Path("/kaggle/input").rglob("v5_7_kj_wt_tiny_finetune.zip"))
work_root = Path("/kaggle/working/v5_7_kj_wt_tiny_finetune")
if work_root.exists():
    shutil.rmtree(work_root)
shutil.unpack_archive(package_zip, Path("/kaggle/working"))
print("Package:", package_zip)
print("Work root:", work_root)
```

## Cell 2: locate and verify the warm-start checkpoint
```python
from pathlib import Path
import hashlib, json

provenance = json.loads((work_root / "checkpoint_provenance.json").read_text())
candidates = list(Path("/kaggle/input").rglob("best.pt"))
expected = provenance["checkpoint_sha256"]

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

matches = [path for path in candidates if sha256(path) == expected]
assert len(matches) == 1, f"Expected one matching checkpoint, found {matches}"
warm_start = matches[0]
print("Warm start:", warm_start)
```

## Cell 3: environment check
```python
import torch
print("torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

## Cell 4: prepare the ROI-aware 2.5D dataset
```python
%cd /kaggle/working/v5_7_kj_wt_tiny_finetune/code
!python prepare_dataset.py --config ../kaggle_finetune.yaml
```

## Cell 5: fine-tune from the verified compatible checkpoint
```python
!python train_unet25d.py \
  --config ../kaggle_finetune.yaml \
  --warm-start "$warm_start"
```

The run writes `best.pt`, `last.pt`, and snapshots for epochs 3, 6, 9, and 12
under `/kaggle/working/v57_kj_wt_finetune_outputs/checkpoints`. Compare
inference overlays from all snapshots before selecting the new checkpoint.
"""
    (output / "KAGGLE_CELLS.md").write_text(kaggle_cells, encoding="utf-8")

    archive = shutil.make_archive(str(output), "zip", output.parent, output.name)
    print(f"Prepared {len(records)} annotation images: {output}")
    print(f"Train: {len(records) - len(valid_z)}; validation: {len(valid_z)}")
    print(f"Archive: {archive}")


if __name__ == "__main__":
    main()
