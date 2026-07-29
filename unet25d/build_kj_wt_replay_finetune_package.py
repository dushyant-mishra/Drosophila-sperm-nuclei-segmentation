import argparse
import csv
import json
import re
import shutil
from pathlib import Path

import numpy as np
import tifffile
import yaml
from PIL import Image, ImageDraw


STACK_PATTERN = re.compile(
    r"^Project001_Series002_z(\d+)_ch00\.tif{1,2}$",
    re.IGNORECASE,
)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def annotation_mask(annotation, image):
    mask = Image.new("1", (int(image["width"]), int(image["height"])), 0)
    draw = ImageDraw.Draw(mask)
    segmentation = annotation.get("segmentation", [])
    polygons = (
        segmentation
        if segmentation and isinstance(segmentation[0], list)
        else [segmentation]
    )
    for polygon in polygons:
        if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:
            continue
        draw.polygon(list(zip(polygon[0::2], polygon[1::2])), fill=1)
    return np.asarray(mask, dtype=bool)


def filter_annotations_to_roi(coco, roi_for_file, source_name):
    images = {image["id"]: image for image in coco["images"]}
    kept = []
    audit = []
    for annotation in coco["annotations"]:
        image = images[annotation["image_id"]]
        file_name = Path(image["file_name"]).name
        mask = annotation_mask(annotation, image)
        roi = np.asarray(roi_for_file(file_name), dtype=bool)
        if roi.shape != mask.shape:
            raise ValueError(
                f"ROI {roi.shape} does not match annotation image {mask.shape}: "
                f"{file_name}"
            )
        pixels = int(mask.sum())
        inside = int((mask & roi).sum())
        keep = pixels > 0 and inside == pixels
        audit.append(
            {
                "source": source_name,
                "image_file": file_name,
                "annotation_id": annotation.get("id"),
                "mask_pixels": pixels,
                "inside_roi_pixels": inside,
                "inside_roi_fraction": round(inside / max(pixels, 1), 6),
                "action": "keep" if keep else "exclude_roi_boundary_or_outside",
            }
        )
        if keep:
            kept.append(annotation)
    filtered = dict(coco)
    filtered["annotations"] = kept
    return filtered, audit


def stack_files(directory):
    by_z = {}
    for path in Path(directory).iterdir():
        if not path.is_file():
            continue
        match = STACK_PATTERN.match(path.name)
        if not match:
            continue
        z = int(match.group(1))
        if z in by_z:
            raise ValueError(f"Duplicate replay Z index {z}: {path}")
        by_z[z] = path
    if not by_z:
        raise FileNotFoundError(f"No matching replay TIFFs in {directory}")
    return by_z


def nearest_z(by_z, requested):
    if requested in by_z:
        return requested
    return min(by_z, key=lambda z: (abs(z - requested), z))


def normalize_display(image, roi):
    values = image[roi].astype(np.float32)
    lo, hi = np.percentile(values, [1.0, 99.5])
    normalized = np.clip(
        (image.astype(np.float32) - lo) / max(float(hi - lo), 1.0),
        0.0,
        1.0,
    )
    return np.round(normalized * 255).astype(np.uint8)


def remap_replay(
    coco,
    replay_stack,
    replay_roi,
    output,
    start_z=300,
    spacing=10,
):
    by_z = stack_files(replay_stack)
    raw_dir = output / "raw_tiffs"
    roi_dir = output / "roi_masks"
    review_dir = output / "replay_annotation_images"
    review_dir.mkdir(parents=True, exist_ok=True)

    old_images = {image["id"]: image for image in coco["images"]}
    annotations_by_image = {image_id: [] for image_id in old_images}
    for annotation in coco["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)

    remapped_images = []
    remapped_annotations = []
    source_rows = []
    next_annotation_id = 1
    sorted_images = sorted(
        coco["images"],
        key=lambda image: int(
            re.search(r"_z(\d+)_ch00", Path(image["file_name"]).name).group(1)
        ),
    )
    for index, image in enumerate(sorted_images):
        source_name = Path(image["file_name"]).name
        match = re.search(r"_z(\d+)_ch00", source_name)
        if not match:
            raise ValueError(f"Could not parse replay Z index: {source_name}")
        source_z = int(match.group(1))
        synthetic_z = int(start_z + index * spacing)
        synthetic_name = (
            f"Project001_Series002_z{synthetic_z:04d}_ch00.png"
        )
        new_image_id = len(remapped_images) + 1
        remapped_images.append(
            {
                **image,
                "id": new_image_id,
                "file_name": synthetic_name,
            }
        )

        context_sources = []
        for offset in (-1, 0, 1):
            context_z = nearest_z(by_z, source_z + offset)
            context_sources.append(context_z)
            destination = (
                raw_dir
                / f"Project001_Series002_z{synthetic_z + offset:04d}_ch00.tif"
            )
            shutil.copy2(by_z[context_z], destination)
        np.save(
            roi_dir / f"Project001_Series002_z{synthetic_z:04d}_ch00.npy",
            replay_roi.astype(bool),
        )
        center = np.asarray(tifffile.imread(by_z[source_z]))
        if center.ndim > 2:
            center = center[..., 0]
        Image.fromarray(normalize_display(center, replay_roi)).save(
            review_dir / synthetic_name
        )

        for annotation in annotations_by_image[image["id"]]:
            remapped_annotations.append(
                {
                    **annotation,
                    "id": next_annotation_id,
                    "image_id": new_image_id,
                    "category_id": 1,
                }
            )
            next_annotation_id += 1
        source_rows.append(
            {
                "training_source": "previous_annotation_replay",
                "split": "train",
                "source_file": source_name,
                "source_z": source_z,
                "synthetic_file": synthetic_name,
                "synthetic_z": synthetic_z,
                "context_source_z": "|".join(str(z) for z in context_sources),
                "annotation_count": len(annotations_by_image[image["id"]]),
            }
        )
    return remapped_images, remapped_annotations, source_rows


def merge_coco(new_coco, replay_images, replay_annotations):
    merged_images = []
    merged_annotations = []
    image_id_map = {}
    next_image_id = 1
    next_annotation_id = 1

    for image in new_coco["images"]:
        image_id_map[("new", image["id"])] = next_image_id
        merged_images.append(
            {
                **image,
                "id": next_image_id,
                "file_name": Path(image["file_name"]).name,
            }
        )
        next_image_id += 1
    for image in replay_images:
        image_id_map[("replay", image["id"])] = next_image_id
        merged_images.append({**image, "id": next_image_id})
        next_image_id += 1

    for annotation in new_coco["annotations"]:
        merged_annotations.append(
            {
                **annotation,
                "id": next_annotation_id,
                "image_id": image_id_map[("new", annotation["image_id"])],
                "category_id": 1,
            }
        )
        next_annotation_id += 1
    for annotation in replay_annotations:
        merged_annotations.append(
            {
                **annotation,
                "id": next_annotation_id,
                "image_id": image_id_map[
                    ("replay", annotation["image_id"])
                ],
                "category_id": 1,
            }
        )
        next_annotation_id += 1

    return {
        "images": merged_images,
        "annotations": merged_annotations,
        "categories": [{"id": 1, "name": "sperm_nucleus"}],
    }


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def update_config(output, new_train_z):
    config_path = output / "kaggle_finetune.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    package_name = output.name
    project_root = f"/kaggle/working/{package_name}"
    config["project_root"] = project_root
    config["stack_image_dir"] = f"{project_root}/raw_tiffs"
    config["annotation_manifest"] = (
        f"{project_root}/annotations/_annotations.coco.json"
    )
    config["roi_mask_dir"] = f"{project_root}/roi_masks"
    config["output_dir"] = "/kaggle/working/v57_kj_wt_replay_outputs"
    config["checkpoint_mirror_dir"] = (
        "/kaggle/working/v57_kj_wt_replay_outputs/checkpoints"
    )
    config["train_repeat_z_indices"] = list(new_train_z)
    config["train_repeat_factor"] = 2
    config["photometric_augment_probability"] = 0.65
    config["photometric_gain_range"] = [0.70, 1.20]
    config["photometric_gamma_range"] = [0.85, 1.30]
    config["photometric_noise_std_max"] = 0.02
    config["brightness_validation_enable"] = True
    config["brightness_validation_thresholds"] = [
        0.05,
        0.10,
        0.20,
        0.30,
        0.50,
    ]
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )


def refresh_code_and_notebook(output):
    source_dir = Path(__file__).resolve().parent
    code_dir = output / "code"
    code_dir.mkdir(exist_ok=True)
    for name in (
        "prepare_dataset.py",
        "train_unet25d.py",
        "torch_device.py",
        "infer_tiled_unet25d.py",
        "evaluate_brightness_recall.py",
        "requirements.txt",
    ):
        shutil.copy2(source_dir / name, code_dir / name)

    package_name = output.name
    cells = f"""# Kaggle notebook cells

## Cell 1: locate and extract the combined package
```python
from pathlib import Path
import shutil

package_zip = next(Path("/kaggle/input").rglob("{package_name}.zip"))
work_root = Path("/kaggle/working/{package_name}")
if work_root.exists():
    shutil.rmtree(work_root)
shutil.unpack_archive(package_zip, Path("/kaggle/working"))
print("Package:", package_zip)
print("Work root:", work_root)
```

## Cell 2: verify the warm-start checkpoint
```python
from pathlib import Path
import hashlib, json

provenance = json.loads((work_root / "checkpoint_provenance.json").read_text())
expected = provenance["checkpoint_sha256"]

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

candidates = list(Path("/kaggle/input").rglob("best.pt"))
matches = [path for path in candidates if sha256(path) == expected]
assert len(matches) == 1, f"Expected one compatible checkpoint, found: {{matches}}"
warm_start = matches[0]
print("Warm start:", warm_start)
```

## Cell 3: verify the GPU
```python
import torch
print("torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("Enable a Kaggle GPU accelerator before training.")
print("GPU:", torch.cuda.get_device_name(0))
```

## Cell 4: prepare the ROI-aware 2.5D samples
```python
%cd /kaggle/working/{package_name}/code
!python prepare_dataset.py --config ../kaggle_finetune.yaml
```

## Cell 5: fine-tune from the verified checkpoint
```python
!python train_unet25d.py \\
  --config ../kaggle_finetune.yaml \\
  --warm-start "$warm_start"
```

## Cell 6: compare baseline and epoch snapshots by brightness
```python
from pathlib import Path
import subprocess

checkpoint_dir = Path("/kaggle/working/v57_kj_wt_replay_outputs/checkpoints")
checkpoints = {{
    "baseline": warm_start,
    "epoch_003": checkpoint_dir / "epoch_003.pt",
    "epoch_006": checkpoint_dir / "epoch_006.pt",
    "epoch_009": checkpoint_dir / "epoch_009.pt",
    "epoch_012": checkpoint_dir / "epoch_012.pt",
}}
cmd = [
    "python", "evaluate_brightness_recall.py",
    "--config", "../kaggle_finetune.yaml",
    "--output", "/kaggle/working/v57_kj_wt_replay_outputs/brightness_validation",
]
for label, path in checkpoints.items():
    assert Path(path).exists(), path
    cmd.extend(["--checkpoint", f"{{label}}={{path}}"])
subprocess.run(cmd, check=True)
```

## Cell 7: archive all outputs for download
```python
import shutil

archive = shutil.make_archive(
    "/kaggle/working/v57_kj_wt_replay_outputs",
    "zip",
    "/kaggle/working/v57_kj_wt_replay_outputs",
)
print("Download:", archive)
```
"""
    (output / "KAGGLE_CELLS.md").write_text(cells, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Build the ROI-cleaned new-plus-replay v5.7 fine-tuning package."
    )
    parser.add_argument("--new-package", required=True, type=Path)
    parser.add_argument("--replay-coco", required=True, type=Path)
    parser.add_argument("--replay-stack", required=True, type=Path)
    parser.add_argument("--replay-roi", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    source = args.new_package.resolve()
    output = args.output.resolve()
    if output.exists():
        shutil.rmtree(output)
    shutil.copytree(source, output)

    new_source = source / "annotations" / "_annotations.sreeni_original.coco.json"
    if not new_source.exists():
        new_source = source / "annotation_images" / "_annotations.json"
    new_coco = load_json(new_source)

    def new_roi(file_name):
        return np.load(
            output / "roi_masks" / f"{Path(file_name).stem}.npy"
        ).astype(bool)

    new_clean, new_audit = filter_annotations_to_roi(
        new_coco,
        new_roi,
        "new_kj_wt",
    )
    annotations_dir = output / "annotations"
    write_json(
        annotations_dir / "_annotations.new_only_roi_clean.coco.json",
        new_clean,
    )

    replay_roi = np.load(args.replay_roi).astype(bool)
    replay_coco = load_json(args.replay_coco)
    replay_clean, replay_audit = filter_annotations_to_roi(
        replay_coco,
        lambda _file_name: replay_roi,
        "previous_annotation_replay",
    )
    write_json(
        annotations_dir / "_annotations.replay_roi_clean.coco.json",
        replay_clean,
    )
    replay_images, replay_annotations, replay_rows = remap_replay(
        replay_clean,
        args.replay_stack,
        replay_roi,
        output,
    )
    merged = merge_coco(new_clean, replay_images, replay_annotations)
    write_json(annotations_dir / "_annotations.coco.json", merged)
    write_csv(annotations_dir / "combined_roi_annotation_audit.csv", new_audit + replay_audit)

    split_rows = list(
        csv.DictReader(
            (output / "split_manifest.csv").open(
                newline="",
                encoding="utf-8",
            )
        )
    )
    new_train_z = [
        int(row["synthetic_z"])
        for row in split_rows
        if row["split"] == "train"
    ]
    source_rows = [
        {
            "training_source": "new_kj_wt",
            "split": row["split"],
            "source_file": "",
            "source_z": "",
            "synthetic_file": row["annotation_file"],
            "synthetic_z": row["synthetic_z"],
            "context_source_z": "",
            "annotation_count": sum(
                annotation["image_id"] == image["id"]
                for image in new_clean["images"]
                if Path(image["file_name"]).name == row["annotation_file"]
                for annotation in new_clean["annotations"]
            ),
        }
        for row in split_rows
    ]
    write_csv(output / "combined_training_sources.csv", source_rows + replay_rows)
    update_config(output, new_train_z)
    refresh_code_and_notebook(output)

    summary = {
        "new_images": len(new_clean["images"]),
        "new_annotations_original": len(new_coco["annotations"]),
        "new_annotations_kept": len(new_clean["annotations"]),
        "replay_images": len(replay_clean["images"]),
        "replay_annotations_original": len(replay_coco["annotations"]),
        "replay_annotations_kept": len(replay_clean["annotations"]),
        "combined_images": len(merged["images"]),
        "combined_annotations": len(merged["annotations"]),
        "new_train_z_repeated_twice": new_train_z,
        "validation_z": [190, 200, 210, 220],
        "replay_context_note": (
            "Missing edge context is filled from the nearest available replay Z plane."
        ),
    }
    write_json(output / "combined_package_summary.json", summary)
    archive = shutil.make_archive(
        str(output),
        "zip",
        root_dir=output.parent,
        base_dir=output.name,
    )
    print(json.dumps({**summary, "package": str(output), "archive": archive}, indent=2))


if __name__ == "__main__":
    main()
