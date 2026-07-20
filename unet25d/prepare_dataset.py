import argparse
import json
import re
from pathlib import Path

import numpy as np
import tifffile
import yaml
from PIL import Image, ImageDraw


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_z(file_name, z_regex):
    match = re.search(z_regex, file_name)
    if not match:
        raise ValueError(f"Could not parse z index from {file_name!r}")
    return int(match.group(1))


def read_tiff(path):
    arr = tifffile.imread(str(path))
    if arr.ndim > 2:
        arr = arr[..., 0]
    return arr


def robust_normalize_stack(stack):
    out = stack.astype(np.float32)
    lo = np.percentile(out, 1.0)
    hi = np.percentile(out, 99.5)
    if hi <= lo:
        hi = lo + 1.0
    out = (out - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def load_context(stack_dir, image_pattern, z):
    planes = []
    for zz in (z - 1, z, z + 1):
        zz = max(0, min(87, zz))
        path = Path(stack_dir) / image_pattern.format(z=zz)
        if not path.exists():
            raise FileNotFoundError(path)
        planes.append(read_tiff(path))
    return robust_normalize_stack(np.stack(planes, axis=0))


def dilate_binary(mask, radius):
    radius = int(radius)
    if radius <= 0:
        return mask.astype(bool)

    src = mask.astype(bool)
    out = src.copy()
    h, w = src.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            y_src0 = max(0, -dy)
            y_src1 = min(h, h - dy)
            x_src0 = max(0, -dx)
            x_src1 = min(w, w - dx)
            y_dst0 = max(0, dy)
            y_dst1 = min(h, h + dy)
            x_dst0 = max(0, dx)
            x_dst1 = min(w, w + dx)
            out[y_dst0:y_dst1, x_dst0:x_dst1] |= src[y_src0:y_src1, x_src0:x_src1]
    return out


def make_supervision_mask(context, label_mask, cfg):
    if not cfg.get("partial_labels", False):
        return np.ones(label_mask.shape, dtype=np.uint8)

    center = context[1]
    percentile = float(cfg.get("ignore_unlabeled_intensity_percentile", 94.0))
    label_dilate = int(cfg.get("ignore_labeled_dilate_px", 3))
    candidate_dilate = int(cfg.get("ignore_candidate_dilate_px", 1))

    threshold = np.percentile(center, percentile)
    likely_foreground = center >= threshold
    if candidate_dilate:
        likely_foreground = dilate_binary(likely_foreground, candidate_dilate)

    protected_label = dilate_binary(label_mask > 0, label_dilate)
    supervision = np.ones(label_mask.shape, dtype=np.uint8)
    supervision[likely_foreground & ~protected_label] = 0
    supervision[label_mask > 0] = 1
    return supervision


def rasterize_coco(coco_path, z_regex):
    with open(coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {im["id"]: im for im in coco["images"]}
    anns_by_image = {im_id: [] for im_id in images}
    for ann in coco.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    samples = []
    for image_id, im in images.items():
        width = int(im["width"])
        height = int(im["height"])
        mask_img = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask_img)

        for ann in anns_by_image.get(image_id, []):
            segmentation = ann.get("segmentation", [])
            if isinstance(segmentation, list):
                for poly in segmentation:
                    if len(poly) >= 6:
                        xy = [(float(poly[i]), float(poly[i + 1])) for i in range(0, len(poly), 2)]
                        draw.polygon(xy, outline=1, fill=1)

        mask = np.asarray(mask_img, dtype=np.uint8)
        samples.append(
            {
                "file_name": im["file_name"],
                "z": parse_z(im["file_name"], z_regex),
                "mask": mask,
                "annotation_count": len(anns_by_image.get(image_id, [])),
            }
        )
    return samples


def write_split(split_name, coco_path, cfg):
    out_dir = Path(cfg["output_dir"]) / "dataset" / split_name
    out_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = out_dir / "masks_png"
    masks_dir.mkdir(parents=True, exist_ok=True)

    samples = rasterize_coco(coco_path, cfg["z_regex"])
    manifest_rows = [
        "split,file_name,z,npz_path,mask_png,supervision_png,annotation_count,mask_pixels,supervised_pixels"
    ]

    for sample in samples:
        z = sample["z"]
        x = load_context(cfg["stack_image_dir"], cfg["image_pattern"], z)
        y = sample["mask"].astype(np.uint8)
        supervision = make_supervision_mask(x, y, cfg)

        stem = Path(sample["file_name"]).stem
        npz_path = out_dir / f"{stem}.npz"
        mask_path = masks_dir / f"{stem}_mask.png"
        supervision_path = masks_dir / f"{stem}_supervision.png"

        np.savez_compressed(
            npz_path,
            image=x.astype(np.float32),
            mask=y,
            supervision_mask=supervision,
            z=np.array([z], dtype=np.int16),
            file_name=np.array([sample["file_name"]]),
        )
        Image.fromarray(y * 255).save(mask_path)
        Image.fromarray(supervision * 255).save(supervision_path)

        manifest_rows.append(
            f"{split_name},{sample['file_name']},{z},{npz_path},{mask_path},{supervision_path},"
            f"{sample['annotation_count']},{int(y.sum())},{int(supervision.sum())}"
        )

    manifest_path = out_dir / "manifest.csv"
    manifest_path.write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
    return len(samples), manifest_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot_unet25d.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_count, train_manifest = write_split("train", cfg["train_coco"], cfg)
    valid_count, valid_manifest = write_split("valid", cfg["valid_coco"], cfg)

    print(f"Prepared train samples: {train_count} -> {train_manifest}")
    print(f"Prepared valid samples: {valid_count} -> {valid_manifest}")


if __name__ == "__main__":
    main()
