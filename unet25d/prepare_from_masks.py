import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from prepare_dataset import load_config, load_context


def read_mask(path):
    mask = Image.open(path).convert("L")
    return (np.asarray(mask) > 0).astype(np.uint8)


def write_split(split_name, z_indices, cfg):
    out_dir = Path(cfg["output_dir"]) / "dataset" / split_name
    out_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = out_dir / "masks_png"
    masks_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = [
        "split,file_name,z,npz_path,mask_png,supervision_png,annotation_count,mask_pixels,supervised_pixels"
    ]
    for z in z_indices:
        z = int(z)
        file_name = cfg["image_pattern"].format(z=z)
        mask_path = Path(cfg["mask_dir"]) / f"z{z:02d}_mask.png"
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)

        x = load_context(cfg["stack_image_dir"], cfg["image_pattern"], z)
        y = read_mask(mask_path)
        supervision = np.ones(y.shape, dtype=np.uint8)

        stem = Path(file_name).stem
        npz_path = out_dir / f"{stem}.npz"
        out_mask_path = masks_dir / f"{stem}_mask.png"
        supervision_path = masks_dir / f"{stem}_supervision.png"

        np.savez_compressed(
            npz_path,
            image=x.astype(np.float32),
            mask=y,
            supervision_mask=supervision,
            z=np.array([z], dtype=np.int16),
            file_name=np.array([file_name]),
        )
        Image.fromarray(y * 255).save(out_mask_path)
        Image.fromarray(supervision * 255).save(supervision_path)
        manifest_rows.append(
            f"{split_name},{file_name},{z},{npz_path},{out_mask_path},{supervision_path},"
            f"pseudo,{int(y.sum())},{int(supervision.sum())}"
        )

    manifest_path = out_dir / "manifest.csv"
    manifest_path.write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
    return len(z_indices), manifest_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_count, train_manifest = write_split("train", cfg["train_z_indices"], cfg)
    valid_count, valid_manifest = write_split("valid", cfg["valid_z_indices"], cfg)
    print(f"Prepared pseudo-label train samples: {train_count} -> {train_manifest}")
    print(f"Prepared pseudo-label valid samples: {valid_count} -> {valid_manifest}")


if __name__ == "__main__":
    main()
