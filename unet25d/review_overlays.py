import argparse
from pathlib import Path

import numpy as np
import tifffile
import yaml
from PIL import Image, ImageDraw, JpegImagePlugin

from prepare_dataset import load_config


def normalize_display(arr):
    arr = arr.astype(np.float32)
    lo = np.percentile(arr, 1)
    hi = np.percentile(arr, 99.5)
    if hi <= lo:
        hi = lo + 1
    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    return (arr * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot_unet25d.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    out_dir = Path(cfg["output_dir"]) / "inference"
    overlay_dir = out_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    pages = []
    for z in cfg["infer_z_indices"]:
        raw_path = Path(cfg["stack_image_dir"]) / cfg["image_pattern"].format(z=int(z))
        mask_path = out_dir / "binary_masks" / f"z{int(z):02d}_mask.png"
        if not mask_path.exists():
            print(f"Missing mask for z{int(z):02d}: {mask_path}")
            continue

        raw = tifffile.imread(str(raw_path))
        base = Image.fromarray(normalize_display(raw)).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        red = Image.new("RGBA", base.size, (255, 40, 40, 110))
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        overlay = Image.composite(red, overlay, mask)
        combined = Image.alpha_composite(base.convert("RGBA"), overlay)
        draw = ImageDraw.Draw(combined)
        draw.text((10, 10), f"U-Net 2.5D z{int(z):02d}", fill=(255, 255, 255, 255))

        out_path = overlay_dir / f"z{int(z):02d}_overlay.png"
        combined.convert("RGB").save(out_path)
        pages.append(combined.convert("RGB"))
        print(f"Saved {out_path}")

    if pages:
        pdf_path = overlay_dir / "unet25d_overlay_review.pdf"
        pages[0].save(pdf_path, save_all=True, append_images=pages[1:])
        print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()

