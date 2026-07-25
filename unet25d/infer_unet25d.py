import argparse
import csv
import re
from collections import deque
from pathlib import Path

import numpy as np
import tifffile
import torch
import yaml
from PIL import Image

from prepare_dataset import load_context, load_config
from torch_device import describe_torch_device, select_torch_device
from train_unet25d import build_model


def connected_components(mask):
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    current = 0
    rows = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or labels[y, x]:
                continue
            current += 1
            q = deque([(y, x)])
            labels[y, x] = current
            coords = []
            while q:
                cy, cx = q.popleft()
                coords.append((cy, cx))
                for ny in (cy - 1, cy, cy + 1):
                    for nx in (cx - 1, cx, cx + 1):
                        if ny < 0 or nx < 0 or ny >= h or nx >= w:
                            continue
                        if labels[ny, nx] or not mask[ny, nx]:
                            continue
                        labels[ny, nx] = current
                        q.append((ny, nx))
            arr = np.asarray(coords)
            rows.append(
                {
                    "label": current,
                    "area_px": int(len(coords)),
                    "centroid_y": float(arr[:, 0].mean()),
                    "centroid_x": float(arr[:, 1].mean()),
                    "bbox_y0": int(arr[:, 0].min()),
                    "bbox_x0": int(arr[:, 1].min()),
                    "bbox_y1": int(arr[:, 0].max()),
                    "bbox_x1": int(arr[:, 1].max()),
                }
            )
    return labels, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot_unet25d.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    checkpoint = Path(args.checkpoint) if args.checkpoint else Path(cfg["output_dir"]) / "checkpoints" / "best.pt"
    out_dir = Path(cfg["output_dir"]) / "inference"
    prob_dir = out_dir / "probability_maps"
    mask_dir = out_dir / "binary_masks"
    prob_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    device = select_torch_device()
    print(f"PyTorch inference device: {describe_torch_device(device)}")
    payload = torch.load(checkpoint, map_location="cpu")
    model_cfg = payload.get("config", cfg)
    model = build_model(model_cfg)
    model.load_state_dict(payload["model"])
    model = model.to(device)
    model.eval()

    rows = []
    for z in cfg["infer_z_indices"]:
        x = load_context(cfg["stack_image_dir"], cfg["image_pattern"], int(z))
        tensor = torch.from_numpy(x[None, ...]).to(device)
        with torch.inference_mode():
            prob = torch.sigmoid(model(tensor))[0, 0].detach().cpu().numpy()
        mask = prob >= float(cfg["threshold"])
        labels, components = connected_components(mask)

        stem = f"z{int(z):02d}"
        tifffile.imwrite(prob_dir / f"{stem}_probability.tif", prob.astype(np.float32))
        Image.fromarray(np.clip(prob * 255, 0, 255).astype(np.uint8)).save(prob_dir / f"{stem}_probability.png")
        Image.fromarray(mask.astype(np.uint8) * 255).save(mask_dir / f"{stem}_mask.png")

        rows.append({"z": z, "count": len(components), "mask_pixels": int(mask.sum())})
        with open(mask_dir / f"{stem}_components.csv", "w", newline="", encoding="utf-8") as f:
            fieldnames = ["label", "area_px", "centroid_y", "centroid_x", "bbox_y0", "bbox_x0", "bbox_y1", "bbox_x1"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(components)
        print(f"z{int(z):02d}: count={len(components)} mask_pixels={int(mask.sum())}")

    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["z", "count", "mask_pixels"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
