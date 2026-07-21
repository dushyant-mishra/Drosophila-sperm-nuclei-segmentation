import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tifffile
import torch
from PIL import Image, ImageDraw

from prepare_dataset import load_config, load_context
from train_unet25d import build_model


def normalize_display(arr):
    arr = arr.astype(np.float32)
    lo = np.percentile(arr, 1.0)
    hi = np.percentile(arr, 99.5)
    if hi <= lo:
        hi = lo + 1.0
    return (np.clip((arr - lo) / (hi - lo), 0.0, 1.0) * 255).astype(np.uint8)


def load_roi(path, shape):
    if not path:
        return np.ones(shape, dtype=bool)

    roi_path = Path(path)
    if not roi_path.exists():
        raise FileNotFoundError(roi_path)

    if roi_path.suffix.lower() == ".npy":
        roi = np.load(roi_path)
    else:
        roi = np.asarray(Image.open(roi_path).convert("L")) > 0

    if roi.shape != shape:
        raise ValueError(f"ROI shape {roi.shape} does not match image shape {shape}")
    return roi.astype(bool)


def roi_bbox(roi, padding, shape):
    ys, xs = np.where(roi)
    if len(ys) == 0:
        raise ValueError("ROI is empty")
    h, w = shape
    y0 = max(0, int(ys.min()) - padding)
    y1 = min(h, int(ys.max()) + padding + 1)
    x0 = max(0, int(xs.min()) - padding)
    x1 = min(w, int(xs.max()) + padding + 1)
    return y0, y1, x0, x1


def tile_starts(start, stop, tile_size, overlap):
    span = stop - start
    if span <= tile_size:
        return [start]

    step = max(1, tile_size - overlap)
    starts = list(range(start, stop - tile_size + 1, step))
    last = stop - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def blend_window(tile_h, tile_w):
    wy = np.hanning(tile_h) if tile_h > 2 else np.ones(tile_h, dtype=np.float32)
    wx = np.hanning(tile_w) if tile_w > 2 else np.ones(tile_w, dtype=np.float32)
    win = np.outer(wy, wx).astype(np.float32)
    win = np.maximum(win, 0.05)
    return win


def predict_tiled_probability(model, context, roi, cfg, device):
    _, h, w = context.shape
    tile_size = int(cfg.get("unet_tile_size", 256))
    overlap = int(cfg.get("unet_tile_overlap", 64))
    padding = int(cfg.get("unet_roi_padding_px", 32))

    if tile_size <= 0:
        raise ValueError("unet_tile_size must be positive")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("unet_tile_overlap must be >= 0 and < unet_tile_size")

    y0, y1, x0, x1 = roi_bbox(roi, padding, (h, w))
    prob_sum = np.zeros((h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)

    y_starts = tile_starts(y0, y1, tile_size, overlap)
    x_starts = tile_starts(x0, x1, tile_size, overlap)

    with torch.inference_mode():
        for yy in y_starts:
            for xx in x_starts:
                patch = context[:, yy : yy + tile_size, xx : xx + tile_size]
                _, ph, pw = patch.shape
                if ph != tile_size or pw != tile_size:
                    padded = np.zeros((3, tile_size, tile_size), dtype=np.float32)
                    padded[:, :ph, :pw] = patch
                    patch = padded

                tensor = torch.from_numpy(patch[None, ...]).to(device)
                pred = torch.sigmoid(model(tensor))[0, 0].detach().cpu().numpy()
                pred = pred[:ph, :pw].astype(np.float32)
                win = blend_window(ph, pw)
                prob_sum[yy : yy + ph, xx : xx + pw] += pred * win
                weight_sum[yy : yy + ph, xx : xx + pw] += win

    prob = np.zeros((h, w), dtype=np.float32)
    valid = weight_sum > 0
    prob[valid] = prob_sum[valid] / weight_sum[valid]
    if bool(cfg.get("unet_outside_roi_zero", True)):
        prob[~roi] = 0.0
    return np.clip(prob, 0.0, 1.0)


def connected_components(mask):
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    rows = []
    label = 0
    stack = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or labels[y, x]:
                continue
            label += 1
            stack.append((y, x))
            labels[y, x] = label
            coords = []
            while stack:
                cy, cx = stack.pop()
                coords.append((cy, cx))
                for ny in (cy - 1, cy, cy + 1):
                    for nx in (cx - 1, cx, cx + 1):
                        if ny < 0 or nx < 0 or ny >= h or nx >= w:
                            continue
                        if labels[ny, nx] or not mask[ny, nx]:
                            continue
                        labels[ny, nx] = label
                        stack.append((ny, nx))
            arr = np.asarray(coords)
            rows.append(
                {
                    "label": label,
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


def save_overlay(raw, mask, out_path, title):
    base = Image.fromarray(normalize_display(raw)).convert("RGB")
    mask_img = Image.fromarray(mask.astype(np.uint8) * 255).convert("L")
    red = Image.new("RGBA", base.size, (255, 40, 40, 110))
    clear = Image.new("RGBA", base.size, (0, 0, 0, 0))
    overlay = Image.composite(red, clear, mask_img)
    combined = Image.alpha_composite(base.convert("RGBA"), overlay)
    ImageDraw.Draw(combined).text((10, 10), title, fill=(255, 255, 255, 255))
    combined.convert("RGB").save(out_path)
    return combined.convert("RGB")


def main():
    parser = argparse.ArgumentParser(description="ROI-aware tiled 2.5D U-Net inference")
    parser.add_argument("--config", default="configs/pilot_resatt_partial_labels_tight_colab.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--roi", default=None, help="Optional ROI mask path (.npy or image).")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--candidate-threshold", type=float, default=None)
    parser.add_argument("--seed-threshold", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoint = Path(args.checkpoint)
    out_dir = Path(args.output_dir) if args.output_dir else Path(cfg["output_dir"]) / "tiled_inference"
    prob_dir = out_dir / "probability_maps"
    mask_dir = out_dir / "soft_masks"
    overlay_dir = out_dir / "overlays"
    component_dir = out_dir / "components"
    for path in (prob_dir, mask_dir, overlay_dir, component_dir):
        path.mkdir(parents=True, exist_ok=True)

    candidate_threshold = float(
        args.candidate_threshold
        if args.candidate_threshold is not None
        else cfg.get("unet_candidate_threshold", 0.05)
    )
    seed_threshold = float(
        args.seed_threshold if args.seed_threshold is not None else cfg.get("unet_seed_threshold", 0.30)
    )

    if not (0.0 <= candidate_threshold <= seed_threshold <= 1.0):
        raise ValueError("Require 0 <= candidate_threshold <= seed_threshold <= 1")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = torch.load(checkpoint, map_location=device)
    model_cfg = payload.get("config", cfg)
    model = build_model(model_cfg).to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    summary_rows = []
    candidate_pages = []
    seed_pages = []
    for z in cfg["infer_z_indices"]:
        z = int(z)
        context = load_context(cfg["stack_image_dir"], cfg["image_pattern"], z)
        raw = context[1]
        roi = load_roi(args.roi or cfg.get("roi_mask_path", ""), raw.shape)
        prob = predict_tiled_probability(model, context, roi, cfg, device)

        candidate_mask = prob >= candidate_threshold
        seed_mask = prob >= seed_threshold
        candidate_labels, candidate_components = connected_components(candidate_mask)
        seed_labels, seed_components = connected_components(seed_mask)

        stem = f"z{z:02d}"
        tifffile.imwrite(prob_dir / f"{stem}_unet_probability.tif", prob.astype(np.float32))
        Image.fromarray(np.clip(prob * 255, 0, 255).astype(np.uint8)).save(prob_dir / f"{stem}_unet_probability.png")
        Image.fromarray(candidate_mask.astype(np.uint8) * 255).save(mask_dir / f"{stem}_candidate_mask.png")
        Image.fromarray(seed_mask.astype(np.uint8) * 255).save(mask_dir / f"{stem}_seed_mask.png")
        tifffile.imwrite(mask_dir / f"{stem}_candidate_labels.tif", candidate_labels.astype(np.uint16))
        tifffile.imwrite(mask_dir / f"{stem}_seed_labels.tif", seed_labels.astype(np.uint16))

        with open(component_dir / f"{stem}_candidate_components.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(candidate_components[0].keys()) if candidate_components else ["label"])
            writer.writeheader()
            writer.writerows(candidate_components)
        with open(component_dir / f"{stem}_seed_components.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(seed_components[0].keys()) if seed_components else ["label"])
            writer.writeheader()
            writer.writerows(seed_components)

        candidate_pages.append(
            save_overlay(raw, candidate_mask, overlay_dir / f"{stem}_candidate_overlay.png", f"candidate >= {candidate_threshold:g} z{z:02d}")
        )
        seed_pages.append(
            save_overlay(raw, seed_mask, overlay_dir / f"{stem}_seed_overlay.png", f"seed >= {seed_threshold:g} z{z:02d}")
        )

        summary_rows.append(
            {
                "z": z,
                "candidate_threshold": candidate_threshold,
                "seed_threshold": seed_threshold,
                "candidate_count": len(candidate_components),
                "seed_count": len(seed_components),
                "candidate_mask_pixels": int(candidate_mask.sum()),
                "seed_mask_pixels": int(seed_mask.sum()),
                "roi_pixels": int(roi.sum()),
                "outside_roi_probability_sum": float(prob[~roi].sum()),
            }
        )
        print(
            f"z{z:02d}: candidate_count={len(candidate_components)} "
            f"seed_count={len(seed_components)} candidate_pixels={int(candidate_mask.sum())}"
        )

    with open(out_dir / "soft_threshold_summary.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "z",
            "candidate_threshold",
            "seed_threshold",
            "candidate_count",
            "seed_count",
            "candidate_mask_pixels",
            "seed_mask_pixels",
            "roi_pixels",
            "outside_roi_probability_sum",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    metadata = {
        "output_type": "stitched_full_frame_probability_maps",
        "measurement_note": (
            "Probability maps are stitched before thresholding and are not length/width/count measurements. "
            "Candidate and seed masks are review aids; Saturn should perform final ROI-aware geometry, "
            "tracking, and biological QC."
        ),
        "stitching_mode": str(cfg.get("unet_stitch_mode", "weighted_average")),
        "tile_size": int(cfg.get("unet_tile_size", 256)),
        "tile_overlap": int(cfg.get("unet_tile_overlap", 64)),
        "roi_padding_px": int(cfg.get("unet_roi_padding_px", 32)),
        "outside_roi_zero": bool(cfg.get("unet_outside_roi_zero", True)),
        "candidate_threshold": candidate_threshold,
        "seed_threshold": seed_threshold,
        "checkpoint": str(checkpoint),
        "config": str(args.config),
    }
    with open(out_dir / "inference_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if candidate_pages:
        candidate_pages[0].save(
            overlay_dir / "candidate_overlay_review.pdf",
            save_all=True,
            append_images=candidate_pages[1:],
        )
    if seed_pages:
        seed_pages[0].save(overlay_dir / "seed_overlay_review.pdf", save_all=True, append_images=seed_pages[1:])

    print(f"Saved tiled inference outputs: {out_dir}")


if __name__ == "__main__":
    main()
