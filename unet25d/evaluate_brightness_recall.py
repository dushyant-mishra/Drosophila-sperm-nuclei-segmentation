import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
import tifffile
import torch
from PIL import Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from infer_tiled_unet25d import predict_tiled_probability
from prepare_dataset import (
    dilate_binary,
    load_config,
    load_context,
    load_sample_roi,
    parse_z,
)
from torch_device import describe_torch_device, select_torch_device
from train_unet25d import build_model


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
        if isinstance(polygon, list) and len(polygon) >= 6 and len(polygon) % 2 == 0:
            draw.polygon(list(zip(polygon[0::2], polygon[1::2])), fill=1)
    return np.asarray(mask, dtype=bool)


def load_validation_annotations(cfg):
    coco = json.loads(Path(cfg["annotation_manifest"]).read_text(encoding="utf-8"))
    valid_z = {int(z) for z in cfg["valid_z_indices"]}
    images = {
        image["id"]: image
        for image in coco["images"]
        if parse_z(Path(image["file_name"]).name, cfg["z_regex"]) in valid_z
    }
    by_image = {image_id: [] for image_id in images}
    for annotation in coco["annotations"]:
        if annotation["image_id"] in by_image:
            by_image[annotation["image_id"]].append(annotation)
    return images, by_image


def robust_display(image, roi):
    values = image[roi].astype(np.float32)
    lo, hi = np.percentile(values, [1.0, 99.5])
    return np.clip(
        (image.astype(np.float32) - lo) / max(float(hi - lo), 1.0),
        0.0,
        1.0,
    )


def object_metrics(raw, probability, mask, roi):
    outer = dilate_binary(mask, 10)
    inner = dilate_binary(mask, 2)
    ring = outer & ~inner & roi
    if not ring.any():
        ring = roi & ~mask
    signal = float(np.median(raw[mask]))
    background = float(np.median(raw[ring]))
    noise = float(1.4826 * np.median(np.abs(raw[ring] - background)))
    contrast = signal - background
    values = probability[mask]
    return {
        "raw_median_intensity": signal,
        "local_background_median": background,
        "local_contrast": contrast,
        "local_contrast_to_noise": contrast / max(noise, 1e-6),
        "unet_probability_mean": float(values.mean()),
        "unet_probability_median": float(np.median(values)),
        "unet_probability_p90": float(np.percentile(values, 90)),
        "unet_probability_max": float(values.max()),
    }


def assign_brightness_bins(rows):
    contrasts = np.asarray([row["local_contrast"] for row in rows], dtype=float)
    low, high = np.quantile(contrasts, [1 / 3, 2 / 3])
    for row in rows:
        value = row["local_contrast"]
        row["brightness_group"] = (
            "faint" if value <= low else "intermediate" if value <= high else "bright"
        )
    return float(low), float(high)


def save_summary(rows, thresholds, output):
    fieldnames = list(rows[0])
    with (output / "nucleus_level_brightness_recall.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summaries = []
    for checkpoint in sorted({row["checkpoint"] for row in rows}):
        checkpoint_rows = [row for row in rows if row["checkpoint"] == checkpoint]
        for threshold in thresholds:
            key = f"supported_at_{threshold:.2f}"
            for group in ("all", "faint", "intermediate", "bright"):
                selected = (
                    checkpoint_rows
                    if group == "all"
                    else [
                        row
                        for row in checkpoint_rows
                        if row["brightness_group"] == group
                    ]
                )
                summaries.append(
                    {
                        "checkpoint": checkpoint,
                        "threshold": threshold,
                        "brightness_group": group,
                        "annotated_nuclei": len(selected),
                        "probability_supported": sum(int(row[key]) for row in selected),
                        "probability_support_recall": (
                            sum(int(row[key]) for row in selected) / max(len(selected), 1)
                        ),
                        "mean_nucleus_probability": float(
                            np.mean(
                                [row["unet_probability_mean"] for row in selected]
                            )
                        ),
                    }
                )
    with (output / "brightness_recall_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"faint": "#d62728", "intermediate": "#ffbf00", "bright": "#2ca02c"}
    for checkpoint in sorted({row["checkpoint"] for row in summaries}):
        for group in ("faint", "intermediate", "bright"):
            selected = [
                row
                for row in summaries
                if row["checkpoint"] == checkpoint
                and row["brightness_group"] == group
            ]
            ax.plot(
                [row["threshold"] for row in selected],
                [row["probability_support_recall"] for row in selected],
                marker="o",
                color=colors[group],
                alpha=0.75,
                label=f"{checkpoint}: {group}",
            )
    ax.set(xlabel="U-Net probability threshold", ylabel="Annotated-nucleus support recall")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output / "brightness_recall_by_threshold.png", dpi=180)
    plt.close(fig)


def draw_review_page(pdf, raw, roi, probability, gt_masks, checkpoint, z, thresholds):
    display = robust_display(raw, roi)
    gt_union = np.logical_or.reduce(gt_masks) if gt_masks else np.zeros(raw.shape, bool)
    panels = [
        ("Raw TIFF", display, "gray"),
        ("ROI-normalized", display * roi, "gray"),
        ("Ground truth", gt_union, "gray"),
        ("Continuous probability", probability, "magma"),
    ]
    panels.extend(
        (f"Probability >= {threshold:.2f}", probability >= threshold, "gray")
        for threshold in thresholds
    )
    columns = 3
    rows = int(np.ceil(len(panels) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(13, 4.1 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, image, cmap) in zip(axes, panels):
        ax.imshow(image, cmap=cmap, vmin=0, vmax=1)
        ax.contour(roi, levels=[0.5], colors=["cyan"], linewidths=0.4)
        if title != "Ground truth":
            ax.contour(gt_union, levels=[0.5], colors=["lime"], linewidths=0.45)
        ax.set_title(title)
        ax.axis("off")
    for ax in axes[len(panels) :]:
        ax.axis("off")
    fig.suptitle(f"{checkpoint}, validation z{z:04d}", fontsize=15)
    fig.tight_layout()
    pdf.savefig(fig, dpi=160)
    plt.close(fig)


def load_model(checkpoint, cfg, device):
    payload = torch.load(checkpoint, map_location="cpu")
    model = build_model(payload.get("config", cfg))
    model.load_state_dict(payload["model"])
    model = model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate U-Net probability support as a function of nucleus brightness."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Repeat as LABEL=PATH for baseline and epoch snapshots.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--coverage-fraction", type=float, default=0.25)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    thresholds = [
        float(value)
        for value in cfg.get(
            "brightness_validation_thresholds",
            [0.05, 0.10, 0.20, 0.30, 0.50],
        )
    ]
    checkpoints = {}
    for item in args.checkpoint:
        if "=" not in item:
            raise ValueError("--checkpoint must use LABEL=PATH")
        label, path = item.split("=", 1)
        checkpoints[label] = Path(path)

    images, annotations_by_image = load_validation_annotations(cfg)
    device = select_torch_device()
    print(f"Brightness validation device: {describe_torch_device(device)}")
    all_rows = []
    bin_limits = None
    with PdfPages(output / "checkpoint_image_versions.pdf") as pdf:
        for label, checkpoint in checkpoints.items():
            model = load_model(checkpoint, cfg, device)
            checkpoint_rows = []
            for image_id, image in sorted(
                images.items(),
                key=lambda item: parse_z(
                    Path(item[1]["file_name"]).name,
                    cfg["z_regex"],
                ),
            ):
                file_name = Path(image["file_name"]).name
                z = parse_z(file_name, cfg["z_regex"])
                context = load_context(cfg["stack_image_dir"], cfg["image_pattern"], z)
                raw_path = Path(cfg["stack_image_dir"]) / cfg["image_pattern"].format(z=z)
                raw = np.asarray(tifffile.imread(raw_path))
                if raw.ndim > 2:
                    raw = raw[..., 0]
                roi = load_sample_roi(cfg, z, raw.shape)
                probability = predict_tiled_probability(model, context, roi, cfg, device)
                tifffile.imwrite(
                    output / f"{label}_z{z:04d}_probability.tif",
                    probability.astype(np.float32),
                )

                gt_masks = []
                for annotation in annotations_by_image[image_id]:
                    mask = annotation_mask(annotation, image) & roi
                    if not mask.any():
                        continue
                    gt_masks.append(mask)
                    metrics = object_metrics(raw, probability, mask, roi)
                    row = {
                        "checkpoint": label,
                        "z": z,
                        "image_file": file_name,
                        "annotation_id": annotation["id"],
                        "annotation_pixels": int(mask.sum()),
                        **metrics,
                    }
                    for threshold in thresholds:
                        coverage = float((probability[mask] >= threshold).mean())
                        row[f"coverage_at_{threshold:.2f}"] = coverage
                        row[f"supported_at_{threshold:.2f}"] = int(
                            coverage >= float(args.coverage_fraction)
                        )
                    checkpoint_rows.append(row)
                draw_review_page(
                    pdf,
                    raw,
                    roi,
                    probability,
                    gt_masks,
                    label,
                    z,
                    thresholds,
                )
            if bin_limits is None:
                bin_limits = assign_brightness_bins(checkpoint_rows)
            else:
                low, high = bin_limits
                for row in checkpoint_rows:
                    value = row["local_contrast"]
                    row["brightness_group"] = (
                        "faint"
                        if value <= low
                        else "intermediate"
                        if value <= high
                        else "bright"
                    )
            all_rows.extend(checkpoint_rows)
            del model
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()

    save_summary(all_rows, thresholds, output)
    write_json = {
        "checkpoints": {label: str(path) for label, path in checkpoints.items()},
        "thresholds": thresholds,
        "coverage_fraction_required": float(args.coverage_fraction),
        "brightness_bin_local_contrast_boundaries": {
            "faint_upper": bin_limits[0],
            "intermediate_upper": bin_limits[1],
        },
        "metric_note": (
            "Probability support recall asks whether at least the configured fraction "
            "of an annotated nucleus exceeds a threshold. It is not instance-splitting "
            "accuracy and does not replace final Saturn object QC."
        ),
    }
    (output / "brightness_validation_metadata.json").write_text(
        json.dumps(write_json, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved brightness-stratified validation: {output}")


if __name__ == "__main__":
    main()
