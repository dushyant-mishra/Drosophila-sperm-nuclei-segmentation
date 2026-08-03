"""Compare control and annotation-tolerant U-Net checkpoints on fixed masks."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt, label
from scipy.optimize import linear_sum_assignment
from skimage.segmentation import watershed

from infer_tiled_unet25d import predict_tiled_probability
from prepare_dataset import dilate_binary, load_config
from torch_device import describe_torch_device, select_torch_device
from train_unet25d import build_model


def safe_divide(numerator, denominator):
    return float(numerator / denominator) if denominator else np.nan


def load_model(checkpoint_path, fallback_cfg, device):
    payload = torch.load(checkpoint_path, map_location="cpu")
    model_cfg = payload.get("config", fallback_cfg)
    model = build_model(model_cfg)
    model.load_state_dict(payload.get("model", payload))
    model.to(device).eval()
    return model


def binary_boundary(mask):
    mask = np.asarray(mask, dtype=bool)
    return mask & ~binary_erosion(mask, border_value=0)


def boundary_metrics(predicted, target, valid, tolerance=1):
    predicted_boundary = binary_boundary(predicted) & valid
    target_boundary = binary_boundary(target) & valid
    if not np.any(predicted_boundary) or not np.any(target_boundary):
        return np.nan, np.nan
    target_tolerance = dilate_binary(target_boundary, tolerance)
    predicted_tolerance = dilate_binary(predicted_boundary, tolerance)
    precision = safe_divide(
        np.count_nonzero(predicted_boundary & target_tolerance),
        np.count_nonzero(predicted_boundary),
    )
    recall = safe_divide(
        np.count_nonzero(target_boundary & predicted_tolerance),
        np.count_nonzero(target_boundary),
    )
    boundary_f1 = safe_divide(2.0 * precision * recall, precision + recall)
    distance_to_target = distance_transform_edt(~target_boundary)
    distance_to_predicted = distance_transform_edt(~predicted_boundary)
    mean_distance = 0.5 * (
        float(distance_to_target[predicted_boundary].mean())
        + float(distance_to_predicted[target_boundary].mean())
    )
    return boundary_f1, mean_distance


def remove_small_components(binary_mask, minimum_area):
    labels, count = label(binary_mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0 or minimum_area <= 1:
        return labels
    sizes = np.bincount(labels.ravel())
    keep = sizes >= int(minimum_area)
    keep[0] = False
    filtered = keep[labels]
    return label(filtered, structure=np.ones((3, 3), dtype=np.uint8))[0]


def marker_controlled_instances(
    foreground_probability,
    core_probability,
    foreground_threshold,
    core_threshold,
    roi,
    minimum_area,
):
    foreground = (foreground_probability >= foreground_threshold) & roi
    foreground_labels = remove_small_components(foreground, minimum_area)
    foreground = foreground_labels > 0
    core = (core_probability >= core_threshold) & foreground
    markers, _ = label(core, structure=np.ones((3, 3), dtype=np.uint8))

    # Preserve a foreground component even if the learned core head has no
    # marker at this threshold. The deterministic maximum-probability pixel is
    # a fallback marker, not an additional biological acceptance gate.
    next_marker = int(markers.max())
    for component_id in range(1, int(foreground_labels.max()) + 1):
        component = foreground_labels == component_id
        if np.any(markers[component] > 0):
            continue
        positions = np.argwhere(component)
        values = foreground_probability[component]
        best = positions[int(np.argmax(values))]
        next_marker += 1
        markers[int(best[0]), int(best[1])] = next_marker

    if next_marker == 0:
        return np.zeros(foreground.shape, dtype=np.int32)
    return watershed(
        -foreground_probability,
        markers=markers,
        mask=foreground,
        connectivity=np.ones((3, 3), dtype=np.uint8),
    ).astype(np.int32)


def overlap_table(reference_labels, predicted_labels):
    reference_count = int(reference_labels.max())
    predicted_count = int(predicted_labels.max())
    overlaps = np.zeros((reference_count + 1, predicted_count + 1), dtype=np.int64)
    np.add.at(
        overlaps,
        (reference_labels.ravel(), predicted_labels.ravel()),
        1,
    )
    return overlaps


def instance_metrics(reference_labels, predicted_labels, iou_threshold, touching_ids):
    overlaps = overlap_table(reference_labels, predicted_labels)
    reference_area = overlaps.sum(axis=1)
    predicted_area = overlaps.sum(axis=0)
    intersection = overlaps[1:, 1:]
    union = (
        reference_area[1:, None]
        + predicted_area[None, 1:]
        - intersection
    )
    iou = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float64),
        where=union > 0,
    )
    matched_reference = set()
    matched_predicted = set()
    matched_iou = {}
    if iou.size:
        rows, columns = linear_sum_assignment(1.0 - iou)
        for row, column in zip(rows, columns):
            if iou[row, column] >= iou_threshold:
                reference_id = int(row + 1)
                predicted_id = int(column + 1)
                matched_reference.add(reference_id)
                matched_predicted.add(predicted_id)
                matched_iou[reference_id] = float(iou[row, column])

    overlap_fraction_reference = np.divide(
        intersection,
        reference_area[1:, None],
        out=np.zeros_like(intersection, dtype=np.float64),
        where=reference_area[1:, None] > 0,
    )
    meaningful = overlap_fraction_reference >= 0.10
    merged_predictions = int(np.sum(meaningful.sum(axis=0) >= 2)) if meaningful.size else 0
    split_references = int(np.sum(meaningful.sum(axis=1) >= 2)) if meaningful.size else 0
    reference_count = len(reference_area) - 1
    predicted_count = len(predicted_area) - 1
    true_positive = len(matched_reference)
    precision = safe_divide(true_positive, predicted_count)
    recall = safe_divide(true_positive, reference_count)
    f1 = safe_divide(2.0 * precision * recall, precision + recall)
    touching_ids = set(int(value) for value in touching_ids)
    touching_recall = safe_divide(
        len(touching_ids & matched_reference),
        len(touching_ids),
    )
    return {
        "reference_count": reference_count,
        "predicted_count": predicted_count,
        "instance_true_positive": true_positive,
        "instance_precision": precision,
        "instance_recall": recall,
        "instance_f1": f1,
        "count_error": predicted_count - reference_count,
        "count_absolute_error": abs(predicted_count - reference_count),
        "merged_prediction_count": merged_predictions,
        "split_reference_count": split_references,
        "missed_reference_count": reference_count - true_positive,
        "duplicate_prediction_count": predicted_count - true_positive,
        "touching_reference_count": len(touching_ids),
        "touching_instance_recall": touching_recall,
        "matched_reference_ids": matched_reference,
        "matched_iou": matched_iou,
    }


def pixel_metrics(predicted, target, valid):
    predicted = predicted & valid
    target = target & valid
    true_positive = int(np.count_nonzero(predicted & target))
    false_positive = int(np.count_nonzero(predicted & ~target))
    false_negative = int(np.count_nonzero(~predicted & target))
    precision = safe_divide(true_positive, true_positive + false_positive)
    recall = safe_divide(true_positive, true_positive + false_negative)
    dice = safe_divide(2 * true_positive, 2 * true_positive + false_positive + false_negative)
    iou = safe_divide(true_positive, true_positive + false_positive + false_negative)
    return {
        "pixel_precision": precision,
        "pixel_recall": recall,
        "pixel_dice": dice,
        "pixel_iou": iou,
        "predicted_area_over_annotated_area": safe_divide(
            np.count_nonzero(predicted),
            np.count_nonzero(target),
        ),
    }


def load_group_map(path):
    if not path:
        return {}
    rows = {}
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            rows[Path(row["annotation_file"]).stem] = row.get("group", "unknown")
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_numeric_rows(rows, group_fields, value_fields):
    grouped = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        grouped.setdefault(key, []).append(row)
    summaries = []
    for key, group_rows in sorted(grouped.items(), key=lambda item: item[0]):
        summary = dict(zip(group_fields, key))
        summary["row_count"] = len(group_rows)
        for field in value_fields:
            values = np.asarray(
                [float(row[field]) for row in group_rows], dtype=np.float64
            )
            finite = values[np.isfinite(values)]
            summary[f"mean_{field}"] = (
                float(finite.mean()) if finite.size else np.nan
            )
            summary[f"median_{field}"] = (
                float(np.median(finite)) if finite.size else np.nan
            )
        summaries.append(summary)
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-a", required=True)
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--config-b", required=True)
    parser.add_argument("--checkpoint-b", required=True)
    parser.add_argument("--config-c", default=None)
    parser.add_argument("--checkpoint-c", default=None)
    parser.add_argument("--reference-dataset", default=None)
    parser.add_argument("--group-key", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.20)
    parser.add_argument("--minimum-component-px", type=int, default=3)
    args = parser.parse_args()

    cfg_a = load_config(args.config_a)
    cfg_b = load_config(args.config_b)
    if bool(args.config_c) != bool(args.checkpoint_c):
        parser.error("--config-c and --checkpoint-c must be supplied together")
    cfg_c = load_config(args.config_c) if args.config_c else None
    thresholds = sorted(
        set(float(value) for value in cfg_a.get("validation_thresholds", []))
        | set(float(value) for value in cfg_b.get("validation_thresholds", []))
        | set(
            float(value)
            for value in (cfg_c or {}).get("validation_thresholds", [])
        )
    )
    if not thresholds:
        thresholds = [0.03, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    reference_dataset = Path(
        args.reference_dataset
        or Path(cfg_b["output_dir"]) / "dataset" / "valid"
    )
    sample_paths = sorted(reference_dataset.glob("*.npz"))
    if not sample_paths:
        raise FileNotFoundError(f"No validation NPZ files in {reference_dataset}")

    device = select_torch_device()
    print(f"Evaluation device: {describe_torch_device(device)}")
    models = {
        "model_a_replay_control": (
            load_model(args.checkpoint_a, cfg_a, device),
            cfg_a,
        ),
        "model_b_annotation_tolerant": (
            load_model(args.checkpoint_b, cfg_b, device),
            cfg_b,
        ),
    }
    if cfg_c is not None:
        models["model_c_dual_head"] = (
            load_model(args.checkpoint_c, cfg_c, device),
            cfg_c,
        )
    group_map = load_group_map(args.group_key)
    pixel_rows = []
    image_rows = []
    object_rows = []

    for sample_path in sample_paths:
        with np.load(sample_path) as sample:
            context = sample["image"].astype(np.float32)
            target = sample["raw_annotation_mask"].astype(bool)
            reference_labels = sample["instance_labels"].astype(np.int32)
            roi = sample["roi_mask"].astype(bool)
            partial_valid = (
                sample["partial_label_supervision_mask"].astype(bool)
                if "partial_label_supervision_mask" in sample
                else roi.copy()
            )
        stem = sample_path.stem
        group = group_map.get(stem, "unknown")
        touching_pairs = _touching_pairs(reference_labels)
        touching_ids = {value for pair in touching_pairs for value in pair}
        center = context[1]
        object_intensity = {}
        for reference_id in range(1, int(reference_labels.max()) + 1):
            values = center[reference_labels == reference_id]
            object_intensity[reference_id] = float(values.mean()) if values.size else np.nan
        finite_intensity = np.asarray(
            [value for value in object_intensity.values() if np.isfinite(value)]
        )
        faint_cut, bright_cut = (
            np.quantile(finite_intensity, [1 / 3, 2 / 3])
            if finite_intensity.size
            else (np.nan, np.nan)
        )

        for model_name, (model, cfg) in models.items():
            probability = predict_tiled_probability(
                model, context, roi, cfg, device, output_key="foreground"
            )
            probability_path = output / f"{model_name}_{stem}_probability.npy"
            np.save(probability_path, probability.astype(np.float32))
            core_probability = None
            if model_name == "model_c_dual_head":
                core_probability = predict_tiled_probability(
                    model, context, roi, cfg, device, output_key="core"
                )
                np.save(
                    output / f"{model_name}_{stem}_core_probability.npy",
                    core_probability.astype(np.float32),
                )
            for threshold in thresholds:
                if core_probability is None:
                    predicted = probability >= threshold
                    predicted_labels = remove_small_components(
                        predicted & roi,
                        args.minimum_component_px,
                    )
                    instance_method = "connected_components"
                else:
                    predicted_labels = marker_controlled_instances(
                        probability,
                        core_probability,
                        threshold,
                        float(cfg.get("core_seed_threshold", 0.50)),
                        roi,
                        args.minimum_component_px,
                    )
                    instance_method = "core_marker_watershed"
                predicted = predicted_labels > 0
                valid = roi & partial_valid
                pixels = pixel_metrics(predicted, target, valid)
                boundary_f1, contour_distance = boundary_metrics(
                    predicted,
                    target,
                    valid,
                )
                instances = instance_metrics(
                    reference_labels,
                    predicted_labels,
                    args.iou_threshold,
                    touching_ids,
                )
                base = {
                    "model": model_name,
                    "threshold": threshold,
                    "image": stem,
                    "group": group,
                    "instance_method": instance_method,
                }
                pixel_rows.append(
                    {
                        **base,
                        **pixels,
                        "boundary_f1_tolerance_1px": boundary_f1,
                        "mean_symmetric_contour_distance_px": contour_distance,
                    }
                )
                image_rows.append(
                    {
                        **base,
                        **{
                            key: value
                            for key, value in instances.items()
                            if key not in {"matched_reference_ids", "matched_iou"}
                        },
                    }
                )
                for reference_id, intensity in object_intensity.items():
                    brightness = (
                        "faint"
                        if intensity <= faint_cut
                        else "bright"
                        if intensity >= bright_cut
                        else "intermediate"
                    )
                    object_rows.append(
                        {
                            **base,
                            "reference_instance_id": reference_id,
                            "matched": reference_id in instances["matched_reference_ids"],
                            "matched_iou": instances["matched_iou"].get(reference_id, np.nan),
                            "brightness_group": brightness,
                            "mean_normalized_intensity": intensity,
                            "touching_annotation": reference_id in touching_ids,
                        }
                    )
            print(f"{model_name}: {stem} complete")

    write_csv(output / "pixel_boundary_metrics.csv", pixel_rows)
    write_csv(output / "instance_image_metrics.csv", image_rows)
    write_csv(output / "instance_object_metrics.csv", object_rows)
    write_csv(
        output / "pixel_boundary_summary.csv",
        aggregate_numeric_rows(
            pixel_rows,
            ["model", "threshold", "group"],
            [
                "pixel_precision",
                "pixel_recall",
                "pixel_dice",
                "pixel_iou",
                "predicted_area_over_annotated_area",
                "boundary_f1_tolerance_1px",
                "mean_symmetric_contour_distance_px",
            ],
        ),
    )
    write_csv(
        output / "instance_summary.csv",
        aggregate_numeric_rows(
            image_rows,
            ["model", "threshold", "group"],
            [
                "instance_precision",
                "instance_recall",
                "instance_f1",
                "count_error",
                "merged_prediction_count",
                "split_reference_count",
                "touching_instance_recall",
            ],
        ),
    )
    write_csv(
        output / "object_recall_summary.csv",
        aggregate_numeric_rows(
            object_rows,
            ["model", "threshold", "group", "brightness_group", "touching_annotation"],
            ["matched", "matched_iou"],
        ),
    )
    metadata = {
        "models": {
            "model_a_replay_control": str(Path(args.checkpoint_a).resolve()),
            "model_b_annotation_tolerant": str(Path(args.checkpoint_b).resolve()),
        },
        "thresholds": thresholds,
        "validation_images": [path.stem for path in sample_paths],
        "iou_threshold": args.iou_threshold,
        "minimum_component_px": args.minimum_component_px,
        "selection_note": (
            "Do not select by Dice alone. Review instance recall, count error, "
            "merges, splits, boundary behavior, threshold stability, brightness, "
            "touching objects, and group balance."
        ),
    }
    if args.checkpoint_c:
        metadata["models"]["model_c_dual_head"] = str(
            Path(args.checkpoint_c).resolve()
        )
    (output / "evaluation_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def _touching_pairs(instance_labels):
    pairs = set()
    labels_array = np.asarray(instance_labels)
    for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
        first = labels_array[
            max(0, dy) : labels_array.shape[0] + min(0, dy),
            max(0, dx) : labels_array.shape[1] + min(0, dx),
        ]
        second = labels_array[
            max(0, -dy) : labels_array.shape[0] - max(0, dy),
            max(0, -dx) : labels_array.shape[1] - max(0, dx),
        ]
        different = (first > 0) & (second > 0) & (first != second)
        for left, right in zip(first[different], second[different]):
            pairs.add(tuple(sorted((int(left), int(right)))))
    return sorted(pairs)


if __name__ == "__main__":
    main()
