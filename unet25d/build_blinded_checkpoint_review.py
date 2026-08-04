"""Build a blinded event-level review from cached Model C probability maps."""

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import label as connected_components
from skimage.measure import regionprops
from skimage.morphology import skeletonize
from skimage.segmentation import find_boundaries

from evaluate_annotation_tolerant_ab import (
    marker_controlled_instances,
    partial_label_audit,
)
from prepare_dataset import decode_binary_mask_rle


REVIEW_FIELDS = [
    "review_classification",
    "review_confidence",
    "reviewer",
    "review_date",
    "review_notes",
]
PUBLIC_COLUMNS = [
    "event_id",
    "image_id",
    "group",
    "event_type",
    "reference_ids",
    "method_a_prediction_ids",
    "method_b_prediction_ids",
    "difficulty_tags",
    "brightness_stratum",
    "thumbnail_path",
    *REVIEW_FIELDS,
]
VALID_CLASSIFICATIONS = [
    "correct separation",
    "probable correct separation despite reference ambiguity",
    "false longitudinal split",
    "false transverse split",
    "remaining merge",
    "probable unannotated nucleus",
    "probable false positive",
    "ambiguous",
]


def write_csv(path, rows):
    if not rows:
        return
    public_rows = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in rows
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PUBLIC_COLUMNS)
        writer.writeheader()
        writer.writerows(public_rows)


def load_group_map(path):
    rows = {}
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            rows[Path(row["annotation_file"]).stem] = row.get("group", "unknown")
    return rows


def blinded_assignment(first_label, second_label, seed):
    labels = [first_label, second_label]
    random.Random(seed).shuffle(labels)
    return {"Method A": labels[0], "Method B": labels[1]}


def reference_masks(reference_dataset, stem, reference_labels):
    records_path = (
        reference_dataset / "instance_records" / f"{stem}_instances_rle.json"
    )
    if records_path.exists():
        records = json.loads(records_path.read_text(encoding="utf-8"))
        return {
            int(record["local_instance_id"]): decode_binary_mask_rle(record["rle"])
            for record in records
        }
    return {
        reference_id: reference_labels == reference_id
        for reference_id in range(1, int(reference_labels.max()) + 1)
    }


def meaningful_relations(references, predicted_labels, minimum_fraction=0.10):
    reference_to_predictions = {reference_id: [] for reference_id in references}
    prediction_to_references = {
        predicted_id: []
        for predicted_id in range(1, int(predicted_labels.max()) + 1)
    }
    for reference_id, reference_mask in references.items():
        area = int(np.count_nonzero(reference_mask))
        if not area:
            continue
        ids, counts = np.unique(predicted_labels[reference_mask], return_counts=True)
        for predicted_id, count in zip(ids, counts):
            predicted_id = int(predicted_id)
            if predicted_id and count / area >= minimum_fraction:
                reference_to_predictions[reference_id].append(predicted_id)
                prediction_to_references[predicted_id].append(reference_id)
    return reference_to_predictions, prediction_to_references


def mask_measurements(mask):
    mask = np.asarray(mask, dtype=bool)
    area = int(np.count_nonzero(mask))
    if not area:
        return {"area_px": 0, "centerline_px": 0, "width_proxy_px": np.nan,
                "curvature_proxy": np.nan, "solidity": np.nan}
    skeleton = skeletonize(mask)
    centerline = int(np.count_nonzero(skeleton))
    props = regionprops(mask.astype(np.uint8))[0]
    major = max(float(props.axis_major_length), 1e-6)
    return {
        "area_px": area,
        "centerline_px": centerline,
        "width_proxy_px": float(area / max(centerline, 1)),
        "curvature_proxy": float(centerline / major),
        "solidity": float(props.solidity),
    }


def union_masks(masks, ids, shape):
    result = np.zeros(shape, dtype=bool)
    for value in ids:
        if int(value) in masks:
            result |= masks[int(value)]
    return result


def selected_prediction_mask(labels, ids):
    ids = [int(value) for value in ids]
    return np.isin(labels, ids) if ids else np.zeros(labels.shape, dtype=bool)


def crop_slices(mask, shape, padding=24):
    positions = np.argwhere(mask)
    if not positions.size:
        return slice(0, shape[0]), slice(0, shape[1])
    y0, x0 = positions.min(axis=0)
    y1, x1 = positions.max(axis=0) + 1
    return (
        slice(max(0, int(y0) - padding), min(shape[0], int(y1) + padding)),
        slice(max(0, int(x0) - padding), min(shape[1], int(x1) + padding)),
    )


def overlay_boundaries(raw, reference_mask, predicted_mask, roi, crop):
    values = raw[roi]
    lo, hi = np.percentile(values, [1, 99.5]) if values.size else (0, 1)
    image = np.repeat(
        np.clip((raw - lo) / max(hi - lo, 1e-6), 0, 1)[..., None], 3, axis=2
    )
    image[find_boundaries(reference_mask, mode="outer")] = (0.1, 1.0, 0.1)
    image[find_boundaries(predicted_mask, mode="outer")] = (0.0, 0.9, 1.0)
    return image[crop]


def add_image(axis, panel, title, cmap="gray", vmin=None, vmax=None):
    axis.imshow(panel, cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_title(title, fontsize=9)
    axis.axis("off")


def render_event_thumbnail(path, event, context, references, roi, methods):
    focus = event["_focus_mask"]
    crop = crop_slices(focus, roi.shape)
    raw = context[1]
    values = raw[roi]
    lo, hi = np.percentile(values, [1, 99.5]) if values.size else (0, 1)
    reference_ids = event["_reference_ids"]
    reference_mask = union_masks(references, reference_ids, roi.shape)
    fig, axes = plt.subplots(4, 5, figsize=(18, 14), constrained_layout=True)
    for index, label_name in enumerate(("z-1", "z", "z+1")):
        add_image(axes[0, index], context[index][crop], label_name, "gray", lo, hi)
    add_image(
        axes[0, 3],
        overlay_boundaries(raw, reference_mask, np.zeros_like(reference_mask), roi, crop),
        "Manual reference boundary (green)",
    )
    axes[0, 4].axis("off")
    axes[0, 4].text(
        0, 1,
        f"Event: {event['event_id']}\nType: {event['event_type']}\n"
        f"Image: {event['image_id']}\nGroup: {event['group']}\n"
        f"Reference IDs: {event['reference_ids']}\nDifficulty: {event['difficulty_tags']}",
        va="top", fontsize=10,
    )

    for row, method_name in enumerate(("Method A", "Method B"), start=1):
        data = methods[method_name]
        ids = event[f"_{method_name.lower().replace(' ', '_')}_ids"]
        prediction_mask = selected_prediction_mask(data["labels"], ids)
        components, _ = connected_components(data["foreground"])
        add_image(axes[row, 0], data["probability"][crop], f"{method_name}: foreground probability", "magma", 0, 1)
        add_image(axes[row, 1], data["core_probability"][crop], f"{method_name}: core probability", "viridis", 0, 1)
        add_image(axes[row, 2], data["markers"][crop], f"{method_name}: core markers", "nipy_spectral")
        add_image(axes[row, 3], components[crop], f"{method_name}: pre-watershed components", "nipy_spectral")
        add_image(axes[row, 4], data["labels"][crop], f"{method_name}: watershed instances", "nipy_spectral")

        measurements = mask_measurements(prediction_mask)
        add_image(
            axes[3, row - 1],
            overlay_boundaries(raw, reference_mask, prediction_mask, roi, crop),
            f"{method_name}: green reference; cyan prediction",
        )
        axes[3, row + 1].axis("off")
        axes[3, row + 1].text(
            0, 1,
            f"{method_name} prediction IDs: {';'.join(map(str, ids)) or 'none'}\n"
            f"Area: {measurements['area_px']} px\n"
            f"Centerline: {measurements['centerline_px']} px\n"
            f"Area/centerline width proxy: {measurements['width_proxy_px']:.2f} px",
            va="top", fontsize=9,
        )
    axes[3, 4].axis("off")
    fig.suptitle("Blinded checkpoint event review", fontsize=15)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def contact_sheet(path, events, title, maximum=None):
    selected = events[:maximum] if maximum else events
    if not selected:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        for start in range(0, len(selected), 4):
            page = selected[start : start + 4]
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            for axis, event in zip(axes.ravel(), page):
                axis.imshow(plt.imread(event["_thumbnail_path"]))
                axis.set_title(event["event_id"])
                axis.axis("off")
            for axis in axes.ravel()[len(page) :]:
                axis.axis("off")
            fig.suptitle(title)
            fig.tight_layout()
            pdf.savefig(fig, dpi=120)
            plt.close(fig)


def prediction_ids_for_references(method, reference_ids):
    values = set()
    for reference_id in reference_ids:
        values.update(method["reference_to_predictions"].get(reference_id, []))
    return sorted(values)


def build_events(stem, group, context, references, supervision, roi, methods, touching_ids):
    events = []
    shape = roi.shape
    reference_measurements = {
        reference_id: mask_measurements(mask)
        for reference_id, mask in references.items()
    }
    areas = np.asarray([value["area_px"] for value in reference_measurements.values()])
    lengths = np.asarray([value["centerline_px"] for value in reference_measurements.values()])
    widths = np.asarray([value["width_proxy_px"] for value in reference_measurements.values()])
    curves = np.asarray([value["curvature_proxy"] for value in reference_measurements.values()])
    solidities = np.asarray([value["solidity"] for value in reference_measurements.values()])
    thresholds = {
        "short": np.nanquantile(lengths, 0.25),
        "wide": np.nanquantile(widths, 0.75),
        "curved": np.nanquantile(curves, 0.75),
        "irregular": np.nanquantile(solidities, 0.25),
        "fused-looking": np.nanquantile(areas, 0.90),
    }
    raw = context[1]
    intensities = {
        reference_id: float(raw[mask].mean())
        for reference_id, mask in references.items() if np.any(mask)
    }
    faint_cut = np.quantile(list(intensities.values()), 1 / 3)

    def append_event(event_type, reference_ids, method_a_ids, method_b_ids, focus, tags=""):
        events.append({
            "image_id": stem,
            "group": group,
            "event_type": event_type,
            "reference_ids": ";".join(map(str, reference_ids)),
            "method_a_prediction_ids": ";".join(map(str, method_a_ids)),
            "method_b_prediction_ids": ";".join(map(str, method_b_ids)),
            "difficulty_tags": tags,
            "brightness_stratum": "",
            "thumbnail_path": "",
            **{field: "" for field in REVIEW_FIELDS},
            "_reference_ids": list(reference_ids),
            "_method_a_ids": list(method_a_ids),
            "_method_b_ids": list(method_b_ids),
            "_focus_mask": focus,
        })

    difficult_candidates = []
    for reference_id, reference_mask in references.items():
        a_ids = methods["Method A"]["reference_to_predictions"].get(reference_id, [])
        b_ids = methods["Method B"]["reference_to_predictions"].get(reference_id, [])
        if len(a_ids) >= 2 or len(b_ids) >= 2:
            append_event("split", [reference_id], a_ids, b_ids, reference_mask)

        if group == "KJ":
            metrics = reference_measurements[reference_id]
            tags = []
            if intensities.get(reference_id, np.inf) <= faint_cut: tags.append("faint")
            if metrics["centerline_px"] <= thresholds["short"]: tags.append("short")
            if metrics["width_proxy_px"] >= thresholds["wide"]: tags.append("wide")
            if metrics["curvature_proxy"] >= thresholds["curved"]: tags.append("curved")
            if metrics["solidity"] <= thresholds["irregular"]: tags.append("irregular")
            if metrics["area_px"] >= thresholds["fused-looking"]: tags.append("fused-looking")
            if reference_id in touching_ids: tags.append("touching")
            if tags:
                difficult_candidates.append((reference_id, sorted(set(tags))))

    # Difficult morphology is a stratified review set, not an acceptance gate.
    # Keep up to 12 deterministic examples per tag and include each reference once.
    selected_difficult = set()
    for tag in (
        "faint", "curved", "short", "wide", "touching",
        "fused-looking", "irregular",
    ):
        candidates = [
            reference_id
            for reference_id, tags in difficult_candidates
            if tag in tags
        ]
        if len(candidates) > 12:
            positions = np.linspace(0, len(candidates) - 1, 12).round().astype(int)
            candidates = [candidates[position] for position in positions]
        selected_difficult.update(candidates)
    for reference_id, tags in difficult_candidates:
        if reference_id not in selected_difficult:
            continue
        reference_mask = references[reference_id]
        append_event(
            "difficult_kj",
            [reference_id],
            methods["Method A"]["reference_to_predictions"].get(reference_id, []),
            methods["Method B"]["reference_to_predictions"].get(reference_id, []),
            reference_mask,
            ";".join(tags),
        )

    merge_groups = set()
    for method in methods.values():
        for reference_ids in method["prediction_to_references"].values():
            if len(reference_ids) >= 2:
                merge_groups.add(tuple(sorted(reference_ids)))
    for reference_ids in sorted(merge_groups):
        a_ids = [pid for pid, refs in methods["Method A"]["prediction_to_references"].items()
                 if set(reference_ids).issubset(refs)]
        b_ids = [pid for pid, refs in methods["Method B"]["prediction_to_references"].items()
                 if set(reference_ids).issubset(refs)]
        append_event("merge", reference_ids, a_ids, b_ids,
                     union_masks(references, reference_ids, shape))

    for method_name, other_name in (("Method A", "Method B"), ("Method B", "Method A")):
        method = methods[method_name]
        other = methods[other_name]
        for predicted_id in range(1, int(method["labels"].max()) + 1):
            prediction_mask = method["labels"] == predicted_id
            area = int(np.count_nonzero(prediction_mask))
            if area >= 3:
                continue
            other_ids = sorted(int(value) for value in np.unique(other["labels"][prediction_mask]) if value)
            a_ids = [predicted_id] if method_name == "Method A" else other_ids
            b_ids = [predicted_id] if method_name == "Method B" else other_ids
            append_event("tiny_child", [], a_ids, b_ids, prediction_mask)

    unmatched = {}
    for method_name, method in methods.items():
        unmatched[method_name] = [
            predicted_id
            for predicted_id, reference_ids in method["prediction_to_references"].items()
            if not reference_ids
        ]
    used_b = set()
    pairs = []
    for a_id in unmatched["Method A"]:
        a_mask = methods["Method A"]["labels"] == a_id
        best_id, best_iou = None, 0.0
        for b_id in unmatched["Method B"]:
            if b_id in used_b: continue
            b_mask = methods["Method B"]["labels"] == b_id
            intersection = np.count_nonzero(a_mask & b_mask)
            union = np.count_nonzero(a_mask | b_mask)
            iou = intersection / union if union else 0.0
            if iou > best_iou: best_id, best_iou = b_id, iou
        if best_id is not None and best_iou >= 0.10:
            used_b.add(best_id); pairs.append(([a_id], [best_id]))
        else:
            pairs.append(([a_id], []))
    pairs.extend(([], [b_id]) for b_id in unmatched["Method B"] if b_id not in used_b)
    ignored = roi & ~supervision
    ignored_intensities = []
    for a_ids, b_ids in pairs:
        focus = selected_prediction_mask(methods["Method A"]["labels"], a_ids)
        focus |= selected_prediction_mask(methods["Method B"]["labels"], b_ids)
        ignored_fraction = np.count_nonzero(focus & ignored) / max(np.count_nonzero(focus), 1)
        event_type = "ignored_region" if ignored_fraction >= 0.5 else "supervised_background"
        append_event(event_type, [], a_ids, b_ids, focus)
        events[-1]["_mean_intensity"] = float(raw[focus].mean()) if np.any(focus) else np.nan
        if event_type == "ignored_region": ignored_intensities.append(events[-1]["_mean_intensity"])
    if ignored_intensities:
        low, high = np.nanquantile(ignored_intensities, [1 / 3, 2 / 3])
        for event in events:
            if event["event_type"] != "ignored_region": continue
            value = event.get("_mean_intensity", np.nan)
            event["brightness_stratum"] = "faint" if value <= low else "bright" if value >= high else "intermediate"
    return events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--reference-dataset", required=True)
    parser.add_argument("--group-key", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--key-output",
        help="Sealed method key outside the reviewer package",
    )
    parser.add_argument("--first-label", default="epoch_003")
    parser.add_argument("--second-label", default="epoch_012")
    parser.add_argument("--foreground-threshold", type=float, default=0.60)
    parser.add_argument("--core-threshold", type=float, default=0.50)
    parser.add_argument("--minimum-component-px", type=int, default=3)
    parser.add_argument("--seed", type=int, default=5710312)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    reference_dataset = Path(args.reference_dataset)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    thumbnails = output / "event_thumbnails"
    mapping = blinded_assignment(args.first_label, args.second_label, args.seed)
    group_map = load_group_map(args.group_key)
    audit_path = reference_dataset / "target_generation_audit.json"
    audit_by_file = {
        row["file_name"]: row
        for row in json.loads(audit_path.read_text(encoding="utf-8"))
    } if audit_path.exists() else {}
    all_events = []

    for sample_path in sorted(reference_dataset.glob("*.npz")):
        with np.load(sample_path) as sample:
            context = sample["image"].astype(np.float32)
            reference_labels = sample["instance_labels"].astype(np.int32)
            roi = sample["roi_mask"].astype(bool)
            supervision = sample["partial_label_supervision_mask"].astype(bool)
        stem = sample_path.stem
        references = reference_masks(reference_dataset, stem, reference_labels)
        sample_audit = audit_by_file.get(f"{stem}.png", {})
        touching_ids = {
            int(value)
            for pair in sample_audit.get("touching_pairs", [])
            for value in pair
        }
        touching_ids.update(
            int(value)
            for pair in sample_audit.get("overlap_pairs", [])
            for value in pair
        )
        methods = {}
        for method_name, source_label in mapping.items():
            prefix = f"model_c_dual_head__{source_label}_{stem}"
            probability = np.load(cache_dir / f"{prefix}_probability.npy")
            core_probability = np.load(cache_dir / f"{prefix}_core_probability.npy")
            labels, foreground, core, markers = marker_controlled_instances(
                probability, core_probability, args.foreground_threshold,
                args.core_threshold, roi, args.minimum_component_px,
                return_diagnostics=True,
            )
            reference_to_predictions, prediction_to_references = meaningful_relations(
                references, labels
            )
            methods[method_name] = {
                "probability": probability,
                "core_probability": core_probability,
                "labels": labels,
                "foreground": foreground,
                "core": core,
                "markers": markers,
                "reference_to_predictions": reference_to_predictions,
                "prediction_to_references": prediction_to_references,
                "partial_label_audit": partial_label_audit(
                    labels, reference_labels > 0, supervision, roi
                ),
            }
        events = build_events(
            stem, group_map.get(stem, "unknown"), context, references,
            supervision, roi, methods, touching_ids
        )
        for event in events:
            event["event_id"] = f"EVT-{len(all_events) + 1:05d}"
            path = thumbnails / event["event_type"] / f"{event['event_id']}.png"
            event["thumbnail_path"] = str(path.relative_to(output))
            event["_thumbnail_path"] = path
            render_event_thumbnail(path, event, context, references, roi, methods)
            all_events.append(event)
        print(f"{stem}: {len(events)} events")

    write_csv(output / "master_review.csv", all_events)
    counts = {}
    for event_type in sorted({event["event_type"] for event in all_events}):
        selected = [event for event in all_events if event["event_type"] == event_type]
        counts[event_type] = len(selected)
        contact_sheet(
            output / "contact_sheets" / f"{event_type}.pdf",
            selected,
            f"Blinded {event_type.replace('_', ' ')} events",
        )
    ignored = [event for event in all_events if event["event_type"] == "ignored_region"]
    stratified = []
    for group in sorted({event["group"] for event in ignored}):
        for brightness in ("faint", "intermediate", "bright"):
            candidates = [event for event in ignored if event["group"] == group and event["brightness_stratum"] == brightness]
            stratified.extend(candidates[:12])
    contact_sheet(
        output / "contact_sheets" / "ignored_region_stratified.pdf",
        stratified,
        "Stratified ignored-region examples",
    )
    key = {
        "review_status": "sealed_until_manual_review_complete",
        "seed": args.seed,
        "method_mapping": mapping,
        "foreground_threshold": args.foreground_threshold,
        "core_threshold": args.core_threshold,
    }
    key_output = (
        Path(args.key_output)
        if args.key_output
        else output.parent / f"{output.name}_SEALED_METHOD_KEY.json"
    )
    key_output.parent.mkdir(parents=True, exist_ok=True)
    key_output.write_text(
        json.dumps(key, indent=2), encoding="utf-8"
    )
    manifest = {
        "event_count": len(all_events),
        "event_counts": counts,
        "validation_images": sorted({event["image_id"] for event in all_events}),
        "valid_review_classifications": VALID_CLASSIFICATIONS,
        "ranking_performed": False,
        "checkpoint_selected": False,
        "method_key_in_review_package": False,
    }
    (output / "review_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# Blinded Model C checkpoint review\n\n"
        "Review Method A and Method B without attempting to identify their "
        "checkpoint epochs. Complete the five blank review columns in "
        "`master_review.csv`; every row links to its event thumbnail. Use only "
        "the classifications listed in `review_manifest.json`.\n\n"
        "The grouped PDFs cover split, merge, tiny-child, supervised-background, "
        "ignored-region, and difficult-KJ events. The full ignored-region PDF "
        "contains every event; `ignored_region_stratified.pdf` is the shorter "
        "faint/intermediate/bright reading set.\n\n"
        "No checkpoint has been ranked or selected. Difficulty tags and pixel "
        "measurements are review aids only: they are not morphology rejection "
        "rules and do not replace calibrated Saturn morphometry. The method key "
        "is intentionally stored outside this reviewer package and should be "
        "opened only after review is complete.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
