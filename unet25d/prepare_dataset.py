import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import tifffile
import yaml
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion, distance_transform_edt, find_objects, label


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
        path = Path(stack_dir) / image_pattern.format(z=zz)
        if not path.exists():
            raise FileNotFoundError(path)
        planes.append(read_tiff(path))
    return robust_normalize_stack(np.stack(planes, axis=0))


def load_sample_roi(cfg, z, expected_shape):
    """Load an optional per-sample ROI mask keyed by the synthetic z index."""
    roi_dir = cfg.get("roi_mask_dir")
    roi_pattern = cfg.get("roi_mask_pattern")
    if not roi_dir or not roi_pattern:
        return np.ones(expected_shape, dtype=bool)
    path = Path(roi_dir) / str(roi_pattern).format(z=z)
    if not path.exists():
        raise FileNotFoundError(path)
    roi = np.asarray(np.load(path), dtype=bool)
    if roi.shape != tuple(expected_shape):
        raise ValueError(
            f"ROI shape {roi.shape} does not match image shape {tuple(expected_shape)}: {path}"
        )
    return roi


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


def array_sha256(array):
    """Return a deterministic digest including shape, dtype, and array bytes."""
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _polygon_mask(width, height, segmentations):
    image = Image.new("L", (int(width), int(height)), 0)
    draw = ImageDraw.Draw(image)
    for polygon_values in segmentations:
        if len(polygon_values) < 6:
            continue
        xy = [
            (float(polygon_values[index]), float(polygon_values[index + 1]))
            for index in range(0, len(polygon_values), 2)
        ]
        draw.polygon(xy, outline=1, fill=1)
    return np.asarray(image, dtype=np.uint8) > 0


def _touching_label_pairs(instance_labels):
    """Return deterministic label pairs touching in an 8-neighborhood."""
    labels = np.asarray(instance_labels)
    pairs = set()
    for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
        first = labels[
            max(0, dy) : labels.shape[0] + min(0, dy),
            max(0, dx) : labels.shape[1] + min(0, dx),
        ]
        second = labels[
            max(0, -dy) : labels.shape[0] - max(0, dy),
            max(0, -dx) : labels.shape[1] - max(0, dx),
        ]
        different = (first > 0) & (second > 0) & (first != second)
        for left, right in zip(first[different], second[different]):
            pairs.add(tuple(sorted((int(left), int(right)))))
    return sorted(pairs)


def _connected_pair_groups(pairs):
    graph = {}
    for left, right in pairs:
        graph.setdefault(left, set()).add(right)
        graph.setdefault(right, set()).add(left)
    groups = []
    remaining = set(graph)
    while remaining:
        start = min(remaining)
        stack = [start]
        group = set()
        while stack:
            current = stack.pop()
            if current in group:
                continue
            group.add(current)
            stack.extend(graph.get(current, ()))
        remaining -= group
        groups.append(sorted(group))
    return groups


def rasterize_instances(width, height, annotations):
    """Rasterize annotations while retaining deterministic local instance IDs."""
    instance_labels = np.zeros((int(height), int(width)), dtype=np.uint32)
    source_ids = {}
    zero_pixel_source_ids = []
    overlap_pairs = set()
    overlap_pixel_count = 0
    ordered = sorted(annotations, key=lambda item: int(item.get("id", 0)))
    for local_id, annotation in enumerate(ordered, start=1):
        source_id = int(annotation.get("id", local_id))
        source_ids[local_id] = source_id
        segmentation = annotation.get("segmentation", [])
        if not isinstance(segmentation, list):
            zero_pixel_source_ids.append(source_id)
            continue
        if segmentation and isinstance(segmentation[0], (int, float)):
            segmentation = [segmentation]
        instance = _polygon_mask(width, height, segmentation)
        if not np.any(instance):
            zero_pixel_source_ids.append(source_id)
            continue
        occupied = instance & (instance_labels > 0)
        overlap_pixel_count += int(np.count_nonzero(occupied))
        for existing in np.unique(instance_labels[occupied]):
            overlap_pairs.add(tuple(sorted((int(existing), int(local_id)))))
        instance_labels[instance & (instance_labels == 0)] = local_id
    touching_pairs = _touching_label_pairs(instance_labels)
    return instance_labels, {
        "source_instance_ids": source_ids,
        "zero_pixel_source_ids": zero_pixel_source_ids,
        "overlap_pairs": sorted(overlap_pairs),
        "overlap_pixel_count": overlap_pixel_count,
        "touching_pairs": touching_pairs,
        "touching_groups": _connected_pair_groups(touching_pairs),
    }


def make_boundary_ignore_mask(instance_labels, radius=1):
    """Build the union of per-instance inside/outside uncertainty rings."""
    radius = int(radius)
    labels_array = np.asarray(instance_labels)
    boundary = np.zeros(labels_array.shape, dtype=bool)
    if radius <= 0:
        return boundary
    for instance_id, object_slice in enumerate(find_objects(labels_array), start=1):
        if object_slice is None:
            continue
        y_slice, x_slice = object_slice
        y0 = max(0, y_slice.start - radius)
        y1 = min(labels_array.shape[0], y_slice.stop + radius)
        x0 = max(0, x_slice.start - radius)
        x1 = min(labels_array.shape[1], x_slice.stop + radius)
        instance = labels_array[y0:y1, x0:x1] == instance_id
        dilated = dilate_binary(instance, radius)
        eroded = binary_erosion(instance, iterations=radius, border_value=0)
        boundary[y0:y1, x0:x1] |= dilated & ~eroded
    return boundary


def make_instance_core_labels(instance_labels, cfg):
    """Generate one connected, conservative core for every surviving instance."""
    labels_array = np.asarray(instance_labels)
    core_labels = np.zeros(labels_array.shape, dtype=np.uint32)
    core_fraction = float(cfg.get("instance_core_distance_fraction", 0.55))
    minimum_distance = float(cfg.get("instance_core_min_distance_px", 1.0))
    missing = []
    for instance_id, object_slice in enumerate(find_objects(labels_array), start=1):
        if object_slice is None:
            continue
        instance = labels_array[object_slice] == instance_id
        distances = distance_transform_edt(instance)
        maximum = float(distances.max())
        if maximum <= 0:
            missing.append(instance_id)
            continue
        threshold = min(maximum, max(minimum_distance, maximum * core_fraction))
        candidate = distances >= threshold
        candidate_labels, count = label(candidate)
        if count:
            sizes = np.bincount(candidate_labels.ravel())
            sizes[0] = 0
            candidate = candidate_labels == int(np.argmax(sizes))
        if not np.any(candidate):
            candidate = distances == maximum
        core_target = core_labels[object_slice]
        core_target[candidate] = instance_id
    return core_labels, missing


def make_loss_weight_mask(supervision, boundary_ignore, cfg):
    weights = np.asarray(supervision, dtype=np.float32).copy()
    boundary_weight = float(cfg.get("boundary_loss_weight", 0.1))
    if not 0.0 <= boundary_weight <= 1.0:
        raise ValueError("boundary_loss_weight must be within [0, 1]")
    weights[boundary_ignore] = np.minimum(weights[boundary_ignore], boundary_weight)
    return weights


def make_training_mask(label_mask, cfg):
    """
    Build the target used for training without modifying the saved raw annotation.

    A small dilation makes training tolerant of slightly tight hand masks. This is
    deliberately prepare-time only; inference still writes probability maps, and
    Saturn measurement should use its own geometry/QC rules.
    """
    radius = int(cfg.get("train_mask_dilate_px", 0))
    if radius <= 0:
        return label_mask.astype(np.uint8)
    return dilate_binary(label_mask > 0, radius).astype(np.uint8)


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
        annotations = anns_by_image.get(image_id, [])
        instance_labels, instance_audit = rasterize_instances(
            width,
            height,
            annotations,
        )
        samples.append(
            {
                "file_name": im["file_name"],
                "z": parse_z(im["file_name"], z_regex),
                "instance_labels": instance_labels,
                "annotation_count": len(annotations),
                "instance_audit": instance_audit,
            }
        )
    return samples


def _annotation_root(annotation_path):
    return Path(annotation_path).resolve().parent


def _resolve_manifest_image_path(annotation_path, image_name):
    root = _annotation_root(annotation_path)
    pure = Path(str(image_name).replace("\\", "/"))
    candidates = [
        root / pure,
        root / "images" / pure.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find manifest image {image_name!r} relative to {root}")


def rasterize_sreeni_manifest(manifest_path, z_regex):
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    samples = []
    for im in manifest.get("images", []):
        file_name = Path(str(im["image"]).replace("\\", "/")).name
        image_path = _resolve_manifest_image_path(manifest_path, im["image"])
        width, height = Image.open(image_path).size
        instances = im.get("instances", [])
        annotations = [
            {
                "id": index,
                "segmentation": [instance.get("segmentation", [])],
            }
            for index, instance in enumerate(instances, start=1)
        ]
        instance_labels, instance_audit = rasterize_instances(
            width,
            height,
            annotations,
        )
        samples.append(
            {
                "file_name": file_name,
                "z": parse_z(file_name, z_regex),
                "instance_labels": instance_labels,
                "annotation_count": len(instances),
                "instance_audit": instance_audit,
            }
        )
    return samples


def load_annotation_samples(annotation_path, z_regex):
    with open(annotation_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if "annotations" in payload and "categories" in payload:
        return rasterize_coco(annotation_path, z_regex)
    if "classes" in payload and "images" in payload:
        return rasterize_sreeni_manifest(annotation_path, z_regex)
    raise ValueError(f"Unsupported annotation format: {annotation_path}")


def filter_samples(samples, include_z=None, exclude_z=None):
    include = {int(z) for z in include_z} if include_z else None
    exclude = {int(z) for z in exclude_z} if exclude_z else set()
    out = []
    for sample in samples:
        z = int(sample["z"])
        if include is not None and z not in include:
            continue
        if z in exclude:
            continue
        out.append(sample)
    return out


def write_samples(split_name, samples, cfg):
    """Write instance-aware targets while preserving the legacy training keys."""
    out_dir = Path(cfg["output_dir"]) / "dataset" / split_name
    out_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = out_dir / "masks_png"
    masks_dir.mkdir(parents=True, exist_ok=True)
    target_mode = str(cfg.get("target_preparation_mode", "legacy_dilated"))
    if target_mode not in {"legacy_dilated", "annotation_tolerant"}:
        raise ValueError(
            "target_preparation_mode must be legacy_dilated or annotation_tolerant"
        )

    manifest_rows = [
        "split,file_name,z,npz_path,mask_png,raw_mask_png,supervision_png,"
        "annotation_count,generated_instance_count,raw_mask_pixels,"
        "training_mask_pixels,supervised_weight_sum,boundary_pixels,core_instances"
    ]
    audit_rows = []

    for sample in samples:
        z = int(sample["z"])
        context = load_context(cfg["stack_image_dir"], cfg["image_pattern"], z)
        instance_labels = sample["instance_labels"].astype(np.uint32)
        roi = load_sample_roi(cfg, z, instance_labels.shape)
        instance_labels[~roi] = 0
        raw_target = (instance_labels > 0).astype(np.uint8)
        training_target = (
            raw_target.copy()
            if target_mode == "annotation_tolerant"
            else make_training_mask(raw_target, cfg)
        )
        training_target[~roi] = 0

        partial_supervision = make_supervision_mask(context, raw_target, cfg)
        partial_supervision[~roi] = 0
        boundary_ignore = make_boundary_ignore_mask(
            instance_labels,
            cfg.get("boundary_ignore_radius_px", 1),
        )
        boundary_ignore[~roi] = False
        confident_interior = raw_target.astype(bool) & ~boundary_ignore
        core_labels, missing_core_local_ids = make_instance_core_labels(
            instance_labels,
            cfg,
        )
        core_labels[~roi] = 0
        loss_weights = (
            make_loss_weight_mask(partial_supervision, boundary_ignore, cfg)
            if target_mode == "annotation_tolerant"
            else partial_supervision.astype(np.float32)
        )
        loss_weights[~roi] = 0.0

        local_ids = sorted(int(value) for value in np.unique(instance_labels) if value)
        core_ids = sorted(int(value) for value in np.unique(core_labels) if value)
        source_audit = sample.get("instance_audit", {})
        source_map = source_audit.get("source_instance_ids", {})
        missing_after_roi = [
            int(source_map.get(local_id, local_id))
            for local_id in sorted(set(source_map) - set(local_ids))
        ]
        missing_core_source_ids = [
            int(source_map.get(local_id, local_id))
            for local_id in sorted(set(local_ids) - set(core_ids))
        ]
        missing_core_source_ids.extend(
            int(source_map.get(local_id, local_id))
            for local_id in missing_core_local_ids
            if int(source_map.get(local_id, local_id)) not in missing_core_source_ids
        )
        touching_pairs = _touching_label_pairs(instance_labels)
        audit_pass = (
            not source_audit.get("zero_pixel_source_ids", [])
            and not missing_after_roi
            and not missing_core_source_ids
            and len(local_ids) == int(sample["annotation_count"])
        )

        stem = Path(sample["file_name"]).stem
        npz_path = out_dir / f"{stem}.npz"
        mask_path = masks_dir / f"{stem}_mask.png"
        raw_mask_path = masks_dir / f"{stem}_raw_annotation_mask.png"
        supervision_path = masks_dir / f"{stem}_supervision.png"
        boundary_path = masks_dir / f"{stem}_boundary_ignore.png"
        interior_path = masks_dir / f"{stem}_confident_interior.png"
        instance_path = masks_dir / f"{stem}_instance_labels.tif"
        core_path = masks_dir / f"{stem}_instance_core_labels.tif"

        np.savez_compressed(
            npz_path,
            image=context.astype(np.float32),
            mask=training_target,
            foreground_target=training_target,
            raw_annotation_mask=raw_target,
            instance_labels=instance_labels,
            confident_interior=confident_interior.astype(np.uint8),
            boundary_ignore_mask=boundary_ignore.astype(np.uint8),
            loss_weight_mask=loss_weights.astype(np.float32),
            supervision_mask=loss_weights.astype(np.float32),
            partial_label_supervision_mask=partial_supervision.astype(np.uint8),
            instance_core_labels=core_labels,
            roi_mask=roi.astype(np.uint8),
            z=np.array([z], dtype=np.int16),
            file_name=np.array([sample["file_name"]]),
        )
        Image.fromarray(training_target * 255).save(mask_path)
        Image.fromarray(raw_target * 255).save(raw_mask_path)
        Image.fromarray(
            np.clip(loss_weights * 255, 0, 255).astype(np.uint8)
        ).save(supervision_path)
        Image.fromarray(boundary_ignore.astype(np.uint8) * 255).save(boundary_path)
        Image.fromarray(confident_interior.astype(np.uint8) * 255).save(interior_path)
        tifffile.imwrite(instance_path, instance_labels.astype(np.uint16))
        tifffile.imwrite(core_path, core_labels.astype(np.uint16))

        manifest_rows.append(
            f"{split_name},{sample['file_name']},{z},{npz_path},{mask_path},"
            f"{raw_mask_path},{supervision_path},{sample['annotation_count']},"
            f"{len(local_ids)},{int(raw_target.sum())},{int(training_target.sum())},"
            f"{float(loss_weights.sum()):.3f},{int(boundary_ignore.sum())},{len(core_ids)}"
        )
        audit_rows.append(
            {
                "split": split_name,
                "file_name": sample["file_name"],
                "z": z,
                "target_preparation_mode": target_mode,
                "source_annotation_count": int(sample["annotation_count"]),
                "generated_instance_count": len(local_ids),
                "core_instance_count": len(core_ids),
                "zero_pixel_source_ids": source_audit.get("zero_pixel_source_ids", []),
                "missing_after_roi_source_ids": missing_after_roi,
                "missing_core_source_ids": sorted(set(missing_core_source_ids)),
                "touching_pairs": touching_pairs,
                "touching_groups": _connected_pair_groups(touching_pairs),
                "overlap_pairs": source_audit.get("overlap_pairs", []),
                "overlap_pixel_count": int(source_audit.get("overlap_pixel_count", 0)),
                "foreground_pixels": int(raw_target.sum()),
                "training_target_pixels": int(training_target.sum()),
                "confident_interior_pixels": int(confident_interior.sum()),
                "ignored_boundary_pixels": int(boundary_ignore.sum()),
                "loss_weight_sum": float(loss_weights.sum()),
                "instance_labels_sha256": array_sha256(instance_labels),
                "foreground_target_sha256": array_sha256(training_target),
                "boundary_ignore_sha256": array_sha256(boundary_ignore.astype(np.uint8)),
                "loss_weight_sha256": array_sha256(loss_weights.astype(np.float32)),
                "instance_core_labels_sha256": array_sha256(core_labels),
                "audit_pass": bool(audit_pass),
            }
        )

    manifest_path = out_dir / "manifest.csv"
    manifest_path.write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
    audit_json_path = out_dir / "target_generation_audit.json"
    audit_json_path.write_text(json.dumps(audit_rows, indent=2), encoding="utf-8")
    csv_rows = [
        {
            key: json.dumps(value, separators=(",", ":"))
            if isinstance(value, (list, dict))
            else value
            for key, value in row.items()
        }
        for row in audit_rows
    ]
    audit_csv_path = out_dir / "target_generation_audit.csv"
    if csv_rows:
        with open(audit_csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
    failures = [row for row in audit_rows if not row["audit_pass"]]
    summary = {
        "split": split_name,
        "target_preparation_mode": target_mode,
        "sample_count": len(audit_rows),
        "source_annotation_count": int(sum(row["source_annotation_count"] for row in audit_rows)),
        "generated_instance_count": int(sum(row["generated_instance_count"] for row in audit_rows)),
        "core_instance_count": int(sum(row["core_instance_count"] for row in audit_rows)),
        "samples_with_touching_instances": int(sum(bool(row["touching_pairs"]) for row in audit_rows)),
        "touching_pair_count": int(sum(len(row["touching_pairs"]) for row in audit_rows)),
        "samples_with_overlaps": int(sum(bool(row["overlap_pairs"]) for row in audit_rows)),
        "overlap_pair_count": int(sum(len(row["overlap_pairs"]) for row in audit_rows)),
        "overlap_pixel_count": int(sum(row["overlap_pixel_count"] for row in audit_rows)),
        "foreground_pixel_count": int(sum(row["foreground_pixels"] for row in audit_rows)),
        "ignored_boundary_pixel_count": int(sum(row["ignored_boundary_pixels"] for row in audit_rows)),
        "audit_failure_count": len(failures),
        "audit_pass": not failures,
        "audit_json": str(audit_json_path),
        "audit_csv": str(audit_csv_path),
    }
    (out_dir / "target_generation_audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if failures and bool(cfg.get("strict_target_audit", True)):
        failed_names = ", ".join(row["file_name"] for row in failures[:10])
        raise ValueError(
            f"Target-generation audit failed for {len(failures)} {split_name} samples: "
            f"{failed_names}. Inspect {audit_json_path}."
        )
    return len(samples), manifest_path


def write_split(split_name, annotation_path, cfg, include_z=None, exclude_z=None):
    samples = load_annotation_samples(annotation_path, cfg["z_regex"])
    samples = filter_samples(samples, include_z=include_z, exclude_z=exclude_z)
    return write_samples(split_name, samples, cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot_unet25d.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    annotation_manifest = cfg.get("annotation_manifest")
    valid_z_indices = cfg.get("valid_z_indices", [])
    train_z_indices = cfg.get("train_z_indices", [])

    if annotation_manifest:
        samples = load_annotation_samples(annotation_manifest, cfg["z_regex"])
        train_samples = filter_samples(
            samples,
            include_z=train_z_indices,
            exclude_z=valid_z_indices if not train_z_indices else None,
        )
        valid_samples = filter_samples(samples, include_z=valid_z_indices)
        if not train_samples:
            raise ValueError("No training samples selected from annotation_manifest")
        if not valid_samples:
            raise ValueError("No validation samples selected from annotation_manifest")
        train_count, train_manifest = write_samples("train", train_samples, cfg)
        valid_count, valid_manifest = write_samples("valid", valid_samples, cfg)
    else:
        train_count, train_manifest = write_split("train", cfg["train_coco"], cfg)
        valid_count, valid_manifest = write_split("valid", cfg["valid_coco"], cfg)

    split_summaries = []
    for split_name in ("train", "valid"):
        summary_path = output_dir / "dataset" / split_name / "target_generation_audit_summary.json"
        if summary_path.exists():
            split_summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
    combined_audit = {
        "target_preparation_mode": cfg.get("target_preparation_mode", "legacy_dilated"),
        "split_count": len(split_summaries),
        "sample_count": int(sum(item["sample_count"] for item in split_summaries)),
        "source_annotation_count": int(sum(item["source_annotation_count"] for item in split_summaries)),
        "generated_instance_count": int(sum(item["generated_instance_count"] for item in split_summaries)),
        "core_instance_count": int(sum(item["core_instance_count"] for item in split_summaries)),
        "touching_pair_count": int(sum(item["touching_pair_count"] for item in split_summaries)),
        "overlap_pair_count": int(sum(item["overlap_pair_count"] for item in split_summaries)),
        "overlap_pixel_count": int(sum(item["overlap_pixel_count"] for item in split_summaries)),
        "ignored_boundary_pixel_count": int(sum(item["ignored_boundary_pixel_count"] for item in split_summaries)),
        "audit_failure_count": int(sum(item["audit_failure_count"] for item in split_summaries)),
        "audit_pass": bool(split_summaries) and all(item["audit_pass"] for item in split_summaries),
        "splits": split_summaries,
    }
    combined_path = output_dir / "dataset" / "target_generation_audit_summary.json"
    combined_path.write_text(json.dumps(combined_audit, indent=2), encoding="utf-8")
    print(f"Prepared train samples: {train_count} -> {train_manifest}")
    print(f"Prepared valid samples: {valid_count} -> {valid_manifest}")
    print(
        "Target audit: "
        f"annotations={combined_audit['source_annotation_count']} "
        f"instances={combined_audit['generated_instance_count']} "
        f"cores={combined_audit['core_instance_count']} "
        f"failures={combined_audit['audit_failure_count']} "
        f"pass={combined_audit['audit_pass']} -> {combined_path}"
    )


if __name__ == "__main__":
    main()
