"""Three-slice synthetic/real smoke harness for Saturn v5.7 U-Net-primary."""

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_saturn():
    spec = importlib.util.spec_from_file_location(
        "saturn_v57_unet_primary_smoke",
        ROOT / "sperm_segmentation_saturnv5.7.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_csv_values(text, cast=str):
    return [cast(item.strip()) for item in str(text).split(",") if item.strip()]


def validate_target_values(values, allow_large_run=False):
    targets = [int(value) for value in values]
    if not targets:
        raise ValueError("At least one target Z value is required")
    if len(targets) != len(set(targets)):
        raise ValueError("Target Z values must be unique")
    if len(targets) > 6 and not allow_large_run:
        raise ValueError(
            "Refusing more than six targets without --allow-large-run"
        )
    return targets


def resolve_target_files(files_by_z, targets):
    """Return only explicitly requested targets; neighbors remain context-only."""
    missing = [z for z in targets if z not in files_by_z]
    if missing:
        raise ValueError(f"Requested Z values not found: {missing}")
    return {z: files_by_z[z] for z in targets}


def load_parameters(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    for key in ("parameters", "best_parameters", "config"):
        if isinstance(payload.get(key), dict):
            return payload[key]
    if not isinstance(payload, dict):
        raise ValueError("Base parameter JSON must contain a dictionary")
    return payload


def label_hash(labels):
    array = np.ascontiguousarray(labels, dtype=np.int32)
    return hashlib.sha256(array.tobytes()).hexdigest()


def unique_label_count(labels):
    return int(np.count_nonzero(np.unique(labels) > 0))


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalize_probability(probability):
    return np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)


def colored_instances(saturn, labels):
    labels = np.asarray(labels, dtype=np.int32)
    count = int(labels.max())
    base = np.zeros((*labels.shape, 3), dtype=np.float32)
    if count:
        colors = plt.cm.tab20(np.linspace(0, 1, max(count, 2)))[:, :3]
        for value in range(1, count + 1):
            base[labels == value] = colors[(value - 1) % len(colors)]
    return base


def source_overlay(saturn, image, seg):
    base = saturn.normalize_display(image)
    rgb = np.stack([base, base, base], axis=-1)
    labels = seg["unet_primary_instance_labels"]
    sources = seg.get("unet_primary_instance_sources", {})
    for value in (int(v) for v in np.unique(labels) if v):
        color = (
            np.array([0.0, 1.0, 1.0])
            if sources.get(value, "unet_primary") == "unet_primary"
            else np.array([0.1, 0.9, 0.2])
        )
        mask = labels == value
        rgb[mask] = 0.35 * rgb[mask] + 0.65 * color
    rejected = seg.get(
        "unet_primary_rejected_reason", np.zeros(labels.shape, dtype=np.uint8)
    ) > 0
    rgb[rejected] = 0.25 * rgb[rejected] + 0.75 * np.array([1.0, 0.0, 0.0])
    return np.clip(rgb, 0, 1)


def hysteresis_overlay(saturn, image, seg):
    base = saturn.normalize_display(image)
    rgb = np.stack([base, base, base], axis=-1)
    support = seg["unet_primary_hysteresis_mask"]
    seeds = seg["unet_seed_mask"]
    rgb[support] = 0.3 * rgb[support] + 0.7 * np.array([0.0, 1.0, 1.0])
    rgb[seeds] = 0.2 * rgb[seeds] + 0.8 * np.array([1.0, 0.9, 0.0])
    return np.clip(rgb, 0, 1)


def save_review_panel(saturn, path, image, probability, hybrid, primary):
    hybrid_seg, hybrid_measured = hybrid
    primary_seg, primary_measured = primary
    panels = [
        ("Raw image", saturn.normalize_display(image), "gray"),
        ("U-Net probability", probability, "magma"),
        (
            "Low/high hysteresis support",
            hysteresis_overlay(saturn, image, primary_seg),
            None,
        ),
        (
            "Filled instance labels",
            colored_instances(
                saturn, primary_seg["unet_primary_instance_labels"]
            ),
            None,
        ),
        (
            "Accepted filled masks",
            source_overlay(saturn, image, primary_seg),
            None,
        ),
        (
            "Measured centerlines",
            saturn.make_overlay(image, primary_measured["skel_label"]),
            None,
        ),
        (
            "Current hybrid result",
            saturn.make_overlay(image, hybrid_measured["skel_label"]),
            None,
        ),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for axis, (title, panel, cmap) in zip(axes.flat, panels):
        axis.imshow(panel, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")
    comparison = np.concatenate([
        saturn.make_overlay(image, hybrid_measured["skel_label"]),
        saturn.make_overlay(image, primary_measured["skel_label"]),
    ], axis=1)
    axes.flat[7].imshow(comparison)
    axes.flat[7].set_title("Hybrid (left) vs U-Net-primary (right)")
    axes.flat[7].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def summarize_primary(seg, measured, valid, roi, exclusion):
    debug = seg["unet_primary_debug"]
    audit = seg.get("unet_primary_component_audit", [])
    marker_by_parent = {}
    for row in audit:
        parent = int(row.get("parent_component_id", 0))
        marker_by_parent[parent] = max(
            marker_by_parent.get(parent, 0), int(row.get("marker_count", 0))
        )
    failures = seg.get("unet_primary_technical_failures", [])
    accepted_ids = [
        int(value)
        for value in np.unique(seg["unet_primary_instance_labels"])
        if value
    ]
    seed_mask = np.asarray(seg["unet_seed_mask"], dtype=bool)
    accepted_without_seed = sum(
        not np.any(
            seed_mask & (seg["unet_primary_instance_labels"] == instance_id)
        )
        for instance_id in accepted_ids
    )
    result_ids = [int(row["label"]) for row in measured["results"]]
    rejected_audit_count = sum(
        row.get("disposition") == "rejected" for row in audit
    )
    unresolved = sum(
        row.get("technical_reason") == "unresolved_multi_instance_merge"
        for row in audit
    )
    return {
        "candidate_pixels": int(debug["candidate_pixels"]),
        "seed_pixels": int(debug["seed_pixels"]),
        "hysteresis_component_count": int(
            debug["hysteresis_component_count"]
        ),
        "marker_count": int(sum(marker_by_parent.values())),
        "split_instance_count": int(debug["split_instance_count"]),
        "accepted_instance_count": len(measured["results"]),
        "morphology_warning_count": sum(
            bool(row.get("morphology_warning", False))
            for row in measured["results"]
        ),
        "hard_technical_failure_count": int(
            len(failures) + rejected_audit_count
        ),
        "unresolved_merge_count": int(unresolved),
        "saturn_only_additions": int(debug["saturn_only_additions"]),
        "final_measured_object_count": len(measured["results"]),
        "outside_roi_pixels": int(np.count_nonzero(
            seg["unet_primary_instance_labels"] & ~roi
        )),
        "exclusion_mask_pixels": int(np.count_nonzero(
            seg["unet_primary_instance_labels"] & exclusion
        )),
        "valid_pixel_count": int(np.count_nonzero(valid)),
        "accepted_without_seed_count": int(accepted_without_seed),
        "duplicate_instance_id_count": int(
            len(result_ids) - len(set(result_ids))
        ),
        "instance_measurement_mapping_mismatch": bool(
            set(result_ids) != set(accepted_ids)
        ),
        "unet_inference_enabled": bool(
            seg.get("unet_debug", {}).get("unet_enabled", False)
        ),
    }


def summarize_hybrid(seg, measured, valid, roi, exclusion):
    labels = measured["skel_label"]
    return {
        "candidate_pixels": int(np.count_nonzero(seg["unet_candidate_mask"])),
        "seed_pixels": int(np.count_nonzero(seg["unet_seed_mask"])),
        "hysteresis_component_count": int(
            unique_label_count(seg["skel_labeled"])
        ),
        "marker_count": 0,
        "split_instance_count": 0,
        "accepted_instance_count": len(measured["results"]),
        "morphology_warning_count": sum(
            bool(row.get("unet_rescue_morphology_warning", False))
            for row in measured["results"]
        ),
        "hard_technical_failure_count": int(sum(
            measured.get("unet_rescue_rejected_counts", {}).values()
        )),
        "unresolved_merge_count": 0,
        "saturn_only_additions": 0,
        "final_measured_object_count": len(measured["results"]),
        "outside_roi_pixels": int(np.count_nonzero(labels & ~roi)),
        "exclusion_mask_pixels": int(np.count_nonzero(labels & exclusion)),
        "valid_pixel_count": int(np.count_nonzero(valid)),
        "accepted_without_seed_count": 0,
        "duplicate_instance_id_count": 0,
        "instance_measurement_mapping_mismatch": False,
        "unet_inference_enabled": bool(
            seg.get("unet_debug", {}).get("unet_enabled", False)
        ),
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--unet-model", required=True)
    parser.add_argument("--base-params", required=True)
    parser.add_argument("--roi-mask", required=True)
    parser.add_argument("--exclusion-mask", default="")
    parser.add_argument("--z-values", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--engines", default="hybrid,unet_primary")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--allow-large-run", action="store_true")
    return parser


def run(args):
    saturn = load_saturn()
    targets = validate_target_values(
        parse_csv_values(args.z_values, int),
        allow_large_run=args.allow_large_run,
    )
    engines = parse_csv_values(args.engines, str)
    if not engines or any(e not in {"hybrid", "unet_primary"} for e in engines):
        raise ValueError("Smoke engines must be hybrid and/or unet_primary")
    if args.repeat < 1:
        raise ValueError("--repeat must be at least 1")

    outdir = Path(args.outdir)
    panel_dir = outdir / "review_panels"
    probability_dir = outdir / "probability_maps"
    panel_dir.mkdir(parents=True, exist_ok=True)
    probability_dir.mkdir(parents=True, exist_ok=True)

    cfg = saturn.CONFIG.copy()
    cfg.update(load_parameters(args.base_params))
    cfg.update({
        "UNET_MODEL_PATH": str(Path(args.unet_model).resolve()),
        "DO_TRACKING": False,
        "SAVE_DEBUG_IMAGES": False,
        "UNET_PRIMARY_CLASSICAL_ADDITIONS_ENABLE": False,
    })
    files, z_values = saturn.load_batch_files(
        args.input_dir, cfg["FILE_PATTERN"]
    )
    files_by_z = {}
    for file_path, z_value in zip(files, z_values):
        if int(z_value) in files_by_z:
            raise ValueError(f"Duplicate source Z index: {z_value}")
        files_by_z[int(z_value)] = file_path
    target_files = resolve_target_files(files_by_z, targets)

    first = saturn.robust_imread(target_files[targets[0]])
    roi = saturn.load_roi_mask_file(args.roi_mask, expected_shape=first.shape)
    exclusion = (
        saturn.load_roi_mask_file(
            args.exclusion_mask, expected_shape=first.shape
        )
        if args.exclusion_mask else np.zeros(first.shape, dtype=bool)
    )
    valid = roi & ~exclusion
    preprocess = saturn.build_stack_preprocess_context(
        files, roi, cfg, exclusion_mask=exclusion
    )

    probabilities = {}
    for z_value in targets:
        image = saturn.robust_imread(target_files[z_value])
        context = saturn._make_unet_context_from_paths(files_by_z, z_value)
        inference_cfg = cfg.copy()
        inference_cfg["SEGMENTATION_ENGINE"] = "unet_primary"
        result = saturn._apply_unet_candidate_support(
            np.zeros(image.shape, dtype=bool),
            np.zeros(image.shape, dtype=np.float32),
            valid,
            image.shape,
            (0, image.shape[0], 0, image.shape[1]),
            roi,
            inference_cfg,
            context,
            z_idx=z_value,
        )
        probabilities[z_value] = result[2]
        np.save(
            probability_dir / f"z{z_value:03d}_unet_probability.npy",
            result[2],
        )
        plt.imsave(
            probability_dir / f"z{z_value:03d}_unet_probability.png",
            normalize_probability(result[2]),
            cmap="magma",
            vmin=0,
            vmax=1,
        )

    summary_rows = []
    audit_rows = []
    failure_rows = []
    first_results = {}
    hashes = {}
    for repeat_index in range(1, args.repeat + 1):
        for z_value in targets:
            image = saturn.robust_imread(target_files[z_value])
            context = saturn._make_unet_context_from_paths(files_by_z, z_value)
            for engine in engines:
                run_cfg = cfg.copy()
                run_cfg.update({
                    "SEGMENTATION_ENGINE": engine,
                    "_UNET_PROBABILITY_CACHE": {
                        z_value: probabilities[z_value]
                    },
                })
                seg = saturn.segment_slice(
                    image,
                    run_cfg,
                    z_idx=z_value,
                    roi_mask=roi,
                    exclusion_mask=exclusion,
                    preprocess_context=preprocess,
                    unet_context_stack=context,
                )
                measured = saturn.measure_spermatids(seg, run_cfg)
                metrics = (
                    summarize_primary(seg, measured, valid, roi, exclusion)
                    if engine == "unet_primary"
                    else summarize_hybrid(seg, measured, valid, roi, exclusion)
                )
                row = {
                    "z": z_value,
                    "engine": engine,
                    "repeat": repeat_index,
                    **metrics,
                    "deterministic_label_hash": label_hash(
                        measured["skel_label"]
                    ),
                }
                summary_rows.append(row)
                hashes.setdefault((z_value, engine), []).append(
                    row["deterministic_label_hash"]
                )
                if repeat_index == 1:
                    first_results[(z_value, engine)] = (seg, measured)
                if engine == "unet_primary":
                    for audit in seg.get(
                        "unet_primary_component_audit", []
                    ):
                        audit_row = {
                            "z": z_value,
                            "engine": engine,
                            "repeat": repeat_index,
                            **audit,
                        }
                        audit_rows.append(audit_row)
                        if audit.get("disposition") == "rejected":
                            failure_rows.append(dict(audit_row))
                    for failure in measured.get(
                        "unet_primary_technical_failures", []
                    ):
                        failure_rows.append({
                            "z": z_value,
                            "engine": engine,
                            "repeat": repeat_index,
                            **failure,
                        })

    failures = []
    for row in summary_rows:
        if row["outside_roi_pixels"]:
            failures.append(f"outside-ROI output: {row}")
        if row["exclusion_mask_pixels"]:
            failures.append(f"exclusion overlap: {row}")
        if row["accepted_without_seed_count"]:
            failures.append(f"accepted instance without seed: {row}")
        if row["duplicate_instance_id_count"]:
            failures.append(f"duplicate instance IDs: {row}")
        if row["instance_measurement_mapping_mismatch"]:
            failures.append(f"instance/measurement mapping mismatch: {row}")
        if not row["unet_inference_enabled"]:
            failures.append(f"U-Net inference disabled or fell back: {row}")
    for key, values in hashes.items():
        if len(set(values)) != 1:
            failures.append(f"nondeterministic labels for {key}: {values}")
    if sorted({row["z"] for row in summary_rows}) != sorted(targets):
        failures.append("processed target set differs from requested target set")

    if {"hybrid", "unet_primary"}.issubset(set(engines)):
        for z_value in targets:
            save_review_panel(
                saturn,
                panel_dir / f"z{z_value:03d}_hybrid_vs_unet_primary.png",
                saturn.robust_imread(target_files[z_value]),
                probabilities[z_value],
                first_results[(z_value, "hybrid")],
                first_results[(z_value, "unet_primary")],
            )

    payload = {
        "requested_z_values": targets,
        "processed_z_values": sorted({row["z"] for row in summary_rows}),
        "engines": engines,
        "repeat": args.repeat,
        "tracking_enabled": False,
        "base_parameters": str(Path(args.base_params).resolve()),
        "unet_model": str(Path(args.unet_model).resolve()),
        "quality_gate_failures": failures,
        "quality_gates_passed": not failures,
        "rows": summary_rows,
    }
    with (outdir / "smoke_summary_v5_7.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2)
    write_csv(outdir / "smoke_summary_v5_7.csv", summary_rows)
    write_csv(outdir / "instance_audit_v5_7.csv", audit_rows)
    write_csv(outdir / "technical_failures_v5_7.csv", failure_rows)
    if failures:
        raise RuntimeError("; ".join(failures))
    return payload


def main():
    args = build_parser().parse_args()
    payload = run(args)
    print(json.dumps({
        "quality_gates_passed": payload["quality_gates_passed"],
        "processed_z_values": payload["processed_z_values"],
        "engines": payload["engines"],
    }, indent=2))


if __name__ == "__main__":
    main()
