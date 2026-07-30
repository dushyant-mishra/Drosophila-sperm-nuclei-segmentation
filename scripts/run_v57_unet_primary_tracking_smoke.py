"""Five-slice tracking smoke harness for Saturn v5.7 U-Net-primary."""

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd

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

def validate_target_values(values):
    targets = [int(value) for value in values]
    if len(targets) < 3 or len(targets) > 7:
        raise ValueError(f"Target slice set must be 3-7 slices long. Got {len(targets)} slices.")
    sorted_z = sorted(targets)
    if len(set(sorted_z)) != len(sorted_z):
        raise ValueError(f"Target slice set contains duplicate Z values: {sorted_z}")
    for i in range(len(sorted_z) - 1):
        if sorted_z[i+1] - sorted_z[i] != 1:
            raise ValueError(f"Target slice set must be consecutive. Got: {sorted_z}")
    return sorted_z

def resolve_target_files(files_by_z, targets):
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

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--unet-model", required=True)
    parser.add_argument("--base-params", required=True)
    parser.add_argument("--roi-mask", required=True)
    parser.add_argument("--exclusion-mask", default="")
    parser.add_argument("--z-values", default="33,34,35,36,37")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--tracking-backend", default="hybrid_repair")
    return parser

def write_csv(path, df):
    if not df.empty:
        df.to_csv(path, index=False)
    else:
        path.write_text("", encoding="utf-8")

SUPPORTED_TRACKING_BACKENDS = {
    "legacy",
    "global_assignment",
    "hybrid_repair",
    "unet_primary_assignment",
}


def validate_run_options(
    *,
    repeat: int,
    tracking_backend: str,
) -> None:
    if int(repeat) < 1:
        raise ValueError("--repeat must be >= 1")

    if tracking_backend not in SUPPORTED_TRACKING_BACKENDS:
        allowed = ", ".join(sorted(SUPPORTED_TRACKING_BACKENDS))
        raise ValueError(
            "Invalid tracking backend "
            f"{tracking_backend!r}. Expected one of: {allowed}"
        )


def compute_membership_hash(
    tracked_df: pd.DataFrame,
) -> str:
    required = {"track_id", "source_instance_key"}
    missing = required - set(tracked_df.columns)

    if missing:
        raise ValueError(
            "Cannot compute membership hash; missing columns: "
            f"{sorted(missing)}"
        )

    if tracked_df.empty:
        canonical_groups = ()
    else:
        grouped_memberships = []

        for _, group in tracked_df.groupby(
            "track_id",
            sort=False,
            dropna=False,
        ):
            members = tuple(
                sorted(
                    str(value)
                    for value in group[
                        "source_instance_key"
                    ].tolist()
                )
            )
            grouped_memberships.append(members)

        canonical_groups = tuple(sorted(grouped_memberships))

    serialized = json.dumps(
        canonical_groups,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


def compute_track_span_metrics(
    tracked_df: pd.DataFrame,
    requested_z_values: list[int],
) -> dict:
    metrics = {
        "total_tracks": 0,
        "tracks_with_1_slice": 0,
        "tracks_with_2_slices": 0,
        "tracks_with_3_slices": 0,
        "tracks_with_4_slices": 0,
        "tracks_with_5_slices": 0,
        "tracks_with_6_slices": 0,
        "tracks_with_7_slices": 0,
        "boundary_single_slice_tracks": 0,
        "interior_single_slice_tracks": 0,
    }

    if tracked_df.empty:
        return metrics

    required = {"track_id", "z_slice"}
    missing = required - set(tracked_df.columns)

    if missing:
        raise ValueError(
            "Cannot compute track spans; missing columns: "
            f"{sorted(missing)}"
        )

    requested = sorted(int(z) for z in requested_z_values)

    track_spans = (
        tracked_df.groupby("track_id", dropna=False)["z_slice"]
        .nunique()
        .astype(int)
    )

    metrics["total_tracks"] = int(len(track_spans))

    for span in range(1, 8):
        metrics[f"tracks_with_{span}_slices"] = int(
            (track_spans == span).sum()
        )

    single_track_ids = track_spans[
        track_spans == 1
    ].index.tolist()

    if not single_track_ids:
        return metrics

    single_track_z = (
        tracked_df[
            tracked_df["track_id"].isin(single_track_ids)
        ]
        .groupby("track_id", dropna=False)["z_slice"]
        .first()
    )

    if requested:
        lower_boundary = requested[0]
        upper_boundary = requested[-1]

        boundary_mask = single_track_z.isin(
            [lower_boundary, upper_boundary]
        )

        metrics["boundary_single_slice_tracks"] = int(
            boundary_mask.sum()
        )
        metrics["interior_single_slice_tracks"] = int(
            (~boundary_mask).sum()
        )

    return metrics


def evaluate_tracking_integrity(
    *,
    pretracking_df: pd.DataFrame,
    tracked_df: pd.DataFrame,
    requested_z_values: list[int],
) -> dict:
    failures = []

    result = {
        "duplicate_same_z_group_count": 0,
        "duplicate_same_z_excess_observation_count": 0,
        "dropped_instance_count": 0,
        "duplicated_source_instance_count": 0,
        "multiply_assigned_instance_count": 0,
        "missing_track_id_count": 0,
        "unexpected_z_observation_count": 0,
        "quality_gates_passed": True,
        "integrity_failures": failures,
    }

    required_pre = {
        "source_instance_key",
        "z_slice",
    }
    required_post = {
        "source_instance_key",
        "z_slice",
        "track_id",
    }

    missing_pre = required_pre - set(pretracking_df.columns)
    missing_post = required_post - set(tracked_df.columns)

    if missing_pre:
        failures.append(
            "Pretracking dataframe missing columns: "
            f"{sorted(missing_pre)}"
        )

    if missing_post:
        failures.append(
            "Tracked dataframe missing columns: "
            f"{sorted(missing_post)}"
        )

    if missing_pre or missing_post:
        result["quality_gates_passed"] = False
        return result

    pre_keys = pretracking_df[
        "source_instance_key"
    ].astype(str)

    post_keys = tracked_df[
        "source_instance_key"
    ].astype(str)

    pre_key_set = set(pre_keys)
    post_key_set = set(post_keys)

    dropped_keys = pre_key_set - post_key_set
    result["dropped_instance_count"] = len(dropped_keys)

    if dropped_keys:
        failures.append(
            "Tracked output is missing source instances: "
            f"{sorted(dropped_keys)}"
        )

    post_key_counts = post_keys.value_counts()
    duplicated_keys = post_key_counts[
        post_key_counts > 1
    ].index.tolist()

    result["duplicated_source_instance_count"] = len(
        duplicated_keys
    )

    if duplicated_keys:
        failures.append(
            "Tracked output contains duplicated source instances: "
            f"{sorted(duplicated_keys)}"
        )

    track_counts_per_source = (
        tracked_df.assign(
            source_instance_key=post_keys
        )
        .groupby("source_instance_key")["track_id"]
        .nunique(dropna=True)
    )

    multiply_assigned_keys = track_counts_per_source[
        track_counts_per_source > 1
    ].index.tolist()

    result["multiply_assigned_instance_count"] = len(
        multiply_assigned_keys
    )

    if multiply_assigned_keys:
        failures.append(
            "Source instances belong to multiple track IDs: "
            f"{sorted(multiply_assigned_keys)}"
        )

    result["missing_track_id_count"] = int(
        tracked_df["track_id"].isna().sum()
    )

    if result["missing_track_id_count"]:
        failures.append(
            "Tracked output contains observations without track_id"
        )

    same_z_sizes = (
        tracked_df.groupby(
            ["track_id", "z_slice"],
            dropna=False,
        )
        .size()
    )

    duplicate_same_z_groups = same_z_sizes[
        same_z_sizes > 1
    ]

    result["duplicate_same_z_group_count"] = int(
        len(duplicate_same_z_groups)
    )

    result[
        "duplicate_same_z_excess_observation_count"
    ] = int(
        (duplicate_same_z_groups - 1).sum()
    )

    if result["duplicate_same_z_group_count"]:
        failures.append(
            "Final tracks contain multiple observations "
            "from the same Z plane"
        )

    requested_z = {
        int(value)
        for value in requested_z_values
    }
    observed_z = tracked_df["z_slice"].astype(int)

    unexpected_z_mask = ~observed_z.isin(requested_z)

    result["unexpected_z_observation_count"] = int(
        unexpected_z_mask.sum()
    )

    if result["unexpected_z_observation_count"]:
        failures.append(
            "Tracked output contains observations from "
            "unrequested Z planes"
        )

    result["quality_gates_passed"] = not failures
    return result

def run(args):
    validate_run_options(
        repeat=args.repeat,
        tracking_backend=args.tracking_backend,
    )

    saturn = load_saturn()
    targets = validate_target_values(parse_csv_values(args.z_values, int))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = saturn.CONFIG.copy()
    cfg.update(load_parameters(args.base_params))
    cfg.update({
        "UNET_MODEL_PATH": str(Path(args.unet_model).resolve()),
        "DO_TRACKING": True,
        "TRACKING_BACKEND": args.tracking_backend,
        "SAVE_DEBUG_IMAGES": False,
        "UNET_PRIMARY_CLASSICAL_ADDITIONS_ENABLE": False,
        "SEGMENTATION_ENGINE": "unet_primary",
    })

    files, z_values = saturn.load_batch_files(args.input_dir, cfg["FILE_PATTERN"])
    files_by_z = {}
    for file_path, z_value in zip(files, z_values):
        if int(z_value) in files_by_z:
            raise ValueError(f"Duplicate source Z index: {z_value}")
        files_by_z[int(z_value)] = file_path

    target_files = resolve_target_files(files_by_z, targets)

    first = saturn.robust_imread(target_files[targets[0]])
    roi = saturn.load_roi_mask_file(args.roi_mask, expected_shape=first.shape)
    exclusion = (
        saturn.load_roi_mask_file(args.exclusion_mask, expected_shape=first.shape)
        if args.exclusion_mask else np.zeros(first.shape, dtype=bool)
    )
    preprocess = saturn.build_stack_preprocess_context(files, roi, cfg, exclusion_mask=exclusion)

    results = []
    for z_value in targets:
        image = saturn.robust_imread(target_files[z_value])
        context = saturn._make_unet_context_from_paths(files_by_z, z_value)
        seg = saturn.segment_slice(
            image,
            cfg,
            z_idx=z_value,
            roi_mask=roi,
            exclusion_mask=exclusion,
            preprocess_context=preprocess,
            unet_context_stack=context,
        )
        measured = saturn.measure_spermatids(seg, cfg)
        for r in measured["results"]:
            r["source_instance_key"] = f"z{z_value:03d}_i{r.get('label', 0):05d}"
        results.extend(saturn.rows_from_results(measured["results"], z_value, cfg["UM_PER_PX_XY"]))

    detections_df = pd.DataFrame(results)
    if detections_df.empty:
        detections_df["source_instance_key"] = []
        detections_df["z_slice"] = []
        detections_df["centroid_x"] = []
        detections_df["centroid_y"] = []

    write_csv(outdir / "pretracking_instances_v5_7.csv", detections_df)

    integrity_failures = []

    if not detections_df.empty:
        if len(detections_df) != len(results):
            integrity_failures.append("Mismatch in rows vs instances")
        if detections_df["source_instance_key"].nunique() != len(detections_df):
            integrity_failures.append("source_instance_key is not unique")
        if not detections_df["centroid_x"].notnull().all():
            integrity_failures.append("Null centroid_x in detections")
        if not detections_df["centroid_y"].notnull().all():
            integrity_failures.append("Null centroid_y in detections")

    processed_z_values = sorted(detections_df["z_slice"].unique().tolist()) if not detections_df.empty else []

    repeat_hashes = []
    final_df_tracked = pd.DataFrame()
    final_track_summary = pd.DataFrame()

    for _ in range(args.repeat):
        df_tracked, track_summary = saturn.track_across_slices(detections_df.copy(), cfg)
        repeat_hashes.append(compute_membership_hash(df_tracked))
        final_df_tracked = df_tracked
        final_track_summary = track_summary

    span_metrics = compute_track_span_metrics(
        final_df_tracked,
        requested_z_values=targets,
    )

    integrity = evaluate_tracking_integrity(
        pretracking_df=detections_df,
        tracked_df=final_df_tracked,
        requested_z_values=targets,
    )

    failure_rows = pd.DataFrame(
        {
            "integrity_failure_reason": integrity[
                "integrity_failures"
            ]
        }
    )

    write_csv(
        outdir / "tracking_integrity_failures_v5_7.csv",
        failure_rows,
    )

    write_csv(outdir / "tracked_observations_v5_7.csv", final_df_tracked)
    write_csv(outdir / "track_summary_v5_7.csv", final_track_summary)

    # Backward compatibility
    write_csv(outdir / "smoke_tracking_results_v5_7.csv", final_df_tracked)
    write_csv(outdir / "smoke_tracking_summary_v5_7.csv", final_track_summary)

    payload = {
        "requested_z_values": targets,
        "processed_z_values": processed_z_values,
        "engines": ["unet_primary"],
        "repeat": args.repeat,
        "tracking_enabled": True,
        "tracking_backend": args.tracking_backend,
        "base_parameters": str(Path(args.base_params).resolve()),
        "unet_model": str(Path(args.unet_model).resolve()),
        "total_2d_instances": len(detections_df),
    }

    payload["repeat_membership_hashes"] = repeat_hashes
    payload["deterministic_membership_hash"] = (
        repeat_hashes[-1] if repeat_hashes else ""
    )
    payload["deterministic_repeats_passed"] = (
        len(set(repeat_hashes)) <= 1
    )

    payload.update(span_metrics)

    payload.update(
        {
            key: value
            for key, value in integrity.items()
            if key != "integrity_failures"
        }
    )

    payload["quality_gates_passed"] = bool(
        integrity["quality_gates_passed"]
        and payload["deterministic_repeats_passed"]
    )

    with (
        outdir / "tracking_smoke_summary_v5_7.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    pd.DataFrame([payload]).drop(
        columns=["repeat_membership_hashes"],
        errors="ignore",
    ).to_csv(
        outdir / "tracking_smoke_summary_v5_7.csv",
        index=False,
    )

    return payload

def main():
    args = build_parser().parse_args()
    payload = run(args)
    print(json.dumps(payload, indent=2))

if __name__ == "__main__":
    main()
