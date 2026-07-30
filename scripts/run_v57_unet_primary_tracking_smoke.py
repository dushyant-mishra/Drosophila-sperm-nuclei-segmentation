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

def compute_membership_hash(df):
    if df.empty or "track_id" not in df.columns:
        return ""
    grouped = df.groupby("track_id")["source_instance_key"].apply(lambda x: tuple(sorted(x)))
    canonical_groups = tuple(sorted(grouped.values))
    digest = hashlib.sha256(str(canonical_groups).encode("utf-8")).hexdigest()
    return digest

def run(args):
    if args.repeat < 1:
        raise ValueError("--repeat must be >= 1")

    valid_backends = {"legacy", "global_assignment", "hybrid_repair"}
    if args.tracking_backend not in valid_backends:
        raise ValueError(f"Invalid tracking backend. Must be one of {valid_backends}")

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

    if len(set(repeat_hashes)) > 1:
        integrity_failures.append("Repeat membership hashes differ")

    final_hash = repeat_hashes[-1] if repeat_hashes else ""

    total_tracks = 0
    tracks_with_n_slices = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    boundary_single_slice_tracks = 0
    interior_single_slice_tracks = 0
    duplicate_same_z_count = 0
    dropped_instance_count = 0
    multiply_assigned_instance_count = 0

    if not final_df_tracked.empty:
        total_tracks = final_df_tracked["track_id"].nunique()
        track_counts = final_df_tracked.groupby("track_id")["z_slice"].nunique()
        for i in range(1, 6):
            tracks_with_n_slices[i] = int((track_counts == i).sum())

        single_tracks = track_counts[track_counts == 1].index
        if len(single_tracks) > 0:
            single_df = final_df_tracked[final_df_tracked["track_id"].isin(single_tracks)]
            if len(processed_z_values) > 2:
                interior = single_df["z_slice"].isin(processed_z_values[1:-1])
                interior_single_slice_tracks = int(interior.sum())
                boundary_single_slice_tracks = int((~interior).sum())
            else:
                boundary_single_slice_tracks = len(single_tracks)

        dup_z = final_df_tracked.groupby(["track_id", "z_slice"]).size()
        duplicate_same_z_count = int((dup_z > 1).sum())
        if duplicate_same_z_count > 0:
            integrity_failures.append("Final track contains duplicate observations at the same z_slice")

        pre_keys = set(detections_df["source_instance_key"])
        post_keys = final_df_tracked["source_instance_key"].tolist()

        missing_keys = pre_keys - set(post_keys)
        if missing_keys:
            dropped_instance_count = len(missing_keys)
            integrity_failures.append("source_instance_key is missing after tracking")

        dup_keys = [k for k, v in pd.Series(post_keys).value_counts().items() if v > 1]
        if dup_keys:
            integrity_failures.append("source_instance_key occurs more than once")

        if final_df_tracked.groupby("source_instance_key")["track_id"].nunique().max() > 1:
            multiply_assigned_instance_count = len(final_df_tracked[final_df_tracked.groupby("source_instance_key")["track_id"].transform('nunique') > 1])
            integrity_failures.append("source_instance_key belongs to multiple track IDs")

        if final_df_tracked["track_id"].isnull().any():
            integrity_failures.append("Missing track_id in tracked df")

    if integrity_failures:
        pd.DataFrame({"integrity_failure_reason": integrity_failures}).to_csv(outdir / "tracking_integrity_failures_v5_7.csv", index=False)
    else:
        write_csv(outdir / "tracking_integrity_failures_v5_7.csv", pd.DataFrame(columns=["integrity_failure_reason"]))

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
        "base_parameters": str(Path(args.base_params).resolve()),
        "unet_model": str(Path(args.unet_model).resolve()),
        "total_2d_instances": len(detections_df),
        "total_tracks": total_tracks,
        "tracks_with_1_slice": tracks_with_n_slices[1],
        "tracks_with_2_slices": tracks_with_n_slices[2],
        "tracks_with_3_slices": tracks_with_n_slices[3],
        "tracks_with_4_slices": tracks_with_n_slices[4],
        "tracks_with_5_slices": tracks_with_n_slices[5],
        "boundary_single_slice_tracks": boundary_single_slice_tracks,
        "interior_single_slice_tracks": interior_single_slice_tracks,
        "duplicate_same_z_count": duplicate_same_z_count,
        "dropped_instance_count": dropped_instance_count,
        "multiply_assigned_instance_count": multiply_assigned_instance_count,
        "deterministic_membership_hash": final_hash,
        "quality_gates_passed": len(integrity_failures) == 0,
    }
    with (outdir / "tracking_smoke_summary_v5_7.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return payload

def main():
    args = build_parser().parse_args()
    payload = run(args)
    print(json.dumps(payload, indent=2))

if __name__ == "__main__":
    main()
