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

def run(args):
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

    # 1. Run Segmentation and collect measurements
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
            if "source_instance_key" not in r:
                r["source_instance_key"] = f"{z_value}_{r['label']}"
        results.extend(saturn.rows_from_results(measured["results"], z_value, cfg["UM_PER_PX_XY"]))
        
    detections_df = pd.DataFrame(results)
    if detections_df.empty:
        detections_df["source_instance_key"] = []
        detections_df["z_slice"] = []
        detections_df["centroid_x"] = []
        detections_df["centroid_y"] = []
        
    # Pre-tracking checks
    if not detections_df.empty:
        assert len(detections_df) == len(results), "Mismatch in rows vs instances"
        assert detections_df["source_instance_key"].nunique() == len(detections_df), "source_instance_key is not unique"
        assert detections_df["centroid_x"].notnull().all()
        assert detections_df["centroid_y"].notnull().all()
        assert set(detections_df["z_slice"].unique()).issubset(set(targets))

    # 2. Track
    df_tracked, track_summary = saturn.track_across_slices(detections_df, cfg)
    
    # Post-tracking checks
    if not df_tracked.empty:
        # Check uniqueness of source_instance_key 
        assert df_tracked["source_instance_key"].nunique() == len(df_tracked), "source_instance_key lost uniqueness"
        assert df_tracked["track_id"].notnull().all(), "Missing track_id"
        
        # Add Track Boundary Flags
        df_tracked["touches_lower_test_boundary"] = df_tracked.groupby("track_id")["z_slice"].transform("min") == targets[0]
        df_tracked["touches_upper_test_boundary"] = df_tracked.groupby("track_id")["z_slice"].transform("max") == targets[-1]
        df_tracked["fully_internal_to_test_window"] = ~(df_tracked["touches_lower_test_boundary"] | df_tracked["touches_upper_test_boundary"])

        # Diagnostics Note
        df_tracked["assignment_cost"] = np.nan
        df_tracked["repair_cost"] = np.nan
        df_tracked["link_type"] = "derived" # if reconstructed

    write_csv(outdir / "smoke_tracking_results_v5_7.csv", df_tracked)
    write_csv(outdir / "smoke_tracking_summary_v5_7.csv", track_summary)

    payload = {
        "requested_z_values": targets,
        "processed_z_values": sorted(targets),
        "engines": ["unet_primary"],
        "repeat": args.repeat,
        "tracking_enabled": True,
        "base_parameters": str(Path(args.base_params).resolve()),
        "unet_model": str(Path(args.unet_model).resolve()),
    }
    with (outdir / "smoke_tracking_metadata_v5_7.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return payload

def main():
    args = build_parser().parse_args()
    payload = run(args)
    print(json.dumps(payload, indent=2))

if __name__ == "__main__":
    main()
