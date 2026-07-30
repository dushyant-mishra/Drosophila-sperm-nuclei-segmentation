"""Run balanced two-group U-Net-primary segmentation and tracking tuning."""

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    ROOT
    / "parameter_tuning_results_v5_7"
    / "mixed_wt_kj_retune"
    / "mixed_tuner_manifest.csv"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "parameter_tuning_results_v5_7"
    / "mixed_unet_primary"
)
DEFAULT_CHECKPOINT = (
    ROOT
    / "Kaggle notebook outputs"
    / "v57_kj_wt_training_export"
    / "checkpoints"
    / "epoch_003.pt"
)
DEFAULT_BASE_PRESET = (
    ROOT
    / "parameter_tuning_results_v5_7"
    / "epoch003_kj_wt_shared"
    / "shared_unet_rescue_params_v5_7_001.json"
)
TUNER = ROOT / "utils" / "tune_parameters_Saturnv5_7.py"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_tuning_inputs(run_root, manifest, checkpoint, base_preset):
    """Copy campaign inputs into a self-contained settings directory."""
    settings_dir = Path(run_root) / "settings"
    settings_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for role, source, destination_name in (
        ("mixed_tuning_manifest", manifest, "mixed_tuner_manifest.csv"),
        ("base_analysis_profile", base_preset, "analysis_profile_used.json"),
        ("unet_checkpoint", checkpoint, Path(checkpoint).name),
    ):
        source = Path(source).resolve()
        destination = (settings_dir / destination_name).resolve()
        if source != destination:
            temporary = destination.with_name(destination.name + ".tmp")
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)
        records.append(
            {
                "role": role,
                "original_path": str(source),
                "copied_path": str(destination),
                "size_bytes": destination.stat().st_size,
                "sha256": file_sha256(destination),
            }
        )
    settings_manifest = settings_dir / "settings_manifest.json"
    settings_manifest.write_text(
        json.dumps({"files": records}, indent=2),
        encoding="utf-8",
    )
    return settings_dir


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--base-preset", default=str(DEFAULT_BASE_PRESET))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--segmentation-candidates", type=int, default=24)
    parser.add_argument("--tracking-candidates", type=int, default=24)
    parser.add_argument("--tracking-slice-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def load_manifest(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 4:
        raise ValueError(f"Expected four balanced strata, found {len(rows)}")
    groups = [row["group"].strip() for row in rows]
    unique_groups = sorted(set(groups), key=str.casefold)
    if len(unique_groups) != 2 or any(
        groups.count(group) != 2 for group in unique_groups
    ):
        raise ValueError(
            "Manifest must contain exactly two named groups with two strata "
            "per group"
        )
    for row in rows:
        image_dir = Path(row["image_dir"])
        roi_path = Path(row["roi_path"])
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not roi_path.is_file():
            raise FileNotFoundError(f"ROI not found: {roi_path}")
        expected = int(row["source_slice_count"])
        actual = len(
            [
                path
                for path in image_dir.iterdir()
                if path.is_file()
                and path.suffix.lower() in {".tif", ".tiff"}
                and "_ch00" in path.stem.lower()
                and "_z" in path.stem.lower()
            ]
        )
        if actual != expected:
            raise ValueError(
                f"{row['specimen_id']}: expected {expected} slices, found {actual}"
            )
    return rows


def tracking_slices(row, count):
    if count < 3:
        raise ValueError("Tracking slice count must be at least three")
    selected = [
        int(value)
        for value in row["selected_z_indices"].split(",")
        if value.strip()
    ]
    center = selected[len(selected) // 2]
    z_min = 0
    z_max = int(row["source_slice_count"]) - 1
    start = max(z_min, min(center - count // 2, z_max - count + 1))
    return list(range(start, start + count))


def run_command(arguments):
    command = [sys.executable, str(TUNER), *map(str, arguments)]
    print("\n$", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def newest_result(directory, pattern):
    matches = sorted(Path(directory).glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No {pattern} found in {directory}")
    return matches[-1]


def aggregate(mode, results, base_preset, checkpoint, outdir, role):
    arguments = [
        "--mode",
        mode,
        "--base-params",
        base_preset,
        "--unet-model",
        checkpoint,
        "--shared-candidate-role",
        role,
        "--outdir",
        outdir,
    ]
    for result in results:
        arguments.extend(["--aggregate-stratum-results", result])
    run_command(arguments)
    return newest_result(outdir, f"shared_{mode}_params_v5_7_*.json")


def main():
    args = parse_args()
    manifest = Path(args.manifest).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    base_preset = Path(args.base_preset).resolve()
    for required in (manifest, checkpoint, base_preset, TUNER):
        if not required.is_file():
            raise FileNotFoundError(required)
    rows = load_manifest(manifest)
    print(
        "Validated mixed strata: "
        + ", ".join(f"{row['specimen_id']} [{row['group']}]" for row in rows)
    )
    if args.validate_only:
        return

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.output_root).resolve() / run_id
    if run_root.exists():
        raise FileExistsError(f"Run directory already exists: {run_root}")
    run_root.mkdir(parents=True)
    archive_tuning_inputs(
        run_root,
        manifest,
        checkpoint,
        base_preset,
    )
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "manifest": str(manifest),
        "checkpoint": str(checkpoint),
        "base_preset": str(base_preset),
        "segmentation_candidates": args.segmentation_candidates,
        "tracking_candidates": args.tracking_candidates,
        "tracking_slice_count": args.tracking_slice_count,
        "seed": args.seed,
        "segmentation_backend": "unet_primary",
        "tracking_backend": "global_assignment",
    }
    (run_root / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    segmentation_results = []
    segmentation_root = run_root / "01_unet_primary_segmentation"
    for row in rows:
        outdir = segmentation_root / row["specimen_id"]
        run_command(
            [
                "--mode",
                "unet_primary",
                "--dir",
                row["image_dir"],
                "--slices",
                row["selected_z_indices"],
                "--roi-mask",
                row["roi_path"],
                "--auto-calibration",
                "--base-params",
                base_preset,
                "--unet-model",
                checkpoint,
                "--outdir",
                outdir,
                "--maxiter",
                args.segmentation_candidates,
                "--seed",
                args.seed,
                "--review-candidates",
                8,
                "--review-candidate-role",
                "evidence_support_0.05_seed_0.30",
            ]
        )
        segmentation_results.append(
            outdir / "tuning_results_saturnv5_7_unet_primary.json"
        )

    shared_segmentation_dir = segmentation_root / "shared_two_group"
    shared_segmentation = aggregate(
        "unet_primary",
        segmentation_results,
        base_preset,
        checkpoint,
        shared_segmentation_dir,
        "evidence_support_0.05_seed_0.30",
    )

    tracking_results = []
    tracking_root = run_root / "02_global_tracking"
    for row in rows:
        outdir = tracking_root / row["specimen_id"]
        slices = tracking_slices(row, args.tracking_slice_count)
        run_command(
            [
                "--mode",
                "unet_primary_tracking",
                "--dir",
                row["image_dir"],
                "--slices",
                ",".join(map(str, slices)),
                "--roi-mask",
                row["roi_path"],
                "--auto-calibration",
                "--base-params",
                shared_segmentation,
                "--unet-model",
                checkpoint,
                "--unet-primary-tracking-backend",
                "global_assignment",
                "--outdir",
                outdir,
                "--maxiter",
                args.tracking_candidates,
                "--seed",
                args.seed,
                "--review-candidates",
                8,
                "--review-candidate-role",
                "reviewed_base",
            ]
        )
        tracking_results.append(
            outdir / "tuning_results_saturnv5_7_unet_primary_tracking.json"
        )

    shared_tracking_dir = tracking_root / "shared_two_group"
    reviewed_preset = aggregate(
        "unet_primary_tracking",
        tracking_results,
        shared_segmentation,
        checkpoint,
        shared_tracking_dir,
        "reviewed_base",
    )
    completion = {
        **metadata,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "shared_segmentation_preset": str(shared_segmentation),
        "shared_tracking_reviewed_base_preset": str(reviewed_preset),
        "tracking_results": [str(path) for path in tracking_results],
        "selection_status": "requires_cross_stratum_review",
    }
    (run_root / "completed_run.json").write_text(
        json.dumps(completion, indent=2),
        encoding="utf-8",
    )
    print(f"\nMixed U-Net-primary tuning complete: {run_root}")


if __name__ == "__main__":
    main()
