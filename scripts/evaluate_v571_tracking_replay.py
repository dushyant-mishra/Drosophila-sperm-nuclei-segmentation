"""Replay v5.7.1 tracking on cached 2D detections without rerunning inference."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = ROOT / "sperm_segmentation_saturnv5.7.1.py"
DEFAULT_PILOT_ROOT = ROOT / "scratch" / "v571_dual_head_full_pilot"
DEFAULT_PROFILE = ROOT / "production_profiles" / "saturn_v5_7_1_model_c_epoch003.json"


def load_pipeline():
    spec = importlib.util.spec_from_file_location("saturn_v571", PIPELINE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def summarize(specimen: str, name: str, tracked: pd.DataFrame, tracks: pd.DataFrame):
    slice_counts = tracks["n_slices"].astype(float)
    lengths = pd.to_numeric(tracks["projection_z_extent_um"], errors="coerce")
    duplicate_z = int(
        tracked.groupby("track_id")["z_slice"].apply(lambda values: values.duplicated().any()).sum()
    )
    distances = pd.to_numeric(tracked["track_link_distance_um"], errors="coerce")
    return {
        "specimen": specimen,
        "candidate": name,
        "detections_2d": int(len(tracked)),
        "tracks": int(len(tracks)),
        "single_slice_tracks": int((slice_counts == 1).sum()),
        "single_slice_fraction": float((slice_counts == 1).mean()),
        "median_slices_per_track": float(slice_counts.median()),
        "mean_slices_per_track": float(slice_counts.mean()),
        "median_projection_z_extent_um": float(lengths.median()),
        "tracks_15_to_20_um": int(((lengths >= 15.0) & (lengths <= 20.0)).sum()),
        "tracks_over_20_um": int((lengths > 20.0).sum()),
        "duplicate_z_tracks": duplicate_z,
        "median_link_distance_um": float(distances.median()),
        "p95_link_distance_um": float(distances.quantile(0.95)),
        "max_link_distance_um": float(distances.max()),
    }


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def specimen_outputs(root):
    samples = Path(root) / "samples"
    if samples.is_dir():
        result = []
        for sample_dir in sorted(samples.iterdir()):
            attempts = sorted(sample_dir.glob("attempt_*"))
            complete = [path for path in attempts if (path / "sample_complete.json").is_file()]
            if complete:
                result.append((sample_dir.name, complete[-1]))
        return result
    return [
        (path.name, path)
        for path in sorted(Path(root).iterdir())
        if path.is_dir() and list(path.glob("spermatid_measurements_*.csv"))
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-root", type=Path, default=DEFAULT_PILOT_ROOT)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "scratch" / "v571_tracking_replay",
    )
    parser.add_argument(
        "--production-only",
        action="store_true",
        help="Replay only the frozen morphology-neutral production candidate.",
    )
    args = parser.parse_args()

    saturn = load_pipeline()
    base = json.loads(args.profile.read_text(encoding="utf-8"))
    candidates = {
        "legacy_morphology_weighted": {
            "COMPARATIVE_TRACKING_MORPHOLOGY_NEUTRAL": False,
            "ASSIGNMENT_LENGTH_WEIGHT": 2.0,
            "ASSIGNMENT_WIDTH_WEIGHT": 1.2,
            "ASSIGNMENT_AREA_WEIGHT": 0.9,
        },
        "production_morphology_neutral": {
            "COMPARATIVE_TRACKING_MORPHOLOGY_NEUTRAL": True,
        },
    }
    if args.production_only:
        candidates = {
            "production_morphology_neutral": candidates[
                "production_morphology_neutral"
            ]
        }

    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    provenance = []
    for specimen, specimen_dir in specimen_outputs(args.pilot_root):
        source = next(specimen_dir.glob("spermatid_measurements_*.csv"))
        detections = pd.read_csv(source)
        runtime_path = specimen_dir / "settings" / "runtime_parameters.json"
        specimen_cfg = dict(base)
        if runtime_path.exists():
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
            specimen_cfg.update(runtime)
        specimen_cfg["TRACKING_BACKEND"] = "global_assignment"

        for name, overrides in candidates.items():
            cfg = dict(specimen_cfg)
            cfg.update(overrides)
            tracked, tracks = saturn.track_across_slices_global_assignment(detections, cfg)
            tracks = saturn.flag_quality_tracks(tracks, cfg)
            rows.append(summarize(specimen, name, tracked, tracks))
            tracked_path = args.output / f"{specimen}_{name}_tracked.csv"
            tracks_path = args.output / f"{specimen}_{name}_tracks.csv"
            tracked.to_csv(tracked_path, index=False)
            tracks.to_csv(tracks_path, index=False)
            provenance.append(
                {
                    "specimen": specimen,
                    "candidate": name,
                    "source_2d_detections": str(source.resolve()),
                    "source_2d_detections_sha256": sha256(source),
                    "tracked_csv": str(tracked_path.resolve()),
                    "tracked_csv_sha256": sha256(tracked_path),
                    "tracks_csv": str(tracks_path.resolve()),
                    "tracks_csv_sha256": sha256(tracks_path),
                }
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(args.output / "tracking_replay_summary.csv", index=False)
    (args.output / "tracking_replay_candidates.json").write_text(
        json.dumps(candidates, indent=2), encoding="utf-8"
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    (args.output / "tracking_replay_manifest.json").write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit_at_generation": commit,
                "pipeline_sha256": sha256(PIPELINE_PATH),
                "profile": str(args.profile.resolve()),
                "profile_sha256": sha256(args.profile),
                "note": (
                    "Deterministic downstream replay from frozen 2D detections; "
                    "U-Net inference was not rerun."
                ),
                "artifacts": provenance,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
