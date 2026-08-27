"""Create unobscured, stratified cross-slice tracking evidence for v5.7.1."""

import argparse
import hashlib
import importlib.util
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import binary_dilation
from skimage.segmentation import find_boundaries


CATEGORIES = (
    "faint",
    "bright",
    "short",
    "long",
    "wide",
    "curved",
    "touching",
    "irregular",
)
COLORS = plt.get_cmap("tab10").colors
ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "sperm_segmentation_saturnv5.7.1.py"
DEFAULT_PROFILE = ROOT / "production_profiles" / "saturn_v5_7_1_model_c_epoch003.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha256(path, commit="HEAD"):
    relative = Path(path).resolve().relative_to(ROOT).as_posix()
    content = subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=ROOT
    )
    return hashlib.sha256(content).hexdigest()


def load_pipeline():
    spec = importlib.util.spec_from_file_location("saturn_v571_tracking_evidence", PIPELINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_csv(folder, prefix):
    matches = sorted(Path(folder).glob(f"{prefix}*.csv"))
    if prefix == "track_summary":
        matches = [path for path in matches if "technical_failures" not in path.name]
    if not matches:
        raise FileNotFoundError(f"No {prefix}*.csv in {folder}")
    return matches[0]


def source_files(input_dir):
    records = {}
    for path in Path(input_dir).iterdir():
        if not path.is_file() or path.suffix.lower() not in {".tif", ".tiff"}:
            continue
        import re
        match = re.search(r"_z(\d+)_ch\d+\.tiff?$", path.name, re.IGNORECASE)
        if match:
            records[int(match.group(1))] = path
    return records


def choose_tracks(tracks, detections):
    tracks = tracks.copy()
    if "technical_valid" in tracks:
        valid = tracks["technical_valid"].astype(str).str.lower().isin({"true", "1"})
        tracks = tracks.loc[valid].copy()
    probability_column = next(
        (name for name in ("unet_mean_probability", "unet_max_probability") if name in detections),
        None,
    )
    if probability_column:
        probability = detections.groupby("track_id")[probability_column].mean()
        tracks["_probability"] = tracks["track_id"].map(probability)
    else:
        tracks["_probability"] = np.nan

    def numeric(name, default=np.nan):
        return pd.to_numeric(tracks.get(name, default), errors="coerce")

    rankings = {
        "faint": numeric("_probability").sort_values(ascending=True),
        "bright": numeric("_probability").sort_values(ascending=False),
        "short": numeric("projection_z_extent_um").sort_values(ascending=True),
        "long": numeric("projection_z_extent_um").sort_values(ascending=False),
        "wide": numeric("representative_body_width_um").sort_values(ascending=False),
        "curved": numeric("tortuosity_3d").sort_values(ascending=False),
        "touching": numeric("nearest_neighbor_um").sort_values(ascending=True),
        "irregular": (
            numeric("representative_body_width_iqr_um").fillna(0)
            + numeric("tortuosity_3d").fillna(1).sub(1).clip(lower=0)
        ).sort_values(ascending=False),
    }
    selected = []
    used = set()
    for category in CATEGORIES:
        for index in rankings[category].index:
            track_id = tracks.loc[index, "track_id"]
            if track_id not in used and np.isfinite(rankings[category].loc[index]):
                selected.append((category, tracks.loc[index]))
                used.add(track_id)
                break
    return selected


def crop_for_rows(rows, shape, padding=45):
    x = pd.to_numeric(rows["centroid_x"], errors="coerce")
    y = pd.to_numeric(rows["centroid_y"], errors="coerce")
    y0 = max(int(np.floor(y.min())) - padding, 0)
    y1 = min(int(np.ceil(y.max())) + padding + 1, shape[0])
    x0 = max(int(np.floor(x.min())) - padding, 0)
    x1 = min(int(np.ceil(x.max())) + padding + 1, shape[1])
    return slice(y0, y1), slice(x0, x1)


def overlay_instance_evidence(raw, segmentation, labels, sperm_id, color, cfg):
    finite = np.asarray(raw, dtype=np.float32)
    lo, hi = np.percentile(finite[np.isfinite(finite)], [1, 99.7])
    normalized = np.clip((finite - lo) / max(hi - lo, 1e-9), 0, 1)
    rgb = np.repeat(normalized[..., None], 3, axis=2)
    centerline = labels == int(sperm_id)
    instances = np.asarray(segmentation["unet_primary_instance_labels"])
    foreground_probability = np.asarray(segmentation["unet_probability"])
    supported = foreground_probability >= float(cfg["UNET_FOREGROUND_THRESHOLD"])
    rgb[supported] = 0.82 * rgb[supported] + 0.18 * np.asarray([0.0, 0.85, 0.9])

    labels_on_centerline = instances[binary_dilation(centerline, iterations=1)]
    labels_on_centerline = labels_on_centerline[labels_on_centerline > 0]
    target_label = 0
    if labels_on_centerline.size:
        values, counts = np.unique(labels_on_centerline, return_counts=True)
        target_label = int(values[np.argmax(counts)])
    target_mask = instances == target_label if target_label else np.zeros_like(instances, bool)
    all_boundaries = find_boundaries(instances, mode="inner")
    target_boundary = find_boundaries(target_mask, mode="inner")
    neighbor_boundaries = all_boundaries & ~target_boundary
    rgb[target_mask] = 0.68 * rgb[target_mask] + 0.32 * np.asarray(color)
    rgb[neighbor_boundaries] = np.asarray([1.0, 1.0, 1.0])
    rgb[target_boundary] = np.asarray(color)
    # Two-pixel display stroke only. Measurements use the original one-pixel
    # centerline labels, so visualization cannot alter count or morphometry.
    rgb[binary_dilation(centerline, iterations=1)] = color
    return rgb, target_label


def consecutive_window(rows, available_z, width=4):
    observed = sorted({int(value) for value in rows["z_slice"]})
    if not observed:
        return []
    center = observed[len(observed) // 2]
    candidates = sorted(int(value) for value in available_z)
    windows = []
    for start_index in range(max(len(candidates) - width + 1, 1)):
        window = candidates[start_index : start_index + width]
        if window:
            overlap = len(set(window) & set(observed))
            distance = abs(np.mean(window) - center)
            windows.append((-overlap, distance, window))
    return min(windows)[2] if windows else observed[:width]


def render_specimen(
    specimen,
    output_dir,
    input_dir,
    destination,
    tracks_path=None,
    detections_path=None,
    roi_path=None,
    profile_path=DEFAULT_PROFILE,
):
    tracks = pd.read_csv(tracks_path or find_csv(output_dir, "track_summary"))
    detections = pd.read_csv(
        detections_path or find_csv(output_dir, "measurements_with_tracks")
    )
    sources = source_files(input_dir)
    saturn = load_pipeline()
    cfg, _ = saturn.load_analysis_profile(profile_path, saturn.CONFIG)
    ordered_files = [str(sources[index]) for index in sorted(sources)]
    saturn.resolve_stack_microscope_calibration(
        cfg, ordered_files, input_dir=Path(input_dir)
    )
    first_raw = saturn.ensure_2d_image(
        saturn.robust_imread(ordered_files[0]), Path(ordered_files[0]).name
    )
    roi = saturn.load_roi_mask_file(roi_path, expected_shape=first_raw.shape)
    preprocess_context = saturn.build_stack_preprocess_context(
        ordered_files, roi, cfg
    )
    selected = choose_tracks(tracks, detections)
    if not selected:
        return []
    fig, axes = plt.subplots(
        len(selected),
        5,
        figsize=(14, 2.8 * len(selected)),
        constrained_layout=True,
        squeeze=False,
    )
    records = []
    raw_cache = {}
    label_cache = {}
    segmentation_cache = {}

    def current_segmentation(z_index):
        if z_index not in segmentation_cache:
            raw = saturn.ensure_2d_image(
                saturn.robust_imread(str(sources[z_index])), sources[z_index].name
            )
            unet_context = saturn._make_unet_context_from_paths(
                {index: str(path) for index, path in sources.items()}, z_index
            )
            segmentation = saturn.segment_slice(
                raw,
                cfg,
                z_idx=z_index,
                roi_mask=roi,
                preprocess_context=preprocess_context,
                unet_context_stack=unet_context,
            )
            measured = saturn.measure_spermatids(segmentation, cfg)
            segmentation_cache[z_index] = (raw, segmentation, measured)
        return segmentation_cache[z_index]
    for row_index, (category, track) in enumerate(selected):
        track_id = track["track_id"]
        rows = detections[detections["track_id"] == track_id].sort_values("z_slice")
        shown_z = consecutive_window(rows, sources, width=4)
        representative = rows.iloc[len(rows) // 2]
        rep_z = int(representative["z_slice"])
        if rep_z not in raw_cache:
            raw_cache[rep_z] = tifffile.imread(sources[rep_z])
        crop = crop_for_rows(rows, raw_cache[rep_z].shape)
        crop_bounds = [crop[0].start, crop[0].stop, crop[1].start, crop[1].stop]
        crop_touches_edge = bool(
            crop[0].start == 0
            or crop[1].start == 0
            or crop[0].stop == raw_cache[rep_z].shape[0]
            or crop[1].stop == raw_cache[rep_z].shape[1]
        )
        axes[row_index, 0].imshow(raw_cache[rep_z][crop], cmap="gray")
        axes[row_index, 0].set_title(f"{category}: raw z{rep_z:03d}", fontsize=9)
        axes[row_index, 0].set_ylabel(
            f"Track {track_id}\nProjection+Z={track.get('projection_z_extent_um', np.nan):.2f} um\n"
            f"W={track.get('representative_body_width_um', np.nan):.2f} um",
            fontsize=8,
        )
        axes[row_index, 0].axis("off")
        color = COLORS[row_index % len(COLORS)]
        for column in range(1, 5):
            axis = axes[row_index, column]
            if column - 1 >= len(shown_z):
                axis.axis("off")
                continue
            z_index = int(shown_z[column - 1])
            observations = rows[pd.to_numeric(rows["z_slice"], errors="coerce") == z_index]
            if observations.empty:
                raw, segmentation, measured = current_segmentation(z_index)
                rendered, _target_instance = overlay_instance_evidence(
                    raw,
                    segmentation,
                    measured["skel_label"],
                    -1,
                    color,
                    cfg,
                )
                axis.imshow(rendered[crop])
                axis.set_title(
                    f"z{z_index:03d}: no target; context only", fontsize=9
                )
                axis.axis("off")
                continue
            detection = observations.iloc[0]
            raw, segmentation, measured = current_segmentation(z_index)
            current_rows = pd.DataFrame(measured["results"])
            distance = np.hypot(
                pd.to_numeric(current_rows["centroid_x"], errors="coerce")
                - float(detection["centroid_x"]),
                pd.to_numeric(current_rows["centroid_y"], errors="coerce")
                - float(detection["centroid_y"]),
            )
            current_id = int(current_rows.loc[distance.idxmin(), "label"])
            rendered, target_instance = overlay_instance_evidence(
                raw,
                segmentation,
                measured["skel_label"],
                current_id,
                color,
                cfg,
            )
            axis.imshow(rendered[crop])
            axis.set_title(f"z{z_index:03d}: instance {target_instance}", fontsize=9)
            axis.axis("off")
        records.append(
            {
                "specimen": specimen,
                "category": category,
                "track_id": int(track_id),
                "displayed_consecutive_z_indices": shown_z,
                "observed_z_indices": [int(value) for value in rows["z_slice"]],
                "projection_z_extent_um": float(
                    track.get("projection_z_extent_um", np.nan)
                ),
                "body_width_um": float(track.get("representative_body_width_um", np.nan)),
                "tortuosity": float(track.get("tortuosity_3d", np.nan)),
                "nearest_neighbor_um": float(track.get("nearest_neighbor_um", np.nan)),
                "mean_unet_probability": float(track.get("_probability", np.nan)),
                "crop_bounds_y0_y1_x0_x1": crop_bounds,
                "source_shape": list(raw_cache[rep_z].shape),
                "crop_touches_source_edge": crop_touches_edge,
                "selection_pool_size": int(len(tracks)),
                "selection_rationale": f"deterministic {category} metric extremum",
            }
        )
    fig.suptitle(
        f"{specimen}: stratified v5.7.1 tracking evidence\n"
        "First panel is unobscured raw data. Cyan is U-Net support; white marks neighboring "
        "instance boundaries; track color marks target mask, boundary, and centerline.",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-manifest", required=True)
    parser.add_argument("--study-output", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE))
    parser.add_argument(
        "--tracking-replay",
        default="",
        help="Optional folder containing current *_tracked.csv and *_tracks.csv.",
    )
    args = parser.parse_args()
    pilot = json.loads(Path(args.pilot_manifest).read_text(encoding="utf-8"))
    inputs = {record["specimen"]: record["input_dir"] for record in pilot}
    rois = {record["specimen"]: record["roi_path"] for record in pilot}
    study_root = Path(args.study_output)
    destination_root = Path(args.output_dir)
    destination_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for sample_dir in sorted((study_root / "samples").iterdir()):
        attempts = sorted(sample_dir.glob("attempt_*"))
        completed = [path for path in attempts if (path / "sample_complete.json").is_file()]
        if not completed:
            continue
        output_dir = completed[-1]
        specimen = "KJ-01" if sample_dir.name.startswith("kj_") else "WT-01"
        tracks_path = None
        detections_path = None
        if args.tracking_replay:
            replay = Path(args.tracking_replay)
            tracks_path = replay / (
                f"{sample_dir.name}_production_morphology_neutral_tracks.csv"
            )
            detections_path = replay / (
                f"{sample_dir.name}_production_morphology_neutral_tracked.csv"
            )
        destination = destination_root / f"{specimen}_stratified_tracking.png"
        records = render_specimen(
            specimen,
            output_dir,
            inputs[specimen],
            destination,
            tracks_path=tracks_path,
            detections_path=detections_path,
            roi_path=rois[specimen],
            profile_path=args.profile,
        )
        if not records:
            manifest.append(
                {
                    "specimen": specimen,
                    "analysis_output": str(output_dir.resolve()),
                    "artifact": None,
                    "records": [],
                    "note": "No eligible technical-valid tracks were available.",
                }
            )
            continue
        manifest.append(
            {
                "specimen": specimen,
                "analysis_output": str(output_dir.resolve()),
                "artifact": str(destination.resolve()),
                "artifact_repository_path": destination.resolve().relative_to(ROOT).as_posix(),
                "artifact_sha256": sha256(destination),
                "records": records,
            }
        )
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
        ).strip()
    except Exception:
        git_commit = ""
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_at_generation": git_commit,
        "pipeline_working_copy_sha256": sha256(PIPELINE),
        "pipeline_git_blob_sha256": git_blob_sha256(PIPELINE, git_commit),
        "profile_working_copy_sha256": sha256(args.profile),
        "profile_git_blob_sha256": git_blob_sha256(args.profile, git_commit),
        "generator_working_copy_sha256": sha256(Path(__file__)),
        "generator_git_blob_sha256": git_blob_sha256(Path(__file__), git_commit),
        "study_output": str(study_root.resolve()),
        "records": manifest,
    }
    (destination_root / "tracking_evidence_manifest.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
