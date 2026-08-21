"""Create unobscured, stratified cross-slice tracking evidence for v5.7.1."""

import argparse
import hashlib
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


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        "short": numeric("total_3d_length_um").sort_values(ascending=True),
        "long": numeric("total_3d_length_um").sort_values(ascending=False),
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


def overlay_centerline(raw, labels, sperm_id, color):
    finite = np.asarray(raw, dtype=np.float32)
    lo, hi = np.percentile(finite[np.isfinite(finite)], [1, 99.7])
    normalized = np.clip((finite - lo) / max(hi - lo, 1e-9), 0, 1)
    rgb = np.repeat(normalized[..., None], 3, axis=2)
    mask = labels == int(sperm_id)
    context = (labels > 0) & ~mask
    rgb[context] = 0.55 * rgb[context] + 0.45 * np.asarray([0.65, 0.65, 0.65])
    rgb[mask] = color
    return rgb


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


def render_specimen(specimen, output_dir, input_dir, destination):
    tracks = pd.read_csv(find_csv(output_dir, "track_summary"))
    detections = pd.read_csv(find_csv(output_dir, "measurements_with_tracks"))
    sources = source_files(input_dir)
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
    for row_index, (category, track) in enumerate(selected):
        track_id = track["track_id"]
        rows = detections[detections["track_id"] == track_id].sort_values("z_slice")
        shown_z = consecutive_window(rows, sources, width=4)
        representative = rows.iloc[len(rows) // 2]
        rep_z = int(representative["z_slice"])
        if rep_z not in raw_cache:
            raw_cache[rep_z] = tifffile.imread(sources[rep_z])
        crop = crop_for_rows(rows, raw_cache[rep_z].shape)
        axes[row_index, 0].imshow(raw_cache[rep_z][crop], cmap="gray")
        axes[row_index, 0].set_title(f"{category}: raw z{rep_z:03d}", fontsize=9)
        axes[row_index, 0].set_ylabel(
            f"Track {track_id}\nL={track.get('total_3d_length_um', np.nan):.2f} um\n"
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
            if z_index not in raw_cache:
                raw_cache[z_index] = tifffile.imread(sources[z_index])
            observations = rows[pd.to_numeric(rows["z_slice"], errors="coerce") == z_index]
            if observations.empty:
                axis.imshow(raw_cache[z_index][crop], cmap="gray")
                axis.set_title(f"z{z_index:03d}: no target observation", fontsize=9)
                axis.axis("off")
                continue
            detection = observations.iloc[0]
            label_path = Path(output_dir) / f"z{z_index:02d}_skel_labels.tif"
            if z_index not in label_cache:
                label_cache[z_index] = tifffile.imread(label_path)
            rendered = overlay_centerline(
                raw_cache[z_index],
                label_cache[z_index],
                detection["sperm_id"],
                color,
            )
            axis.imshow(rendered[crop])
            axis.set_title(f"z{z_index:03d}", fontsize=9)
            axis.axis("off")
        records.append(
            {
                "specimen": specimen,
                "category": category,
                "track_id": int(track_id),
                "displayed_consecutive_z_indices": shown_z,
                "observed_z_indices": [int(value) for value in rows["z_slice"]],
                "length_um": float(track.get("total_3d_length_um", np.nan)),
                "body_width_um": float(track.get("representative_body_width_um", np.nan)),
                "tortuosity": float(track.get("tortuosity_3d", np.nan)),
                "nearest_neighbor_um": float(track.get("nearest_neighbor_um", np.nan)),
                "mean_unet_probability": float(track.get("_probability", np.nan)),
            }
        )
    fig.suptitle(
        f"{specimen}: stratified v5.7.1 tracking evidence\n"
        "First panel is unobscured raw data; colored one-pixel centerlines retain track identity.",
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
    args = parser.parse_args()
    pilot = json.loads(Path(args.pilot_manifest).read_text(encoding="utf-8"))
    inputs = {record["specimen"]: record["input_dir"] for record in pilot}
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
        destination = destination_root / f"{specimen}_stratified_tracking.png"
        records = render_specimen(
            specimen,
            output_dir,
            inputs[specimen],
            destination,
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
        "generator_sha256": sha256(Path(__file__)),
        "study_output": str(study_root.resolve()),
        "records": manifest,
    }
    (destination_root / "tracking_evidence_manifest.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
