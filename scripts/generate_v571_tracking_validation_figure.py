"""Create four-plane visual audits for revised v5.7.1 tracking."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "scratch" / "v571_dual_head_full_pilot"
REPLAY = ROOT / "scratch" / "v571_tracking_replay"
OUTPUT = ROOT / "scratch" / "v571_tracking_visual_audit"
Z_PLANES = (34, 35, 36, 37)


def _files_by_z(folder: Path):
    result = {}
    for path in folder.glob("*.tif*"):
        match = re.search(r"_z(\d+)_ch00\.tif{1,2}$", path.name, re.I)
        if match:
            result[int(match.group(1))] = path
    return result


def _select_tracks(df: pd.DataFrame):
    present = (
        df[df["z_slice"].isin(Z_PLANES)]
        .groupby("track_id")["z_slice"]
        .nunique()
    )
    ids = present[present == len(Z_PLANES)].index
    candidates = df[(df["z_slice"] == Z_PLANES[1]) & df["track_id"].isin(ids)].copy()
    candidates = candidates[
        candidates["length_um_geodesic"].between(7.0, 13.0)
        & (candidates["length_body_width_ratio"] >= 4.0)
    ].sort_values(
        ["unet_mean_probability", "length_body_width_ratio"],
        ascending=False,
    )
    if len(candidates) < 3:
        raise RuntimeError("Fewer than three four-plane tracks are available")

    anchor = candidates.iloc[0]
    delta = np.hypot(
        candidates["centroid_x"] - anchor["centroid_x"],
        candidates["centroid_y"] - anchor["centroid_y"],
    )
    nearby = candidates[(delta >= 20.0) & (delta <= 220.0)]
    selected = [anchor]
    for _, row in nearby.iterrows():
        if all(
            np.hypot(
                row["centroid_x"] - other["centroid_x"],
                row["centroid_y"] - other["centroid_y"],
            ) >= 20.0
            for other in selected
        ):
            selected.append(row)
        if len(selected) == 3:
            break
    if len(selected) < 3:
        selected = [row for _, row in candidates.iloc[:3].iterrows()]
    return [int(row["track_id"]) for row in selected]


def _display_range(images):
    values = np.concatenate([image.ravel()[::16] for image in images])
    return tuple(np.percentile(values, (1.0, 99.7)))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        row["specimen"]: row
        for row in json.loads((PILOT / "pilot_manifest.json").read_text(encoding="utf-8"))
    }
    colors = ("#e31a1c", "#1f78b4", "#18a558")
    markers = ("o", "s", "^")

    for specimen in ("KJ-01", "WT-01"):
        tracked = pd.read_csv(
            REPLAY / f"{specimen}_production_morphology_neutral_tracked.csv"
        )
        track_ids = _select_tracks(tracked)
        source_files = _files_by_z(Path(manifest[specimen]["input_dir"]))
        images = [np.asarray(tifffile.imread(source_files[z]), dtype=float) for z in Z_PLANES]
        vmin, vmax = _display_range(images)

        selected = tracked[tracked["track_id"].isin(track_ids)]
        x0 = max(0, int(selected["centroid_x"].min()) - 45)
        x1 = min(images[0].shape[1], int(selected["centroid_x"].max()) + 46)
        y0 = max(0, int(selected["centroid_y"].min()) - 45)
        y1 = min(images[0].shape[0], int(selected["centroid_y"].max()) + 46)

        fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
        fig.suptitle(
            f"{specimen}: same three revised tracks across four optical planes",
            fontsize=15,
            fontweight="bold",
        )
        for ax, z, image in zip(axes.ravel(), Z_PLANES, images):
            ax.imshow(image[y0:y1, x0:x1], cmap="gray", vmin=vmin, vmax=vmax)
            for label, (track_id, color, marker) in enumerate(
                zip(track_ids, colors, markers), start=1
            ):
                row = selected[
                    (selected["track_id"] == track_id) & (selected["z_slice"] == z)
                ].iloc[0]
                ax.scatter(
                    row["centroid_x"] - x0,
                    row["centroid_y"] - y0,
                    s=95,
                    marker=marker,
                    facecolors="none",
                    edgecolors=color,
                    linewidths=2.2,
                    label=f"Nucleus {chr(64 + label)}" if z == Z_PLANES[0] else None,
                )
            ax.set_title(f"z{z:02d}")
            ax.set_axis_off()
        axes.ravel()[0].legend(loc="lower left", fontsize=8, framealpha=0.8)
        fig.text(
            0.5,
            0.01,
            "Colors and symbols identify the same track in every plane; morphology changes are not link vetoes.",
            ha="center",
            fontsize=9,
        )
        path = OUTPUT / f"{specimen}_four_plane_tracking_audit.png"
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(path)


if __name__ == "__main__":
    main()
