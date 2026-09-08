"""Generate object-level evidence for the v5.7.1 apparent body-width claim."""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.segmentation import find_boundaries


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "sperm_segmentation_saturnv5.7.1.py"
PROFILE = ROOT / "production_profiles/saturn_v5_7_1_model_c_epoch003.json"
EVIDENCE_ROOT = ROOT / "audits/evidence/v571_rc6_candidate"
REPLAY = EVIDENCE_ROOT / "provenance/tracking_replay_inputs_outputs.zip"

SPECIMENS = {
    "KJ-01": "kj_sv_40xx0.75-1",
    "WT-01": "w1118_sv_feb_40xx0.75-1",
}


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
    spec = importlib.util.spec_from_file_location("saturn_v571_width_evidence", PIPELINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def retained_settings_dir(stem):
    return EVIDENCE_ROOT / "provenance/specimens" / stem / "settings"


def load_source_files(stem):
    payload = json.loads(
        (retained_settings_dir(stem) / "source_image_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    records = payload["ordered_source_images"]
    files_by_z = {int(row["z_index"]): str(Path(row["path"])) for row in records}
    for z_index, path in files_by_z.items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing source image z{z_index}: {path}")
        if int(next(row["channel"] for row in records if int(row["z_index"]) == z_index)) != 0:
            raise ValueError(f"Nonzero source channel at z{z_index}")
    return files_by_z


def read_replay_table(archive, stem, name):
    with archive.open(f"{stem}/{name}.csv") as handle:
        return pd.read_csv(handle)


def representative_plane_detections(track_summary, detections):
    """Join each track to the detection on its own representative plane."""
    representative = detections.copy()
    representative["z_slice"] = pd.to_numeric(
        representative["z_slice"], errors="coerce"
    )
    representative = representative.merge(
        track_summary[["track_id", "representative_width_z"]],
        on="track_id",
        how="inner",
    )
    return representative[
        representative["z_slice"]
        == pd.to_numeric(representative["representative_width_z"], errors="coerce")
    ]


def select_stratified_tracks(track_summary, detections, target_z=35):
    """Choose one exemplar per morphology and failure category.

    Curating only well-behaved objects makes visual evidence unfalsifiable: a
    reviewer cannot see how the measurement behaves where it is hardest. Each
    category below is selected independently, and a category with no eligible
    track is reported as absent rather than quietly skipped.

    Returns:
        list[tuple[str, pandas.Series]]: (category, track row) pairs.
    """
    tracks = track_summary.copy()
    numeric = {
        "n_slices": pd.to_numeric(tracks.get("n_slices"), errors="coerce"),
        "width": pd.to_numeric(
            tracks.get("representative_body_width_um"), errors="coerce"
        ),
        "z": pd.to_numeric(tracks.get("representative_width_z"), errors="coerce"),
    }
    tracks["_z_distance"] = (numeric["z"] - target_z).abs()
    merged_flag = tracks.get("suspected_multi_object_merge")
    merged_flag = (
        merged_flag.fillna(False).astype(bool)
        if merged_flag is not None
        else pd.Series(False, index=tracks.index)
    )

    representative = representative_plane_detections(tracks, detections)
    branch = pd.to_numeric(
        representative.get("n_branch_nodes"), errors="coerce"
    )
    warning = representative.get("morphology_warning")
    warning = (
        warning.fillna(False).astype(bool)
        if warning is not None
        else pd.Series(False, index=representative.index)
    )
    branched_ids = set(representative.loc[branch > 0, "track_id"])
    warned_ids = set(representative.loc[warning, "track_id"])
    clean_ids = set(
        representative.loc[
            (branch == 0)
            & (
                pd.to_numeric(
                    representative["instance_mask_area_px"], errors="coerce"
                ).between(60, 180)
            )
            & (
                pd.to_numeric(
                    representative["length_body_width_ratio"], errors="coerce"
                )
                > 3.0
            )
            & ~warning,
            "track_id",
        ]
    )

    measurable = numeric["width"].notna()
    multi_slice = numeric["n_slices"] >= 3

    def nearest(mask, order_by=None, ascending=True):
        subset = tracks[mask].copy()
        if subset.empty:
            return None
        columns = ([order_by] if order_by else []) + ["_z_distance", "track_id"]
        directions = ([ascending] if order_by else []) + [True, True]
        return subset.sort_values(
            columns, ascending=directions, kind="mergesort"
        ).iloc[0]

    categories = [
        (
            "clean",
            measurable & multi_slice & ~merged_flag & tracks["track_id"].isin(clean_ids),
            None,
            True,
        ),
        (
            "morphology_warning",
            measurable & tracks["track_id"].isin(warned_ids),
            None,
            True,
        ),
        (
            "branched_centerline",
            measurable & tracks["track_id"].isin(branched_ids),
            None,
            True,
        ),
        ("suspected_merge", merged_flag, None, True),
        ("width_unavailable", ~measurable, None, True),
        ("short_track", measurable & (numeric["n_slices"] < 3), None, True),
        ("narrowest_width", measurable, "representative_body_width_um", True),
        ("widest_width", measurable, "representative_body_width_um", False),
    ]

    selected = []
    seen = set()
    for name, mask, order_by, ascending in categories:
        row = nearest(mask, order_by, ascending)
        if row is None:
            selected.append((name, None))
            continue
        # Allow the same track to illustrate two categories only if nothing else
        # qualifies, so the panel set stays genuinely varied.
        if int(row["track_id"]) in seen:
            alternative = nearest(
                mask & ~tracks["track_id"].isin(seen), order_by, ascending
            )
            if alternative is not None:
                row = alternative
        seen.add(int(row["track_id"]))
        selected.append((name, row))
    return selected


def select_track(track_summary, detections, target_z=35):
    frame = track_summary.copy()
    frame = frame[
        (pd.to_numeric(frame["n_slices"], errors="coerce") >= 3)
        & pd.to_numeric(frame["representative_body_width_um"], errors="coerce").notna()
        & ~frame["suspected_multi_object_merge"].fillna(False).astype(bool)
    ].copy()
    frame["_z_distance"] = (
        pd.to_numeric(frame["representative_width_z"], errors="coerce") - target_z
    ).abs()
    representative = detections.copy()
    representative["z_slice"] = pd.to_numeric(
        representative["z_slice"], errors="coerce"
    )
    representative = representative.merge(
        frame[["track_id", "representative_width_z"]], on="track_id", how="inner"
    )
    representative = representative[
        representative["z_slice"]
        == pd.to_numeric(representative["representative_width_z"], errors="coerce")
    ]
    clean_track_ids = set(
        representative.loc[
            (pd.to_numeric(representative["n_branch_nodes"], errors="coerce") == 0)
            & (pd.to_numeric(representative["instance_mask_area_px"], errors="coerce").between(60, 180))
            & (pd.to_numeric(representative["length_body_width_ratio"], errors="coerce") > 3.0)
            & ~representative["morphology_warning"].fillna(False).astype(bool),
            "track_id",
        ]
    )
    frame = frame[frame["track_id"].isin(clean_track_ids)].copy()
    if frame.empty:
        raise ValueError("No eligible unbranched representative-width track")
    return frame.sort_values(
        ["_z_distance", "track_id"],
        ascending=[True, True],
        kind="mergesort",
    ).iloc[0]


def chord_geometry(saturn, instance_mask, center_coords, cfg):
    path = saturn._resample_smoothed_centerline(
        center_coords,
        cfg["BODY_WIDTH_SAMPLE_SPACING_PX"],
        cfg["BODY_WIDTH_SMOOTH_SIGMA_PX"],
    )
    contour = saturn._subpixel_instance_contour(instance_mask)
    if path.shape[0] < 3 or contour.shape[0] < 4:
        return {"path": path, "contour": contour, "trimmed": path, "accepted": [], "rejected": []}
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(steps)])
    total = float(cumulative[-1])
    trim = float(cfg["BODY_WIDTH_ENDPOINT_TRIM_FRACTION"])
    eligible = (
        (cumulative >= trim * total)
        & (cumulative <= (1.0 - trim) * total)
    )
    points = []
    normals = []
    for index in np.flatnonzero(eligible):
        before = path[max(0, index - 2)]
        after = path[min(path.shape[0] - 1, index + 2)]
        tangent = after - before
        norm = float(np.linalg.norm(tangent))
        if norm > 1e-9:
            tangent /= norm
            points.append(path[index])
            normals.append(np.array([-tangent[1], tangent[0]], dtype=float))
    accepted = []
    rejected = []
    starts = contour[:-1].astype(float)
    segments = np.diff(contour, axis=0).astype(float)
    for point, normal in zip(points, normals):
        denominator = normal[0] * segments[:, 1] - normal[1] * segments[:, 0]
        usable = np.abs(denominator) >= 1e-10
        safe = np.where(usable, denominator, 1.0)
        offset = starts - point
        t_values = (offset[:, 0] * segments[:, 1] - offset[:, 1] * segments[:, 0]) / safe
        u_values = (offset[:, 0] * normal[1] - offset[:, 1] * normal[0]) / safe
        intersects = usable & (u_values >= -1e-9) & (u_values <= 1.0 + 1e-9)
        negative = t_values[intersects & (t_values <= 1e-7)]
        positive = t_values[intersects & (t_values >= -1e-7)]
        if negative.size and positive.size:
            low = float(np.max(negative))
            high = float(np.min(positive))
            if np.isfinite(high - low) and high > low:
                accepted.append(
                    {
                        "start": point + low * normal,
                        "end": point + high * normal,
                        "width_px": high - low,
                    }
                )
                continue
        rejected.append(point)
    return {
        "path": path,
        "contour": contour,
        "trimmed": path[~eligible],
        "accepted": accepted,
        "rejected": np.asarray(rejected),
    }


def crop_bounds(observations, margin, shape):
    y0 = max(0, int(observations["bbox_min_y"].min()) - margin)
    x0 = max(0, int(observations["bbox_min_x"].min()) - margin)
    y1 = min(shape[0], int(observations["bbox_max_y"].max()) + margin)
    x1 = min(shape[1], int(observations["bbox_max_x"].max()) + margin)
    return y0, y1, x0, x1


def display_raw(axis, raw, crop):
    values = np.asarray(raw)[crop]
    finite = values[np.isfinite(values)]
    low, high = np.percentile(finite, [1, 99.7]) if finite.size else (0, 1)
    axis.imshow(values, cmap="gray", vmin=low, vmax=max(high, low + 1e-9))


def draw_geometry(axis, geometry, crop, show_trimmed=True):
    y0, _y1, x0, _x1 = crop
    contour = geometry["contour"]
    if contour.size:
        axis.plot(contour[:, 1] - x0, contour[:, 0] - y0, color="#00e5ff", lw=1.3)
    path = geometry["path"]
    if path.size:
        axis.plot(path[:, 1] - x0, path[:, 0] - y0, color="#33cc33", lw=1.2)
    if show_trimmed and geometry["trimmed"].size:
        axis.scatter(
            geometry["trimmed"][:, 1] - x0,
            geometry["trimmed"][:, 0] - y0,
            s=10,
            color="#bbbbbb",
            marker="x",
            label="trimmed endpoint samples",
        )
    for chord in geometry["accepted"]:
        axis.plot(
            [chord["start"][1] - x0, chord["end"][1] - x0],
            [chord["start"][0] - y0, chord["end"][0] - y0],
            color="#ffb000",
            lw=0.8,
            alpha=0.8,
        )
    rejected = geometry["rejected"]
    if rejected.size:
        axis.scatter(
            rejected[:, 1] - x0,
            rejected[:, 0] - y0,
            s=18,
            color="red",
            marker="x",
            label="rejected chord samples",
        )
    if geometry["accepted"]:
        median = float(np.median([item["width_px"] for item in geometry["accepted"]]))
        chosen = min(geometry["accepted"], key=lambda item: abs(item["width_px"] - median))
        axis.plot(
            [chosen["start"][1] - x0, chosen["end"][1] - x0],
            [chosen["start"][0] - y0, chosen["end"][0] - y0],
            color="#ff2d55",
            lw=2.5,
            label="median chord",
        )


def render_selected(specimen, track_id, observation, data, cfg, destination, category="clean"):
    raw = data["raw"]
    seg = data["segmentation"]
    label = int(observation["sperm_id"])
    mask = np.asarray(seg["unet_primary_instance_labels"]) == label
    center = np.argwhere(np.asarray(seg["unet_primary_centerline_labels"]) == label)
    geometry = data["geometry"]
    y0, y1, x0, x1 = data["crop"]
    crop = np.s_[y0:y1, x0:x1]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    display_raw(axes[0, 0], raw, crop)
    axes[0, 0].set_title("Raw image")
    axes[0, 1].imshow(np.asarray(seg["unet_probability"])[crop], cmap="magma", vmin=0, vmax=1)
    axes[0, 1].set_title("Foreground probability")
    axes[0, 2].imshow(mask[crop], cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title("Filled target mask")
    axes[1, 0].imshow(find_boundaries(mask, mode="inner")[crop], cmap="gray", vmin=0, vmax=1)
    axes[1, 0].plot(center[:, 1] - x0, center[:, 0] - y0, color="#33cc33", lw=1.5)
    axes[1, 0].set_title("Boundary and ordered centerline")
    axes[1, 1].imshow(mask[crop], cmap="gray", vmin=0, vmax=1)
    draw_geometry(axes[1, 1], geometry, (y0, y1, x0, x1))
    axes[1, 1].set_title("Central-body chord samples")
    display_raw(axes[1, 2], raw, crop)
    draw_geometry(axes[1, 2], geometry, (y0, y1, x0, x1), show_trimmed=False)
    axes[1, 2].set_title("Measurement on raw image")
    for axis in axes.ravel():
        axis.axis("off")
    width = pd.to_numeric(
        pd.Series([observation["body_width_um"]]), errors="coerce"
    ).iloc[0]
    width_text = (
        "apparent body width unavailable"
        if pd.isna(width)
        else f"apparent body width {float(width):.3f} um"
    )
    fig.suptitle(
        f"{specimen}, {category.replace('_', ' ')}, track {track_id}, "
        f"z{int(observation['z_slice']):03d}: {width_text}\n"
        "cyan boundary; green centerline; orange accepted chords; red median chord; gray x trimmed endpoints",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_plane_selection(specimen, track_id, observations, plane_data, destination):
    first = next(iter(plane_data.values()))
    crop = first["crop"]
    y0, y1, x0, x1 = crop
    view = np.s_[y0:y1, x0:x1]
    fig, axes = plt.subplots(len(observations), 3, figsize=(11, 3.2 * len(observations)), constrained_layout=True)
    if len(observations) == 1:
        axes = np.asarray([axes])
    areas = pd.to_numeric(observations["instance_mask_area_px"], errors="coerce")
    # A degenerate object can carry no usable area, so fall back to the lowest
    # observed plane rather than failing the whole evidence run.
    selected_z = int(
        observations.loc[areas.idxmax(), "z_slice"]
        if areas.notna().any()
        else observations["z_slice"].min()
    )
    for row_index, observation in enumerate(observations.itertuples(index=False)):
        data = plane_data[int(observation.z_slice)]
        label = int(observation.sperm_id)
        mask = np.asarray(data["segmentation"]["unet_primary_instance_labels"]) == label
        display_raw(axes[row_index, 0], data["raw"], view)
        axes[row_index, 0].set_title(f"z{int(observation.z_slice):03d} raw")
        axes[row_index, 1].imshow(mask[view], cmap="gray", vmin=0, vmax=1)
        draw_geometry(axes[row_index, 1], data["geometry"], crop)
        axes[row_index, 1].set_title(
            f"area {observation.instance_mask_area_px:.0f} px2; width {observation.body_width_um:.3f} um"
        )
        display_raw(axes[row_index, 2], data["raw"], view)
        draw_geometry(axes[row_index, 2], data["geometry"], crop, show_trimmed=False)
        marker = "SELECTED representative plane" if int(observation.z_slice) == selected_z else "observed plane"
        axes[row_index, 2].set_title(marker)
        for axis in axes[row_index]:
            axis.axis("off")
    fig.suptitle(
        f"{specimen}, persistent track {track_id}: largest filled-mask area selects representative width",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return selected_z


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-z", type=int, default=35)
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    saturn = load_pipeline()
    records = []
    with zipfile.ZipFile(REPLAY) as archive:
        for specimen, stem in SPECIMENS.items():
            files_by_z = load_source_files(stem)
            files = [files_by_z[z] for z in sorted(files_by_z)]
            input_dir = Path(files[0]).parent
            roi_path = retained_settings_dir(stem) / "roi_mask_source.npy"
            cfg, _ = saturn.load_analysis_profile(PROFILE, saturn.CONFIG)
            saturn.resolve_stack_microscope_calibration(cfg, files, input_dir=input_dir)
            raw0 = saturn.ensure_2d_image(saturn.robust_imread(files[0]), Path(files[0]).name)
            roi = saturn.load_roi_mask_file(roi_path, expected_shape=raw0.shape)
            context = saturn.build_stack_preprocess_context(files, roi, cfg)
            tracks = read_replay_table(archive, stem, "track_summary")
            detections = read_replay_table(archive, stem, "tracked_detections")
            for category, selected_track in select_stratified_tracks(
                tracks, detections, args.target_z
            ):
                if selected_track is None:
                    records.append(
                        {
                            "specimen": specimen,
                            "source_stem": stem,
                            "category": category,
                            "status": "absent",
                            "note": "no track in this specimen matched the category",
                        }
                    )
                    continue
                track_id = int(selected_track["track_id"])
                observations = detections[detections["track_id"] == track_id].copy()
                observations["z_slice"] = pd.to_numeric(
                    observations["z_slice"], errors="coerce"
                )
                observations = observations.dropna(subset=["z_slice"])
                if observations.empty:
                    records.append(
                        {
                            "specimen": specimen,
                            "source_stem": stem,
                            "category": category,
                            "track_id": track_id,
                            "status": "no_observations",
                        }
                    )
                    continue
                # A track whose width is unavailable has no representative plane,
                # so anchor the view on its observed planes instead.
                raw_target = pd.to_numeric(
                    pd.Series([selected_track.get("representative_width_z")]),
                    errors="coerce",
                ).iloc[0]
                if pd.isna(raw_target):
                    target_z = int(observations["z_slice"].iloc[0])
                    representative_known = False
                else:
                    target_z = int(raw_target)
                    representative_known = True
                observations = observations[
                    observations["z_slice"].isin([target_z - 1, target_z, target_z + 1])
                ].sort_values("z_slice")
                if observations.empty:
                    records.append(
                        {
                            "specimen": specimen,
                            "source_stem": stem,
                            "category": category,
                            "track_id": track_id,
                            "status": "no_adjacent_planes",
                        }
                    )
                    continue
                crop = crop_bounds(observations, margin=18, shape=raw0.shape)
                plane_data = {}
                usable = []
                chord_matches_production = True
                for observation in observations.itertuples(index=False):
                    z_index = int(observation.z_slice)
                    raw = saturn.ensure_2d_image(
                        saturn.robust_imread(files_by_z[z_index]),
                        Path(files_by_z[z_index]).name,
                    )
                    segmentation = saturn.segment_slice(
                        raw,
                        cfg,
                        z_idx=z_index,
                        roi_mask=roi,
                        preprocess_context=context,
                        unet_context_stack=saturn._make_unet_context_from_paths(
                            files_by_z, z_index
                        ),
                    )
                    measured = saturn.measure_spermatids(segmentation, cfg)
                    label = int(observation.sperm_id)
                    result = next(
                        (row for row in measured["results"] if int(row["label"]) == label),
                        None,
                    )
                    if result is None:
                        # A difficult object may not reproduce under fresh
                        # segmentation. That is itself evidence, so record it
                        # rather than aborting the whole evidence run.
                        continue
                    mask = np.asarray(segmentation["unet_primary_instance_labels"]) == label
                    center = np.argwhere(
                        np.asarray(segmentation["unet_primary_centerline_labels"]) == label
                    )
                    geometry = chord_geometry(saturn, mask, center, cfg)
                    accepted = [item["width_px"] for item in geometry["accepted"]]
                    if accepted and np.isfinite(float(result["body_width_px"])):
                        chord_median_px = float(np.median(accepted))
                        if not np.isclose(
                            chord_median_px, float(result["body_width_px"]), atol=1e-7
                        ):
                            chord_matches_production = False
                    plane_data[z_index] = {
                        "raw": raw,
                        "segmentation": segmentation,
                        "geometry": geometry,
                        "crop": crop,
                    }
                    usable.append(z_index)
                if not usable:
                    records.append(
                        {
                            "specimen": specimen,
                            "source_stem": stem,
                            "category": category,
                            "track_id": track_id,
                            "status": "not_reproduced_by_fresh_segmentation",
                        }
                    )
                    continue
                if not chord_matches_production:
                    raise ValueError(
                        "Displayed chord median does not match production measurement "
                        f"for {specimen} track {track_id}"
                    )
                observations = observations[observations["z_slice"].isin(usable)]
                anchor_z = target_z if target_z in plane_data else usable[0]
                anchor_observation = observations[
                    observations["z_slice"] == anchor_z
                ].iloc[0]
                prefix = f"{specimen}_{category}_track{track_id}"
                selected_path = output_dir / f"{prefix}_selected_width.png"
                render_selected(
                    specimen,
                    track_id,
                    anchor_observation,
                    plane_data[anchor_z],
                    cfg,
                    selected_path,
                    category=category,
                )
                planes_path = output_dir / f"{prefix}_representative_plane.png"
                visually_selected_z = render_plane_selection(
                    specimen, track_id, observations, plane_data, planes_path
                )
                if (
                    category == "clean"
                    and representative_known
                    and visually_selected_z != target_z
                ):
                    raise ValueError(
                        "Visual largest-area selection disagrees with retained "
                        "representative Z"
                    )
                width_value = pd.to_numeric(
                    pd.Series([selected_track.get("representative_body_width_um")]),
                    errors="coerce",
                ).iloc[0]
                records.append(
                    {
                        "specimen": specimen,
                        "source_stem": stem,
                        "category": category,
                        "status": "rendered",
                        "track_id": track_id,
                        "displayed_z_indices": [int(value) for value in observations["z_slice"]],
                        "representative_width_z": target_z if representative_known else None,
                        "representative_width_known": representative_known,
                        "visual_largest_area_z": int(visually_selected_z),
                        "representative_body_width_um": (
                            None if pd.isna(width_value) else float(width_value)
                        ),
                        "selection_method": str(
                            selected_track.get("representative_width_selection", "")
                        ),
                        "selected_width_artifact_repository_path": selected_path.relative_to(ROOT).as_posix(),
                        "selected_width_artifact_sha256": sha256(selected_path),
                        "representative_plane_artifact_repository_path": planes_path.relative_to(ROOT).as_posix(),
                        "representative_plane_artifact_sha256": sha256(planes_path),
                        "roi_repository_path": roi_path.relative_to(ROOT).as_posix(),
                        "roi_sha256": sha256(roi_path),
                        "xy_um_per_pixel": float(cfg["UM_PER_PX_XY"]),
                        "z_um_per_slice": float(cfg["UM_PER_SLICE_Z"]),
                        "source_images": [
                            {
                                "z_index": int(z),
                                "path": files_by_z[int(z)],
                                "sha256": sha256(files_by_z[int(z)]),
                            }
                            for z in observations["z_slice"]
                        ],
                    }
                )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {
        "schema_version": "1.0",
        "claim_id": "MEAS-BODY-WIDTH-001",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_at_generation": commit,
        "pipeline_git_blob_sha256": git_blob_sha256(PIPELINE, commit),
        "profile_git_blob_sha256": git_blob_sha256(PROFILE, commit),
        "generator_git_blob_sha256": git_blob_sha256(Path(__file__), commit),
        "checkpoint_sha256": sha256(ROOT / "model_checkpoints/v571_model_c_dual_head_epoch003.pt"),
        "replay_archive_sha256": sha256(REPLAY),
        "measurement_scope": "apparent filled-mask body width; not PSF-corrected chromatin diameter",
        "records": records,
    }
    (output_dir / "body_width_visual_evidence_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
