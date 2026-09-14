"""Visual evidence for the raw-signal FWHM width measurement.

A reviewer cannot check a width by reading a number. These panels show the
mechanism on real objects: where the profile is sampled, what the raw intensity
does across the object, where the half-maximum crossings fall, how far the
segmentation mask boundary sits from them, and why a profile is refused when the
crossing is not observed inside the object's own mask.

Objects are chosen to span the behaviour rather than to look clean, including a
refused profile and a bimodal merge, because evidence curated to easy cases
cannot falsify anything.

Usage:
    python scripts/generate_v571_intensity_width_evidence.py --output <dir>
"""

import argparse
import hashlib
import importlib.util
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "sperm_segmentation_saturnv5.7.1.py"
PROFILE = ROOT / "production_profiles" / "saturn_v5_7_1_model_c_epoch003.json"
DEFAULT_MANIFEST = (
    ROOT / "scratch" / "aborted_v57_full_20260730_0630" / "study_manifest.csv"
)
CAVEAT = (
    "Relative comparison between groups is valid; absolute nucleus diameter is "
    "not established."
)


def load_pipeline():
    spec = importlib.util.spec_from_file_location("saturn_intensity_evidence", PIPELINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def segment_plane(saturn, row, z_index):
    """Segment one plane and return everything the panels need."""
    input_dir = Path(str(row["input_dir"]))
    files = sorted(input_dir.glob(str(row["file_pattern"])))
    files_by_z = {}
    for path in files:
        token = path.stem.split("_z")[-1].split("_")[0]
        try:
            files_by_z[int(token)] = str(path)
        except ValueError:
            continue
    cfg, _ = saturn.load_analysis_profile(PROFILE, saturn.CONFIG)
    ordered = [files_by_z[z] for z in sorted(files_by_z)]
    saturn.resolve_stack_microscope_calibration(cfg, ordered, input_dir=input_dir)
    first = saturn.ensure_2d_image(
        saturn.robust_imread(ordered[0]), Path(ordered[0]).name
    )
    roi = saturn.load_roi_mask_file(
        Path(str(row["roi_path"])), expected_shape=first.shape
    )
    context = saturn.build_stack_preprocess_context(ordered, roi, cfg)
    raw = saturn.ensure_2d_image(
        saturn.robust_imread(files_by_z[z_index]), Path(files_by_z[z_index]).name
    )
    seg = saturn.segment_slice(
        raw,
        cfg,
        z_idx=z_index,
        roi_mask=roi,
        preprocess_context=context,
        unet_context_stack=saturn._make_unet_context_from_paths(files_by_z, z_index),
    )
    measured = saturn.measure_spermatids(seg, cfg)
    return cfg, seg, measured, files_by_z[z_index]


def mid_profile(saturn, cfg, image, instance_mask, center_coords):
    """Recompute one representative profile so the panel shows the real samples."""
    path = saturn._resample_smoothed_centerline(
        center_coords,
        cfg.get("BODY_WIDTH_SAMPLE_SPACING_PX", 1.0),
        cfg.get("BODY_WIDTH_SMOOTH_SIGMA_PX", 1.0),
    )
    if path.shape[0] < 3:
        return None
    index = path.shape[0] // 2
    before = path[max(0, index - 2)]
    after = path[min(path.shape[0] - 1, index + 2)]
    tangent = after - before
    norm = float(np.linalg.norm(tangent))
    if norm <= 1e-9:
        return None
    tangent = tangent / norm
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
    half_extent = float(cfg.get("INTENSITY_WIDTH_PROFILE_HALF_EXTENT_PX", 8.0))
    step = float(cfg.get("INTENSITY_WIDTH_PROFILE_STEP_PX", 0.1))
    offsets = np.arange(-half_extent, half_extent + 1e-9, step)
    points = path[index][None, :] + offsets[:, None] * normal[None, :]
    rows = np.clip(points[:, 0], 0, image.shape[0] - 1)
    columns = np.clip(points[:, 1], 0, image.shape[1] - 1)
    sampled = ndi.map_coordinates(image, [rows, columns], order=1, mode="nearest")
    tail = float(cfg.get("INTENSITY_WIDTH_BACKGROUND_OFFSET_PX", 5.0))
    background = min(
        float(np.median(sampled[offsets <= -tail])),
        float(np.median(sampled[offsets >= tail])),
    )
    inside = ndi.map_coordinates(
        instance_mask.astype(np.uint8), [rows, columns], order=0, mode="constant"
    ).astype(bool)
    return {
        "offsets": offsets,
        "profile": sampled - background,
        "inside": inside,
        "centre": path[index],
        "normal": normal,
        "background": background,
    }


def render(saturn, cfg, category, specimen, z_index, result, seg, destination):
    """Three panels: the object, the profile that measures it, and the verdict."""
    um = float(cfg["UM_PER_PX_XY"])
    label = int(result["label"])
    labels = np.asarray(seg["unet_primary_instance_labels"])
    centers = np.asarray(seg["unet_primary_centerline_labels"])
    image = np.asarray(seg["img_linear"], dtype=float)
    mask = labels == label
    center_coords = np.argwhere(centers == label)
    if center_coords.size == 0:
        return None
    detail = mid_profile(saturn, cfg, image, mask, center_coords)
    if detail is None:
        return None

    ys, xs = np.nonzero(mask)
    pad = 14
    y0, y1 = max(0, ys.min() - pad), min(image.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(image.shape[1], xs.max() + pad + 1)
    view = np.s_[y0:y1, x0:x1]

    fwhm_um = float(result.get("intensity_fwhm_width_um", np.nan))
    mask_um = float(result.get("body_width_px", np.nan)) * um
    method = str(result.get("intensity_width_method", "unavailable"))

    figure, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)

    axes[0].imshow(image[view], cmap="gray")
    axes[0].contour(mask[view], levels=[0.5], colors="#00d4ff", linewidths=1.2)
    axes[0].plot(
        center_coords[:, 1] - x0, center_coords[:, 0] - y0, ".", color="#33cc33", ms=1.6
    )
    centre = detail["centre"]
    normal = detail["normal"]
    span = 8.0
    axes[0].plot(
        [centre[1] - span * normal[1] - x0, centre[1] + span * normal[1] - x0],
        [centre[0] - span * normal[0] - y0, centre[0] + span * normal[0] - y0],
        color="#ff8c00",
        lw=1.6,
    )
    axes[0].set_title("Raw signal, mask boundary (cyan),\ncenterline (green), sampled normal (orange)")
    axes[0].axis("off")

    offsets = detail["offsets"]
    profile = detail["profile"]
    inside = detail["inside"]
    axis = axes[1]
    axis.plot(offsets * um, profile, color="#222222", lw=1.4, label="raw signal - background")
    axis.fill_between(
        offsets * um, 0, profile, where=inside, color="#00d4ff", alpha=0.25,
        label="inside this instance mask",
    )
    owned = np.where(inside, profile, -np.inf)
    if np.isfinite(owned).any() and owned.max() > 0:
        half = owned.max() / 2.0
        axis.axhline(half, color="#cc0000", ls="--", lw=1.2, label="half maximum")
        crossings = np.flatnonzero(np.diff(np.sign(profile - half)) != 0)
        for c in crossings:
            axis.plot(offsets[c] * um, half, "o", color="#cc0000", ms=5)
    axis.axhline(0, color="#888888", lw=0.8)
    # Draw the mask extent measured on THIS normal, not the median chord over all
    # normals, so the two things on this axis are directly comparable. Use lines
    # rather than a second translucent band, because overlapping fills blend into
    # a third colour that means nothing.
    if inside.any():
        edges = offsets[inside]
        for position, label_text in (
            (edges.min() * um, "mask edge on this normal"),
            (edges.max() * um, None),
        ):
            axis.axvline(
                position, color="#ffb000", ls=":", lw=1.6,
                label=label_text,
            )
    axis.set_xlabel("distance across the object (um)")
    axis.set_ylabel("background-corrected signal")
    axis.set_title("Intensity profile along the normal")
    axis.legend(fontsize=7, loc="upper right")

    axes[2].axis("off")
    verdict = [
        f"specimen        {specimen}",
        f"plane           z{z_index:03d}   instance {label}",
        f"category        {category}",
        "",
        f"signal FWHM     {fwhm_um:.3f} um" if np.isfinite(fwhm_um) else "signal FWHM     unavailable",
        f"mask chord      {mask_um:.3f} um" if np.isfinite(mask_um) else "mask chord      unavailable",
    ]
    if np.isfinite(fwhm_um) and np.isfinite(mask_um) and fwhm_um > 0:
        verdict.append(f"mask / signal   {mask_um / fwhm_um:.2f}x")
    verdict += [
        "",
        f"method          {method}",
        f"merge flagged   {bool(result.get('intensity_profile_suspected_merge', False))}",
        "",
        "The mask boundary follows the training annotation",
        "convention. The half-maximum crossings follow the",
        "signal, so the two need not agree.",
        "",
        CAVEAT,
    ]
    axes[2].text(
        0.0, 1.0, "\n".join(verdict), va="top", ha="left", family="monospace",
        fontsize=10, transform=axes[2].transAxes,
    )
    figure.suptitle(
        f"{specimen} z{z_index:03d} instance {label} - {category.replace('_', ' ')}",
        fontsize=13, fontweight="bold",
    )
    figure.savefig(destination, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return {
        "category": category,
        "specimen": specimen,
        "z_index": int(z_index),
        "instance": label,
        "signal_fwhm_width_um": None if not np.isfinite(fwhm_um) else fwhm_um,
        "mask_chord_width_um": None if not np.isfinite(mask_um) else mask_um,
        "intensity_width_method": method,
        "suspected_merge": bool(result.get("intensity_profile_suspected_merge", False)),
        "artifact": destination.relative_to(ROOT).as_posix(),
        "artifact_sha256": sha256(destination),
    }


def choose(results, um):
    """One exemplar per behaviour, including the refusals."""
    frame = pd.DataFrame(
        [
            {
                "i": i,
                "fwhm": float(r.get("intensity_fwhm_width_um", np.nan)),
                "mask": float(r.get("body_width_px", np.nan)) * um,
                "method": str(r.get("intensity_width_method", "")),
                "merge": bool(r.get("intensity_profile_suspected_merge", False)),
            }
            for i, r in enumerate(results)
        ]
    )
    measured = frame[frame["fwhm"].notna()]
    picks = {}
    if not measured.empty:
        typical = measured.iloc[(measured["fwhm"] - measured["fwhm"].median()).abs().argsort()]
        picks["typical"] = int(typical.iloc[0]["i"])
        picks["narrowest_signal"] = int(measured.nsmallest(1, "fwhm").iloc[0]["i"])
        picks["widest_signal"] = int(measured.nlargest(1, "fwhm").iloc[0]["i"])
        widest_mask = measured.nlargest(1, "mask")
        if not widest_mask.empty:
            picks["widest_mask"] = int(widest_mask.iloc[0]["i"])
    merged = frame[frame["merge"]]
    if not merged.empty:
        picks["profile_merge"] = int(merged.iloc[0]["i"])
    clipped = frame[frame["method"] == "unavailable_boundary_clipped_profiles"]
    if not clipped.empty:
        picks["refused_boundary_clipped"] = int(clipped.iloc[0]["i"])
    short = frame[frame["method"] == "unavailable_short_centerline"]
    if not short.empty:
        picks["refused_short_centerline"] = int(short.iloc[0]["i"])
    return picks


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output", required=True)
    parser.add_argument("--plane", type=int, default=35)
    parser.add_argument("--specimens", type=int, default=2)
    args = parser.parse_args(argv)

    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit("refusing to overwrite existing evidence directory: " + str(output))
    output.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    if "include" in manifest.columns:
        manifest = manifest[manifest["include"].astype(str).str.lower() == "true"]
    chosen = pd.concat(
        [block.head(max(1, args.specimens // 2)) for _, block in manifest.groupby("group")]
    )

    saturn = load_pipeline()
    records = []
    unrenderable = []
    for _, row in chosen.iterrows():
        specimen = str(row["sample_id"])
        cfg, seg, measured, _ = segment_plane(saturn, row, args.plane)
        um = float(cfg["UM_PER_PX_XY"])
        results = measured["results"]
        picks = choose(results, um)
        print(f"  {specimen}: {len(results)} instances, categories {sorted(picks)}", flush=True)
        for category, index in sorted(picks.items()):
            destination = output / f"{specimen}_{category}.png"
            record = render(
                saturn, cfg, category, specimen, args.plane, results[index], seg, destination
            )
            if record is not None:
                records.append(record)
            else:
                # A centerline too short to resample cannot produce a profile
                # panel. That is the very reason the object is refused, so record
                # it rather than letting the category vanish from the evidence.
                unrenderable.append(
                    {
                        "specimen": specimen,
                        "category": category,
                        "reason": "centerline too short to resample a profile",
                        "intensity_width_method": str(
                            results[index].get("intensity_width_method", "")
                        ),
                    }
                )

    manifest_payload = {
        "schema_version": "1.0",
        "claim_id": "MEAS-INTENSITY-WIDTH-001",
        "role": "visual_evidence",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "pipeline_sha256": sha256(PIPELINE),
        "profile_sha256": sha256(PROFILE),
        "plane": int(args.plane),
        "interpretation_limit": CAVEAT,
        "selection_note": (
            "One exemplar per behaviour, deliberately including refused profiles "
            "and a bimodal merge. Evidence curated to clean objects cannot "
            "falsify the measurement."
        ),
        "panels": records,
        "categories_without_a_panel": unrenderable,
    }
    (output / "intensity_width_visual_evidence_manifest.json").write_text(
        json.dumps(manifest_payload, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {len(records)} panels to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
