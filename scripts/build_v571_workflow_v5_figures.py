"""Build the figures that showcase the v5.7.1 measurement work.

Every number plotted here was measured during the 2026-09 work and is recorded in
the audit evidence or the findings under `audits/`. Nothing is illustrative or
invented; where a figure shows a limit rather than a result, it says so.

Usage:
    python scripts/build_v571_workflow_v5_figures.py --output docs/v5_7_illustrated_workflow/figures_v5
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]

INK = "#1f2933"
MUTED = "#6b7280"
MASK = "#d97706"
SIGNAL = "#0f766e"
ACCENT = "#1f4e79"
WARN = "#b91c1c"
GRID = "#e5e7eb"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": "#9ca3af",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def finish(fig, path, caption=None, caption_y=-0.04):
    if caption:
        # Below the axes rather than on top of them; bbox_inches="tight" grows
        # the canvas to include it.
        fig.text(0.5, caption_y, caption, fontsize=7.5, color=MUTED,
                 ha="center", va="top", wrap=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _kj_plane(plane=35, group="KJ"):
    """Segment one plane of the study data and hand back everything needed."""
    import importlib.util

    import pandas as pd

    spec = importlib.util.spec_from_file_location(
        "v5_workflow_evidence",
        ROOT / "scripts" / "generate_v571_intensity_width_evidence.py",
    )
    evidence = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evidence)
    saturn = evidence.load_pipeline()
    manifest = pd.read_csv(evidence.DEFAULT_MANIFEST)
    manifest = manifest[manifest["include"].astype(str).str.lower() == "true"]
    row = manifest[manifest["group"].astype(str) == group].iloc[0]
    cfg, seg, measured, path = evidence.segment_plane(saturn, row, plane)
    return evidence, saturn, row, cfg, seg, measured


def fig_region_across_slices(out):
    """The same drawn region applied to consecutive slices of the study data."""
    import importlib.util

    import pandas as pd

    spec = importlib.util.spec_from_file_location(
        "v5_region_evidence",
        ROOT / "scripts" / "generate_v571_intensity_width_evidence.py",
    )
    evidence = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evidence)
    saturn = evidence.load_pipeline()
    manifest = pd.read_csv(evidence.DEFAULT_MANIFEST)
    manifest = manifest[manifest["include"].astype(str).str.lower() == "true"]
    row = manifest[manifest["group"].astype(str) == "KJ"].iloc[0]

    input_dir = Path(str(row["input_dir"]))
    files_by_z = {}
    for candidate in sorted(input_dir.glob(str(row["file_pattern"]))):
        token = candidate.stem.split("_z")[-1].split("_")[0]
        try:
            files_by_z[int(token)] = str(candidate)
        except ValueError:
            continue
    cfg, _ = saturn.load_analysis_profile(
        ROOT / "production_profiles" / "saturn_v5_7_1_model_c_epoch003.json",
        saturn.CONFIG,
    )
    ordered = [files_by_z[z] for z in sorted(files_by_z)]
    saturn.resolve_stack_microscope_calibration(cfg, ordered, input_dir=input_dir)
    um = float(cfg["UM_PER_PX_XY"])
    first = saturn.ensure_2d_image(
        saturn.robust_imread(ordered[0]), Path(ordered[0]).name
    )
    roi = saturn.load_roi_mask_file(Path(str(row["roi_path"])), expected_shape=first.shape)

    planes = [34, 35, 36, 37]
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.7), constrained_layout=True)
    for axis, z in zip(axes, planes):
        raw = saturn.ensure_2d_image(
            saturn.robust_imread(files_by_z[z]), Path(files_by_z[z]).name
        )
        axis.imshow(np.asarray(raw, dtype=float), cmap="gray")
        axis.contour(roi.astype(float), levels=[0.5], colors="#dc2626", linewidths=1.3)
        axis.set_title("slice {}".format(z), fontsize=10, color=ACCENT)
        axis.axis("off")
    bar = 20.0 / um
    axes[0].plot([30, 30 + bar], [first.shape[0] - 40] * 2, color="white", lw=3,
                 solid_capstyle="butt")
    axes[0].text(30 + bar / 2, first.shape[0] - 60, "20 um", color="white",
                 fontsize=9, ha="center")
    fig.suptitle(
        "The same drawn region applied to four consecutive slices",
        fontsize=12, color=ACCENT, fontweight="bold",
    )
    return finish(
        fig, out / "v5_fig10_region_across_slices.png",
        "Specimen KJ-01. Pixels outside the red boundary take no part in the "
        "analysis, and the same boundary is used on every slice.",
    )


def fig_processing_stages(out):
    """What happens to one slice before anything is detected."""
    evidence, saturn, row, cfg, seg, measured = _kj_plane(35)
    y0, y1, x0, x1 = 300, 560, 300, 560
    view = np.s_[y0:y1, x0:x1]
    panels = [
        ("as acquired", seg.get("img_linear"), "gray"),
        ("after normalising", seg.get("img_norm"), "gray"),
        ("after denoising", seg.get("img_denoised"), "gray"),
        ("background removed", seg.get("foreground"), "gray"),
        ("network confidence", seg.get("unet_probability"), "magma"),
        ("nuclei it found", seg.get("unet_primary_instance_labels"), "nipy_spectral"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 7.6), constrained_layout=True)
    for axis, (label, data, cmap) in zip(axes.ravel(), panels):
        if data is None:
            axis.axis("off")
            continue
        array = np.asarray(data, dtype=float)[view]
        if label == "nuclei it found":
            array = np.where(array > 0, array % 19 + 1, 0)
        axis.imshow(array, cmap=cmap)
        axis.set_title(label, fontsize=10, color=ACCENT)
        axis.axis("off")
    um = float(cfg["UM_PER_PX_XY"])
    bar = 10.0 / um
    axes[0, 0].plot([8, 8 + bar], [(y1 - y0) - 10] * 2, color="white", lw=3,
                    solid_capstyle="butt")
    axes[0, 0].text(8 + bar / 2, (y1 - y0) - 14, "10 um", color="white",
                    fontsize=9, ha="center", va="bottom")
    fig.suptitle("From raw slice to found nuclei", fontsize=12, color=ACCENT,
                 fontweight="bold")
    return finish(
        fig, out / "v5_fig11_processing_stages.png",
        "Specimen KJ-01, slice 35. Normalising uses brightness statistics from "
        "inside the drawn region only, so bright tissue outside it cannot set "
        "the thresholds.",
    )


def fig_network_context(out):
    """The three slices the network reads, and what it predicts from them."""
    evidence, saturn, row, cfg, seg, measured = _kj_plane(35)
    input_dir = Path(str(row["input_dir"]))
    files_by_z = {}
    for candidate in sorted(input_dir.glob(str(row["file_pattern"]))):
        token = candidate.stem.split("_z")[-1].split("_")[0]
        try:
            files_by_z[int(token)] = str(candidate)
        except ValueError:
            continue
    y0, y1, x0, x1 = 320, 520, 320, 520
    view = np.s_[y0:y1, x0:x1]

    fig, axes = plt.subplots(1, 5, figsize=(15.0, 3.5), constrained_layout=True)
    for axis, z, label in zip(axes[:3], (34, 35, 36),
                              ("slice below", "slice being analysed", "slice above")):
        raw = saturn.ensure_2d_image(
            saturn.robust_imread(files_by_z[z]), Path(files_by_z[z]).name
        )
        axis.imshow(np.asarray(raw, dtype=float)[view], cmap="gray")
        axis.set_title("{}\n(z{})".format(label, z), fontsize=9.5, color=ACCENT)
        axis.axis("off")
    axes[3].imshow(np.asarray(seg["unet_probability"], dtype=float)[view],
                   cmap="magma", vmin=0, vmax=1)
    axes[3].set_title("how confident it is\na nucleus is there",
                      fontsize=9.5, color=ACCENT)
    axes[3].axis("off")
    core = seg.get("unet_core_probability")
    if core is not None:
        axes[4].imshow(np.asarray(core, dtype=float)[view], cmap="viridis",
                       vmin=0, vmax=1)
        axes[4].set_title("where the dense core is\n(separates touching nuclei)",
                          fontsize=9.5, color=ACCENT)
    axes[4].axis("off")
    fig.suptitle(
        "The network reads three slices at once, and predicts two things",
        fontsize=12, color=ACCENT, fontweight="bold",
    )
    return finish(
        fig, out / "v5_fig12_network_context.png",
        "Specimen KJ-01. Seeing the slices above and below helps the network "
        "recognise a nucleus that is only partly in focus. The second prediction, "
        "the dense core, is what lets it tell two touching nuclei apart.",
    )


def fig_same_nucleus_through_depth(out):
    """One nucleus reappearing on consecutive slices, which is why linking matters."""
    evidence, saturn, row, cfg, seg, measured = _kj_plane(35)
    um = float(cfg["UM_PER_PX_XY"])
    labels = np.asarray(seg["unet_primary_instance_labels"])
    roi = np.asarray(seg.get("roi_mask", np.ones(labels.shape, dtype=bool)), dtype=bool)
    # A long, isolated nucleus is the clearest example of one that persists.
    lengths = [float(r.get("length_px_geodesic", 0.0)) * um for r in measured["results"]]
    median_length = float(np.median([v for v in lengths if v > 0]))
    target = None
    for result in sorted(
        measured["results"],
        key=lambda r: abs(float(r.get("length_px_geodesic", 0.0)) * um - median_length),
    ):
        if int(result.get("n_branch_nodes", 0)) != 0:
            continue
        if bool(result.get("suspected_multi_object_merge", False)):
            continue
        label = int(result["label"])
        ys, xs = np.nonzero(labels == label)
        cy, cx = int(round(ys.mean())), int(round(xs.mean()))
        half = 34
        y0, x0, y1, x1 = cy - half, cx - half, cy + half, cx + half
        if y0 < 0 or x0 < 0 or y1 > labels.shape[0] or x1 > labels.shape[1]:
            continue
        if roi[y0:y1, x0:x1].mean() < 0.99:
            continue
        target = (y0, y1, x0, x1)
        break
    if target is None:
        return None
    y0, y1, x0, x1 = target
    view = np.s_[y0:y1, x0:x1]

    input_dir = Path(str(row["input_dir"]))
    files_by_z = {}
    for candidate in sorted(input_dir.glob(str(row["file_pattern"]))):
        token = candidate.stem.split("_z")[-1].split("_")[0]
        try:
            files_by_z[int(token)] = str(candidate)
        except ValueError:
            continue
    planes = [33, 34, 35, 36, 37]
    fig, axes = plt.subplots(1, len(planes), figsize=(2.5 * len(planes), 3.1),
                             constrained_layout=True)
    for axis, z in zip(axes, planes):
        raw = saturn.ensure_2d_image(
            saturn.robust_imread(files_by_z[z]), Path(files_by_z[z]).name
        )
        axis.imshow(np.asarray(raw, dtype=float)[view], cmap="gray")
        axis.set_title("slice {}".format(z), fontsize=10,
                       color=ACCENT if z == 35 else MUTED)
        axis.axis("off")
    bar = 5.0 / um
    axes[0].plot([4, 4 + bar], [(y1 - y0) - 5] * 2, color="white", lw=2.6,
                 solid_capstyle="butt")
    axes[0].text(4 + bar / 2, (y1 - y0) - 8, "5 um", color="white", fontsize=8,
                 ha="center", va="bottom")
    fig.suptitle(
        "The same nuclei appear on slice after slice",
        fontsize=12, color=ACCENT, fontweight="bold",
    )
    return finish(
        fig, out / "v5_fig13_through_depth.png",
        "Specimen KJ-01, the same small area on five consecutive slices. Counting "
        "detections would count these several times over, which is why detections "
        "are linked through depth and the reconstructed nucleus is what gets counted.",
    )


def fig_how_length_and_width(out, plane=35):
    """Show the geometry of both measurements on one real, typical nucleus."""
    import importlib.util

    import pandas as pd

    spec = importlib.util.spec_from_file_location(
        "v5_method_evidence",
        ROOT / "scripts" / "generate_v571_intensity_width_evidence.py",
    )
    evidence = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evidence)
    saturn = evidence.load_pipeline()
    manifest = pd.read_csv(evidence.DEFAULT_MANIFEST)
    manifest = manifest[manifest["include"].astype(str).str.lower() == "true"]
    row = manifest[manifest["group"].astype(str) == "KJ"].iloc[0]
    cfg, seg, measured, _ = evidence.segment_plane(saturn, row, plane)
    um = float(cfg["UM_PER_PX_XY"])
    labels = np.asarray(seg["unet_primary_instance_labels"])
    centers = np.asarray(seg["unet_primary_centerline_labels"])
    image = np.asarray(seg["img_linear"], dtype=float)
    roi = np.asarray(seg.get("roi_mask", np.ones(labels.shape, dtype=bool)), dtype=bool)

    # Choose a typical nucleus, not the longest one. Sorting by length picks
    # objects several times the median, which are usually several nuclei joined
    # together rather than one unusually long one. A little curvature is
    # preferred so the difference between the two length measures is visible.
    lengths = [float(r.get("length_px_geodesic", 0.0)) * um for r in measured["results"]]
    median_length = float(np.median([v for v in lengths if v > 0]))

    def eligible(result):
        if int(result.get("n_branch_nodes", 0)) != 0:
            return False
        if bool(result.get("suspected_multi_object_merge", False)):
            return False
        if bool(result.get("intensity_profile_suspected_merge", False)):
            return False
        return np.isfinite(float(result.get("intensity_fwhm_width_um", np.nan)))

    def crop_for(result):
        ys, xs = np.nonzero(labels == int(result["label"]))
        cy, cx = int(round(ys.mean())), int(round(xs.mean()))
        half = 30
        y0, x0, y1, x1 = cy - half, cx - half, cy + half, cx + half
        if y0 < 0 or x0 < 0 or y1 > labels.shape[0] or x1 > labels.shape[1]:
            return None
        if roi[y0:y1, x0:x1].mean() < 0.99:
            return None
        return y0, y1, x0, x1

    chosen = None
    for curved_only in (True, False):
        candidates = []
        for result in measured["results"]:
            if not eligible(result):
                continue
            tortuosity = float(result.get("tortuosity", 1.0))
            if curved_only and not 1.04 <= tortuosity <= 1.5:
                continue
            candidates.append(result)
        for result in sorted(
            candidates,
            key=lambda r: abs(float(r.get("length_px_geodesic", 0.0)) * um - median_length),
        ):
            crop = crop_for(result)
            if crop is None:
                continue
            chosen = (result, int(result["label"]), crop)
            break
        if chosen is not None:
            break
    if chosen is None:
        return None

    result, label, (y0, y1, x0, x1) = chosen
    view = np.s_[y0:y1, x0:x1]
    mask = labels == label
    path = saturn._resample_smoothed_centerline(
        np.argwhere(centers == label),
        cfg.get("BODY_WIDTH_SAMPLE_SPACING_PX", 1.0),
        cfg.get("BODY_WIDTH_SMOOTH_SIGMA_PX", 1.0),
    )
    geodesic_um = float(result["length_px_geodesic"]) * um
    tortuosity = float(result.get("tortuosity", 1.0))
    straight_um = geodesic_um / tortuosity if tortuosity > 0 else np.nan
    reported_width = float(result["intensity_fwhm_width_um"])

    local = image[view]
    lo_disp, hi_disp = np.percentile(local, (2.0, 99.7))

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.6), constrained_layout=True)

    # Length: the ordered centre line, end to end.
    axes[0].imshow(local, cmap="gray", vmin=lo_disp, vmax=hi_disp)
    axes[0].contour(mask[view].astype(float), levels=[0.5], colors=MASK,
                    linewidths=1.0)
    axes[0].plot([path[0, 1] - x0, path[-1, 1] - x0],
                 [path[0, 0] - y0, path[-1, 0] - y0], "--", color="#e2e8f0", lw=1.6)
    axes[0].plot(path[:, 1] - x0, path[:, 0] - y0, "-", color=SIGNAL, lw=2.4)
    axes[0].plot([path[0, 1] - x0], [path[0, 0] - y0], "o", color="#facc15", ms=8,
                 markeredgecolor="black", markeredgewidth=0.6)
    axes[0].plot([path[-1, 1] - x0], [path[-1, 0] - y0], "o", color="#f43f5e", ms=8,
                 markeredgecolor="black", markeredgewidth=0.6)
    axes[0].set_title("Length: measured along the centre line",
                      fontsize=10.5, color=ACCENT)
    axes[0].axis("off")
    bar = 5.0 / um
    axes[0].plot([4, 4 + bar], [(y1 - y0) - 5] * 2, color="white", lw=2.6,
                 solid_capstyle="butt")
    axes[0].text(4 + bar / 2, (y1 - y0) - 8, "5 um", color="white", fontsize=8,
                 ha="center", va="bottom")
    axes[0].text(
        0.03, 0.03,
        "along the centre line   {:.2f} um\nstraight tip to tip      {:.2f} um\n"
        "curvature                {:.2f}".format(geodesic_um, straight_um, tortuosity),
        transform=axes[0].transAxes, fontsize=8.5, color="white",
        family="monospace", va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="#0f172a", ec="none", alpha=0.72))

    # Width: perpendicular cuts along the centre line.
    axes[1].imshow(local, cmap="gray", vmin=lo_disp, vmax=hi_disp)
    axes[1].contour(mask[view].astype(float), levels=[0.5], colors=MASK,
                    linewidths=1.0)
    axes[1].plot(path[:, 1] - x0, path[:, 0] - y0, "-", color=SIGNAL, lw=1.3,
                 alpha=0.75)
    profiles = _centerline_profiles(image, mask, path, cfg)
    if not profiles:
        return None
    median_px = float(np.median([p["width_px"] for p in profiles]))
    representative = min(profiles, key=lambda p: abs(p["width_px"] - median_px))
    step = max(1, len(profiles) // 9)
    for entry in profiles[::step]:
        point, normal = entry["point"], entry["normal"]
        p0, p1 = point - 3.2 * normal, point + 3.2 * normal
        axes[1].plot([p0[1] - x0, p1[1] - x0], [p0[0] - y0, p1[0] - y0],
                     color="#38bdf8", lw=1.3, alpha=0.9)
    point, normal = representative["point"], representative["normal"]
    p0, p1 = point - 3.6 * normal, point + 3.6 * normal
    axes[1].plot([p0[1] - x0, p1[1] - x0], [p0[0] - y0, p1[0] - y0],
                 color="#f43f5e", lw=2.2)
    axes[1].set_title("Width: measured across many cuts,\nthen the middle value taken",
                      fontsize=10.5, color=ACCENT)
    axes[1].axis("off")
    axes[1].text(
        0.03, 0.03,
        "{} cuts taken\nreported width   {:.2f} um".format(
            len(profiles), reported_width),
        transform=axes[1].transAxes, fontsize=8.5, color="white",
        family="monospace", va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="#0f172a", ec="none", alpha=0.72))

    # The profile at the cut marked in red: the one typical of this nucleus.
    offsets = representative["offsets"] * um
    profile = representative["profile"]
    peak = representative["peak"]
    half_level = peak / 2.0
    lo = representative["left_px"] * um
    hi = representative["right_px"] * um
    keep = np.abs(offsets) <= 2.6
    axes[2].plot(offsets[keep], profile[keep], color=INK, lw=1.8)
    axes[2].axhline(0, color=MUTED, lw=0.8)
    axes[2].axhline(half_level, color=SIGNAL, ls="--", lw=1.3)
    axes[2].plot([lo, hi], [half_level] * 2, color=SIGNAL, lw=3.4,
                 solid_capstyle="butt")
    axes[2].plot([lo, hi], [half_level] * 2, "o", color=SIGNAL, ms=6)
    axes[2].annotate("{:.2f} um".format(hi - lo),
                     xy=(0, half_level), xytext=(0, half_level + peak * 0.16),
                     ha="center", color=SIGNAL, fontsize=11, fontweight="bold")
    axes[2].text(2.5, peak * 0.97, "peak", color=MUTED, fontsize=8.5, ha="right")
    axes[2].text(2.5, half_level * 1.08, "half the peak", color=SIGNAL,
                 fontsize=8.5, ha="right")
    axes[2].set_xlim(-2.6, 2.6)
    axes[2].set_xlabel("distance across the nucleus (um)")
    axes[2].set_ylabel("brightness above background")
    axes[2].set_title("The marked cut, close up", fontsize=10.5, color=ACCENT)

    fig.suptitle("How length and width are measured", fontsize=13, color=ACCENT,
                 fontweight="bold")
    return finish(
        fig, out / "v5_fig20_how_measured.png",
        "Specimen KJ-01, one typical nucleus. Length follows the centre line from "
        "tip to tip, so a curved nucleus is not shortened; the pale dashed line is "
        "the straight tip-to-tip distance, and the ratio of the two is the "
        "curvature. Width is measured on a cut at right angles to the centre line, "
        "repeated along the middle three quarters of the nucleus, and the middle "
        "value of those {} cuts is what is reported. The red cut is the one shown "
        "on the right. The orange outline is the shape the network found: it says "
        "which pixels belong to this nucleus, so a close neighbour cannot "
        "contaminate the profile, but the width itself is read from the "
        "brightness and not from the outline."
        .format(len(profiles)),
    )


def _join_demonstration_slab(planes=(33, 34, 35, 36, 37), group="KJ"):
    """Segment a few consecutive slices and join them with the pipeline's own code.

    The joining step is what turns detections on separate optical slices into
    whole nuclei, so a figure about it has to run the real linker rather than
    illustrate one. Detections are assembled exactly as ``process_batch`` does,
    through ``rows_from_results``, and handed to ``track_across_slices``.
    """
    import importlib.util

    import pandas as pd

    spec = importlib.util.spec_from_file_location(
        "v5_join_evidence",
        ROOT / "scripts" / "generate_v571_intensity_width_evidence.py",
    )
    evidence = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evidence)
    saturn = evidence.load_pipeline()

    manifest = pd.read_csv(evidence.DEFAULT_MANIFEST)
    manifest = manifest[manifest["include"].astype(str).str.lower() == "true"]
    row = manifest[manifest["group"].astype(str) == str(group)].iloc[0]

    labels, images, rows = {}, {}, []
    cfg = None
    for z in planes:
        cfg, seg, measured, _ = evidence.segment_plane(saturn, row, z)
        um = float(cfg["UM_PER_PX_XY"])
        labels[z] = np.asarray(seg["unet_primary_instance_labels"]).astype(np.int32)
        images[z] = np.asarray(seg["img_linear"], dtype=np.float32)
        rows.extend(saturn.rows_from_results(measured["results"], z, um))
    tracked, summary = saturn.track_across_slices(pd.DataFrame(rows), cfg)
    return {
        "planes": list(planes),
        "labels": labels,
        "images": images,
        "tracked": tracked,
        "summary": summary,
        "um": float(cfg["UM_PER_PX_XY"]),
        "um_z": float(cfg["UM_PER_SLICE_Z"]),
        "max_link_um": float(cfg["TRACK_MAX_DIST_UM"]),
    }


def fig_joining_through_depth(out):
    """How detections on separate slices become one reconstructed nucleus."""
    import zipfile

    import pandas as pd

    slab = _join_demonstration_slab()
    planes = slab["planes"]
    labels, images = slab["labels"], slab["images"]
    tracked, summary = slab["tracked"], slab["summary"]
    um, um_z, max_dist = slab["um"], slab["um_z"], slab["max_link_um"]

    full = summary[summary["n_slices"] == len(planes)]["track_id"].tolist()
    members = tracked[tracked["track_id"].isin(full)]

    # A window holding several nuclei that persist through the whole slab, as
    # far apart from one another as the field allows.
    size = 100
    half = size // 2
    best = None
    for track_id, block in members.groupby("track_id"):
        cy = int(round(block["centroid_y"].mean()))
        cx = int(round(block["centroid_x"].mean()))
        y0, y1, x0, x1 = cy - half, cy + half, cx - half, cx + half
        shape = images[planes[0]].shape
        if y0 < 0 or x0 < 0 or y1 > shape[0] or x1 > shape[1]:
            continue
        inside = []
        for other, other_block in members.groupby("track_id"):
            if (other_block["centroid_y"].between(y0 + 10, y1 - 10).all()
                    and other_block["centroid_x"].between(x0 + 10, x1 - 10).all()):
                inside.append(int(other))
        if len(inside) < 3:
            continue
        points = np.array(
            [[members[members["track_id"] == t]["centroid_x"].mean(),
              members[members["track_id"] == t]["centroid_y"].mean()] for t in inside]
        )
        spread = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
        np.fill_diagonal(spread, np.inf)
        score = (min(len(inside), 4), float(spread.min()))
        if best is None or score > best["score"]:
            best = {"score": score, "crop": (y0, y1, x0, x1), "tracks": inside}
    if best is None:
        return None

    y0, y1, x0, x1 = best["crop"]
    centre = {t: members[members["track_id"] == t]["centroid_x"].mean()
              for t in best["tracks"]}
    order = sorted(best["tracks"], key=lambda t: centre[t])[:4]

    palette = ["#14b8a6", "#f59e0b", "#a855f7", "#f43f5e"]
    colour_of = {t: palette[i % len(palette)] for i, t in enumerate(order)}

    fig = plt.figure(figsize=(14.0, 6.9))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], height_ratios=[1, 1],
                            left=0.01, right=0.985, top=0.90, bottom=0.11,
                            wspace=0.16, hspace=0.62)

    ax = fig.add_subplot(grid[:, 0], projection="3d")
    exaggeration = 11.0
    nx, ny = x1 - x0, y1 - y0
    gx, gy = np.meshgrid((np.arange(nx) - nx / 2) * um,
                         (np.arange(ny) - ny / 2) * um)
    stack = np.stack([images[z][y0:y1, x0:x1] for z in planes])
    lo, hi = np.percentile(stack, (2.0, 99.6))
    for position, z in enumerate(planes):
        shade = np.clip((images[z][y0:y1, x0:x1] - lo) / max(hi - lo, 1e-9), 0, 1)
        colours = plt.cm.gray(shade)
        # Paint each tracked nucleus into the plane itself. Drawn as a separate
        # three-dimensional artist it disappears behind the surface at almost
        # every viewing angle.
        for track_id in order:
            block = tracked[(tracked["track_id"] == track_id)
                            & (tracked["z_slice"] == z)]
            if block.empty:
                continue
            mask = labels[z][y0:y1, x0:x1] == int(block.iloc[0]["sperm_id"])
            if not mask.any():
                continue
            rgb = np.array(matplotlib.colors.to_rgb(colour_of[track_id]))
            colours[mask, :3] = rgb[None, :] * (0.55 + 0.45 * shade[mask])[:, None]
        colours[..., 3] = 0.96
        ax.plot_surface(gx, gy, np.full_like(gx, position * um_z * exaggeration),
                        facecolors=colours, shade=False, rstride=1, cstride=1,
                        linewidth=0, antialiased=False)

    for position, track_id in enumerate(order):
        block = tracked[tracked["track_id"] == track_id].sort_values("z_slice")
        colour = colour_of[track_id]
        px, py, pz = [], [], []
        for _, detection in block.iterrows():
            z = int(detection["z_slice"])
            px.append((float(detection["centroid_x"]) - x0 - nx / 2) * um)
            py.append((float(detection["centroid_y"]) - y0 - ny / 2) * um)
            pz.append(planes.index(z) * um_z * exaggeration)
        # Thin and dashed: the join is the point, but it must not hide the
        # nucleus it is joining.
        line, = ax.plot(px, py, pz, "--", color=colour, lw=1.3, alpha=0.95,
                        dashes=(3.5, 2.5))
        line.set_zorder(20)
        dots = ax.scatter(px, py, pz, s=22, c=colour, depthshade=False,
                          edgecolors="white", linewidths=0.7)
        dots.set_zorder(21)
        ax.text(px[-1], py[-1], pz[-1] + 0.55, str(position + 1), color="white",
                fontsize=9, fontweight="bold", ha="center", va="center", zorder=30,
                bbox=dict(boxstyle="circle,pad=0.2", fc=colour, ec="white", lw=0.8))

    ax.set_xlabel("x (um)", labelpad=-2, fontsize=9)
    ax.set_ylabel("y (um)", labelpad=-2, fontsize=9)
    ax.set_zticks([i * um_z * exaggeration for i in range(len(planes))])
    ax.set_zticklabels(["slice {}".format(z) for z in planes], fontsize=8)
    ax.tick_params(axis="z", pad=6)
    ax.view_init(elev=17, azim=-60)
    ax.set_box_aspect((1.0, 1.0, 0.80), zoom=1.18)
    ax.tick_params(labelsize=8, pad=-1)
    ax.set_title("{} nuclei followed through {} slices, close up".format(
        len(order), len(planes)), fontsize=10.5, color=ACCENT, y=0.97)

    replay = (ROOT / "audits/evidence/v571_rc6_candidate/provenance"
              / "tracking_replay_inputs_outputs.zip")
    with zipfile.ZipFile(replay) as archive:
        with archive.open("kj_sv_40xx0.75-1/tracked_detections.csv") as handle:
            run_detections = pd.read_csv(handle)
        with archive.open("kj_sv_40xx0.75-1/track_summary.csv") as handle:
            run_tracks = pd.read_csv(handle)

    steps = []
    for _, block in run_detections.groupby("track_id"):
        block = block.sort_values("z_slice")
        if block.shape[0] < 2:
            continue
        gap = np.diff(block["z_slice"].to_numpy())
        moved = np.hypot(np.diff(block["centroid_x"].to_numpy()) * um,
                         np.diff(block["centroid_y"].to_numpy()) * um)
        steps.extend(moved[gap == 1].tolist())
    steps = np.asarray(steps)

    ax_b = fig.add_subplot(grid[0, 1])
    ax_b.hist(steps, bins=np.arange(0, max_dist + 0.3, 0.2), color=ACCENT,
              alpha=0.88, edgecolor="white")
    ax_b.axvline(max_dist, color=WARN, ls="--", lw=1.7)
    ax_b.text(max_dist - 0.15, ax_b.get_ylim()[1] * 0.95,
              "beyond {:.1f} um apart{}they are never joined".format(max_dist, "\n"),
              color=WARN, fontsize=8.2, ha="right", va="top")
    ax_b.set_xlabel("how far a nucleus shifts from one slice to the next (um)")
    ax_b.set_ylabel("links")
    ax_b.set_title("A nucleus barely moves between slices, which is what makes "
                   "joining safe", fontsize=10, color=ACCENT)

    ax_c = fig.add_subplot(grid[1, 1])
    n_detections = int(run_detections.shape[0])
    n_tracks = int(run_tracks.shape[0])
    bars = ax_c.bar(["counted once{}per slice".format("\n"),
                     "counted once{}per nucleus".format("\n")],
                    [n_detections, n_tracks], color=[MUTED, SIGNAL], width=0.5)
    for bar, value in zip(bars, (n_detections, n_tracks)):
        ax_c.text(bar.get_x() + bar.get_width() / 2, value + n_detections * 0.04,
                  "{:,}".format(value), ha="center", fontsize=12,
                  fontweight="bold", color=bar.get_facecolor())
    ax_c.set_ylim(0, n_detections * 1.3)
    ax_c.set_ylabel("count in this specimen")
    ax_c.set_title("Whole stack, {} slices: each nucleus counted once".format(
        int(run_detections["z_slice"].nunique())), fontsize=10, color=ACCENT)
    ax_c.text(0.74, 0.46,
              "without joining, the same{}nucleus would be counted{}{:.1f} times over"
              .format("\n", "\n", n_detections / max(n_tracks, 1)),
              transform=ax_c.transAxes, ha="center", fontsize=9, color=MUTED)
    ax_c.grid(axis="x", visible=False)

    fig.suptitle("Joining detections through depth into whole nuclei",
                 fontsize=13.5, color=ACCENT, fontweight="bold", y=0.975)
    fig.text(
        0.5, 0.035,
        "Specimen KJ-01. Left: slices {}-{}, segmented and joined with the "
        "pipeline's own code. The coloured pixels are the nucleus found on each "
        "slice and the dashed line links the ones judged to be the same nucleus. "
        "The slice spacing is drawn {:.0f} times larger than life so the planes "
        "can be told apart; every measurement uses the true {:.2f} um spacing. "
        "Right: the recorded production run for the whole {}-slice stack. A "
        "nucleus missing from a single slice can be bridged across the gap, and "
        "nothing is invented for the missing slice.".format(
            planes[0], planes[-1], exaggeration, um_z,
            int(run_detections["z_slice"].nunique())),
        fontsize=8.5, color=MUTED, ha="center", va="top", wrap=True)
    fig.savefig(out / "v5_fig21_joining_depth.png", dpi=225, bbox_inches="tight")
    plt.close(fig)
    return "v5_fig21_joining_depth.png"


def _centerline_profiles(image, mask, path, cfg, half_extent=8.0, step=0.05):
    """Sample the raw signal across a nucleus, perpendicular to its centre line.

    This mirrors ``measure_intensity_profile_width`` in the pipeline: background
    comes from the far tails of the same profile, the profile is restricted to
    the object being measured, and a width is only formed when the signal falls
    below half its peak on both sides while still inside that object.
    """
    from scipy import ndimage as ndi

    offsets = np.arange(-half_extent, half_extent + 1e-9, step)
    tail = float(cfg.get("INTENSITY_WIDTH_BACKGROUND_OFFSET_PX", 5.0))
    left_tail = offsets <= -tail
    right_tail = offsets >= tail
    arc = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    )
    total = float(arc[-1])
    trim = float(cfg.get("BODY_WIDTH_ENDPOINT_TRIM_FRACTION", 0.125))
    eligible = np.flatnonzero((arc >= trim * total) & (arc <= (1.0 - trim) * total))

    out = []
    for index in eligible:
        before = path[max(0, index - 2)]
        after = path[min(path.shape[0] - 1, index + 2)]
        tangent = after - before
        norm = float(np.linalg.norm(tangent))
        if norm <= 1e-9:
            continue
        tangent = tangent / norm
        normal = np.array([-tangent[1], tangent[0]], dtype=float)
        points = path[index][None, :] + offsets[:, None] * normal[None, :]
        rows = np.clip(points[:, 0], 0, image.shape[0] - 1)
        cols = np.clip(points[:, 1], 0, image.shape[1] - 1)
        sampled = ndi.map_coordinates(image, [rows, cols], order=1, mode="nearest")
        background = min(float(np.median(sampled[left_tail])),
                         float(np.median(sampled[right_tail])))
        profile = sampled - background
        inside = ndi.map_coordinates(mask.astype(np.uint8), [rows, cols],
                                     order=0, mode="constant").astype(bool)
        if not inside.any():
            continue
        owned = np.where(inside, profile, -np.inf)
        peak = float(owned.max())
        if peak <= 0:
            continue
        half = peak / 2.0
        centre = int(np.argmax(owned))
        left = centre
        while left > 0 and inside[left - 1] and profile[left - 1] >= half:
            left -= 1
        right = centre
        while (right < profile.size - 1 and inside[right + 1]
               and profile[right + 1] >= half):
            right += 1
        if not (left > 0 and inside[left - 1] and profile[left - 1] < half):
            continue
        if not (right < profile.size - 1 and inside[right + 1]
                and profile[right + 1] < half):
            continue
        out.append({
            "index": int(index),
            "point": path[index],
            "normal": normal,
            "offsets": offsets,
            "profile": profile,
            "peak": peak,
            "left_px": float(offsets[left]),
            "right_px": float(offsets[right]),
            "width_px": float(offsets[right] - offsets[left]),
            "centre_offset": float(offsets[centre]),
        })
    return out


def fig_hero_neighbourhood(out, plane=35):
    """Several neighbouring nuclei, each measured the way the pipeline does it.

    A single nucleus can be argued about. A crowded field is the real situation,
    and it shows two things at once: the software separates nuclei that sit
    close together, and the width it reports for each one is read from the raw
    signal across that nucleus rather than from the outline drawn around it.
    """
    import importlib.util

    import pandas as pd

    spec = importlib.util.spec_from_file_location(
        "v5_hood_evidence",
        ROOT / "scripts" / "generate_v571_intensity_width_evidence.py",
    )
    evidence = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evidence)
    saturn = evidence.load_pipeline()

    manifest = pd.read_csv(evidence.DEFAULT_MANIFEST)
    manifest = manifest[manifest["include"].astype(str).str.lower() == "true"]
    row = manifest[manifest["group"].astype(str) == "KJ"].iloc[0]
    cfg, seg, measured, _ = evidence.segment_plane(saturn, row, plane)
    um = float(cfg["UM_PER_PX_XY"])
    labels = np.asarray(seg["unet_primary_instance_labels"])
    centers = np.asarray(seg["unet_primary_centerline_labels"])
    image = np.asarray(seg["img_linear"], dtype=float)
    roi = np.asarray(seg.get("roi_mask", np.ones(labels.shape, dtype=bool)), dtype=bool)

    reported = {}
    for result in measured["results"]:
        value = float(result.get("intensity_fwhm_width_um", np.nan))
        if np.isfinite(value) and int(result.get("n_branch_nodes", 0)) == 0:
            reported[int(result["label"])] = value

    # Choose the crop holding the most fully contained, measured nuclei, so every
    # outline in the panel belongs to an object the figure can account for.
    size = 78
    half = size // 2
    boxes = {}
    for value in reported:
        ys, xs = np.nonzero(labels == value)
        boxes[value] = (ys.min(), ys.max(), xs.min(), xs.max(),
                        float(ys.mean()), float(xs.mean()))
    best = None
    for value, (ly, hy, lx, hx, my, mx) in boxes.items():
        cy, cx = int(round(my)), int(round(mx))
        y0, y1, x0, x1 = cy - half, cy + half, cx - half, cx + half
        if y0 < 0 or x0 < 0 or y1 > labels.shape[0] or x1 > labels.shape[1]:
            continue
        if roi[y0:y1, x0:x1].mean() < 0.999:
            continue
        inside = [v for v, (a, b, c, d, _, _) in boxes.items()
                  if a >= y0 and b < y1 and c >= x0 and d < x1]
        if len(inside) < 3:
            continue
        # Prefer crops where nothing intrudes: every labelled pixel in the crop
        # should belong to one of the nuclei the figure names and explains.
        present = set(int(v) for v in np.unique(labels[y0:y1, x0:x1]) if v > 0)
        intruders = len(present - set(inside))
        score = (-intruders, len(inside))
        if best is None or score > best["score"]:
            best = {"score": score, "crop": (y0, y1, x0, x1),
                    "members": sorted(inside), "intruders": intruders}
    if best is None:
        return None

    y0, y1, x0, x1 = best["crop"]
    view = np.s_[y0:y1, x0:x1]
    local = image[view]

    palette = ["#0f766e", "#b45309", "#7c3aed", "#be123c", "#0369a1"]
    picked = []
    for value in best["members"]:
        if len(picked) >= 5:
            break
        mask = labels == value
        coords = np.argwhere(centers == value)
        if coords.shape[0] < 3:
            continue
        path = saturn._resample_smoothed_centerline(
            coords, cfg.get("BODY_WIDTH_SAMPLE_SPACING_PX", 1.0),
            cfg.get("BODY_WIDTH_SMOOTH_SIGMA_PX", 1.0))
        profiles = _centerline_profiles(image, mask, path, cfg)
        if not profiles:
            continue
        median_px = float(np.median([p["width_px"] for p in profiles]))
        # Show the cut that is typical of the object, not the best or the worst.
        representative = min(profiles, key=lambda p: abs(p["width_px"] - median_px))
        picked.append({
            "label": int(value),
            "colour": palette[len(picked) % len(palette)],
            "path": path,
            "profile": representative,
            "reported": reported[int(value)],
            "n_cuts": len(profiles),
            "order": len(picked),
        })
    if len(picked) < 3:
        return None

    fig = plt.figure(figsize=(13.0, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0])

    # Display contrast only. The stretch changes nothing that is measured; the
    # profiles below are read from the unstretched linear image.
    lo_disp, hi_disp = np.percentile(local, (2.0, 99.6))

    ax_raw = fig.add_subplot(grid[0, 0])
    ax_raw.imshow(local, cmap="gray", vmin=lo_disp, vmax=hi_disp)
    ax_raw.set_title("A crowded field, exactly as acquired", fontsize=11, color=ACCENT)
    ax_raw.axis("off")
    bar = 5.0 / um
    ax_raw.plot([5, 5 + bar], [(y1 - y0) - 6] * 2, color="white", lw=3,
                solid_capstyle="butt")
    ax_raw.text(5 + bar / 2, (y1 - y0) - 8, "5 um", color="white", fontsize=9,
                ha="center", va="bottom")

    ax_out = fig.add_subplot(grid[0, 1])
    ax_out.imshow(local, cmap="gray", vmin=lo_disp, vmax=hi_disp)
    for entry in picked:
        shape = (labels[view] == entry["label"]).astype(float)
        ax_out.contour(shape, levels=[0.5], colors=[entry["colour"]], linewidths=1.3)
        path = entry["path"]
        ax_out.plot(path[:, 1] - x0, path[:, 0] - y0, "-", color=entry["colour"],
                    lw=0.9, alpha=0.8)
        rep = entry["profile"]
        point, normal = rep["point"], rep["normal"]
        span = 4.5
        p0 = point - span * normal
        p1 = point + span * normal
        ax_out.plot([p0[1] - x0, p1[1] - x0], [p0[0] - y0, p1[0] - y0],
                    color=entry["colour"], lw=2.2, alpha=0.95)
        ax_out.text(point[1] - x0 + 5.5, point[0] - y0 - 5.5,
                    str(entry["order"] + 1), color="white", fontsize=9,
                    fontweight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="circle,pad=0.18", fc=entry["colour"],
                              ec="none"))
    ax_out.set_title("each nucleus separated, with the cut used to measure it",
                     fontsize=11, color=ACCENT)
    ax_out.axis("off")

    ax_p = fig.add_subplot(grid[1, :])
    # The five profiles sit almost on top of one another, so the widths they
    # produce are stacked as a ruler underneath rather than drawn on the curves.
    base, pitch = -0.13, 0.085
    for entry in picked:
        rep = entry["profile"]
        offsets = (rep["offsets"] - rep["centre_offset"]) * um
        normalised = rep["profile"] / rep["peak"]
        keep = np.abs(offsets) <= 2.2
        ax_p.plot(offsets[keep], normalised[keep], color=entry["colour"], lw=1.8)
        lo = (rep["left_px"] - rep["centre_offset"]) * um
        hi = (rep["right_px"] - rep["centre_offset"]) * um
        level = base - pitch * entry["order"]
        ax_p.plot([lo, hi], [level, level], color=entry["colour"], lw=3.4,
                  solid_capstyle="butt", marker="|", markersize=7,
                  markeredgewidth=1.6)
        ax_p.text(hi + 0.07, level, "nucleus {}: {:.2f} um".format(
            entry["order"] + 1, entry["reported"]), color=entry["colour"],
            fontsize=8.5, va="center", ha="left", fontweight="bold")
    ax_p.axhline(0.5, color=MUTED, lw=0.9, ls="--")
    ax_p.axhline(0.0, color=MUTED, lw=0.8)
    ax_p.text(-2.15, 0.53, "half the peak", color=MUTED, fontsize=8.5, ha="left")
    ax_p.text(-2.15, base - pitch * (len(picked) - 1) / 2.0,
              "the width each nucleus measures", color=MUTED, fontsize=8.5,
              ha="left", va="center")
    ax_p.set_xlim(-2.2, 2.2)
    ax_p.set_ylim(base - pitch * len(picked) - 0.04, 1.12)
    ax_p.set_xlabel("distance across the nucleus, from its centre (um)")
    ax_p.set_ylabel("brightness (fraction of peak)")
    ax_p.set_yticks([0.0, 0.5, 1.0])
    ax_p.set_title(
        "Brightness across each nucleus: close neighbours still give separate, "
        "consistent measurements",
        fontsize=10.5, color=ACCENT,
    )

    fig.suptitle("Measuring nuclei in a densely packed field",
                 fontsize=13, color=ACCENT, fontweight="bold")
    total_cuts = sum(entry["n_cuts"] for entry in picked)
    fig.text(
        0.5, -0.045,
        "Specimen KJ-01, slice {}. Width is never read off the outline. For each "
        "nucleus the raw brightness is sampled along a line at right angles to "
        "its own centre line, background is taken from the quiet ends of that "
        "same line, and the width is the distance between the two points where "
        "the signal falls to half its peak. The cut drawn above is the typical "
        "one for that nucleus; the value reported is the middle value of all "
        "{} cuts taken across these {} nuclei. Curves are scaled to their own "
        "peaks so their shapes can be compared directly.".format(
            plane, total_cuts, len(picked)),
        fontsize=8.5, color=MUTED, ha="center", va="top", wrap=True,
    )
    fig.savefig(out / "v5_fig_hero_neighbourhood.png", dpi=230, bbox_inches="tight")
    plt.close(fig)
    return "v5_fig_hero_neighbourhood.png"


def fig_clean_examples(out, plane=35, per_specimen=3):
    """A gallery of real, fully isolated nuclei with the measurement drawn on.

    Selection is objective: unbranched, not merge-flagged, length in the plausible
    range, and with no other instance anywhere in the crop, so nothing in the
    frame belongs to a neighbour. Nuclei are then ordered by how close their width
    sits to the population median, which picks representative objects rather than
    flattering ones.
    """
    import importlib.util

    import pandas as pd
    from scipy import ndimage as ndi

    spec = importlib.util.spec_from_file_location(
        "v5_gallery_evidence", ROOT / "scripts" / "generate_v571_intensity_width_evidence.py"
    )
    evidence = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evidence)
    saturn = evidence.load_pipeline()

    manifest = pd.read_csv(evidence.DEFAULT_MANIFEST)
    manifest = manifest[manifest["include"].astype(str).str.lower() == "true"]
    chosen = pd.concat([b.head(1) for _, b in manifest.groupby("group")])

    picks = []
    for _, row in chosen.iterrows():
        cfg, seg, measured, _ = evidence.segment_plane(saturn, row, plane)
        um = float(cfg["UM_PER_PX_XY"])
        labels = np.asarray(seg["unet_primary_instance_labels"])
        centers = np.asarray(seg["unet_primary_centerline_labels"])
        image = np.asarray(seg["img_linear"], dtype=float)
        roi_mask = np.asarray(
            seg.get("roi_mask", np.ones(labels.shape, dtype=bool)), dtype=bool
        )
        records = []
        for result in measured["results"]:
            fwhm = float(result.get("intensity_fwhm_width_um", np.nan))
            if not np.isfinite(fwhm) or int(result.get("n_branch_nodes", 0)) != 0:
                continue
            if bool(result.get("intensity_profile_suspected_merge", False)):
                continue
            length = float(result["length_px_geodesic"]) * um
            if not 6.0 <= length <= 12.0:
                continue
            label = int(result["label"])
            mask = labels == label
            ys, xs = np.nonzero(mask)
            # A fixed square window keeps every panel the same size and scale, so
            # the gallery reads as a comparison rather than a collage.
            half = 20
            cy, cx = int(round(ys.mean())), int(round(xs.mean()))
            y0, x0 = cy - half, cx - half
            y1, x1 = cy + half, cx + half
            if y0 < 0 or x0 < 0 or y1 > labels.shape[0] or x1 > labels.shape[1]:
                continue  # too close to the frame edge for an equal-sized crop
            if set(np.unique(labels[y0:y1, x0:x1])) - {0, label}:
                continue  # a neighbour intrudes on the crop
            if roi_mask[y0:y1, x0:x1].mean() < 0.98:
                continue  # the ROI boundary would appear as a dark band
            records.append(
                {"label": label, "fwhm": fwhm, "length": length,
                 "mask": mask, "crop": (y0, y1, x0, x1), "image": image,
                 "centers": centers, "um": um, "cfg": cfg,
                 "specimen": str(row["sample_id"]), "group": str(row["group"])}
            )
        if not records:
            continue
        frame = pd.DataFrame([{k: r[k] for k in ("label", "fwhm")} for r in records])
        order = frame["fwhm"].sub(frame["fwhm"].median()).abs().sort_values().index
        picks.extend(records[i] for i in order[:per_specimen])

    if not picks:
        return None

    columns = len(picks)
    fig, axes = plt.subplots(2, columns, figsize=(2.5 * columns, 5.4),
                             constrained_layout=True)
    if columns == 1:
        axes = axes.reshape(2, 1)
    for column, pick in enumerate(picks):
        y0, y1, x0, x1 = pick["crop"]
        view = np.s_[y0:y1, x0:x1]
        centerline = np.argwhere(pick["centers"] == pick["label"])

        local = pick["image"][view]
        # Display contrast only. Nothing measured is affected; the profiles are
        # read from the unstretched linear image.
        lo_disp, hi_disp = np.percentile(local, (2.0, 99.6))

        top = axes[0, column]
        top.imshow(local, cmap="gray", vmin=lo_disp, vmax=hi_disp)
        top.set_title(f"{pick['group']}  instance {pick['label']}", fontsize=9,
                      color=ACCENT)
        top.axis("off")

        bottom = axes[1, column]
        bottom.imshow(local, cmap="gray", vmin=lo_disp, vmax=hi_disp)
        bottom.contour(pick["mask"][view], levels=[0.5], colors=MASK, linewidths=1.3)
        # Show the cuts the width is actually taken on, not just the centre line.
        path = saturn._resample_smoothed_centerline(
            centerline, pick["cfg"].get("BODY_WIDTH_SAMPLE_SPACING_PX", 1.0),
            pick["cfg"].get("BODY_WIDTH_SMOOTH_SIGMA_PX", 1.0))
        bottom.plot(path[:, 1] - x0, path[:, 0] - y0, "-", color=SIGNAL, lw=1.0,
                    alpha=0.8)
        cuts = _centerline_profiles(pick["image"], pick["mask"], path, pick["cfg"])
        for entry in cuts[::max(1, len(cuts) // 8)]:
            point, normal = entry["point"], entry["normal"]
            a, b = point - 2.6 * normal, point + 2.6 * normal
            bottom.plot([a[1] - x0, b[1] - x0], [a[0] - y0, b[0] - y0],
                        color="#38bdf8", lw=1.1, alpha=0.9)
        bottom.set_title(f"measured width: {pick['fwhm']:.2f} um", fontsize=9)
        bottom.axis("off")
        # A 5 um scale bar, as used in the earlier workflow figures.
        bar_px = 5.0 / pick["um"]
        span = x1 - x0
        bottom.plot([span * 0.06, span * 0.06 + bar_px], [ (y1 - y0) * 0.93 ] * 2,
                    color="white", lw=2.6, solid_capstyle="butt")
        bottom.text(span * 0.06 + bar_px / 2, (y1 - y0) * 0.90, "5 um",
                    color="white", fontsize=7.5, ha="center", va="bottom")

    axes[0, 0].text(-0.08, 0.5, "the image", transform=axes[0, 0].transAxes,
                    rotation=90, va="center", ha="right", fontsize=9, color=MUTED)
    axes[1, 0].text(-0.08, 0.5, "with the measurement", transform=axes[1, 0].transAxes,
                    rotation=90, va="center", ha="right", fontsize=9, color=MUTED)
    handles = [
        Patch(facecolor="none", edgecolor=MASK, label="outline the computer drew"),
        Patch(facecolor=SIGNAL, label="centre line"),
        Patch(facecolor="#38bdf8", label="cuts the width is measured on"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.10))
    fig.suptitle("Real sperm nuclei, one slice, nothing else in the frame", fontsize=11,
                 color=ACCENT, fontweight="bold")
    fig.text(
        0.5, -0.155,
        "Selected objectively: unbranched, not merge-flagged, plausible length, no "
        "other instance in the crop, and clear of the ROI boundary. Every panel is "
        "the same 40 x 40 pixel window, about 15 um across. The width comes from "
        "the brightness across each blue cut, not from the orange outline.",
        fontsize=7.5, color=MUTED, ha="center", va="top", wrap=True,
    )
    fig.savefig(out / "v5_fig00_clean_examples.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return "v5_fig00_clean_examples.png"


def fig_mask_versus_signal(out):
    """The headline: the mask boundary and the signal disagree, unevenly."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5), constrained_layout=True)

    specimens = ["KJ-01", "WT-01"]
    mask = [1.514, 1.629]
    signal = [0.701, 0.730]
    x = np.arange(len(specimens))
    w = 0.34
    axes[0].bar(x - w / 2, mask, w, color=MASK, label="width of the computer-drawn outline")
    axes[0].bar(x + w / 2, signal, w, color=SIGNAL, label="width of the actual glow")
    for i, (m, s) in enumerate(zip(mask, signal)):
        axes[0].text(i - w / 2, m + 0.04, f"{m:.3f}", ha="center", fontsize=8)
        axes[0].text(i + w / 2, s + 0.04, f"{s:.3f}", ha="center", fontsize=8)
    axes[0].set_xticks(x, specimens)
    axes[0].set_ylabel("width (um)")
    axes[0].set_ylim(0, 2.0)
    axes[0].set_title("Two ways of measuring the same nuclei", fontsize=10, color=ACCENT)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")

    ratio = [m / s for m, s in zip(mask, signal)]
    bars = axes[1].bar(specimens, ratio, 0.45, color=[MASK, "#f59e0b"])
    for bar, value in zip(bars, ratio):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2, value + 0.04,
            f"{value:.2f}x", ha="center", fontsize=10, fontweight="bold",
        )
    axes[1].axhline(1.0, color=SIGNAL, lw=1.2, ls="--")
    axes[1].text(1.42, 1.05, "agreement", fontsize=8, color=SIGNAL, ha="right")
    axes[1].set_ylim(0, 2.9)
    axes[1].set_ylabel("outline width divided by glow width")
    axes[1].set_title(
        "And they disagree by different amounts in different samples",
        fontsize=10, color=ACCENT,
    )
    axes[1].annotate(
        "an 8% difference in bias,\nagainst a 0.029 um difference\nin the signal itself",
        xy=(1, ratio[1]), xytext=(0.15, 2.45), fontsize=8, color=WARN,
        arrowprops=dict(arrowstyle="->", color=WARN, lw=1.1),
    )
    return finish(
        fig, out / "v5_fig01_mask_versus_signal.png",
        "Measured on planes 34-36. A bias that differs by specimen cannot be "
        "divided out of a group comparison.",
    )


def fig_why_the_mask_is_wide(out):
    """The root cause: the annotation convention, not the model."""
    fig, ax = plt.subplots(figsize=(8.2, 3.2), constrained_layout=True)
    labels = [
        "Optical width of a nucleus\n(raw intensity FWHM)",
        "Human training annotation\n(median)",
        "Learned mask boundary\n(what the model reproduces)",
    ]
    values = [0.643, 1.606, 1.514]
    colors = [SIGNAL, "#7c3aed", MASK]
    bars = ax.barh(labels, values, color=colors, height=0.55)
    for bar, value in zip(bars, values):
        ax.text(value + 0.03, bar.get_y() + bar.get_height() / 2,
                f"{value:.3f} um", va="center", fontsize=9, fontweight="bold")
    ax.set_xlim(0, 2.0)
    ax.set_xlabel("width (um)")
    ax.set_title(
        "The outline is wide because that is how the training data was drawn",
        fontsize=10, color=ACCENT,
    )
    ax.invert_yaxis()
    return finish(
        fig, out / "v5_fig02_annotation_convention.png",
        "The mask is not a model error. It reproduces how the training data was "
        "drawn, which is why no amount of retraining on the same annotations "
        "would narrow it.",
    )


def fig_sampling_limit(out):
    """What the current acquisition already does well, and where the headroom is."""
    NA, n, lam = 1.3, 1.518, 0.580
    lateral = 0.51 * lam / NA
    axial = 1.4 * n * lam / NA ** 2
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), constrained_layout=True)

    names = ["XY pixel", "Z step"]
    actual = [0.37842, 0.34618]
    nyquist = [lateral / 2, axial / 2]
    x = np.arange(2)
    w = 0.34
    axes[0].bar(x - w / 2, actual, w, color=MASK, label="what we use now")
    axes[0].bar(x + w / 2, nyquist, w, color=SIGNAL,
                label="what would resolve width outright")
    axes[0].set_xticks(x, names)
    axes[0].set_ylabel("microns")
    axes[0].set_title("The slice spacing is already ideal; zoom is where the "
                      "headroom is", fontsize=10, color=ACCENT)
    axes[0].set_ylim(0, 0.52)
    axes[0].legend(fontsize=8, frameon=False, loc="upper right")
    axes[0].text(0, 0.40, "3.3x finer would\nresolve width outright", ha="center",
                 fontsize=8, color=MASK, fontweight="bold")
    axes[0].text(1, 0.40, "already ideal", ha="center", fontsize=8, color=SIGNAL,
                 fontweight="bold")

    true_um = np.array([0.30, 0.40, 0.50, 0.60, 0.75, 1.00, 1.50, 2.00, 3.00])
    measured = np.array([0.732, 0.742, 0.750, 0.760, 0.793, 0.964, 1.503, 2.058, 3.001])
    axes[1].plot(true_um, measured, "o-", color=SIGNAL, lw=1.8, ms=5,
                 label="what we measure now")
    axes[1].plot([0, 3.1], [0, 3.1], ls="--", color=MUTED, lw=1.1,
                 label="what finer pixels would give")
    axes[1].axvspan(0.25, 0.75, color="#fee2e2", zorder=0)
    axes[1].text(0.5, 2.3, "where these\nnuclei live", ha="center", fontsize=8,
                 color=WARN)
    axes[1].set_xlabel("real width of the object (um)")
    axes[1].set_ylabel("width we measure (um)")
    axes[1].set_xlim(0, 3.1)
    axes[1].set_ylim(0, 3.2)
    axes[1].set_title("With finer pixels, width would become absolute too",
                      fontsize=10, color=ACCENT)
    axes[1].legend(fontsize=8, frameon=False, loc="upper left")
    return finish(
        fig, out / "v5_fig03_sampling_limit.png",
        f"Confocal lateral resolution {lateral:.3f} um and axial {axial:.3f} um, "
        "computed from the Leica metadata for this objective. The recovery curve "
        "comes from synthetic rods imaged through the same optical chain: above "
        "about one micron the measurement is already absolute, and raising the "
        "zoom would extend that down to the width of these nuclei.",
    )


def fig_two_measures(out):
    """FWHM and integrated signal fail in opposite directions."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), constrained_layout=True)

    axes[0].bar(["half-maximum\nwidth", "integrated\nsignal"], [2.2, 57.3],
                color=[SIGNAL, "#7c3aed"], width=0.45)
    axes[0].set_ylabel("% change")
    axes[0].set_title("If a nucleus really gets 67% wider...", fontsize=10, color=ACCENT)
    for i, v in enumerate([2.2, 57.3]):
        axes[0].text(i, v + 1.5, f"+{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
    axes[0].text(0.5, 45, "26x better at seeing\na real change", ha="center", fontsize=9, color="#7c3aed")

    axes[1].bar(["half-maximum\nwidth", "integrated\nsignal"], [0.2, 19.8],
                color=[SIGNAL, "#7c3aed"], width=0.45)
    axes[1].set_ylabel("% change")
    axes[1].set_title("...but if the stain is just 20% brighter", fontsize=10, color=ACCENT)
    for i, v in enumerate([0.2, 19.8]):
        axes[1].text(i, v + 0.6, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
    axes[1].text(0, 12, "unaffected", ha="center", fontsize=9, color=SIGNAL)
    axes[1].text(1, 12, "fooled by it", ha="center", fontsize=9, color=WARN)
    return finish(
        fig, out / "v5_fig04_two_measures.png",
        "Their weaknesses are orthogonal, so agreement between them is evidence "
        "and disagreement is a flag. With no staining control in this study, "
        "integrated signal stays quality control only.",
    )


def fig_merge_correction(out):
    """Objective evidence, not length, decides what is one nucleus."""
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.4), constrained_layout=True)

    axes[0].bar(["old rule\n>20 um AND branched", "objective evidence\n>=3 branch nodes\nor bimodal profile"],
                [0.18, 7.51], color=[MUTED, ACCENT], width=0.5)
    axes[0].set_ylabel("% of objects flagged as stuck together")
    axes[0].set_title("Touching nuclei now identified", fontsize=10, color=ACCENT)
    for i, v in enumerate([0.18, 7.51]):
        axes[0].text(i, v + 0.2, f"{v:.2f}%", ha="center", fontsize=10, fontweight="bold")

    bands = ["0-5", "5-8", "8-11", "11-14", "14-17", "17-20", "20-25", "25+"]
    counts = [155, 125, 160, 74, 25, 17, 2, 1]
    colors = [MUTED] * 6 + [WARN] * 2
    axes[1].bar(bands, counts, color=colors, width=0.7)
    axes[1].axvline(5.5, color=WARN, ls="--", lw=1.4)
    axes[1].text(5.6, 150, "old 20 um\ntrigger", fontsize=8, color=WARN)
    axes[1].set_xlabel("length of the detected object (um)")
    axes[1].set_ylabel("number of objects")
    axes[1].set_title("Most clumps are two nuclei, not one long one", fontsize=10, color=ACCENT)
    axes[1].annotate("a pair of ~8 um nuclei\nlands here", xy=(4.4, 40),
                     xytext=(1.4, 120), fontsize=8, color=ACCENT,
                     arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.1))

    x = np.arange(2)
    w = 0.34
    axes[2].bar(x - w / 2, [292, 267], w, color=MUTED, label="before")
    axes[2].bar(x + w / 2, [329, 286], w, color=ACCENT, label="after separating them")
    axes[2].set_xticks(x, ["KJ-01", "WT-01"])
    axes[2].set_ylabel("nuclei counted on one slice")
    axes[2].set_title("A more accurate count", fontsize=10, color=ACCENT)
    axes[2].set_ylim(0, 400)
    axes[2].legend(fontsize=8, frameon=False, loc="lower right")
    for i, (b, a) in enumerate(zip([292, 267], [329, 286])):
        axes[2].text(i + w / 2, a + 6, f"+{100*(a-b)/b:.1f}%", ha="center",
                     fontsize=9, fontweight="bold", color=WARN)
    return finish(
        fig, out / "v5_fig05_merge_correction.png",
        "Median object length is 7.91 um, so a pair of touching nuclei seldom "
        "reached the old 20 um trigger. Recognising clumps from their shape also "
        "evens out a correction that would otherwise have been larger in one "
        "group than the other.",
    )


def fig_availability_bias(out):
    """The fail-closed measurement drops objects; the QC asks whether that biases groups."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), constrained_layout=True)

    reasons = ["boundary\nclipped", "short\ncenterline", "insufficient\nprofiles"]
    shares = [76.9, 16.2, 6.9]
    axes[0].bar(reasons, shares, color=[MASK, "#f59e0b", "#fbbf24"], width=0.55)
    axes[0].set_ylabel("% of the withheld ones")
    axes[0].set_title("Why a width is sometimes withheld (21% of objects)",
                      fontsize=10, color=ACCENT)
    for i, v in enumerate(shares):
        axes[0].text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9)

    groups = ["KJ (n=18)", "WT (n=17)"]
    unavailable = [0.2121, 0.2211]
    errs = [0.0242, 0.0287]
    axes[1].bar(groups, unavailable, yerr=errs, capsize=6, color=[ACCENT, SIGNAL],
                width=0.45)
    axes[1].set_ylabel("fraction withheld")
    axes[1].set_ylim(0, 0.33)
    axes[1].set_title("Withheld at the same rate in both groups,"
                      " so comparisons stay balanced",
                      fontsize=10, color=ACCENT)
    axes[1].text(0.5, 0.29, "Welch p = 0.33", ha="center", fontsize=9, color=SIGNAL,
                 fontweight="bold")
    return finish(
        fig, out / "v5_fig06_availability_bias.png",
        "All 35 specimens of the study, 22,381 detections. The rate is the same "
        "in both groups, which is what keeps a between-group comparison balanced; "
        "the check is run on every study rather than assumed.",
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "docs" / "v5_7_illustrated_workflow" / "figures_v5"))
    args = parser.parse_args(argv)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)

    built = [
        fig_region_across_slices(out),
        fig_how_length_and_width(out),
        fig_joining_through_depth(out),
        fig_processing_stages(out),
        fig_network_context(out),
        fig_same_nucleus_through_depth(out),
        fig_hero_neighbourhood(out),
        fig_clean_examples(out),
        fig_mask_versus_signal(out),
        fig_why_the_mask_is_wide(out),
        fig_sampling_limit(out),
        fig_two_measures(out),
        fig_merge_correction(out),
        fig_availability_bias(out),
    ]
    (out / "figure_manifest.json").write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "figures": built,
                "provenance": (
                    "Every value plotted was measured during the 2026-09 v5.7.1 "
                    "measurement work and is recorded under audits/evidence or "
                    "audits/findings. No value is illustrative."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    built = [name for name in built if name]
    for name in built:
        print("  wrote", name)
    print(f"\n{len(built)} figures in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
