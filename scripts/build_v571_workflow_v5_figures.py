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


def finish(fig, path, caption=None):
    if caption:
        fig.text(0.01, 0.005, caption, fontsize=7, color=MUTED, ha="left", va="bottom")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path.name


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
                 "centers": centers, "um": um,
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

        top = axes[0, column]
        top.imshow(pick["image"][view], cmap="gray")
        top.set_title(f"{pick['group']}  instance {pick['label']}", fontsize=9,
                      color=ACCENT)
        top.axis("off")

        bottom = axes[1, column]
        bottom.imshow(pick["image"][view], cmap="gray")
        bottom.contour(pick["mask"][view], levels=[0.5], colors=MASK, linewidths=1.3)
        bottom.plot(centerline[:, 1] - x0, centerline[:, 0] - y0, ".",
                    color=SIGNAL, ms=1.8)
        bottom.set_title(f"width of the glow: {pick['fwhm']:.2f} um", fontsize=9)
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
    axes[1, 0].text(-0.08, 0.5, "what we measure", transform=axes[1, 0].transAxes,
                    rotation=90, va="center", ha="right", fontsize=9, color=MUTED)
    handles = [
        Patch(facecolor="none", edgecolor=MASK, label="outline the computer drew"),
        Patch(facecolor=SIGNAL, label="line we measure across"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.10))
    fig.suptitle("Real sperm nuclei, one slice, nothing else in the frame", fontsize=11,
                 color=ACCENT, fontweight="bold")
    fig.text(
        0.5, -0.155,
        "Selected objectively: unbranched, not merge-flagged, plausible length, no "
        "other instance in the crop, and clear of the ROI boundary. Every panel is "
        "the same 40 x 40 pixel window, about 15 um across. Note how far the mask boundary sits outside "
        "the visible signal.",
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
    """Why absolute width is not recoverable at the current zoom."""
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
    axes[0].bar(x + w / 2, nyquist, w, color=SIGNAL, label="what is needed to resolve width")
    axes[0].set_xticks(x, names)
    axes[0].set_ylabel("microns")
    axes[0].set_title("The pixels are too big sideways, but the Z step is fine", fontsize=10, color=ACCENT)
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].text(0, 0.40, "3.3x\nundersampled", ha="center", fontsize=8, color=WARN,
                 fontweight="bold")
    axes[0].text(1, 0.40, "already fine", ha="center", fontsize=8, color=SIGNAL,
                 fontweight="bold")

    true_um = np.array([0.30, 0.40, 0.50, 0.60, 0.75, 1.00, 1.50, 2.00, 3.00])
    measured = np.array([0.732, 0.742, 0.750, 0.760, 0.793, 0.964, 1.503, 2.058, 3.001])
    axes[1].plot(true_um, measured, "o-", color=SIGNAL, lw=1.8, ms=5,
                 label="what we measure")
    axes[1].plot([0, 3.1], [0, 3.1], ls="--", color=MUTED, lw=1.1,
                 label="what we would measure with small enough pixels")
    axes[1].axvspan(0.25, 0.75, color="#fee2e2", zorder=0)
    axes[1].text(0.5, 2.3, "where these\nnuclei live", ha="center", fontsize=8,
                 color=WARN)
    axes[1].set_xlabel("real width of the object (um)")
    axes[1].set_ylabel("width we measure (um)")
    axes[1].set_xlim(0, 3.1)
    axes[1].set_ylim(0, 3.2)
    axes[1].set_title("Below about 1 um, real differences get squashed", fontsize=10, color=ACCENT)
    axes[1].legend(fontsize=8, frameon=False, loc="upper left")
    return finish(
        fig, out / "v5_fig03_sampling_limit.png",
        f"Confocal lateral FWHM {lateral:.3f} um, axial {axial:.3f} um from the "
        "Leica metadata. Recovery curve from synthetic rods imaged through the "
        "same optical chain.",
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
    axes[0].set_title("How often we spotted two nuclei stuck together", fontsize=10, color=ACCENT)
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
    axes[1].set_title("The old cut-off was set too high", fontsize=10, color=ACCENT)
    axes[1].annotate("a pair of ~8 um nuclei\nlands here", xy=(4.4, 40),
                     xytext=(1.4, 120), fontsize=8, color=ACCENT,
                     arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.1))

    x = np.arange(2)
    w = 0.34
    axes[2].bar(x - w / 2, [292, 267], w, color=MUTED, label="before")
    axes[2].bar(x + w / 2, [329, 286], w, color=ACCENT, label="after separating them")
    axes[2].set_xticks(x, ["KJ-01", "WT-01"])
    axes[2].set_ylabel("nuclei counted on one slice")
    axes[2].set_title("We were undercounting, and not equally", fontsize=10, color=ACCENT)
    axes[2].legend(fontsize=8, frameon=False)
    for i, (b, a) in enumerate(zip([292, 267], [329, 286])):
        axes[2].text(i + w / 2, a + 6, f"+{100*(a-b)/b:.1f}%", ha="center",
                     fontsize=9, fontweight="bold", color=WARN)
    return finish(
        fig, out / "v5_fig05_merge_correction.png",
        "Median instance length is 7.91 um. Because the correction is larger in "
        "KJ than WT, the previous under-counting was suppressing one group more "
        "than the other.",
    )


def fig_availability_bias(out):
    """The fail-closed measurement drops objects; the QC asks whether that biases groups."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), constrained_layout=True)

    reasons = ["boundary\nclipped", "short\ncenterline", "insufficient\nprofiles"]
    shares = [76.9, 16.2, 6.9]
    axes[0].bar(reasons, shares, color=[MASK, "#f59e0b", "#fbbf24"], width=0.55)
    axes[0].set_ylabel("% of the unmeasurable ones")
    axes[0].set_title("When we cannot measure a width (21% of objects)",
                      fontsize=10, color=ACCENT)
    for i, v in enumerate(shares):
        axes[0].text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9)

    groups = ["KJ (n=18)", "WT (n=17)"]
    unavailable = [0.2121, 0.2211]
    errs = [0.0242, 0.0287]
    axes[1].bar(groups, unavailable, yerr=errs, capsize=6, color=[ACCENT, SIGNAL],
                width=0.45)
    axes[1].set_ylabel("fraction we could not measure")
    axes[1].set_ylim(0, 0.33)
    axes[1].set_title("The same fraction is skipped in both groups, so it is fair", fontsize=10, color=ACCENT)
    axes[1].text(0.5, 0.29, "Welch p = 0.33", ha="center", fontsize=9, color=SIGNAL,
                 fontweight="bold")
    return finish(
        fig, out / "v5_fig06_availability_bias.png",
        "All 35 specimens, 22,381 detections. A shared and equal refusal rate "
        "does not distort a between-group comparison; an unequal one would.",
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "docs" / "v5_7_illustrated_workflow" / "figures_v5"))
    args = parser.parse_args(argv)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)

    built = [
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
