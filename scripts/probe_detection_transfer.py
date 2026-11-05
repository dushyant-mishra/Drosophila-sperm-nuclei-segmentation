"""Measure whether the trained detector transfers to a stack it was not trained on.

The production checkpoint was trained on Leica 40x/1.3 oil data at 0.378 um per
pixel. Applying it to another microscope, another pixel size or another sample
presentation is an assumption, not a given, and the failure mode is quiet: the
run completes, reports thousands of objects and produces a plausible-looking
table of numbers that happen to describe noise.

This probe segments one plane and reports the few quantities that separate a
working detector from a failing one, without needing ground truth:

- the share of the frame the detector claims, against the share that is
  actually above background;
- the median object length against the length a real nucleus has;
- the median reported width against the optical resolution limit, since a width
  below that limit is physically impossible and marks noise.

It does not decide anything. It prints numbers a reviewer can weigh.

Usage:
    python scripts/probe_detection_transfer.py --input <folder of slice TIFFs> \
        --xy-um 0.2247 --z-um 1.0 [--resample-to 0.37841796875] [--roi <mask.tif>]
"""

import argparse
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAINED_PIXEL_UM = 0.37841796875          # the sampling the checkpoint saw
REFERENCE_LENGTH_UM = 10.59               # KJ-01 median, from the RC3 evidence
REFERENCE_WIDTH_UM = 0.71                 # KJ-01 median signal FWHM


def load_pipeline():
    spec = importlib.util.spec_from_file_location(
        "saturn_probe", ROOT / "sperm_segmentation_saturnv5.7.1.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def lateral_resolution_um(wavelength_um, numerical_aperture):
    """Rayleigh-style lateral FWHM. A reported width below this is impossible."""
    return 0.51 * wavelength_um / numerical_aperture


def probe(saturn, slice_paths, xy_um, z_um, roi=None, plane=None):
    cfg, _ = saturn.load_analysis_profile(
        ROOT / "production_profiles" / "saturn_v5_7_1_model_c_epoch003.json",
        saturn.CONFIG,
    )
    cfg["AUTO_LEICA_CALIBRATION"] = False
    cfg["REQUIRE_LEICA_METADATA"] = False
    cfg["UM_PER_PX_XY"] = float(xy_um)
    cfg["UM_PER_SLICE_Z"] = float(z_um)
    saturn.resolve_stack_microscope_calibration(
        cfg, slice_paths, input_dir=Path(slice_paths[0]).parent
    )

    index = len(slice_paths) // 2 if plane is None else int(plane)
    raw = saturn.ensure_2d_image(
        saturn.robust_imread(slice_paths[index]), Path(slice_paths[index]).name
    )
    context = saturn.build_stack_preprocess_context(slice_paths, roi, cfg)
    seg = saturn.segment_slice(
        raw, cfg, z_idx=index, roi_mask=roi, preprocess_context=context,
        unet_context_stack=saturn._make_unet_context_from_paths(
            dict(enumerate(slice_paths)), index
        ),
    )
    measured = saturn.measure_spermatids(seg, cfg)
    labels = np.asarray(seg["unet_primary_instance_labels"])
    image = np.asarray(raw, dtype=float)

    lengths = np.array(
        [float(r["length_px_geodesic"]) * cfg["UM_PER_PX_XY"] for r in measured["results"]]
    )
    widths = np.array(
        [
            float(r.get("intensity_fwhm_width_um", np.nan))
            for r in measured["results"]
        ]
    )
    widths = widths[np.isfinite(widths)]
    return {
        "plane": index,
        "objects": len(measured["results"]),
        "labelled_fraction": float((labels > 0).mean()),
        "above_background_fraction": float((image > 10).mean()),
        "length_median_um": float(np.median(lengths)) if lengths.size else float("nan"),
        "length_over_5um": int((lengths > 5).sum()),
        "width_median_um": float(np.median(widths)) if widths.size else float("nan"),
        "width_count": int(widths.size),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--xy-um", type=float, required=True)
    parser.add_argument("--z-um", type=float, required=True)
    parser.add_argument("--pattern", default="*_ch00.tif")
    parser.add_argument("--plane", default=None)
    parser.add_argument("--roi", default="", help="optional ROI mask file")
    parser.add_argument(
        "--resample-to", type=float, default=0.0,
        help=f"resample to this pixel size first, e.g. {TRAINED_PIXEL_UM} to match training",
    )
    parser.add_argument("--na", type=float, default=1.2)
    parser.add_argument("--wavelength-um", type=float, default=0.488)
    args = parser.parse_args(argv)

    saturn = load_pipeline()
    source = Path(args.input).resolve()
    paths = sorted(str(p) for p in source.glob(args.pattern))
    if not paths:
        raise SystemExit(f"no slices matching {args.pattern} in {source}")

    xy_um = args.xy_um
    if args.resample_to:
        import tempfile

        import tifffile
        from scipy import ndimage as ndi

        factor = args.xy_um / args.resample_to
        staging = Path(tempfile.mkdtemp(prefix="transfer_probe_"))
        rescaled = []
        for index, path in enumerate(paths):
            plane = np.asarray(
                saturn.ensure_2d_image(saturn.robust_imread(path), Path(path).name),
                dtype=float,
            )
            out = staging / f"probe_z{index:02d}_ch00.tif"
            tifffile.imwrite(
                str(out),
                np.clip(ndi.zoom(plane, factor, order=1), 0, 255).astype(np.uint8),
                photometric="minisblack",
            )
            rescaled.append(str(out))
        paths, xy_um = rescaled, args.resample_to
        print(f"resampled {args.xy_um:.4f} -> {xy_um:.4f} um/px (factor {factor:.3f})")

    roi = None
    if args.roi:
        first = saturn.ensure_2d_image(saturn.robust_imread(paths[0]), "first")
        roi = saturn.load_roi_mask_file(Path(args.roi), expected_shape=np.asarray(first).shape)

    result = probe(saturn, paths, xy_um, args.z_um, roi=roi, plane=args.plane)
    limit = lateral_resolution_um(args.wavelength_um, args.na)

    print(f"\nplane {result['plane']} of {len(paths)}, {xy_um:.4f} um/px"
          f"{', region supplied' if roi is not None else ', no region'}")
    print(f"  objects detected          {result['objects']}")
    print(f"  frame claimed by detector {100*result['labelled_fraction']:.3f}%")
    print(f"  frame above background    {100*result['above_background_fraction']:.3f}%")
    print(f"  median object length      {result['length_median_um']:.2f} um"
          f"   (a real nucleus is about {REFERENCE_LENGTH_UM:.1f})")
    print(f"  objects longer than 5 um  {result['length_over_5um']} of {result['objects']}")
    print(f"  median reported width     {result['width_median_um']:.3f} um"
          f" on {result['width_count']}   (reference {REFERENCE_WIDTH_UM:.2f})")
    print(f"  optical limit for this NA {limit:.3f} um")

    # State the reading, but keep it clearly separable from the measurements.
    suspect = []
    if result["labelled_fraction"] > 5 * max(result["above_background_fraction"], 1e-6):
        suspect.append("detector claims far more of the frame than carries signal")
    if result["length_median_um"] < REFERENCE_LENGTH_UM / 3:
        suspect.append("median object is far shorter than a nucleus")
    if np.isfinite(result["width_median_um"]) and result["width_median_um"] < limit:
        suspect.append("median width is below the optical limit, so it cannot be real")
    print("\n  reading: " + ("transfer looks SOUND" if not suspect
                             else "transfer looks UNSOUND -> " + "; ".join(suspect)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
