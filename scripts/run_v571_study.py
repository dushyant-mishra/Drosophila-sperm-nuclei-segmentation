"""Run Saturn v5.7.1 Study Manager reproducibly from the command line."""

import argparse
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = ROOT / "sperm_segmentation_saturnv5.7.1.py"
DEFAULT_PROFILE = (
    ROOT / "production_profiles" / "saturn_v5_7_1_model_c_epoch003.json"
)


def load_pipeline():
    spec = importlib.util.spec_from_file_location("saturn_v571_study", PIPELINE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_config(module, preset_path):
    cfg, _ = module.load_analysis_profile(
        Path(preset_path),
        module.CONFIG,
    )
    cfg.update(
        {
            "RUN_MODE": "batch",
            "ANALYSIS_MODE": "comparative",
            "SEGMENTATION_ENGINE": "unet_primary",
            "TRACKING_BACKEND": "global_assignment",
            "AUTO_LEICA_CALIBRATION": True,
            "DO_TRACKING": True,
            "SHOW_PREVIEW_WINDOW": False,
        }
    )
    checkpoint = Path(str(cfg["UNET_MODEL_PATH"]))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"U-Net checkpoint was not found: {checkpoint}")
    module.validate_analysis_runtime_config(cfg)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--params", default=str(DEFAULT_PROFILE))
    parser.add_argument(
        "--sample-id",
        action="append",
        default=[],
        help="Run only this discovered sample ID; repeat for multiple samples.",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--lean-output",
        action="store_true",
        help="Keep biological overlays and compact reports; omit redundant per-slice diagnostics.",
    )
    args = parser.parse_args()

    module = load_pipeline()
    cfg = load_config(module, args.params)
    if args.lean_output:
        cfg.update(
            {
                "SAVE_DETAIL_FIGURE": False,
                "SAVE_MASK_TIFS": False,
                "SAVE_LABEL_TIFS": True,
                "UNET_SAVE_PROBABILITY_MAPS": False,
                "SAVE_TECHNICAL_REVIEW_OVERLAYS": False,
                "REPORT_MAX_SLICE_PAGES": 6,
            }
        )
    rows = module.discover_multisample_study(args.study_root, base_cfg=cfg)
    selected = set(args.sample_id)
    if selected:
        discovered = {row["sample_id"] for row in rows}
        missing = sorted(selected - discovered)
        if missing:
            raise ValueError(f"Unknown sample IDs: {', '.join(missing)}")
        for row in rows:
            row["include"] = row["sample_id"] in selected

    included = [row for row in rows if row["include"]]
    print(f"Discovered {len(rows)} specimens; running {len(included)}.")
    for row in included:
        print(
            f"  {row['sample_id']} [{row['group']}] "
            f"{row['slice_count']} slices; "
            f"XY={row['xy_um_per_pixel']:.9f} um/px; "
            f"Z={row['z_um_per_slice']:.9f} um/slice"
        )

    def progress(event):
        print(
            f"[{event['position']}/{event['total']}] "
            f"{event['event']}: {event['sample_id']} "
            f"{event['message']}",
            flush=True,
        )

    state, summary = module.run_multisample_study(
        rows,
        args.output_root,
        base_cfg=cfg,
        progress_callback=progress,
        resume=not args.no_resume,
        study_root=args.study_root,
    )
    print(f"Study status: {state.get('run_status')}")
    if not summary.empty:
        columns = [
            column
            for column in (
                "sample_id",
                "group",
                "estimated_unique_nuclei",
                "median_2d_length_um",
                "median_projection_z_extent_um",
            )
            if column in summary.columns
        ]
        print(summary[columns].to_string(index=False))


if __name__ == "__main__":
    main()
