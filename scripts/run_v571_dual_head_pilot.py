"""Run frozen Saturn v5.7.1 Model C settings on one or more full specimens."""

import argparse
import importlib.util
import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]


def load_saturn():
    spec = importlib.util.spec_from_file_location(
        "saturn_v571_dual_head_pilot",
        ROOT / "sperm_segmentation_saturnv5.7.1.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--specimen",
        action="append",
        required=True,
        help="NAME|INPUT_DIR|ROI_PATH; repeat for each specimen",
    )
    args = parser.parse_args()

    saturn = load_saturn()
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for value in args.specimen:
        parts = value.split("|", 2)
        if len(parts) != 3:
            raise ValueError("--specimen must use NAME|INPUT_DIR|ROI_PATH")
        name, input_dir, roi_path = parts
        specimen_output = output_root / name
        if specimen_output.exists() and any(specimen_output.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite nonempty pilot output: {specimen_output}"
            )
        cfg, _ = saturn.load_analysis_profile(args.profile)
        cfg.update({
            "RUN_MODE": "batch",
            "INPUT_DIR": str(Path(input_dir).resolve()),
            "ROI_MASK_PATH": str(Path(roi_path).resolve()),
            "OUTPUT_DIR": str(specimen_output),
            "SHOW_PREVIEW_WINDOW": False,
            "SHOW_DEBUG_PREVIEW": False,
            "SAVE_DEBUG_IMAGES": False,
            "SAVE_DETAIL_FIGURE": False,
            "SAVE_TECHNICAL_REVIEW_OVERLAYS": False,
            "UNET_SAVE_PROBABILITY_MAPS": False,
            "REPORT_MAX_SLICE_PAGES": 6,
        })
        saturn.validate_analysis_runtime_config(cfg)
        started = time.time()
        status = "complete"
        error = ""
        try:
            saturn.process_batch(cfg)
        except Exception as exc:
            status = "failed"
            error = str(exc)
            raise
        finally:
            records.append({
                "specimen": name,
                "input_dir": cfg["INPUT_DIR"],
                "roi_path": cfg["ROI_MASK_PATH"],
                "output_dir": cfg["OUTPUT_DIR"],
                "status": status,
                "error": error,
                "elapsed_seconds": round(time.time() - started, 3),
            })
            (output_root / "pilot_manifest.json").write_text(
                json.dumps(records, indent=2), encoding="utf-8"
            )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
