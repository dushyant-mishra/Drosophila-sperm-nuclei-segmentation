import argparse
import shutil
from pathlib import Path

import yaml

from prepare_dataset import load_config


def write_config(base_cfg, threshold, output_dir, config_path):
    cfg = dict(base_cfg)
    cfg["threshold"] = float(threshold)
    cfg["output_dir"] = str(output_dir)
    cfg.pop("checkpoint_mirror_dir", None)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def run_command(cmd):
    print("Running:", " ".join(str(x) for x in cmd))
    import subprocess

    subprocess.run([str(x) for x in cmd], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.5, 0.6, 0.7])
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    source_output_dir = Path(base_cfg["output_dir"])
    output_root = Path(args.output_root) if args.output_root else source_output_dir.parent / f"{source_output_dir.name}_threshold_sweep"
    output_root.mkdir(parents=True, exist_ok=True)

    for threshold in args.thresholds:
        tag = f"thr_{threshold:.2f}".replace(".", "p")
        run_dir = output_root / tag
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = run_dir / "config.yaml"
        write_config(base_cfg, threshold, run_dir, cfg_path)

        run_command(["python", "infer_unet25d.py", "--config", cfg_path, "--checkpoint", args.checkpoint])
        run_command(["python", "review_overlays.py", "--config", cfg_path])

    if output_root.exists():
        archive_base = output_root.parent / output_root.name
        shutil.make_archive(str(archive_base), "zip", output_root)
        print(f"Saved threshold sweep zip: {archive_base}.zip")


if __name__ == "__main__":
    main()
