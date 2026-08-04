"""Audit package identity, split leakage, environment, and generated targets."""

import argparse
import csv
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path

import numpy as np
import PIL
import scipy
import skimage
import torch
import yaml


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def aggregate_target_hash(sample_paths):
    digest = hashlib.sha256()
    for path in sorted(sample_paths):
        digest.update(path.name.encode("utf-8"))
        with np.load(path) as sample:
            for key in (
                "foreground_target",
                "instance_labels",
                "instance_core_labels",
                "overlap_count_map",
                "loss_weight_mask",
            ):
                value = np.ascontiguousarray(sample[key])
                digest.update(key.encode("ascii"))
                digest.update(str(value.shape).encode("ascii"))
                digest.update(value.dtype.str.encode("ascii"))
                digest.update(value.tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    package = Path(args.package)
    checkpoint = Path(args.checkpoint)
    repo = Path(args.repo)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    annotation_path = package / "annotations" / "_annotations.coco.json"
    split_path = package / "split_manifest.csv"
    combined_sources_path = package / "combined_training_sources.csv"
    key_path = package / "annotation_key_private.csv"
    required = [annotation_path, split_path, key_path, checkpoint]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing preflight inputs: {missing}")

    split_rows = read_rows(split_path)
    all_source_rows = (
        read_rows(combined_sources_path)
        if combined_sources_path.exists()
        else split_rows
    )
    key_rows = read_rows(key_path)
    split_by_name = {
        row.get("annotation_file") or row.get("synthetic_file"): row["split"]
        for row in all_source_rows
    }
    train_names = {name for name, split in split_by_name.items() if split == "train"}
    valid_names = {name for name, split in split_by_name.items() if split == "valid"}
    image_name_overlap = sorted(train_names & valid_names)

    image_hashes = {"train": {}, "valid": {}}
    for row in all_source_rows:
        annotation_file = row.get("annotation_file") or row.get("synthetic_file")
        z = int(row["synthetic_z"])
        image_path = package / "raw_tiffs" / f"Project001_Series002_z{z:04d}_ch00.tif"
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        image_hashes[row["split"]][annotation_file] = sha256_file(image_path)
    exact_hash_overlap = sorted(
        set(image_hashes["train"].values()) & set(image_hashes["valid"].values())
    )

    train_specimens = {
        row["specimen"] for row in key_rows if row.get("split") == "train"
    }
    valid_specimens = {
        row["specimen"] for row in key_rows if row.get("split") == "valid"
    }
    specimen_overlap = sorted(train_specimens & valid_specimens)

    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    image_names = {int(row["id"]): row["file_name"] for row in coco["images"]}
    annotation_counts = Counter(
        image_names[int(annotation["image_id"])] for annotation in coco["annotations"]
    )
    focus_values = np.asarray(
        [float(row["focus_score"]) for row in key_rows], dtype=np.float64
    )
    focus_low, focus_high = np.quantile(focus_values, [1 / 3, 2 / 3])
    validation_summary = []
    for row in key_rows:
        if row.get("split") != "valid":
            continue
        focus = float(row["focus_score"])
        category = "low" if focus <= focus_low else "high" if focus >= focus_high else "intermediate"
        validation_summary.append(
            {
                "annotation_file": row["annotation_file"],
                "genotype": row["group"],
                "specimen": row["specimen"],
                "annotation_count": int(annotation_counts[row["annotation_file"]]),
                "focus_score": focus,
                "focus_category": category,
            }
        )

    file_hashes = {
        "annotation_coco": sha256_file(annotation_path),
        "split_manifest": sha256_file(split_path),
        "warm_start_checkpoint": sha256_file(checkpoint),
        "prepare_dataset_py": sha256_file(repo / "unet25d" / "prepare_dataset.py"),
        "train_unet25d_py": sha256_file(repo / "unet25d" / "train_unet25d.py"),
    }
    if combined_sources_path.exists():
        file_hashes["combined_training_sources"] = sha256_file(
            combined_sources_path
        )
    config_records = []
    target_records = []
    for config_name in args.config:
        config_path = Path(config_name)
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        file_hashes[f"config:{config_path.name}"] = sha256_file(config_path)
        config_records.append(
            {
                "path": str(config_path.resolve()),
                "sha256": file_hashes[f"config:{config_path.name}"],
                "output_dir": cfg.get("output_dir"),
            }
        )
        dataset = Path(cfg["output_dir"]) / "dataset"
        for split in ("train", "valid"):
            manifest = dataset / split / "manifest.csv"
            samples = sorted((dataset / split).glob("*.npz"))
            if manifest.exists() and samples:
                target_records.append(
                    {
                        "config": config_path.name,
                        "split": split,
                        "generated_manifest_sha256": sha256_file(manifest),
                        "generated_target_arrays_sha256": aggregate_target_hash(samples),
                        "sample_count": len(samples),
                    }
                )

    environment = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": np.__version__,
        "scikit_image": skimage.__version__,
        "scipy": scipy.__version__,
        "pillow": PIL.__version__,
    }
    report = {
        "environment": environment,
        "file_hashes": file_hashes,
        "configurations": config_records,
        "generated_targets": target_records,
        "split_audit": {
            "train_image_count": len(train_names),
            "validation_image_count": len(valid_names),
            "train_validation_name_overlap": image_name_overlap,
            "train_validation_exact_image_hash_overlap": exact_hash_overlap,
            "train_validation_specimen_overlap": specimen_overlap,
            "pass": not image_name_overlap and not exact_hash_overlap and not specimen_overlap,
        },
        "validation_summary": validation_summary,
        "validation_limitation": (
            "The engineering validation set contains four images from four specimens. "
            "Annotated nuclei are not independent biological replicates."
        ),
    }
    report_path = output / "preflight_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with open(output / "validation_split_summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(validation_summary[0]))
        writer.writeheader()
        writer.writerows(validation_summary)
    print(json.dumps(report, indent=2))
    if not report["split_audit"]["pass"]:
        raise ValueError(f"Train/validation leakage detected; inspect {report_path}")


if __name__ == "__main__":
    main()
