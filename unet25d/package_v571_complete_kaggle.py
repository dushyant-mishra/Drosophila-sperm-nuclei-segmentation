"""Create one Kaggle upload containing code, checkpoint, and required data."""

import argparse
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE_BUNDLE = (
    ROOT / "Kaggle notebook inputs" / "v571_annotation_tolerant_code_bundle.zip"
)
PACKAGE = ROOT / "training_packages" / "v5_7_kj_wt_replay_finetune"
DEFAULT_OUTPUT = (
    ROOT / "Kaggle notebook inputs" / "v571_annotation_tolerant_complete_kaggle.zip"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    required_package_files = [
        PACKAGE / "annotation_key_private.csv",
        PACKAGE / "checkpoint_provenance.json",
        PACKAGE / "combined_package_summary.json",
        PACKAGE / "combined_training_sources.csv",
        PACKAGE / "split_manifest.csv",
        PACKAGE / "annotations" / "_annotations.coco.json",
    ]
    required_package_files += sorted((PACKAGE / "raw_tiffs").glob("*.tif"))
    required_package_files += sorted((PACKAGE / "roi_masks").glob("*.npy"))
    missing = [path for path in [CODE_BUNDLE, *required_package_files] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing complete-package inputs: {missing}")

    package_root = Path("v5_7_kj_wt_replay_finetune")
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as target:
        with zipfile.ZipFile(CODE_BUNDLE) as source:
            for entry in source.infolist():
                target.writestr(entry, source.read(entry.filename))
        for path in required_package_files:
            target.write(path, package_root / path.relative_to(PACKAGE))

    print(
        f"Created {output} ({output.stat().st_size} bytes) with code, checkpoint, "
        f"COCO annotations, {len(list((PACKAGE / 'raw_tiffs').glob('*.tif')))} TIFFs, "
        f"and {len(list((PACKAGE / 'roi_masks').glob('*.npy')))} ROI masks."
    )


if __name__ == "__main__":
    main()
