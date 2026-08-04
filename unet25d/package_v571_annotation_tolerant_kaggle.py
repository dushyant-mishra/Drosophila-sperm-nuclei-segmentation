"""Build the small code bundle consumed by the v5.7.1 Kaggle notebook."""

import argparse
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "Kaggle notebook inputs"
    / "v571_annotation_tolerant_code_bundle.zip"
)
DEFAULT_CHECKPOINT = (
    ROOT
    / "Kaggle notebook outputs"
    / "v57_kj_wt_training_export"
    / "checkpoints"
    / "epoch_003.pt"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    args = parser.parse_args()
    output = Path(args.output)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)

    files = []
    for path in (ROOT / "unet25d").rglob("*"):
        if not path.is_file():
            continue
        if any(part in {"__pycache__", "outputs", "dataset"} for part in path.parts):
            continue
        if path.suffix.lower() in {".py", ".yaml", ".md", ".txt", ".ipynb"}:
            files.append(path)
    files.append(ROOT / "tests" / "test_unet25d_prepare_dataset.py")

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(set(files)):
            handle.write(path, Path("repo") / path.relative_to(ROOT))
        handle.write(checkpoint, Path("repo") / "warm_start" / "epoch_003.pt")
    print(f"Created {output} with {len(set(files))} files ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
