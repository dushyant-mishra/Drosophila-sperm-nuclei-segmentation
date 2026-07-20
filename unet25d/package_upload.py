import shutil
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    out = here.parent / "unet25d_upload"
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(here, out, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "outputs"))
    zip_path = shutil.make_archive(str(out), "zip", out)
    print(zip_path)


if __name__ == "__main__":
    main()

