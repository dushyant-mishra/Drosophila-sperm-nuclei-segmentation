import importlib.util
from pathlib import Path

import numpy as np
import tifffile


ROOT = Path(__file__).resolve().parents[1]


def load_prepare_dataset():
    spec = importlib.util.spec_from_file_location(
        "unet25d_prepare_dataset_test",
        ROOT / "unet25d" / "prepare_dataset.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_context_loader_supports_synthetic_indices_above_87(tmp_path):
    prepare = load_prepare_dataset()
    pattern = "Project001_Series002_z{z:04d}_ch00.tif"
    for z, value in ((109, 10), (110, 20), (111, 30)):
        tifffile.imwrite(tmp_path / pattern.format(z=z), np.full((8, 9), value, np.uint16))

    context = prepare.load_context(tmp_path, pattern, 110)

    assert context.shape == (3, 8, 9)
    assert float(context[0].mean()) < float(context[1].mean()) < float(context[2].mean())


def test_per_sample_roi_is_loaded_and_shape_checked(tmp_path):
    prepare = load_prepare_dataset()
    pattern = "Project001_Series002_z{z:04d}_ch00.npy"
    roi = np.zeros((7, 8), dtype=bool)
    roi[2:5, 3:6] = True
    np.save(tmp_path / pattern.format(z=120), roi)
    cfg = {"roi_mask_dir": str(tmp_path), "roi_mask_pattern": pattern}

    loaded = prepare.load_sample_roi(cfg, 120, roi.shape)

    np.testing.assert_array_equal(loaded, roi)
