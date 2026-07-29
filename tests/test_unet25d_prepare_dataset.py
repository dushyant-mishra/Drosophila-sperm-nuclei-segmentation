import importlib.util
import sys
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


def load_unet25d_module(name):
    module_path = ROOT / "unet25d" / f"{name}.py"
    sys.path.insert(0, str(module_path.parent))
    try:
        spec = importlib.util.spec_from_file_location(f"unet25d_{name}_test", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


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


def test_training_dataset_can_balance_selected_synthetic_z_indices(tmp_path):
    train = load_unet25d_module("train_unet25d")
    for z in (110, 120, 300, 310):
        (tmp_path / f"Project001_Series002_z{z:04d}_ch00.npz").touch()

    dataset = train.SpermPatchDataset(
        tmp_path,
        patch_size=32,
        patches_per_image=4,
        augment=False,
        seed=7,
        repeat_z_indices=[110, 120],
        repeat_factor=2,
    )

    assert len(dataset.paths) == 6
    assert len(dataset) == 24
    repeated = [path.name for path in dataset.paths]
    assert repeated.count("Project001_Series002_z0110_ch00.npz") == 2
    assert repeated.count("Project001_Series002_z0300_ch00.npz") == 1


def test_photometric_augmentation_changes_images_not_masks(tmp_path):
    train = load_unet25d_module("train_unet25d")
    path = tmp_path / "Project001_Series002_z0110_ch00.npz"
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[3:5, 3:5] = 1
    np.savez_compressed(
        path,
        image=np.full((3, 8, 8), 0.5, dtype=np.float32),
        mask=mask,
        supervision_mask=np.ones((8, 8), dtype=np.uint8),
    )
    dataset = train.SpermPatchDataset(
        tmp_path,
        patch_size=8,
        patches_per_image=1,
        augment=True,
        seed=11,
        photometric_augment_probability=1.0,
        photometric_gain_range=(0.5, 0.5),
        photometric_gamma_range=(1.0, 1.0),
        photometric_noise_std_max=0.0,
    )

    image, target, supervision = dataset[0]

    np.testing.assert_allclose(image.numpy(), 0.25)
    assert int(target.sum()) == int(mask.sum())
    assert int(supervision.sum()) == mask.size
