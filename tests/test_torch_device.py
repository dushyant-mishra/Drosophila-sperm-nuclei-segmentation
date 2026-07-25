import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_device_module():
    spec = importlib.util.spec_from_file_location(
        "saturn_torch_device_test",
        ROOT / "unet25d" / "torch_device.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_torch(cuda=False, mps=False):
    return SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: cuda,
            get_device_name=lambda _index: "Test CUDA GPU",
        ),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: mps),
        ),
    )


def test_device_selection_prefers_cuda_then_mps_then_cpu():
    devices = load_device_module()

    assert devices.select_torch_device(torch_module=fake_torch(cuda=True, mps=True)) == "cuda"
    assert devices.select_torch_device(torch_module=fake_torch(cuda=False, mps=True)) == "mps"
    assert devices.select_torch_device(torch_module=fake_torch(cuda=False, mps=False)) == "cpu"


def test_explicit_unavailable_accelerator_raises():
    devices = load_device_module()
    torch = fake_torch(cuda=False, mps=False)

    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        devices.select_torch_device("cuda", torch_module=torch)
    with pytest.raises(RuntimeError, match="Apple MPS is unavailable"):
        devices.select_torch_device("mps", torch_module=torch)


def test_device_description_names_apple_mps():
    devices = load_device_module()

    assert "Apple Metal Performance Shaders" in devices.describe_torch_device(
        "mps",
        torch_module=fake_torch(mps=True),
    )
