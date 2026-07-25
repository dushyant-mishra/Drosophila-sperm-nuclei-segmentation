"""Consistent PyTorch accelerator selection for Saturn and U-Net utilities."""


def select_torch_device(preferred=None, torch_module=None):
    """
    Select CUDA, Apple Metal (MPS), or CPU in that order.

    An explicit device is validated so an unavailable accelerator cannot
    silently fall back to CPU.
    """
    if torch_module is None:
        import torch as torch_module

    torch = torch_module
    requested = str(preferred).strip().lower() if preferred is not None else ""

    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    mps_available = bool(mps_backend is not None and mps_backend.is_available())
    cuda_available = bool(torch.cuda.is_available())

    if requested:
        device_type = requested.split(":", 1)[0]
        if device_type == "cuda" and not cuda_available:
            raise RuntimeError(f"Requested PyTorch device '{requested}', but CUDA is unavailable")
        if device_type == "mps" and not mps_available:
            raise RuntimeError(f"Requested PyTorch device '{requested}', but Apple MPS is unavailable")
        if device_type not in {"cuda", "mps", "cpu"}:
            raise ValueError(f"Unsupported PyTorch device '{requested}'")
        return requested

    if cuda_available:
        return "cuda"
    if mps_available:
        return "mps"
    return "cpu"


def describe_torch_device(device, torch_module=None):
    """Return a short user-facing description of the selected backend."""
    if torch_module is None:
        import torch as torch_module

    device = str(device)
    if device.startswith("cuda"):
        index = 0
        if ":" in device:
            index = int(device.split(":", 1)[1])
        return f"{device} ({torch_module.cuda.get_device_name(index)})"
    if device == "mps":
        return "mps (Apple Metal Performance Shaders)"
    return "cpu"
