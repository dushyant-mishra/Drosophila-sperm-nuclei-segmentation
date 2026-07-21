"""
Optional 2.5D U-Net inference bridge for Saturn v5.7.

This module is intentionally training-data agnostic. COCO files are used only
for training the U-Net. Saturn runtime inference needs only:

- three adjacent raw image planes, ordered [z-1, z, z+1]
- a trained checkpoint
- the model configuration stored in that checkpoint

Torch is imported lazily so the classical Saturn pipeline can run unchanged on
environments that do not have PyTorch installed.
"""

from pathlib import Path

import numpy as np


def robust_normalize_stack(stack):
    arr = stack.astype(np.float32)
    lo = np.percentile(arr, 1.0)
    hi = np.percentile(arr, 99.5)
    if hi <= lo:
        hi = lo + 1.0
    arr = (arr - lo) / (hi - lo)
    return np.clip(arr, 0.0, 1.0)


def predict_probability(context_stack, checkpoint_path, device=None):
    """
    Return a 2D U-Net probability map for one center slice.

    Args:
        context_stack: numpy array shaped (3, height, width), ordered [z-1, z, z+1].
        checkpoint_path: path to a U-Net checkpoint saved by unet25d/train_unet25d.py.
        device: optional torch device string. Defaults to cuda if available, else cpu.
    """
    import sys

    import torch

    repo_root = Path(__file__).resolve().parents[1]
    unet_dir = repo_root / "unet25d"
    if str(unet_dir) not in sys.path:
        sys.path.insert(0, str(unet_dir))

    from train_unet25d import build_model

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    payload = torch.load(checkpoint_path, map_location=device)
    cfg = payload.get("config", {"architecture": "residual_attention_unet", "base_channels": 24})
    model = build_model(cfg).to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    x = robust_normalize_stack(np.asarray(context_stack, dtype=np.float32))
    tensor = torch.from_numpy(x[None, ...]).to(device)
    with torch.inference_mode():
        prob = torch.sigmoid(model(tensor))[0, 0].detach().cpu().numpy()
    return np.clip(prob.astype(np.float32), 0.0, 1.0)
