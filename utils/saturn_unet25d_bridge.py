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
import hashlib

import numpy as np

_MODEL_CACHE = {}
_REPORTED_DEVICES = set()


def _checkpoint_sha256(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _device_helpers():
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    unet_dir = repo_root / "unet25d"
    if str(unet_dir) not in sys.path:
        sys.path.insert(0, str(unet_dir))
    from torch_device import describe_torch_device, select_torch_device

    return select_torch_device, describe_torch_device


def _resolve_device(device=None):
    select_torch_device, describe_torch_device = _device_helpers()
    resolved = select_torch_device(preferred=device)
    if resolved not in _REPORTED_DEVICES:
        print(f"    U-Net inference device: {describe_torch_device(resolved)}")
        _REPORTED_DEVICES.add(resolved)
    return resolved


def robust_normalize_stack(stack):
    arr = stack.astype(np.float32)
    lo = np.percentile(arr, 1.0)
    hi = np.percentile(arr, 99.5)
    if hi <= lo:
        hi = lo + 1.0
    arr = (arr - lo) / (hi - lo)
    return np.clip(arr, 0.0, 1.0)


def _tile_starts(start, stop, tile_size, overlap):
    span = stop - start
    if span <= tile_size:
        return [start]

    step = max(1, tile_size - overlap)
    starts = list(range(start, stop - tile_size + 1, step))
    last = stop - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _blend_window(tile_h, tile_w):
    wy = np.hanning(tile_h) if tile_h > 2 else np.ones(tile_h, dtype=np.float32)
    wx = np.hanning(tile_w) if tile_w > 2 else np.ones(tile_w, dtype=np.float32)
    win = np.outer(wy, wx).astype(np.float32)
    return np.maximum(win, 0.05)


def _roi_bbox(roi, padding):
    ys, xs = np.where(roi)
    if len(ys) == 0:
        raise ValueError("ROI is empty")
    h, w = roi.shape
    y0 = max(0, int(ys.min()) - int(padding))
    y1 = min(h, int(ys.max()) + int(padding) + 1)
    x0 = max(0, int(xs.min()) - int(padding))
    x1 = min(w, int(xs.max()) + int(padding) + 1)
    return y0, y1, x0, x1


def _cfg_get(cfg, upper_key, lower_key=None, default=None):
    if cfg is None:
        return default
    if upper_key in cfg:
        return cfg[upper_key]
    if lower_key and lower_key in cfg:
        return cfg[lower_key]
    return default


def _load_model(checkpoint_path, device):
    import sys

    import torch

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    cache_key = (
        str(checkpoint_path.resolve()),
        _checkpoint_sha256(checkpoint_path),
        str(device),
    )
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    repo_root = Path(__file__).resolve().parents[1]
    unet_dir = repo_root / "unet25d"
    if str(unet_dir) not in sys.path:
        sys.path.insert(0, str(unet_dir))

    from train_unet25d import build_model

    payload = torch.load(checkpoint_path, map_location="cpu")
    cfg = payload.get("config", {"architecture": "residual_attention_unet", "base_channels": 24})
    model = build_model(cfg)
    model.load_state_dict(payload["model"])
    model = model.to(device)
    model.eval()
    _MODEL_CACHE[cache_key] = (model, cfg)
    return model, cfg


def predict_probability(context_stack, checkpoint_path, device=None):
    """
    Return a 2D U-Net probability map for one center slice.

    Args:
        context_stack: numpy array shaped (3, height, width), ordered [z-1, z, z+1].
        checkpoint_path: path to a U-Net checkpoint saved by unet25d/train_unet25d.py.
        device: optional torch device string. Defaults to CUDA, then Apple MPS,
            then CPU according to availability.
    """
    import torch

    device = _resolve_device(device)

    model, _ = _load_model(checkpoint_path, device)

    x = robust_normalize_stack(np.asarray(context_stack, dtype=np.float32))
    tensor = torch.from_numpy(x[None, ...]).to(device)
    with torch.inference_mode():
        output = model(tensor)
        if isinstance(output, dict):
            output = output["foreground"]
        prob = torch.sigmoid(output)[0, 0].detach().cpu().numpy()
    return np.clip(prob.astype(np.float32), 0.0, 1.0)


def predict_probability_heads_tiled(
    context_stack,
    checkpoint_path,
    roi_mask=None,
    cfg=None,
    device=None,
):
    """
    Return stitched full-frame probability maps for every production head.

    Single-head checkpoints return ``foreground`` only. Dual-head checkpoints
    return both ``foreground`` and ``core``. Thresholding and geometry remain
    in Saturn.
    """
    import torch

    device = _resolve_device(device)
    model, _ = _load_model(checkpoint_path, device)

    context = robust_normalize_stack(np.asarray(context_stack, dtype=np.float32))
    if context.ndim != 3 or context.shape[0] != 3:
        raise ValueError(f"context_stack must have shape (3, H, W), got {context.shape}")
    _, h, w = context.shape
    if roi_mask is None:
        roi = np.ones((h, w), dtype=bool)
    else:
        roi = np.asarray(roi_mask, dtype=bool)
        if roi.shape != (h, w):
            raise ValueError(f"roi_mask shape {roi.shape} does not match image shape {(h, w)}")

    tile_size = int(_cfg_get(cfg, "UNET_TILE_SIZE", "unet_tile_size", 256))
    overlap = int(_cfg_get(cfg, "UNET_TILE_OVERLAP", "unet_tile_overlap", 64))
    padding = int(_cfg_get(cfg, "UNET_ROI_PADDING_PX", "unet_roi_padding_px", 32))
    batch_size = int(_cfg_get(cfg, "UNET_TILE_BATCH_SIZE", "unet_tile_batch_size", 8))
    outside_zero = bool(_cfg_get(cfg, "UNET_OUTSIDE_ROI_ZERO", "unet_outside_roi_zero", True))

    if tile_size <= 0:
        raise ValueError("UNET_TILE_SIZE must be positive")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("UNET_TILE_OVERLAP must be >= 0 and < UNET_TILE_SIZE")
    if batch_size <= 0:
        raise ValueError("UNET_TILE_BATCH_SIZE must be positive")

    y0, y1, x0, x1 = _roi_bbox(roi, padding)
    y_starts = _tile_starts(y0, y1, tile_size, overlap)
    x_starts = _tile_starts(x0, x1, tile_size, overlap)
    prob_sums = {}
    weight_sum = np.zeros((h, w), dtype=np.float32)

    def flush_batch(batch, meta):
        if not batch:
            return
        tensor = torch.from_numpy(np.stack(batch, axis=0)).to(device)
        output = model(tensor)
        outputs = output if isinstance(output, dict) else {"foreground": output}
        if "foreground" not in outputs:
            raise KeyError("U-Net output is missing the foreground head")
        prediction_batches = {
            name: torch.sigmoid(logits)[:, 0].detach().cpu().numpy()
            for name, logits in outputs.items()
            if name in {"foreground", "core"}
        }
        for batch_index, (yy, xx, ph, pw) in enumerate(meta):
            win = _blend_window(ph, pw)
            for name, predictions in prediction_batches.items():
                if name not in prob_sums:
                    prob_sums[name] = np.zeros((h, w), dtype=np.float32)
                pred = predictions[batch_index, :ph, :pw].astype(np.float32)
                prob_sums[name][yy:yy + ph, xx:xx + pw] += pred * win
            weight_sum[yy:yy + ph, xx:xx + pw] += win

    batch = []
    meta = []
    with torch.inference_mode():
        for yy in y_starts:
            for xx in x_starts:
                patch = context[:, yy:yy + tile_size, xx:xx + tile_size]
                _, ph, pw = patch.shape
                if ph != tile_size or pw != tile_size:
                    padded = np.zeros((3, tile_size, tile_size), dtype=np.float32)
                    padded[:, :ph, :pw] = patch
                    patch = padded
                batch.append(patch)
                meta.append((yy, xx, ph, pw))
                if len(batch) >= batch_size:
                    flush_batch(batch, meta)
                    batch = []
                    meta = []
        flush_batch(batch, meta)

    valid = weight_sum > 0
    probabilities = {}
    for name, probability_sum in prob_sums.items():
        probability = np.zeros((h, w), dtype=np.float32)
        probability[valid] = probability_sum[valid] / weight_sum[valid]
        if outside_zero:
            probability[~roi] = 0.0
        probabilities[name] = np.clip(probability, 0.0, 1.0)
    return probabilities


def predict_probability_tiled(
    context_stack,
    checkpoint_path,
    roi_mask=None,
    cfg=None,
    device=None,
):
    """Return the foreground head for backward-compatible Saturn callers."""
    return predict_probability_heads_tiled(
        context_stack,
        checkpoint_path,
        roi_mask=roi_mask,
        cfg=cfg,
        device=device,
    )["foreground"]
