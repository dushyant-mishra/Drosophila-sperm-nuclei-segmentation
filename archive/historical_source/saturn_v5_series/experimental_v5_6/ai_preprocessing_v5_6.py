"""Optional AI preprocessing helpers for the Saturn v5.6 N2V2 pilot.

The production segmentation path must import without CAREamics installed. All
CAREamics/PyTorch imports therefore stay inside adapter functions.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from scipy.ndimage import binary_dilation, distance_transform_edt
from skimage import measure


AI_METHOD = "N2V2"


def discover_n2v2_backend() -> dict[str, Any]:
    info: dict[str, Any] = {"available": False, "method": AI_METHOD}
    try:
        import careamics
        from careamics import CAREamist
        from careamics.config import create_n2v_config
        import torch
    except Exception as exc:  # pragma: no cover - depends on optional env
        info["error"] = repr(exc)
        return info
    info.update({
        "available": True,
        "careamics_version": getattr(careamics, "__version__", "unknown"),
        "torch_version": getattr(torch, "__version__", "unknown"),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "careamist_class": CAREamist.__name__,
        "config_factory": create_n2v_config.__name__,
    })
    return info


def normalize_ai_input(image: np.ndarray, roi_mask: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    arr = np.asarray(image)
    source_dtype = str(arr.dtype)
    source_min = float(np.nanmin(arr))
    source_max = float(np.nanmax(arr))
    arr_f = arr.astype(np.float32, copy=False)
    if np.issubdtype(arr.dtype, np.integer):
        max_value = float(np.iinfo(arr.dtype).max)
        arr_f = arr_f / max(max_value, 1.0)
        scale_mode = f"integer_full_range_{source_dtype}"
    else:
        lo = float(np.nanpercentile(arr_f, 0.1))
        hi = float(np.nanpercentile(arr_f, 99.9))
        arr_f = (arr_f - lo) / max(hi - lo, 1e-6)
        scale_mode = "float_percentile_0p1_99p9"
    arr_f = np.clip(arr_f, 0.0, 1.0).astype(np.float32)
    if roi_mask is not None:
        roi = np.asarray(roi_mask, dtype=bool)
        fill = float(np.median(arr_f[roi])) if np.any(roi) else float(np.median(arr_f))
        arr_f = arr_f.copy()
        arr_f[~roi] = fill
    meta = {
        "source_dtype": source_dtype,
        "source_min": source_min,
        "source_max": source_max,
        "normalized_min": float(np.min(arr_f)),
        "normalized_max": float(np.max(arr_f)),
        "scale_mode": scale_mode,
        "output_dtype": "float32",
        "output_range": [0.0, 1.0],
    }
    return arr_f, meta


def restore_output_scale(prediction: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(prediction, dtype=np.float32), 0.0, 1.0)


def build_data_split(
    z_indices: list[int],
    test_z: list[int],
    seed: int = 560756,
    validation_count: int = 8,
) -> dict[str, Any]:
    z_all = sorted(int(z) for z in z_indices)
    test = sorted(int(z) for z in test_z)
    excluded = sorted({zz for z in test for zz in (z - 1, z, z + 1) if zz in z_all})
    candidates = [z for z in z_all if z not in excluded]
    rng = np.random.default_rng(seed)
    order = list(candidates)
    rng.shuffle(order)
    val = sorted(order[: min(validation_count, len(order) // 4)])
    train = sorted(z for z in candidates if z not in val)
    return {
        "random_seed": seed,
        "training_z_indices": train,
        "validation_z_indices": val,
        "test_z_indices": test,
        "excluded_buffer_z_indices": excluded,
    }


def extract_roi_patches(
    images_by_z: dict[int, np.ndarray],
    roi_mask: np.ndarray,
    z_indices: list[int],
    patch_size: int = 64,
    patches_per_slice: int = 4,
    seed: int = 560756,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    roi = np.asarray(roi_mask, dtype=bool)
    dist = distance_transform_edt(roi)
    margin = patch_size // 2
    valid = np.argwhere(dist >= margin)
    if valid.size == 0:
        raise ValueError("ROI is too small for requested patch size")
    patches = []
    records = []
    for z in z_indices:
        img, _ = normalize_ai_input(images_by_z[z], roi)
        local_valid = valid.copy()
        rng.shuffle(local_valid)
        count = 0
        for cy, cx in local_valid:
            y0, y1 = int(cy - margin), int(cy + margin)
            x0, x1 = int(cx - margin), int(cx + margin)
            patch_roi = roi[y0:y1, x0:x1]
            patch = img[y0:y1, x0:x1]
            if patch.shape != (patch_size, patch_size):
                continue
            if not np.all(patch_roi):
                continue
            if float(np.std(patch)) < 0.002:
                continue
            patches.append(patch.astype(np.float32))
            records.append({"z_index": int(z), "y0": y0, "x0": x0, "patch_size": patch_size})
            count += 1
            if count >= patches_per_slice:
                break
    if not patches:
        raise ValueError("No valid ROI patches extracted")
    return np.stack(patches, axis=0).astype(np.float32), records


def build_n2v2_training_config(
    experiment_name: str,
    patch_size: int = 64,
    batch_size: int = 8,
    num_epochs: int = 2,
    num_steps: int = 8,
    learning_rate: float = 1e-3,
) -> Any:
    from careamics.config import create_n2v_config

    return create_n2v_config(
        experiment_name=experiment_name,
        data_type="array",
        axes="SYX",
        patch_size=[patch_size, patch_size],
        batch_size=batch_size,
        num_epochs=num_epochs,
        num_steps=num_steps,
        augmentations=["x_flip", "y_flip"],
        n_val_patches=8,
        use_n2v2=True,
    )


def train_n2v2_model(
    train_patches: np.ndarray,
    val_patches: np.ndarray,
    work_dir: str | Path,
    config: Any,
) -> dict[str, Any]:
    from careamics import CAREamist

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    careamist = CAREamist(config, work_dir=work_dir, enable_progress_bar=False)
    careamist.train(train_data=train_patches, val_data=val_patches)
    losses = careamist.get_losses()
    checkpoints = [str(p) for p in careamist.get_checkpoints()]
    return {"careamist": careamist, "losses": losses, "checkpoints": checkpoints}


def load_n2v2_model(work_dir: str | Path, checkpoint_path: str | Path | None = None, config_path: str | Path | None = None) -> Any:
    from careamics import CAREamist

    if config_path is not None:
        return CAREamist(config_path, checkpoint_path=checkpoint_path, work_dir=work_dir, enable_progress_bar=False)
    return CAREamist(checkpoint_path=checkpoint_path, work_dir=work_dir, enable_progress_bar=False)


def predict_n2v2(model: Any, image: np.ndarray, tile_size: int = 256) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    pred, _ = model.predict(arr[None, ...], axes="SYX", data_type="array", tile_size=(tile_size, tile_size))
    if isinstance(pred, list):
        pred = pred[0]
    pred = np.asarray(pred)
    pred = np.squeeze(pred)
    return restore_output_scale(pred)


def validate_ai_output(raw_norm: np.ndarray, prediction: np.ndarray, roi_mask: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(prediction)
    raw = np.asarray(raw_norm)
    roi = np.asarray(roi_mask, dtype=bool)
    if pred.shape != raw.shape:
        raise ValueError(f"AI prediction shape {pred.shape} does not match raw shape {raw.shape}")
    return {
        "shape": list(pred.shape),
        "min": float(np.min(pred)),
        "max": float(np.max(pred)),
        "mean_abs_residual_roi": float(np.mean(np.abs(raw[roi] - pred[roi]))) if np.any(roi) else np.nan,
        "outside_roi_abs_change": float(np.mean(np.abs(raw[~roi] - pred[~roi]))) if np.any(~roi) else 0.0,
        "clipped_low_fraction": float(np.mean(pred <= 0.0)),
        "clipped_high_fraction": float(np.mean(pred >= 1.0)),
    }


def calculate_raw_support_metrics(
    label_mask: np.ndarray,
    raw_norm: np.ndarray,
    raw_ridge: np.ndarray,
    roi_mask: np.ndarray,
) -> dict[str, Any]:
    mask = np.asarray(label_mask, dtype=bool)
    if not np.any(mask):
        return {"raw_support_score": 0.0, "supported_fraction": 0.0, "rejection_reason": "empty_mask"}
    dilated = np.logical_and(binary_dilation(mask, iterations=4), roi_mask)
    local = np.logical_and(dilated, ~mask)
    raw_values = raw_norm[mask]
    bg = raw_norm[local] if np.any(local) else raw_norm[roi_mask]
    raw_center = float(np.median(raw_values))
    local_bg = float(np.median(bg)) if bg.size else 0.0
    contrast = raw_center - local_bg
    ridge_values = raw_ridge[mask]
    ridge_bg = raw_ridge[roi_mask]
    ridge_threshold = float(np.percentile(ridge_bg, 65)) if ridge_bg.size else 0.0
    supported_fraction = float(np.mean(ridge_values >= ridge_threshold)) if ridge_values.size else 0.0
    raw_support_score = float(max(0.0, contrast) * 0.6 + supported_fraction * 0.4)
    dist = distance_transform_edt(roi_mask)
    coords = np.argwhere(mask)
    roi_edge_distance_px = float(np.min(dist[coords[:, 0], coords[:, 1]])) if coords.size else 0.0
    reasons = []
    if contrast < 0.02:
        reasons.append("low_raw_contrast")
    if supported_fraction < 0.25:
        reasons.append("weak_raw_ridge_support")
    if roi_edge_distance_px < 2:
        reasons.append("roi_edge")
    return {
        "raw_centerline_intensity": raw_center,
        "local_raw_background": local_bg,
        "raw_local_contrast": float(contrast),
        "raw_ridge_support_median": float(np.median(ridge_values)) if ridge_values.size else 0.0,
        "supported_fraction": supported_fraction,
        "raw_support_score": raw_support_score,
        "roi_edge_distance_px": roi_edge_distance_px,
        "rejection_reason": ",".join(reasons),
        "accepted_by_raw_support": not reasons,
    }


def load_ilastik_probability_map(
    path: str | Path,
    expected_shape: tuple[int, int],
    roi_mask: np.ndarray,
    z_index: int | None = None,
    metadata_path: str | Path | None = None,
    hdf5_dataset_key: str | None = None,
    nucleus_channel: int | None = None,
    expected_class_order: list[str] | None = None,
) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".h5", ".hdf5"):
        try:
            import h5py
        except Exception as exc:
            raise ImportError("h5py is required to read ilastik HDF5 probability maps") from exc
        with h5py.File(path, "r") as h5:
            if hdf5_dataset_key is None:
                keys = []
                h5.visititems(lambda name, obj: keys.append(name) if hasattr(obj, "shape") else None)
                if len(keys) != 1:
                    raise ValueError("Ambiguous HDF5 probability map; provide hdf5_dataset_key")
                hdf5_dataset_key = keys[0]
            if hdf5_dataset_key not in h5:
                raise ValueError(f"HDF5 dataset key not found: {hdf5_dataset_key}")
            arr = np.asarray(h5[hdf5_dataset_key])
    else:
        arr = tifffile.imread(str(path))
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] >= 4 and arr.shape[-1] != 4:
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim != 3:
        raise ValueError(f"Expected probability map with channel dimension, got shape {arr.shape}")
    if arr.shape[:2] != tuple(expected_shape):
        raise ValueError(f"Probability map shape {arr.shape[:2]} does not match expected {expected_shape}")
    if arr.shape[2] < 4:
        raise ValueError(f"Expected at least 4 probability channels, got {arr.shape[2]}")
    if np.isnan(arr).any() or np.isinf(arr).any():
        raise ValueError("Probability map contains NaN or infinity values")
    if float(np.nanmin(arr)) < -1e-6 or float(np.nanmax(arr)) > 1.0 + 1e-6:
        raise ValueError("Probability map values must be in [0, 1]")
    if np.asarray(roi_mask).shape != tuple(expected_shape):
        raise ValueError("ROI mask shape does not match expected shape")
    class_order = None
    if metadata_path is not None and Path(metadata_path).exists():
        meta = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        if z_index is not None and int(meta.get("z_index", z_index)) != int(z_index):
            raise ValueError("Probability map metadata Z index mismatch")
        class_order = meta.get("class_order")
    if expected_class_order is not None and class_order is not None and list(class_order) != list(expected_class_order):
        raise ValueError("Probability map class order does not match expected class order")
    if class_order is None and nucleus_channel is None:
        raise ValueError("Missing class metadata; provide explicit nucleus_channel")
    if nucleus_channel is not None and not (0 <= int(nucleus_channel) < arr.shape[2]):
        raise ValueError("nucleus_channel is outside probability channel range")
    return arr.astype(np.float32)


def checksum_model_weights(model: Any) -> str:
    """Return a stable SHA256 checksum for a torch-like model state dict."""
    if not hasattr(model, "state_dict"):
        raise ValueError("Object has no state_dict; cannot verify real model weights")
    hasher = hashlib.sha256()
    state = model.state_dict()
    if not state:
        raise ValueError("Model state_dict is empty")
    for key in sorted(state):
        value = state[key]
        if hasattr(value, "detach"):
            arr = value.detach().cpu().numpy()
        else:
            arr = np.asarray(value)
        hasher.update(key.encode("utf-8"))
        hasher.update(str(arr.shape).encode("utf-8"))
        hasher.update(arr.tobytes())
    return hasher.hexdigest()


def require_real_n2v2_verification(verification: dict[str, Any]) -> None:
    """Fail if a diagnostic run is not a real, changed-weight N2V2 training run."""
    required_true = [
        "real_careamics_model",
        "n2v2_specific_configuration_active",
        "blind_spot_masking_active",
        "weights_changed",
    ]
    for key in required_true:
        if not verification.get(key):
            raise RuntimeError(f"N2V2 verification failed: {key} is not true")
    if verification.get("fallback_or_mock_used"):
        raise RuntimeError("N2V2 verification failed: fallback/mock implementation was used")
    if int(verification.get("optimizer_steps_completed", 0)) <= 0:
        raise RuntimeError("N2V2 verification failed: no optimizer steps completed")
