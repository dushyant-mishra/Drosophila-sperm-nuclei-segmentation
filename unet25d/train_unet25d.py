import argparse
import csv
import random
import re
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from torch_device import describe_torch_device, select_torch_device


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class SpermPatchDataset(Dataset):
    def __init__(
        self,
        sample_dir,
        patch_size,
        patches_per_image,
        augment,
        seed,
        positive_patch_probability=0.8,
        repeat_z_indices=None,
        repeat_factor=1,
        photometric_augment_probability=0.0,
        photometric_gain_range=(1.0, 1.0),
        photometric_gamma_range=(1.0, 1.0),
        photometric_noise_std_max=0.0,
        return_core_target=False,
    ):
        base_paths = sorted(Path(sample_dir).glob("*.npz"))
        repeat_z_indices = {int(z) for z in (repeat_z_indices or [])}
        repeat_factor = max(1, int(repeat_factor))
        self.paths = []
        for path in base_paths:
            match = re.search(r"_z(\d+)_ch00", path.stem, re.IGNORECASE)
            z = int(match.group(1)) if match else None
            copies = repeat_factor if z in repeat_z_indices else 1
            self.paths.extend([path] * copies)
        self.patch_size = int(patch_size)
        self.patches_per_image = int(patches_per_image)
        self.augment = augment
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.positive_patch_probability = float(positive_patch_probability)
        self.photometric_augment_probability = float(
            photometric_augment_probability
        )
        self.photometric_gain_range = tuple(
            float(value) for value in photometric_gain_range
        )
        self.photometric_gamma_range = tuple(
            float(value) for value in photometric_gamma_range
        )
        self.photometric_noise_std_max = float(photometric_noise_std_max)
        self.return_core_target = bool(return_core_target)
        if not self.paths:
            raise FileNotFoundError(f"No .npz samples found in {sample_dir}")

    def __len__(self):
        return len(self.paths) * self.patches_per_image

    def __getitem__(self, idx):
        path = self.paths[idx % len(self.paths)]
        data = np.load(path)
        image = data["image"].astype(np.float32)
        mask = data["mask"].astype(np.float32)
        supervision = data["supervision_mask"].astype(np.float32) if "supervision_mask" in data else np.ones_like(mask, dtype=np.float32)
        _, h, w = image.shape
        ps = self.patch_size

        positives = np.argwhere(mask > 0)
        if len(positives) and self.rng.random() < self.positive_patch_probability:
            cy, cx = positives[self.rng.randrange(len(positives))]
            y0 = int(np.clip(cy - self.rng.randrange(ps), 0, max(0, h - ps)))
            x0 = int(np.clip(cx - self.rng.randrange(ps), 0, max(0, w - ps)))
        else:
            y0 = self.rng.randrange(max(1, h - ps + 1))
            x0 = self.rng.randrange(max(1, w - ps + 1))

        x = image[:, y0 : y0 + ps, x0 : x0 + ps]
        y = mask[y0 : y0 + ps, x0 : x0 + ps][None, ...]
        valid = supervision[y0 : y0 + ps, x0 : x0 + ps][None, ...]
        core = None
        if self.return_core_target:
            if "instance_core_labels" not in data:
                raise KeyError(
                    f"Dual-head training requires instance_core_labels in {path}"
                )
            core = (data["instance_core_labels"] > 0).astype(np.float32)
            core = core[y0 : y0 + ps, x0 : x0 + ps][None, ...]

        if self.augment:
            if self.rng.random() < 0.5:
                x = x[:, :, ::-1].copy()
                y = y[:, :, ::-1].copy()
                valid = valid[:, :, ::-1].copy()
                if core is not None:
                    core = core[:, :, ::-1].copy()
            if self.rng.random() < 0.5:
                x = x[:, ::-1, :].copy()
                y = y[:, ::-1, :].copy()
                valid = valid[:, ::-1, :].copy()
                if core is not None:
                    core = core[:, ::-1, :].copy()
            k = self.rng.randrange(4)
            if k:
                x = np.rot90(x, k=k, axes=(1, 2)).copy()
                y = np.rot90(y, k=k, axes=(1, 2)).copy()
                valid = np.rot90(valid, k=k, axes=(1, 2)).copy()
                if core is not None:
                    core = np.rot90(core, k=k, axes=(1, 2)).copy()
            if self.rng.random() < self.photometric_augment_probability:
                gain = self.rng.uniform(*self.photometric_gain_range)
                gamma = self.rng.uniform(*self.photometric_gamma_range)
                x = np.clip(x, 0.0, 1.0) ** gamma
                x = x * gain
                if self.photometric_noise_std_max > 0:
                    noise_std = self.rng.uniform(
                        0.0,
                        self.photometric_noise_std_max,
                    )
                    x = x + self.np_rng.normal(0.0, noise_std, size=x.shape)
                x = np.clip(x, 0.0, 1.0).astype(np.float32)

        result = (torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(valid))
        if core is not None:
            result += (torch.from_numpy(core),)
        return result


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self, in_channels=3, base=24):
        super().__init__()
        self.e1 = ConvBlock(in_channels, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.b = ConvBlock(base * 4, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.d3 = ConvBlock(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = ConvBlock(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b = self.b(self.pool(e3))
        d3 = self.d3(torch.cat([self.u3(b), e3], dim=1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], dim=1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], dim=1))
        return self.out(d1)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        residual = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual, inplace=True)


class AttentionGate(nn.Module):
    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(gate_ch, inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch))
        self.skip = nn.Sequential(nn.Conv2d(skip_ch, inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch))
        self.psi = nn.Sequential(nn.Conv2d(inter_ch, 1, 1), nn.Sigmoid())

    def forward(self, gate, skip):
        alpha = self.psi(F.relu(self.gate(gate) + self.skip(skip), inplace=True))
        return skip * alpha


class ResidualAttentionUNet(nn.Module):
    def __init__(self, in_channels=3, base=24):
        super().__init__()
        self.e1 = ResidualBlock(in_channels, base)
        self.e2 = ResidualBlock(base, base * 2)
        self.e3 = ResidualBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.b = ResidualBlock(base * 4, base * 8, dilation=2)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.a3 = AttentionGate(base * 4, base * 4, base * 2)
        self.d3 = ResidualBlock(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.a2 = AttentionGate(base * 2, base * 2, base)
        self.d2 = ResidualBlock(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.a1 = AttentionGate(base, base, max(1, base // 2))
        self.d1 = ResidualBlock(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b = self.b(self.pool(e3))
        u3 = self.u3(b)
        d3 = self.d3(torch.cat([u3, self.a3(u3, e3)], dim=1))
        u2 = self.u2(d3)
        d2 = self.d2(torch.cat([u2, self.a2(u2, e2)], dim=1))
        u1 = self.u1(d2)
        d1 = self.d1(torch.cat([u1, self.a1(u1, e1)], dim=1))
        return self.out(d1)


class DualHeadResidualAttentionUNet(ResidualAttentionUNet):
    """Experimental foreground/core model; existing single-head keys stay loadable."""

    def __init__(self, in_channels=3, base=24, deep_supervision=False):
        super().__init__(in_channels=in_channels, base=base)
        self.core_out = nn.Conv2d(base, 1, 1)
        self.deep_supervision = bool(deep_supervision)
        if self.deep_supervision:
            self.foreground_aux_half = nn.Conv2d(base * 2, 1, 1)
            self.foreground_aux_quarter = nn.Conv2d(base * 4, 1, 1)
            self.core_aux_half = nn.Conv2d(base * 2, 1, 1)
            self.core_aux_quarter = nn.Conv2d(base * 4, 1, 1)

    def forward(self, x):
        output_size = x.shape[-2:]
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b = self.b(self.pool(e3))
        u3 = self.u3(b)
        d3 = self.d3(torch.cat([u3, self.a3(u3, e3)], dim=1))
        u2 = self.u2(d3)
        d2 = self.d2(torch.cat([u2, self.a2(u2, e2)], dim=1))
        u1 = self.u1(d2)
        d1 = self.d1(torch.cat([u1, self.a1(u1, e1)], dim=1))
        outputs = {
            "foreground": self.out(d1),
            "core": self.core_out(d1),
        }
        if self.deep_supervision:
            outputs["foreground_aux"] = [
                F.interpolate(
                    self.foreground_aux_half(d2),
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                ),
                F.interpolate(
                    self.foreground_aux_quarter(d3),
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                ),
            ]
            outputs["core_aux"] = [
                F.interpolate(
                    self.core_aux_half(d2),
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                ),
                F.interpolate(
                    self.core_aux_quarter(d3),
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                ),
            ]
        return outputs


def build_model(cfg):
    architecture = str(cfg.get("architecture", "unet_small")).lower()
    base = int(cfg["base_channels"])
    if architecture == "unet_small":
        return UNetSmall(in_channels=3, base=base)
    if architecture in {"residual_attention_unet", "resatt_unet"}:
        return ResidualAttentionUNet(in_channels=3, base=base)
    if architecture in {
        "dual_head_residual_attention_unet",
        "dual_head_resatt_unet",
    }:
        return DualHeadResidualAttentionUNet(
            in_channels=3,
            base=base,
            deep_supervision=bool(cfg.get("deep_supervision", False)),
        )
    raise ValueError(f"Unknown architecture: {architecture}")


def dice_loss(logits, target, valid=None, eps=1e-6):
    prob = torch.sigmoid(logits)
    if valid is None:
        valid = torch.ones_like(target)
    inter = (prob * target * valid).sum(dim=(1, 2, 3))
    denom = (prob * valid).sum(dim=(1, 2, 3)) + (
        target * valid
    ).sum(dim=(1, 2, 3))
    return 1.0 - ((2.0 * inter + eps) / (denom + eps)).mean()


def masked_bce_loss(logits, target, valid, positive_weight=1.0):
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if positive_weight != 1.0:
        weights = torch.ones_like(target)
        weights = torch.where(target > 0.5, weights * float(positive_weight), weights)
        loss = loss * weights
    valid_sum = valid.sum().clamp_min(1.0)
    return (loss * valid).sum() / valid_sum


def soft_erode(image):
    p1 = -F.max_pool2d(-image, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-image, (1, 3), (1, 1), (0, 1))
    return torch.minimum(p1, p2)


def soft_dilate(image):
    return F.max_pool2d(image, (3, 3), (1, 1), (1, 1))


def soft_skeletonize(image, iterations=10):
    image = image.clamp(0.0, 1.0)
    opened = soft_dilate(soft_erode(image))
    skeleton = F.relu(image - opened)
    for _ in range(int(iterations)):
        image = soft_erode(image)
        opened = soft_dilate(soft_erode(image))
        delta = F.relu(image - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def soft_cldice_loss(logits, target, valid=None, iterations=10, eps=1e-6):
    probability = torch.sigmoid(logits)
    if valid is None:
        valid = torch.ones_like(target)
    probability = probability * valid
    target = target * valid
    predicted_skeleton = soft_skeletonize(probability, iterations)
    target_skeleton = soft_skeletonize(target, iterations)
    topology_precision = (
        (predicted_skeleton * target).sum(dim=(1, 2, 3)) + eps
    ) / (predicted_skeleton.sum(dim=(1, 2, 3)) + eps)
    topology_sensitivity = (
        (target_skeleton * probability).sum(dim=(1, 2, 3)) + eps
    ) / (target_skeleton.sum(dim=(1, 2, 3)) + eps)
    score = (
        2.0
        * topology_precision
        * topology_sensitivity
        / (topology_precision + topology_sensitivity + eps)
    )
    return 1.0 - score.mean()


def segmentation_loss(logits, target, valid, positive_weight=1.0):
    return masked_bce_loss(logits, target, valid, positive_weight) + dice_loss(
        logits, target, valid
    )


def batch_metrics(logits, target, valid=None, threshold=0.5):
    pred = (torch.sigmoid(logits) > threshold).float()
    if valid is None:
        valid = torch.ones_like(target)
    else:
        valid = (valid > 0).float()
    pred = pred * valid
    target = target * valid
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    dice = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    return float(dice.detach().cpu())


def run_epoch(model, loader, optimizer, device, train, cfg=None):
    cfg = cfg or {}
    model.train(train)
    total_loss = 0.0
    total_dice = 0.0
    n = 0
    for batch in loader:
        x, y, valid = batch[:3]
        core = batch[3] if len(batch) > 3 else None
        x = x.to(device)
        y = y.to(device)
        valid = valid.to(device)
        if core is not None:
            core = core.to(device)
        with torch.set_grad_enabled(train):
            output = model(x)
            positive_weight = float(getattr(loader.dataset, "positive_loss_weight", 1.0))
            if isinstance(output, dict):
                logits = output["foreground"]
                if core is None:
                    raise ValueError("Dual-head output requires a core target")
                loss = segmentation_loss(logits, y, valid, positive_weight)
                core_weight = float(cfg.get("core_loss_weight", 0.25))
                loss = loss + core_weight * segmentation_loss(
                    output["core"], core, valid, positive_weight
                )
                cldice_weight = float(cfg.get("cldice_loss_weight", 0.0))
                if cldice_weight > 0:
                    loss = loss + cldice_weight * soft_cldice_loss(
                        logits,
                        y,
                        valid,
                        int(cfg.get("cldice_iterations", 10)),
                    )
                auxiliary_weight = float(cfg.get("deep_supervision_weight", 0.0))
                if auxiliary_weight > 0:
                    auxiliary_losses = []
                    for auxiliary_logits in output.get("foreground_aux", []):
                        auxiliary_losses.append(
                            segmentation_loss(
                                auxiliary_logits, y, valid, positive_weight
                            )
                        )
                    for auxiliary_logits in output.get("core_aux", []):
                        auxiliary_losses.append(
                            core_weight
                            * segmentation_loss(
                                auxiliary_logits, core, valid, positive_weight
                            )
                        )
                    if auxiliary_losses:
                        loss = loss + auxiliary_weight * torch.stack(
                            auxiliary_losses
                        ).mean()
            else:
                logits = output
                loss = segmentation_loss(logits, y, valid, positive_weight)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        bs = x.shape[0]
        total_loss += float(loss.detach().cpu()) * bs
        total_dice += batch_metrics(logits, y, valid) * bs
        n += bs
    return total_loss / max(1, n), total_dice / max(1, n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot_unet25d.yaml")
    parser.add_argument(
        "--warm-start",
        default=None,
        help="Optional checkpoint path used to initialize model weights before a new training run.",
    )
    parser.add_argument(
        "--allow-partial-warm-start",
        action="store_true",
        help="Load only matching checkpoint tensors. Use for architecture experiments, not strict continuation.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    architecture = str(cfg.get("architecture", "unet_small")).lower()
    dual_head = architecture in {
        "dual_head_residual_attention_unet",
        "dual_head_resatt_unet",
    }

    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    out_dir = Path(cfg["output_dir"])
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    mirror_dir = Path(cfg["checkpoint_mirror_dir"]) if cfg.get("checkpoint_mirror_dir") else None
    if mirror_dir:
        mirror_dir.mkdir(parents=True, exist_ok=True)
        print(f"Mirroring checkpoints to: {mirror_dir}")

    positive_patch_probability = float(cfg.get("positive_patch_probability", 0.8))
    train_ds = SpermPatchDataset(
        out_dir / "dataset" / "train",
        cfg["patch_size"],
        cfg["patches_per_image"],
        True,
        seed,
        positive_patch_probability,
        cfg.get("train_repeat_z_indices", []),
        cfg.get("train_repeat_factor", 1),
        cfg.get("photometric_augment_probability", 0.0),
        cfg.get("photometric_gain_range", [1.0, 1.0]),
        cfg.get("photometric_gamma_range", [1.0, 1.0]),
        cfg.get("photometric_noise_std_max", 0.0),
        dual_head,
    )
    valid_ds = SpermPatchDataset(
        out_dir / "dataset" / "valid",
        cfg["patch_size"],
        max(8, cfg["patches_per_image"] // 4),
        False,
        seed + 1,
        positive_patch_probability,
        return_core_target=dual_head,
    )
    train_ds.positive_loss_weight = float(cfg.get("positive_loss_weight", 1.0))
    valid_ds.positive_loss_weight = float(cfg.get("positive_loss_weight", 1.0))

    train_loader = DataLoader(train_ds, batch_size=int(cfg["batch_size"]), shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=0)

    device = select_torch_device()
    print(f"PyTorch training device: {describe_torch_device(device)}")
    model = build_model(cfg).to(device)
    warm_start_info = ""
    if args.warm_start:
        warm_start_path = Path(args.warm_start)
        checkpoint = torch.load(warm_start_path, map_location="cpu")
        state = checkpoint.get("model", checkpoint)
        if args.allow_partial_warm_start:
            current = model.state_dict()
            compatible = {k: v for k, v in state.items() if k in current and current[k].shape == v.shape}
            current.update(compatible)
            model.load_state_dict(current)
            print(f"Partially loaded {len(compatible)} / {len(current)} tensors from warm-start checkpoint.")
        else:
            model.load_state_dict(state)
        source_epoch = checkpoint.get("epoch", "unknown") if isinstance(checkpoint, dict) else "unknown"
        warm_start_info = f"{warm_start_path} (source_epoch={source_epoch})"
        print(f"Loaded warm-start checkpoint: {warm_start_info}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))

    history_path = out_dir / "train_history.csv"
    best_valid = -1.0
    rows = []
    snapshot_epochs = {int(value) for value in cfg.get("snapshot_epochs", [])}

    for epoch in range(1, int(cfg["epochs"]) + 1):
        train_loss, train_dice = run_epoch(
            model, train_loader, optimizer, device, True, cfg
        )
        valid_loss, valid_dice = run_epoch(
            model, valid_loader, optimizer, device, False, cfg
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_dice": train_dice,
            "valid_loss": valid_loss,
            "valid_dice": valid_dice,
            "warm_start": warm_start_info,
        }
        rows.append(row)
        print(row)

        last_path = ckpt_dir / "last.pt"
        torch.save(
            {"model": model.state_dict(), "config": cfg, "epoch": epoch, "warm_start": warm_start_info},
            last_path,
        )
        if valid_dice > best_valid:
            best_valid = valid_dice
            best_path = ckpt_dir / "best.pt"
            torch.save(
                {"model": model.state_dict(), "config": cfg, "epoch": epoch, "warm_start": warm_start_info},
                best_path,
            )
        snapshot_path = None
        if epoch in snapshot_epochs:
            snapshot_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
            shutil.copy2(last_path, snapshot_path)

        with open(history_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        if mirror_dir and mirror_dir.resolve() != ckpt_dir.resolve():
            shutil.copy2(last_path, mirror_dir / "last.pt")
            if (ckpt_dir / "best.pt").exists():
                shutil.copy2(ckpt_dir / "best.pt", mirror_dir / "best.pt")
            if snapshot_path is not None:
                shutil.copy2(snapshot_path, mirror_dir / snapshot_path.name)
            shutil.copy2(history_path, mirror_dir / "train_history.csv")

    print(f"Best validation Dice: {best_valid:.4f}")
    print(f"Saved checkpoints to {ckpt_dir}")
    if mirror_dir and mirror_dir.resolve() != ckpt_dir.resolve():
        print(f"Mirrored checkpoints to {mirror_dir}")


if __name__ == "__main__":
    main()
