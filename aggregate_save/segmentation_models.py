"""
3D segmentation backbones for comparing architectures from ``segmentation_tune.py``.

- **3dunet**: MONAI ``UNet`` (same style as the original script).
- **nnunet**: MONAI ``DynUNet`` (residual blocks, anisotropic-capable U-Net family used in nnU-Net-style setups).
- **vista3d**: ``SegResNetDS2`` **auto (label) branch only** — same encoder/decoder family as the VISTA3D image encoder in MONAI, with a standard multi-class logits head. The full ``VISTA3D`` class is prompt-based (class / point) and is not drop-in compatible with ``outputs = model(images)`` + ``DiceCELoss`` here; use this variant for fair throughput / parameter comparisons.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from monai.networks.nets import DynUNet, SegResNetDS2, UNet

ARCH_CHOICES: tuple[str, ...] = ("3dunet", "nnunet", "vista3d")


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _build_monai_unet(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    channels: tuple[int, ...] = (32, 64, 128, 256, 512),
    strides: tuple[int, ...] = (2, 2, 2, 2),
    num_res_units: int = 2,
    norm: str | tuple = "batch",
    dropout: float = 0.1,
) -> UNet:
    return UNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
        norm=norm,
        dropout=dropout,
    )


def _build_dynunet_nnunet_style(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    deep_supervision: bool = False,
) -> DynUNet:
    # Isotropic 3D schedule; input spatial size should be divisible by 2**(len(strides)-1) on each axis.
    strides: tuple[tuple[int, ...], ...] = (
        (1, 1, 1),
        (2, 2, 2),
        (2, 2, 2),
        (2, 2, 2),
        (2, 2, 2),
    )
    kernel_size: tuple[tuple[int, ...], ...] = tuple((3, 3, 3) for _ in range(len(strides)))
    upsample_kernel_size = strides[1:]
    return DynUNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        strides=strides,
        upsample_kernel_size=upsample_kernel_size,
        norm_name=("instance", {"affine": True}),
        act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        deep_supervision=deep_supervision,
        deep_supr_num=1,
        res_block=True,
        trans_bias=False,
    )


class _SegResNetDS2AutoOnly(nn.Module):
    """Wraps SegResNetDS2 so forward returns only the label (auto) branch logits."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.backbone = SegResNetDS2(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, auto = self.backbone(x, with_point=False, with_label=True)
        if isinstance(auto, list):
            return auto[0]
        return auto


def _build_vista3d_style_segresnetds2(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    init_filters: int = 48,
    blocks_down: tuple[int, ...] = (1, 2, 2, 4, 4),
    dsdepth: int = 1,
) -> nn.Module:
    # Matches MONAI ``vista3d132`` encoder width / depth (SegResNetDS2 inside VISTA3D).
    return _SegResNetDS2AutoOnly(
        spatial_dims=spatial_dims,
        init_filters=init_filters,
        in_channels=in_channels,
        out_channels=out_channels,
        act="relu",
        norm="instance",
        blocks_down=blocks_down,
        blocks_up=None,
        dsdepth=dsdepth,
        preprocess=None,
        upsample_mode="deconv",
        resolution=None,
    )


def build_segmentation_model(
    name: str,
    *,
    spatial_dims: int = 3,
    in_channels: int = 1,
    out_channels: int = 9,
) -> nn.Module:
    """
    Return a segmentation model with ``forward(x) -> logits`` of shape
    ``(B, out_channels, *spatial)`` suitable for ``DiceCELoss(..., softmax=True)``.

    Args:
        name: One of ``ARCH_CHOICES`` (case-insensitive).
        spatial_dims: Must be 3 for this project.
        in_channels: CT channels (1).
        out_channels: Number of classes (including background).
    """
    key = name.strip().lower()
    if key not in ARCH_CHOICES:
        raise ValueError(f"Unknown architecture {name!r}. Choose one of: {ARCH_CHOICES}")

    if spatial_dims != 3:
        raise ValueError("Only spatial_dims=3 is supported for these backbones.")

    if key == "3dunet":
        return _build_monai_unet(spatial_dims, in_channels, out_channels)
    if key == "nnunet":
        return _build_dynunet_nnunet_style(spatial_dims, in_channels, out_channels, deep_supervision=False)
    return _build_vista3d_style_segresnetds2(spatial_dims, in_channels, out_channels)


def default_checkpoint_path(arch: str) -> str:
    return f"best_{arch.lower()}_3d_seg.pth"
