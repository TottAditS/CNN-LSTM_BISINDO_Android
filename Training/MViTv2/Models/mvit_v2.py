"""
MViTv2-S from torchvision, pretrained on Kinetics-400, used as the comparison model.

torchvision.models.video.mvit_v2_s:
  conv_proj (Conv3d) -> pos_encoding -> blocks (16 MultiscaleBlocks)
  -> norm (LayerNorm) -> head (Dropout(0.5), Linear(768, 400))
"""

from typing import List, Dict

import torch
import torch.nn as nn
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights

HEAD_IN_FEATURES = 768  # output channels of the last MViTv2-S block


def build_mvit_v2(
    num_classes: int,
    freeze_until_block: int = 12,
    pretrained: bool = True,
    dropout: float = 0.5,
) -> nn.Module:
    """
    Build MViTv2-S with a new num_classes head and blocks[:freeze_until_block] frozen.

    Args:
        num_classes       : number of output classes (32 for WL-BISINDO)
        freeze_until_block: blocks[0:freeze_until_block] are frozen;
                            0 = full fine-tuning, 16 = head only
        pretrained        : load the Kinetics-400 weights
        dropout           : dropout of the new head

    Returns:
        nn.Module mapping (B, C, T, H, W) to logits (B, num_classes)
    """
    weights = MViT_V2_S_Weights.KINETICS400_V1 if pretrained else None
    model = mvit_v2_s(weights=weights)

    num_blocks = len(model.blocks)
    if not (0 <= freeze_until_block <= num_blocks):
        raise ValueError(f"freeze_until_block must be in [0, {num_blocks}], got {freeze_until_block}")

    model.head = nn.Sequential(
        nn.Dropout(dropout, inplace=True),
        nn.Linear(HEAD_IN_FEATURES, num_classes),
    )

    # Freeze conv_proj, pos_encoding, and blocks[:freeze_until_block];
    # the remaining blocks, norm, and the new head stay trainable.
    for param in model.conv_proj.parameters():
        param.requires_grad = False
    for param in model.pos_encoding.parameters():
        param.requires_grad = False
    for block in model.blocks[:freeze_until_block]:
        for param in block.parameters():
            param.requires_grad = False

    return model


def get_param_groups(model: nn.Module, backbone_lr: float, head_lr: float) -> List[Dict]:
    """
    Optimizer parameter groups: the trainable backbone parameters use backbone_lr
    and the new head uses head_lr. Frozen parameters are left out.
    """
    head_params = list(model.head.parameters())
    head_param_ids = {id(p) for p in head_params}

    backbone_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in head_param_ids
    ]

    return [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": [p for p in head_params if p.requires_grad], "lr": head_lr},
    ]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_mvit_v2(num_classes=32, freeze_until_block=12).to(device)

    dummy_input = torch.randn(2, 3, 16, 224, 224).to(device)
    out = model(dummy_input)
    print("Output shape:", out.shape)  # (2, 32)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params     : {total_params:,}")
    print(f"Trainable params : {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
