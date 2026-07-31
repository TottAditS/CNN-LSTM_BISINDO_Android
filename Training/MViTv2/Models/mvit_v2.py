"""
Training/MViTv2/Models/mvit_v2.py
====================================
Wrapper untuk MViTv2 (torchvision, pretrained Kinetics-400) sebagai baseline
transformer pembanding CNNsa-BiLSTM.

Struktur torchvision.models.video.mvit_v2_s (dikonfirmasi via inspeksi langsung):
  conv_proj (Conv3d) -> pos_encoding -> blocks (ModuleList, 16x MultiscaleBlock)
  -> norm (LayerNorm) -> head (Sequential(Dropout(0.5), Linear(768, 400)))
  34.5M parameter, pretrained di Kinetics-400 (400 kelas).
"""

from typing import List, Dict

import torch
import torch.nn as nn
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights

HEAD_IN_FEATURES = 768  # output_channels block terakhir MViTv2-S (tetap, arsitektur fixed)


def build_mvit_v2(
    num_classes: int,
    freeze_until_block: int = 12,
    pretrained: bool = True,
    dropout: float = 0.5,
) -> nn.Module:
    """
    Bangun MViTv2-S, ganti head ke num_classes, freeze blocks[:freeze_until_block].

    Args:
        num_classes       : jumlah kelas output (32 untuk WLBisindo)
        freeze_until_block: blocks[0:freeze_until_block] dibekukan (requires_grad=False).
                             0 = full fine-tune, 16 = semua block dibekukan (cuma head yang dilatih).
        pretrained         : True = load bobot Kinetics-400
        dropout             : dropout rate di head baru

    Returns:
        nn.Module — forward(x) dengan x shape (B, C, T, H, W) -> logits (B, num_classes)
    """
    weights = MViT_V2_S_Weights.KINETICS400_V1 if pretrained else None
    model = mvit_v2_s(weights=weights)

    num_blocks = len(model.blocks)
    if not (0 <= freeze_until_block <= num_blocks):
        raise ValueError(f"freeze_until_block harus di [0, {num_blocks}], dapat {freeze_until_block}")

    # Ganti classifier head (400 kelas Kinetics -> num_classes kelas BISINDO)
    model.head = nn.Sequential(
        nn.Dropout(dropout, inplace=True),
        nn.Linear(HEAD_IN_FEATURES, num_classes),
    )

    # Freeze conv_proj + pos_encoding + blocks[:freeze_until_block].
    # norm + blocks[freeze_until_block:] + head baru tetap trainable.
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
    Param groups untuk differential learning rate: backbone (blocks yang di-fine-tune +
    norm) pakai LR kecil (sudah pretrained), head baru pakai LR lebih besar (random init).
    Parameter yang di-freeze (requires_grad=False) tidak dimasukkan (tidak akan diupdate
    optimizer manapun).
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
    print("Output shape:", out.shape)  # Expected: (2, 32)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params     : {total_params:,}")
    print(f"Trainable params : {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
