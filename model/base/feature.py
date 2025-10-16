"""Utilities for extracting intermediate backbone features."""

from typing import Callable, Iterable, Optional

import torch


def _is_stage_match(stage: int, hook_stage: Optional[Iterable[int]]) -> bool:
    if hook_stage is None:
        return False
    if isinstance(hook_stage, Iterable) and not isinstance(hook_stage, (torch.Tensor, bytes, str)):
        return stage in hook_stage
    return stage == hook_stage


def extract_feat_vgg(
    img: torch.Tensor,
    backbone,
    feat_ids,
    bottleneck_ids=None,
    lids=None,
    low_level_hook: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    hook_stage: Optional[Iterable[int]] = None,
):
    """Extract intermediate features from VGG backbones."""

    del bottleneck_ids, lids  # Unused for VGG but kept for API compatibility

    feats = []
    feat = img
    for lid, module in enumerate(backbone.features):
        feat = module(feat)
        if lid in feat_ids:
            feats.append(feat.clone())

    if low_level_hook is not None and hook_stage is not None and feats:
        # VGG integration is not stage-aware; hook on the shallowest extracted feature
        with torch.enable_grad():
            feats[0] = low_level_hook(feats[0].requires_grad_())

    return feats


def extract_feat_res(
    img: torch.Tensor,
    backbone,
    feat_ids,
    bottleneck_ids,
    lids,
    low_level_hook: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    hook_stage: Optional[Iterable[int]] = None,
):
    """Extract intermediate features from ResNet backbones."""

    feats = []

    # Layer 0
    feat = backbone.conv1.forward(img)
    feat = backbone.bn1.forward(feat)
    feat = backbone.relu.forward(feat)
    feat = backbone.maxpool.forward(feat)

    # Track the last block id for each stage so that hooks are applied once per stage
    stage_last_block = {}
    for bid, lid in zip(bottleneck_ids, lids):
        stage_last_block[lid] = bid

    # Layer 1-4
    for hid, (bid, lid) in enumerate(zip(bottleneck_ids, lids)):
        res = feat
        layer = backbone.__getattr__(f"layer{lid}")[bid]

        feat = layer.conv1.forward(feat)
        feat = layer.bn1.forward(feat)
        feat = layer.relu.forward(feat)
        feat = layer.conv2.forward(feat)
        feat = layer.bn2.forward(feat)
        feat = layer.relu.forward(feat)
        feat = layer.conv3.forward(feat)
        feat = layer.bn3.forward(feat)

        if bid == 0:
            res = layer.downsample.forward(res)

        feat = feat + res

        if (
            low_level_hook is not None
            and hook_stage is not None
            and _is_stage_match(lid, hook_stage)
            and bid == stage_last_block[lid]
        ):
            with torch.enable_grad():
                feat = low_level_hook(feat.requires_grad_())

        if hid + 1 in feat_ids:
            feats.append(feat.clone())

        feat = layer.relu.forward(feat)


