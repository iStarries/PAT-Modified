"""Prototype-based metric losses for few-shot segmentation."""
from __future__ import annotations

from typing import Tuple

import torch


def compute_prototype_losses(
    feat_map: torch.Tensor,
    labels: torch.Tensor,
    max_points_per_class: int = 128,
    ignore_label: int = 255,
    detach_prototype: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute intra-class compactness and inter-class separation losses.

    Args:
        feat_map: Tensor of shape (B, D, H, W) containing dense feature embeddings.
        labels: Tensor of shape (B, H, W) with integer class ids. Pixels with
            ``ignore_label`` are skipped.
        max_points_per_class: Maximum number of pixels sampled per class to keep
            the loss balanced and memory usage bounded.
        ignore_label: Label id that should be excluded from the statistics.
        detach_prototype: When ``True`` the class prototypes are detached from
            the computation graph to stabilise training.

    Returns:
        Tuple containing ``(loss_intra, loss_inter)`` as scalar tensors located
        on the same device as ``feat_map``.
    """

    if feat_map.dim() != 4:
        raise ValueError(f"Expected feat_map with 4 dimensions, got shape {feat_map.shape}")
    if labels.dim() != 3:
        raise ValueError(f"Expected labels with 3 dimensions, got shape {labels.shape}")

    device = feat_map.device
    loss_zero = feat_map.new_tensor(0.0)

    b, c, h, w = feat_map.shape
    if labels.shape != (b, h, w):
        raise ValueError(
            "Label shape must match feature map spatial dimensions: "
            f"expected {(b, h, w)}, got {tuple(labels.shape)}"
        )

    feat_map = feat_map.permute(0, 2, 3, 1).reshape(b, h * w, c)
    labels = labels.reshape(b, h * w)

    intra_losses = []
    inter_losses = []

    for batch_idx in range(b):
        sample_labels = labels[batch_idx]
        sample_feats = feat_map[batch_idx]

        valid_mask = sample_labels != ignore_label
        if not torch.any(valid_mask):
            continue

        valid_labels = sample_labels[valid_mask]
        unique_labels = torch.unique(valid_labels)
        if unique_labels.numel() == 0:
            continue

        sample_prototypes = []

        for class_id in unique_labels:
            class_mask = valid_mask & (sample_labels == class_id)
            idx = torch.nonzero(class_mask, as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue

            if max_points_per_class > 0 and idx.numel() > max_points_per_class:
                perm = torch.randperm(idx.numel(), device=device)[:max_points_per_class]
                idx = idx[perm]

            class_features = sample_feats.index_select(0, idx)
            prototype = class_features.mean(dim=0, keepdim=True)

            prototype_for_intra = prototype.detach() if detach_prototype else prototype
            diffs = class_features - prototype_for_intra
            class_intra = (diffs.pow(2).sum(dim=1)).mean()
            intra_losses.append(class_intra)

            sample_prototypes.append(prototype)

        if not sample_prototypes:
            continue

        prototypes_tensor = torch.cat(sample_prototypes, dim=0)
        prototypes_for_inter = (
            prototypes_tensor.detach() if detach_prototype else prototypes_tensor
        )

        if prototypes_tensor.size(0) < 2:
            continue

        diff = prototypes_for_inter.unsqueeze(0) - prototypes_for_inter.unsqueeze(1)
        dist_sq = (diff.pow(2)).sum(dim=-1)
        triu_idx = torch.triu_indices(
            prototypes_tensor.size(0), prototypes_tensor.size(0), offset=1, device=device
        )
        pairwise_dist = dist_sq[triu_idx[0], triu_idx[1]]
        inter_losses.append(-pairwise_dist.mean())

    if not intra_losses:
        return loss_zero, loss_zero.clone()

    loss_intra = torch.stack(intra_losses).mean()

    if not inter_losses:
        loss_inter = loss_zero.clone()
    else:
        loss_inter = torch.stack(inter_losses).mean()

    return loss_intra, loss_inter
