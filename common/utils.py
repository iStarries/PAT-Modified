r""" Helper functions """
import random

import torch
import numpy as np


def fix_randseed(seed):
    r""" Set random seeds for reproducibility """
    if seed is None:
        seed = int(random.random() * 1e5)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mean(x):
    return sum(x) / len(x) if len(x) > 0 else 0.0


def to_cuda(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.cuda()
    return batch


def to_cpu(tensor):
    return tensor.detach().clone().cpu()

def poly_learning_rate(optimizer, base_lr, curr_iter, max_iter, power=0.9, warmup=False, warmup_iters=100):
    """
    Applies a polynomial learning rate decay schedule with an optional warmup phase.
    """
    # 1. Warmup 阶段
    if warmup and curr_iter < warmup_iters:
        # 学习率从 0 线性增长到 base_lr
        lr = base_lr * (float(curr_iter) / warmup_iters)
    # 2. Poly LR 衰减阶段
    else:
        # 学习率按多项式曲线衰减
        lr = base_lr * (1 - float(curr_iter) / max_iter) ** power

    # 将计算出的学习率应用到优化器
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr