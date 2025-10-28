import torch
from typing import Callable, Iterable, Optional


def _is_stage_match(stage: int, hook_stage: Optional[Iterable[int]]) -> bool:
    if hook_stage is None:
        return False
    # 既支持 hook_stage=2 也支持 hook_stage=[2,3]
    if isinstance(hook_stage, Iterable) and not isinstance(hook_stage, (torch.Tensor, bytes, str)):
        return stage in hook_stage
    return stage == hook_stage


@torch.no_grad()
def extract_feat_res(
    img: torch.Tensor,
    backbone,
    feat_ids,
    bottleneck_ids,
    lids,
    low_level_hook: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    hook_stage: Optional[Iterable[int]] = None,
):
    """
    从 ResNet (resnet50) 中抽取若干中间层特征，用于 few-shot 分割。

    返回:
        feats: list[Tensor], 长度通常是 13
               feats[i] 形状是 [B, C, H, W]
               顺序与 PATNetwork.self.feat_ids 对齐
    """

    feats = []

    # stem
    feat = backbone.conv1(img)
    feat = backbone.bn1(feat)
    feat = backbone.relu(feat)
    feat = backbone.maxpool(feat)

    # 我们需要依次跑 layer1, layer2, layer3, layer4
    # 同时要知道：
    # - hid: 全局第几个 bottleneck block (0-based)
    # - lid: 第几层 stage (1,2,3,4)
    # - bid: 该 stage 内第几个 bottleneck (0-based)
    # 这样我们就可以：
    #   1. 判断是否调用 low_level_hook
    #   2. 判断这个 block 的输出是否应该收集进 feats

    layers = [backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4]

    hid = 0  # 全局 block 计数
    for stage_idx, layer in enumerate(layers, start=1):       # stage_idx = 1..4
        num_blocks_this_stage = len(layer)                    # e.g. 3,4,6,3
        last_block_id = num_blocks_this_stage - 1             # e.g. 2,3,5,2

        for bid, block in enumerate(layer):                   # bid = 0..(num_blocks-1)
            feat = block(feat)

            # 是否在这个 stage 的最后一个 block 之后注入低层增强模块(LEM)
            # 只有当:
            #   - low_level_hook 存在
            #   - hook_stage 匹配当前 stage
            #   - 我们正在这个 stage 的最后一个 bottleneck
            if (
                low_level_hook is not None
                and hook_stage is not None
                and _is_stage_match(stage_idx, hook_stage)
                and bid == last_block_id
            ):
                # LEM 需要梯度（它是可训练模块），所以局部打开 grad
                with torch.enable_grad():
                    feat = low_level_hook(feat.requires_grad_())

            # 这里决定是否把这一层的输出保存到 feats 里
            # 全局 block id 是 hid，从 0 开始；feat_ids 是从 1 开始的编号
            if (hid + 1) in feat_ids:
                feats.append(feat.clone())

            hid += 1

    return feats  # <-- 非常重要！！！


@torch.no_grad()
def extract_feat_vgg(
    img: torch.Tensor,
    backbone,
    feat_ids,
    bottleneck_ids=None,
    lids=None,
    low_level_hook: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    hook_stage: Optional[Iterable[int]] = None,
):
    """
    从 VGG16 中提取多个 conv block 的输出。
    逻辑和 ResNet 版类似：我们会遍历 features 里的卷积+ReLU，按顺序计数，
    把编号在 feat_ids 里的那些特征保留下来。
    对 VGG 来说 bottleneck_ids / lids 没有实际作用，只是为了兼容签名。
    """

    feats = []
    feat = img

    # 统计每个 "stage" 的最后一层，用来决定是否插 LEM。
    # 对 VGG16 通常我们把每个池化前的一组 conv 当成一个 stage。
    # 下面这块实现可以按你的原始逻辑来，但关键仍然是 return feats。
    stage_id = 1
    conv_id_in_stage = 0

    # 我们需要知道每个 stage 的最后一个 conv index，方便决定什么时候 hook。
    # 这里假设每遇到 MaxPool 就说明这个 stage 结束。
    # 我们会先扫一遍 backbone.features 来记录这些位置。
    stage_last_block = {}
    tmp_stage = 1
    tmp_bid = -1
    for layer in backbone.features:
        from torch import nn as _nn
        if isinstance(layer, torch.nn.Conv2d):
            tmp_bid += 1
        if isinstance(layer, torch.nn.MaxPool2d):
            stage_last_block[tmp_stage] = tmp_bid
            tmp_stage += 1
            tmp_bid = -1
    if tmp_bid >= 0:
        stage_last_block[tmp_stage] = tmp_bid  # 最后一段如果没pool也记录一下

    hid = 0  # 全局 conv block 计数
    bid = -1 # 当前stage里的conv计数
    for lid, layer in enumerate(backbone.features):
        # 卷积
        if isinstance(layer, torch.nn.Conv2d):
            bid += 1
            feat = layer(feat)

            # 是否在这一层之后做 low_level_hook
            if (
                low_level_hook is not None
                and hook_stage is not None
                and _is_stage_match(stage_id, hook_stage)
                and bid == stage_last_block.get(stage_id, -1)
            ):
                with torch.enable_grad():
                    feat = low_level_hook(feat.requires_grad_())

            # 保存特征
            if (hid + 1) in feat_ids:
                feats.append(feat.clone())

            hid += 1

        # ReLU / BN / 池化等
        elif isinstance(layer, torch.nn.ReLU):
            feat = layer(feat)
        elif isinstance(layer, torch.nn.BatchNorm2d):
            feat = layer(feat)
        elif isinstance(layer, torch.nn.MaxPool2d):
            feat = layer(feat)
            # 新的 stage
            stage_id += 1
            bid = -1
        else:
            # 其他层 (Dropout等)，直接前向
            feat = layer(feat)

    return feats  # <-- 同样，必须返回

