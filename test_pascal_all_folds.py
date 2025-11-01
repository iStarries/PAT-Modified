"""Evaluate PATNet on the Pascal-5i dataset across all folds.

This script ensures that every sample from each Pascal fold is visited once
by using a sequential variant of the DatasetPASCAL class whose length matches
its metadata size. As a result, iterating through a fold processes each query
image exactly once while still randomly sampling the support set.
"""
import argparse
from types import SimpleNamespace
import statistics

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import yaml

from model.patnet import PATNetwork
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset
from data.pascal import DatasetPASCAL


class SequentialDatasetPASCAL(DatasetPASCAL):
    """Variant of DatasetPASCAL whose length equals its metadata size.

    The default implementation fixes the length to 1000 when ``split`` is set
    to ``'val'`` or ``'test'`` so that dataloaders can draw more episodes than
    there are images. For exhaustive fold evaluation we override ``__len__`` to
    match the true number of query images, guaranteeing that each sample is
    seen once per epoch when ``shuffle=False``.
    """

    def __len__(self):  # noqa: D401 - short override
        return len(self.img_metadata)


def evaluate_fold(model, dataloader, nshot):
    """Evaluate ``model`` on a dataloader and return mIoU / FB-IoU metrics."""

    utils.fix_randseed(0)
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):
        batch = utils.to_cuda(batch)
        pred_mask = model.module.predict_mask_nshot(batch, nshot=nshot)
        assert pred_mask.size() == batch['query_mask'].size()

        area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=1)

    average_meter.write_result('Test', 0)
    miou, fb_iou = average_meter.compute_iou()
    return miou, fb_iou


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate PATNet on Pascal folds.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the YAML config file.')
    parser.add_argument('--img-size', type=int, default=400, help='Resize dimension used by the dataset loader.')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of workers for the evaluation dataloaders.')
    cli_args = parser.parse_args()

    with open(cli_args.config, 'r') as f:
        config = yaml.safe_load(f)

    args = SimpleNamespace(**config['test'])
    args.load = args.load_model_path
    args.benchmark = 'pascal'
    args.split = 'val'
    args.fold = 'all'

    Logger.initialize(args, training=False)

    model = PATNetwork(args.backbone)
    model.eval()
    Logger.log_params(model)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model = nn.DataParallel(model)
    model.to(device)

    if args.load == '':
        raise Exception('Pretrained model not specified.')
    state_dict = torch.load(args.load, map_location=device)

    # 兼容不同保存方式的权重文件：既支持直接保存的 state_dict，也支持字典包装。
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    target_model = model.module if isinstance(model, nn.DataParallel) else model

    # 如果权重是从 DataParallel 模型中直接保存的，键名会包含 "module." 前缀，需要去掉。
    if state_dict and next(iter(state_dict)).startswith('module.'):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}

    target_model.load_state_dict(state_dict)

    Evaluator.initialize()

    FSSDataset.initialize(img_size=cli_args.img_size, datapath=args.datapath)
    transform = FSSDataset.transform

    fold_metrics = []
    for fold in range(4):
        dataset = SequentialDatasetPASCAL(FSSDataset.datapath, fold=fold, transform=transform, split=args.split, shot=args.nshot)
        dataloader = DataLoader(dataset, batch_size=args.bsz, shuffle=False, num_workers=cli_args.num_workers)

        Logger.info('\n==================== Evaluating Fold %d ====================' % fold)
        Logger.info('Number of query episodes: %d' % len(dataset))

        with torch.no_grad():
            miou, fb_iou = evaluate_fold(model, dataloader, args.nshot)

        Logger.info('Fold %d -> mIoU: %5.2f \t FB-IoU: %5.2f' % (fold, miou.item(), fb_iou.item()))
        fold_metrics.append((miou.item(), fb_iou.item()))

    miou_mean = statistics.mean(metric[0] for metric in fold_metrics)
    fb_iou_mean = statistics.mean(metric[1] for metric in fold_metrics)
    Logger.info('\n==================== Pascal Fold Summary ====================')
    for fold, (miou_value, fb_iou_value) in enumerate(fold_metrics):
        Logger.info('Fold %d -> mIoU: %5.2f \t FB-IoU: %5.2f' % (fold, miou_value, fb_iou_value))
    Logger.info('Overall -> mIoU: %5.2f \t FB-IoU: %5.2f' % (miou_mean, fb_iou_mean))
    Logger.info('==================== Finished Testing ====================')
