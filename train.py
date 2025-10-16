r""" PATNet training (validation) code """
import os
import sys
sys.path.insert(0, "../")

import argparse

import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch

from model.patnet import PATNetwork
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset
import time  # <-- 添加这一行
import datetime  # <-- 添加这一行

def train(epoch, model, dataloader, optimizer, training):
    r""" Train PATNet """

    # Force randomness during training / freeze randomness during testing
    utils.fix_randseed(None) if training else utils.fix_randseed(0)
    model.module.train_mode() if training else model.module.eval()
    average_meter = AverageMeter(dataloader.dataset)

    # if training:
    for idx, batch in enumerate(dataloader):
        # if idx >= 10:
        #     break
        current_iter = epoch * len(dataloader) + idx
        if training:
            utils.poly_learning_rate(optimizer, args.lr, current_iter, max_iter, power=args.power,
                                     warmup=args.warmup, warmup_iters=args.warmup_iters)

        # 1. PATNetworks forward pass
        batch = utils.to_cuda(batch)

        logit_mask, feat_map = model(batch['query_img'], batch['support_imgs'].squeeze(1), batch['support_masks'].squeeze(1))
        loss_seg = model.module.compute_objective(logit_mask, batch['query_mask'])

        if training:
            loss_intra, loss_inter = model.module.compute_metric_losses(
                feat_map,
                batch['query_mask'],
                max_points_per_class=args.max_points_per_class,
                detach_prototype=args.detach_prototype,
                ignore_label=getattr(args, 'ignore_label', 255),
            )
        else:
            zero = loss_seg.new_zeros(1).squeeze(0)
            loss_intra, loss_inter = zero.clone(), zero.clone()

        loss = loss_seg + args.lambda_intra * loss_intra + args.lambda_inter * loss_inter

        pred_mask = logit_mask.argmax(dim=1)

        # 2. Compute loss & update model parameters
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 3. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask, batch)
        loss_components = {
            'total': loss.detach().clone(),
            'seg': loss_seg.detach().clone(),
            'intra': loss_intra.detach().clone(),
            'inter': loss_inter.detach().clone(),
        }
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone(), loss_components)
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)
    print()
    # Write evaluation results
    average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return miou, fb_iou


if __name__ == '__main__':

    import yaml
    from types import SimpleNamespace

    # Arguments parsing from config file
    parser = argparse.ArgumentParser(description='Training script for PATNet.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the YAML config file.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 将字典转换为对象，以便使用 args.variable 的形式访问
    args = SimpleNamespace(**config['train'])

    Logger.initialize(args, training=True)

    # Model initialization
    model = PATNetwork(args.backbone)
    Logger.log_params(model)

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model = nn.DataParallel(model)
    model.to(device)

    # Helper classes (for training) initialization
    optimizer = optim.Adam([{"params": model.parameters(), "lr": args.lr}])


    # ======================= 恢复训练配置 =======================
    # 初始化起始轮次和最佳mIoU
    start_epoch = 0
    best_val_miou = float('-inf')

    # 如果 config 文件中 resume 为 True，则加载模型
    if hasattr(args, 'resume') and args.resume:
        if os.path.isfile(args.load_path):
            Logger.info(f"Loading checkpoint from: {args.load_path}")
            # 加载权重文件
            checkpoint = torch.load(args.load_path)
            # 加载模型权重
            model.module.load_state_dict(checkpoint)
            # 设置起始轮次
            start_epoch = args.start_epoch
            Logger.info(f"Resuming training from epoch {start_epoch}")
        else:
            Logger.info(f"Checkpoint not found at: {args.load_path}")
    # ==========================================================

    Evaluator.initialize()

    # Dataset initialization
    FSSDataset.initialize(img_size=400, datapath=args.datapath)
    dataloader_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn')
    dataloader_val = FSSDataset.build_dataloader('fss', args.bsz, args.nworker, '0', 'val')
    max_iter = args.niter * len(dataloader_trn)

    # Train HSNet
    # best_val_miou = float('-inf')
    best_val_loss = float('inf')
    Logger.info(f"============================================================================")
    Logger.info(f"Training started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    Logger.info(f"============================================================================\n")

    epochs_without_improvement = 0
    for epoch in range(start_epoch, args.niter):
        epoch_start_time = time.time()
        trn_miou, trn_fb_iou = train(epoch, model, dataloader_trn, optimizer, training=True)
        with torch.no_grad():
            val_miou, val_fb_iou = train(epoch, model, dataloader_val, optimizer, training=False)


        # Save the best model
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            Logger.save_model_miou(model, epoch, val_miou)
            epochs_without_improvement = 0  # 重置计数器
        else:
            epochs_without_improvement += 1  # 性能没有提升，计数器加一



        if epochs_without_improvement >= args.early_stopping_patience:
            Logger.info(f"Early stopping triggered after {args.early_stopping_patience} epochs without improvement.")
            break  # 退出训练循环

        epoch_duration_sec = time.time() - epoch_start_time
        end_time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        epoch_duration_min, epoch_duration_sec = divmod(epoch_duration_sec, 60)

        # --- 将日志信息分为两行，并为时间信息添加边框 ---
        duration_info = f'Epoch [{epoch:04d}/{args.niter - 1:04d}] finished. Duration: {int(epoch_duration_min)}min {epoch_duration_sec:.2f}s.'
        Logger.info(f"============================================================================")
        time_info = f'End Time: {end_time_str}'
        Logger.info(f'{duration_info}\n{time_info}')
        Logger.info(f"============================================================================\n")
    # Logger.tbd_writer.close()
    Logger.info('==================== Finished Training ====================')
