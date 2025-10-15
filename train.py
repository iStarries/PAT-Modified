r""" PATNet training (validation) code """
import sys
sys.path.insert(0, "../")

import argparse

import torch.optim as optim
import torch.nn as nn
import torch

from model.patnet import PATNetwork
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset

from torch.cuda.amp import autocast, GradScaler # <-- 添加这一行
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

        # 1. PATNetworks forward pass
        batch = utils.to_cuda(batch)

        with autocast():
            logit_mask = model(batch['query_img'], batch['support_imgs'].squeeze(1), batch['support_masks'].squeeze(1))
            loss = model.module.compute_objective(logit_mask, batch['query_mask'])

        pred_mask = logit_mask.argmax(dim=1)

        # 2. Compute loss & update model parameters
        if training:
            optimizer.zero_grad()
            # 使用 scaler 缩放损失并进行反向传播
            scaler.scale(loss).backward()
            # 使用 scaler 更新权重
            scaler.step(optimizer)
            scaler.update()

        # 3. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask, batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)

    # Write evaluation results
    average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou


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
    scaler = GradScaler()  # <-- 添加这一行
    Evaluator.initialize()

    # Dataset initialization
    FSSDataset.initialize(img_size=400, datapath=args.datapath)
    dataloader_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn')
    FSSDataset.initialize(img_size=400, datapath=args.datapath)
    dataloader_val = FSSDataset.build_dataloader('fss', args.bsz, args.nworker, '0', 'val')

    # Train HSNet
    best_val_miou = float('-inf')
    best_val_loss = float('inf')
    Logger.info(f"============================================================================")
    Logger.info(f"Training started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    Logger.info(f"============================================================================\n")
    for epoch in range(args.niter):
        epoch_start_time = time.time()
        trn_loss, trn_miou, trn_fb_iou = train(epoch, model, dataloader_trn, optimizer, training=True)
        with torch.no_grad():
            val_loss, val_miou, val_fb_iou = train(epoch, model, dataloader_val, optimizer, training=False)

        # Save the best model
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            Logger.save_model_miou(model, epoch, val_miou)

        Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
        Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
        Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
        Logger.tbd_writer.flush()

        epoch_duration_sec = time.time() - epoch_start_time
        end_time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        epoch_duration_min, epoch_duration_sec = divmod(epoch_duration_sec, 60)

        # --- 将日志信息分为两行，并为时间信息添加边框 ---
        duration_info = f'Epoch [{epoch:04d}/{args.niter - 1:04d}] finished. Duration: {int(epoch_duration_min)}min {epoch_duration_sec:.2f}s.'
        Logger.info(f"============================================================================")
        time_info = f'End Time: {end_time_str}'
        Logger.info(f'{duration_info}\n{time_info}')
        Logger.info(f"============================================================================\n")
    Logger.tbd_writer.close()
    Logger.info('==================== Finished Training ====================')
