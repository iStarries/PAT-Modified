r""" Cross-Domain Few-Shot Semantic Segmentation testing code """
import argparse

import torch.nn as nn
import torch

from model.patnet import PATNetwork
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset


def test(model, dataloader, nshot):
    r""" Test PATNet """

    # Freeze randomness during testing for reproducibility if needed
    utils.fix_randseed(0)
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):
        # 1. PATNetworks forward pass
        batch = utils.to_cuda(batch)
        pred_mask = model.module.predict_mask_nshot(batch, nshot=nshot)
        assert pred_mask.size() == batch['query_mask'].size()
        # 2. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=1)

    # Write evaluation results
    average_meter.write_result('Test', 0)
    miou, fb_iou = average_meter.compute_iou()

    return miou, fb_iou


if __name__ == '__main__':

    import yaml
    from types import SimpleNamespace

    # Arguments parsing from config file
    parser = argparse.ArgumentParser(description='Testing script for PATNet.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the YAML config file.')
    cli_args = parser.parse_args()  # 读取命令行参数以获取配置文件路径

    with open(cli_args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 将字典转换为对象，以便使用 args.variable 的形式访问
    args = SimpleNamespace(**config['test'])

    # 在测试脚本中，我们需要手动设置 load 路径，因为它来自配置
    args.load = args.load_model_path
    args.fold = 4  # fold 参数在测试时通常为0或不使用，这里设为0

    Logger.initialize(args, training=False)

    # Model initialization
    model = PATNetwork(args.backbone)
    model.eval()
    Logger.log_params(model)

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model = nn.DataParallel(model)
    model.to(device)

    # Load trained model
    if args.load == '': raise Exception('Pretrained model not specified.')
    model.load_state_dict(torch.load(args.load))

    # Helper classes (for testing) initialization
    Evaluator.initialize()

    # Dataset initialization
    FSSDataset.initialize(img_size=400, datapath=args.datapath)
    dataloader_test = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'test', args.nshot)

    # Test PATNet
    with torch.no_grad():
        test_miou, test_fb_iou = test(model, dataloader_test, args.nshot)
    Logger.info('mIoU: %5.2f \t FB-IoU: %5.2f' % (test_miou.item(), test_fb_iou.item()))
    Logger.info('==================== Finished Testing ====================')
