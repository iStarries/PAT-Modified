r""" Logging during training/testing """
import datetime
import logging
import os
import sys

from tensorboardX import SummaryWriter
import torch


class AverageMeter:
    r""" Stores loss, evaluation results """
    def __init__(self, dataset):
        self.benchmark = dataset.benchmark
        if self.benchmark == 'pascal':
            self.class_ids_interest = dataset.class_ids
            self.class_ids_interest = torch.tensor(self.class_ids_interest).cuda()
            self.nclass = 20
        elif self.benchmark == 'fss':
            self.class_ids_interest = dataset.class_ids
            self.class_ids_interest = torch.tensor(self.class_ids_interest).cuda()
            self.nclass = 1000
        elif self.benchmark == 'deepglobe':
            self.class_ids_interest = dataset.class_ids
            self.class_ids_interest = torch.tensor(self.class_ids_interest).cuda()
            self.nclass = 6
        elif self.benchmark == 'isic':
            self.class_ids_interest = dataset.class_ids
            self.class_ids_interest = torch.tensor(self.class_ids_interest).cuda()
            self.nclass = 3
        elif self.benchmark == 'lung':
            self.class_ids_interest = dataset.class_ids
            self.class_ids_interest = torch.tensor(self.class_ids_interest).cuda()
            self.nclass = 1

        self.intersection_buf = torch.zeros([2, self.nclass]).float().cuda()
        self.union_buf = torch.zeros([2, self.nclass]).float().cuda()
        self.ones = torch.ones_like(self.union_buf)
        self.loss_buf = []
        self.loss_components = {
            'total': [],
            'seg': [],
            'intra': [],
            'inter': [],
        }
    def update(self, inter_b, union_b, class_id, loss=None, loss_components=None):
        self.intersection_buf.index_add_(1, class_id, inter_b.float())
        self.union_buf.index_add_(1, class_id, union_b.float())
        if loss is None:
            loss = torch.tensor(0.0, device=self.intersection_buf.device)
        self.loss_buf.append(loss)
        if loss_components is not None:
            for key, value in loss_components.items():
                if key not in self.loss_components:
                    self.loss_components[key] = []
                self.loss_components[key].append(value)

    def compute_iou(self):
        iou = self.intersection_buf.float() / \
              torch.max(torch.stack([self.union_buf, self.ones]), dim=0)[0]
        iou = iou.index_select(1, self.class_ids_interest)
        miou = iou[1].mean() * 100

        fb_iou = (self.intersection_buf.index_select(1, self.class_ids_interest).sum(dim=1) /
                  self.union_buf.index_select(1, self.class_ids_interest).sum(dim=1)).mean() * 100

        return miou, fb_iou

    def write_result(self, split, epoch):
        iou, fb_iou = self.compute_iou()

        loss_buf = torch.stack(self.loss_buf)
        msg = '\n*** %s ' % split
        msg += '[@Epoch %02d] ' % epoch
        msg += 'Avg L: %6.5f  ' % loss_buf.mean().item()
        if self.loss_components['seg']:
            seg_mean = torch.stack(self.loss_components['seg']).mean().item()
            intra_mean = torch.stack(self.loss_components['intra']).mean().item()
            inter_mean = torch.stack(self.loss_components['inter']).mean().item()
            msg += f"Seg: {seg_mean:.5f}   "
            msg += f"Intra: {intra_mean:.5f}   "
            msg += f"Inter: {inter_mean:.5f}   "
        msg += 'mIoU: %5.2f   ' % iou
        msg += 'FB-IoU: %5.2f   ' % fb_iou

        msg += '***\n'
        Logger.info(msg)

    def write_process(self, batch_idx, datalen, epoch, write_batch_idx=20):
        if batch_idx % write_batch_idx == 0:
            # 1. 计算指标 (与原来相同)
            iou, fb_iou = self.compute_iou()
            loss_buf = torch.stack(self.loss_buf)
            avg_loss = loss_buf.mean().item()
            seg_text = intra_text = inter_text = ''
            if self.loss_components['seg']:
                seg_mean = torch.stack(self.loss_components['seg']).mean().item()
                intra_mean = torch.stack(self.loss_components['intra']).mean().item()
                inter_mean = torch.stack(self.loss_components['inter']).mean().item()
                seg_text = f" | Seg: {seg_mean:.5f}"
                intra_text = f" | Intra: {intra_mean:.5f}"
                inter_text = f" | Inter: {inter_mean:.5f}"

            # 2. 构建简化的单行消息
            #    - \r: 将光标移动到行首
            #    - 空格填充: 确保新内容能完全覆盖旧内容
            msg = (f"\r[Epoch: {epoch:02d}] [Batch: {batch_idx + 1:05d}/{datalen:05d}] | "
                   f"Avg L: {avg_loss:.5f}{seg_text}{intra_text}{inter_text} | "
                   f"mIoU: {iou:5.2f} | "
                   f"FB-IoU: {fb_iou:5.2f}      ")

            # 3. 直接写入标准输出，不换行
            sys.stdout.write(msg)
            sys.stdout.flush()


class Logger:
    r""" Writes evaluation results of training/testing """
    @classmethod
    def initialize(cls, args, training):
        logtime = datetime.datetime.now().__format__('_%m%d_%H%M%S')
        logpath = args.logpath if training else '_TEST_' + args.load.split('/')[-2].split('.')[0] + logtime
        if logpath == '': logpath = logtime

        cls.logpath = os.path.join('logs', logpath)
        cls.benchmark = args.benchmark
        os.makedirs(cls.logpath, exist_ok=True)

        logging.basicConfig(filemode='a',
                            filename=os.path.join(cls.logpath, 'log.txt'),
                            level=logging.INFO,
                            format='%(message)s',
                            datefmt='%m-%d %H:%M:%S')

        # ======================= 恢复训练配置 =======================
        # 如果是恢复训练，则在日志中添加一个清晰的分隔符
        if hasattr(args, 'resume') and args.resume:
            logging.info('\n' + '=' * 60)
            logtime = datetime.datetime.now().__format__('%Y-%m-%d %H:%M:%S')
            logging.info(f'RESUMING TRAINING AT: {logtime}')
            logging.info('=' * 60 + '\n')
        # ==========================================================


        # Console log config
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)


        # Log arguments
        logging.info('\n:=========== Cross-Domain Few-shot Seg. with PATNet ===========')
        for arg_key in args.__dict__:
            logging.info('| %20s: %-24s' % (arg_key, str(args.__dict__[arg_key])))
        logging.info(':==============================================================\n')

    @classmethod
    def info(cls, msg):
        r""" Writes log message to log.txt """
        logging.info(msg)

    @classmethod
    def save_model_miou(cls, model, epoch, val_miou):
        # 获取模型权重
        state_dict = model.module.state_dict()

        # 1. 构造新的文件名，格式为 mIoU 在前, epoch 在后
        checkpoint_name = f'model_miou_{val_miou:.2f}_epoch_{epoch}.pt'
        checkpoint_path = os.path.join(cls.logpath, checkpoint_name)

        # 2. 保存新的权重文件
        torch.save(state_dict, checkpoint_path)

        # 3. 打印清晰的日志信息，指明新保存的文件路径
        cls.info(f'New best model saved to: {checkpoint_path}\n')

    @classmethod
    def log_params(cls, model):
        backbone_param = 0
        learner_param = 0
        for k in model.state_dict().keys():
            n_param = model.state_dict()[k].view(-1).size(0)
            if k.split('.')[0] in 'backbone':
                if k.split('.')[1] in ['classifier', 'fc']:  # as fc layers are not used in HSNet
                    continue
                backbone_param += n_param
            else:
                learner_param += n_param
        Logger.info('Backbone # param.: %d' % backbone_param)
        Logger.info('Learnable # param.: %d' % learner_param)
        Logger.info('Total # param.: %d' % (backbone_param + learner_param))

