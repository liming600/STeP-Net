import torch
import torch.nn as nn
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter

import numpy as np
from datasets import data_transforms
from pointnet2_ops import pointnet2_utils
from torchvision import transforms

# 导入 scikit-learn 用于计算更全面的评估指标
from sklearn.metrics import confusion_matrix, classification_report

train_transforms = transforms.Compose(
    [
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)

test_transforms = transforms.Compose(
    [
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)


class Acc_Metric:
    def __init__(self, acc=0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        elif type(acc).__name__ == 'Acc_Metric':
            self.acc = acc.acc
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict


def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, val_dataloader), = builder.dataset_builder(args, config.dataset.train), \
        builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model)

    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger=logger)
        best_metrics = Acc_Metric(best_metrics)
    else:
        if args.ckpts is not None:
            base_model.load_model_from_ckpt(args.ckpts)
        else:
            print_log('Training from scratch', logger=logger)

    if args.use_gpu:
        base_model.to(args.local_rank)

    if args.distributed:
        # (分布式逻辑)
        pass
    else:
        print_log('Using Data parallel ...', logger=logger)
        base_model = nn.DataParallel(base_model).cuda()

    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)

    # trainval
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['loss', 'acc'])
        num_iter = 0
        n_batches = len(train_dataloader)

        batch_start_time = time.time()
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx
            data_time.update(time.time() - batch_start_time)

            points = data[0].cuda()
            label = data[1].cuda()

            # 对于单帧模型，可能需要进行在线的采样 (此逻辑来自原文件)
            # 对于序列模型，points已经是(B,T,N,3)，不需要这步
            if len(points.shape) == 3 and hasattr(config, 'npoints'):
                npoints = config.npoints
                if npoints == 1024:
                    point_all = 1200
                elif npoints == 2048:
                    point_all = 2400
                else:
                    point_all = points.size(1)

                if points.size(1) < point_all: point_all = points.size(1)
                fps_idx = pointnet2_utils.furthest_point_sample(points, point_all)
                fps_idx = fps_idx[:, np.random.choice(point_all, npoints, False)]
                points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1,
                                                                                                                  2).contiguous()

            points = train_transforms(points)
            ret = base_model(points)
            loss, acc = base_model.module.get_loss_acc(ret, label)

            _loss = loss
            _loss.backward()

            if num_iter == config.step_per_update:
                if config.get('grad_norm_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                pass
            else:
                losses.update([loss.item(), acc.item()])

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

        # (后续训练循环逻辑保持不变)
        if isinstance(scheduler, list):
            for item in scheduler: item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
                  (epoch, epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],
                   optimizer.param_groups[0]['lr']), logger=logger)

        if epoch % args.val_freq == 0 and epoch != 0:
            metrics = validate(base_model, val_dataloader, epoch, val_writer, args, config, logger=logger)
            better = metrics.better_than(best_metrics)
            if better:
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args,
                                        logger=logger)

        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger=logger)

    if train_writer is not None: train_writer.close()
    if val_writer is not None: val_writer.close()


def validate(base_model, val_dataloader, epoch, val_writer, args, config, logger=None):
    base_model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(val_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            # 验证时，单帧模型也使用简单的FPS采样
            if len(points.shape) == 3 and hasattr(config, 'npoints'):
                points = misc.fps(points, config.npoints)

            logits = base_model(points)
            pred = logits.argmax(-1).view(-1)

            all_preds.append(pred.detach())
            all_labels.append(label.detach())

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    acc = (all_preds == all_labels).sum() / float(all_labels.size(0)) * 100.
    print_log('[Validation] EPOCH: %d  acc = %.4f' % (epoch, acc), logger=logger)

    if val_writer is not None: val_writer.add_scalar('Metric/ACC', acc, epoch)
    return Acc_Metric(acc)


# ====================================================================
# ---               这是最终修正后的 test_net 函数                 ---
# ====================================================================
def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger=logger)

    # 1. 构建测试集数据加载器
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)

    # 2. 构建模型并加载最佳权重
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.ckpts, logger=logger)
    if args.use_gpu:
        base_model.to(args.local_rank)
        base_model = nn.DataParallel(base_model).cuda()  # 包装以进行评估

    # 3. 在测试集上进行评估
    base_model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            # --- [核心修改] ---
            # 移除错误的FPS调用，直接将数据送入模型
            # 无论是 (B, N, 3) 还是 (B, T, N, 3)，模型都能正确处理
            logits = base_model(points)

            pred = logits.argmax(-1).view(-1)

            all_preds.append(pred.detach())
            all_labels.append(label.detach())

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # 4. 计算并打印所有评估指标
    pred_np = all_preds.cpu().numpy()
    label_np = all_labels.cpu().numpy()

    acc = (pred_np == label_np).sum() / float(label_np.shape[0]) * 100.
    print_log('[TEST] Overall Accuracy = %.4f' % acc, logger=logger)

    cm = confusion_matrix(label_np, pred_np)
    print_log('Confusion Matrix:', logger=logger)
    print_log('\n' + str(cm), logger=logger)

    class_names = ['Daily Activity (0)', 'Fall (1)']
    report = classification_report(label_np, pred_np, target_names=class_names, digits=4)
    print_log('Classification Report:', logger=logger)
    print_log('\n' + report, logger=logger)