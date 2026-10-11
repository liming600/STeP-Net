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

from sklearn.metrics import confusion_matrix, classification_report

# 数据增强：data_transforms 内部会自动判断是 3D（单帧）还是 4D（时序）并进行相应处理
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
        if isinstance(acc, dict):
            self.acc = acc.get('acc', 0.)
        elif isinstance(acc, Acc_Metric):
            self.acc = acc.acc
        else:
            self.acc = acc

    def better_than(self, other):
        return self.acc > other.acc

    def state_dict(self):
        return {'acc': self.acc}


def run_net(args, config, train_writer=None, val_writer=None, train_dataloader=None, val_dataloader=None,
            train_sampler=None, early_stop_patience=10):
    logger = get_logger(args.log_name)

    # 1. 构建数据集
    if train_dataloader is None or val_dataloader is None:
        (train_sampler, train_dataloader), (_, val_dataloader) = \
            builder.dataset_builder(args, config.dataset.train), \
                builder.dataset_builder(args, config.dataset.val)

    # 2. 构建模型
    base_model = builder.model_builder(config.model)

    # --- [关键修改] 双重保险：确保 Encoder 在微调模式下被冻结 ---
    if config.model.get('freeze_encoder', False):
        print_log(f'[Runner] Ensuring encoder parameters are frozen (freeze_encoder=True)...', logger=logger)
        frozen_count = 0
        for name, param in base_model.named_parameters():
            if 'point_mae_encoder' in name or 'MAE_encoder' in name:
                param.requires_grad = False
                frozen_count += 1
        print_log(f'[Runner] Frozen {frozen_count} parameter groups in encoder.', logger=logger)
    # --------------------------------------------------------

    # 3. 加载权重 (Resume / Finetune / Scratch)
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    if args.resume:
        start_epoch, best_metric_dict = builder.resume_model(base_model, args, logger=logger)
        best_metrics = Acc_Metric(best_metric_dict)
    elif args.finetune_model or args.scratch_model:
        if args.ckpts is not None and not args.scratch_model:
            base_model.load_model_from_ckpt(args.ckpts)

    # 4. GPU 设置
    if args.use_gpu:
        base_model.to(args.local_rank)

    # 5. 并行设置 (DDP / DP)
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        base_model = nn.parallel.DistributedDataParallel(base_model,
                                                         device_ids=[args.local_rank % torch.cuda.device_count()],
                                                         find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        print_log('Using Data parallel ...', logger=logger)
        base_model = nn.DataParallel(base_model).cuda()

    # 6. 优化器与调度器
    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)

    # 7. 训练循环
    patience_counter = 0  # 早停计数器

    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        base_model.train()
        # 如果冻结了 encoder，在 train 模式下也要保持 encoder 为 eval 模式 (关闭 Dropout/BN 更新)
        if config.model.get('freeze_encoder', False):
            if hasattr(base_model.module, 'point_mae_encoder'):
                base_model.module.point_mae_encoder.eval()

        epoch_start_time = time.time()
        losses = AverageMeter(['loss', 'acc'])
        num_iter = 0
        n_batches = len(train_dataloader)

        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            num_iter += 1
            points, label = data[0].cuda(), data[1].cuda()
            if epoch == start_epoch and idx == 0:
                print("DEBUG -- points.shape:", points.shape)

            # --- [关键修改] 移除 FPS 采样 ---
            # 原因：数据预处理已保证点数统一，在线 FPS 计算量大且无必要
            # if len(points.shape) == 3 and hasattr(config, 'npoints'):
            #     npoints = config.npoints
            #     if npoints < points.size(1):
            #         points = misc.fps(points, npoints)

            # 数据增强 (自动处理 3D/4D)
            points = train_transforms(points)

            # 前向传播
            ret = base_model(points)
            loss, acc = base_model.module.get_loss_acc(ret, label)

            _loss = loss / config.get('step_per_update', 1)
            _loss.backward()

            # 梯度累积与更新
            if num_iter == config.get('step_per_update', 1):
                if config.get('grad_norm_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            losses.update([loss.item(), acc.item()])

        # 调度器更新
        if isinstance(scheduler, list):
            for item in scheduler: item.step(epoch)
        else:
            scheduler.step(epoch)

        # 打印训练日志
        if args.local_rank == 0:
            print_log(
                f'[Training] EPOCH: {epoch} EpochTime = {time.time() - epoch_start_time:.3f} (s) Losses = {["%.4f" % l for l in losses.avg()]} lr = {optimizer.param_groups[0]["lr"]:.6f}',
                logger=logger)

        # 验证与早停逻辑
        if epoch % args.val_freq == 0 and epoch != 0:
            if not args.distributed or args.local_rank == 0:
                metrics = validate(base_model, val_dataloader, epoch, val_writer, args, config, logger=logger)

                if metrics.better_than(best_metrics):
                    best_metrics = metrics
                    builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args,
                                            logger=logger)
                    patience_counter = 0  # 重置早停计数
                    print_log(f'[Early Stopping] Validation performance improved. Patience counter reset to 0.',
                              logger=logger)
                else:
                    patience_counter += 1
                    print_log(
                        f'[Early Stopping] No improvement for {patience_counter} validation steps. Patience: {early_stop_patience}.',
                        logger=logger)

        # 保存最新检查点
        if args.local_rank == 0:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args,
                                    logger=logger)

        # 触发早停
        if patience_counter >= early_stop_patience:
            print_log(f'Early stopping triggered after {early_stop_patience} validation intervals with no improvement.',
                      logger=logger)
            break

    if train_writer is not None: train_writer.close()
    if val_writer is not None: val_writer.close()

    return best_metrics


def validate(base_model, val_dataloader, epoch, val_writer, args, config, logger=None):
    base_model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(val_dataloader):
            points, label = data[0].cuda(), data[1].cuda()

            # --- [关键修改] 移除 FPS 采样 ---
            # if len(points.shape) == 3 and hasattr(config, 'npoints'):
            #     points = misc.fps(points, config.npoints)

            logits = base_model(points)
            pred = logits.argmax(-1).view(-1)
            all_preds.append(pred.detach())
            all_labels.append(label.detach())

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    acc = (all_preds == all_labels).sum() / float(all_labels.size(0)) * 100.

    if args.local_rank == 0:
        print_log(f'[Validation] EPOCH: {epoch}  acc = {acc:.4f}', logger=logger)
        if val_writer is not None: val_writer.add_scalar('Metric/ACC', acc, epoch)

    return Acc_Metric(acc)


def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger=logger)

    # 构建测试集
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)

    # 构建模型
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.ckpts, logger=logger)

    if args.use_gpu:
        base_model.to(args.local_rank)
        base_model = nn.DataParallel(base_model).cuda()

    base_model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points, label = data[0].cuda(), data[1].cuda()

            # --- [关键修改] 移除 FPS 采样 ---
            # if len(points.shape) == 3 and hasattr(config, 'npoints'):
            #    points = misc.fps(points, config.npoints)

            logits = base_model(points)
            pred = logits.argmax(-1).view(-1)
            all_preds.append(pred.detach())
            all_labels.append(label.detach())

    all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
    all_labels = torch.cat(all_labels, dim=0).cpu().numpy()

    # --- 详细指标报告 ---
    metrics_to_return = {}
    try:
        # 假设标签 0 是日常活动，1 是跌倒 (Fall)
        class_names = ['Daily Activity (0)', 'Fall (1)']

        # 获取字典格式报告，用于程序处理
        report_dict = classification_report(all_labels, all_preds, target_names=class_names, digits=4, output_dict=True,
                                            zero_division=0)

        # 获取字符串格式报告，用于日志打印
        report_str = classification_report(all_labels, all_preds, target_names=class_names, digits=4, zero_division=0)

        cm = confusion_matrix(all_labels, all_preds)

        print_log(f"[TEST] Overall Accuracy = {report_dict['accuracy'] * 100:.4f}%", logger=logger)
        print_log('Confusion Matrix:', logger=logger)
        print_log('\n' + str(cm), logger=logger)
        print_log('Classification Report:', logger=logger)
        print_log('\n' + report_str, logger=logger)

        # 提取用于 LOSO 汇总的关键指标
        # 针对跌倒检测，我们特别关注 'Fall (1)' 类的召回率 (Recall)
        metrics_to_return = {
            'accuracy': report_dict['accuracy'],
            'precision': report_dict['weighted avg']['precision'],
            'recall': report_dict['weighted avg']['recall'],
            'f1-score': report_dict['weighted avg']['f1-score'],
            'fall_recall': report_dict['Fall (1)']['recall'],  # 单独记录跌倒召回率
            'fall_precision': report_dict['Fall (1)']['precision']  # 单独记录跌倒精确率
        }
    except Exception as e:
        print_log(f"Could not generate detailed classification report: {e}", logger=logger)
        # 降级处理：仅返回准确率
        acc = (all_preds == all_labels).sum() / float(len(all_labels))
        metrics_to_return = {'accuracy': acc}

    return metrics_to_return