# tools/runner_finetune.py (Final Version with Full Metrics & Early Stopping)

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

    if train_dataloader is None or val_dataloader is None:
        (train_sampler, train_dataloader), (_, val_dataloader), = builder.dataset_builder(args, config.dataset.train), \
            builder.dataset_builder(args, config.dataset.val)

    base_model = builder.model_builder(config.model)

    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    if args.resume:
        start_epoch, best_metric_dict = builder.resume_model(base_model, args, logger=logger)
        best_metrics = Acc_Metric(best_metric_dict)
    elif args.finetune_model or args.scratch_model:
        if args.ckpts is not None and not args.scratch_model:
            base_model.load_model_from_ckpt(args.ckpts)

    if args.use_gpu:
        base_model.to(args.local_rank)

    if args.distributed:
        base_model = nn.parallel.DistributedDataParallel(base_model,
                                                         device_ids=[args.local_rank % torch.cuda.device_count()])
    else:
        print_log('Using Data parallel ...', logger=logger)
        base_model = nn.DataParallel(base_model).cuda()

    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)

    patience_counter = 0

    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        losses = AverageMeter(['loss', 'acc'])
        num_iter = 0
        n_batches = len(train_dataloader)

        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            num_iter += 1
            points, label = data[0].cuda(), data[1].cuda()

            if len(points.shape) == 3 and hasattr(config, 'npoints'):
                npoints = config.npoints
                if npoints < points.size(1):
                    points = misc.fps(points, npoints)

            points = train_transforms(points)
            ret = base_model(points)
            loss, acc = base_model.module.get_loss_acc(ret, label)

            _loss = loss / config.get('step_per_update', 1)
            _loss.backward()

            if num_iter == config.get('step_per_update', 1):
                if config.get('grad_norm_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            losses.update([loss.item(), acc.item()])

        if isinstance(scheduler, list):
            for item in scheduler: item.step(epoch)
        else:
            scheduler.step(epoch)

        if args.local_rank == 0:
            print_log(
                f'[Training] EPOCH: {epoch} EpochTime = {time.time() - epoch_start_time:.3f} (s) Losses = {["%.4f" % l for l in losses.avg()]} lr = {optimizer.param_groups[0]["lr"]:.6f}',
                logger=logger)

        if epoch % args.val_freq == 0 and epoch != 0:
            if not args.distributed or args.local_rank == 0:
                metrics = validate(base_model, val_dataloader, epoch, val_writer, args, config, logger=logger)

                if metrics.better_than(best_metrics):
                    best_metrics = metrics
                    builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args,
                                            logger=logger)
                    patience_counter = 0
                    print_log(f'[Early Stopping] Validation performance improved. Patience counter reset to 0.',
                              logger=logger)
                else:
                    patience_counter += 1
                    print_log(
                        f'[Early Stopping] No improvement for {patience_counter} validation steps. Patience: {early_stop_patience}.',
                        logger=logger)

        if args.local_rank == 0:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args,
                                    logger=logger)

        if patience_counter >= early_stop_patience:
            print_log(f'Early stopping triggered after {early_stop_patience} epochs with no improvement.',
                      logger=logger)
            break

    if train_writer is not None: train_writer.close()
    if val_writer is not None: val_writer.close()

    return best_metrics


def validate(base_model, val_dataloader, epoch, val_writer, args, config, logger=None):
    # ... (此函数保持不变) ...
    base_model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(val_dataloader):
            points, label = data[0].cuda(), data[1].cuda()
            if len(points.shape) == 3 and hasattr(config, 'npoints'):
                points = misc.fps(points, config.npoints)
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

    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)

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
            logits = base_model(points)
            pred = logits.argmax(-1).view(-1)
            all_preds.append(pred.detach())
            all_labels.append(label.detach())

    all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
    all_labels = torch.cat(all_labels, dim=0).cpu().numpy()

    # --- [核心修改] 使用 classification_report 的字典输出功能 ---
    metrics_to_return = {}
    try:
        class_names = ['Daily Activity (0)', 'Fall (1)']
        report_dict = classification_report(all_labels, all_preds, target_names=class_names, digits=4, output_dict=True,
                                            zero_division=0)

        report_str = classification_report(all_labels, all_preds, target_names=class_names, digits=4, zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)

        print_log(f"[TEST] Overall Accuracy = {report_dict['accuracy'] * 100:.4f}", logger=logger)
        print_log('Confusion Matrix:', logger=logger)
        print_log('\n' + str(cm), logger=logger)
        print_log('Classification Report:', logger=logger)
        print_log('\n' + report_str, logger=logger)

        # 提取关键指标
        metrics_to_return = {
            'accuracy': report_dict['accuracy'],
            'precision': report_dict['weighted avg']['precision'],
            'recall': report_dict['weighted avg']['recall'],
            'f1-score': report_dict['weighted avg']['f1-score'],
        }
    except Exception as e:
        print_log(f"Could not generate classification report: {e}", logger=logger)

    return metrics_to_return