"""PointNet++ 帧级分类器外部基线 - 独立 LOSO 训练脚本。

对比试验专用脚本，不修改项目主实验代码 (runner_finetune.py 等)。

设计依据 (与 Ayyagari et al., 2025 论文一致)：
  - 单帧输入：每个点云帧 = 一个独立样本（无序列/滑窗），与原论文逐帧分类一致
  - 数据增强：random rotation / translation / scaling / jitter (仅训练集，
    config 里 dataset.train.others.use_augmentation=True 启用；默认 False 不影响主实验)
  - 逐帧监督：每个样本一个标签，标准 CE 训练
  - 逐帧评估：测试时对每一帧独立判断为'日常行为'或'跌倒' (LOSO 仅按志愿者划分数据)
  - 输入 1024 点 (与 STeP-Net 对齐，保证公平对比)
  - 架构：标准 PointNet++ 分类器 (3 SA + 3 FC, radii [0.2,0.4], mlp [64,128,256,512,1024])

Usage:
    python loso_pointnetpp_frame.py
"""
import sys, os, copy, torch, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import torch.nn as nn
import torch.nn.functional as F
from utils import parser, misc
from utils.config import get_config
from utils.logger import get_root_logger, print_log
from tools import builder
from tools.runner_finetune import Acc_Metric

CFG_PATH = 'cfgs/finetune_sequence_pointnetpp_frame.yaml'
TOTAL_SUBJECTS = 9
MAX_EPOCH = 101          # 与主 LOSO 协议一致
EARLY_STOP_PATIENCE = 15
SEED = 42


def build_train_val_test(args, config, test_subj, val_subj):
    cfg = copy.deepcopy(config)
    for subset in ['train', 'val', 'test']:
        cfg.dataset[subset]._base_.NAME = 'SubjectSplitDataset'   # 单帧数据集
        cfg.dataset[subset].others['test_subject_id'] = test_subj
        cfg.dataset[subset].others['val_subject_id'] = val_subj
    (train_sampler, train_dl) = builder.dataset_builder(args, cfg.dataset.train)
    (_, val_dl)              = builder.dataset_builder(args, cfg.dataset.val)
    (_, test_dl)             = builder.dataset_builder(args, cfg.dataset.test)
    return train_sampler, train_dl, val_dl, test_dl


def predict_all_frames(base_model, dataloader):
    """逐帧评估：对测试集每一帧独立判断（论文式），返回 (preds, labels)。"""
    base_model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for _, _, data in dataloader:
            pts, lbl = data[0].cuda(), data[1].cuda()
            logits = base_model(pts)   # (B, C)
            preds = logits.argmax(-1)
            all_preds.append(preds.cpu())
            all_labels.append(lbl.cpu())
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


def validate_frames(base_model, val_dl, epoch, args, config, logger=None):
    """逐帧验证准确率，用于早停。"""
    preds, labels = predict_all_frames(base_model, val_dl)
    acc = (preds == labels).sum() / float(len(labels)) * 100.
    print_log(f'[Validation] EPOCH: {epoch}  acc (per-frame) = {acc:.4f}', logger=logger)
    return Acc_Metric(acc)


def train_fold(args, config, test_subj, val_subj):
    misc.set_random_seed(SEED + test_subj)
    train_sampler, train_dl, val_dl, test_dl = build_train_val_test(args, config, test_subj, val_subj)

    base_model = builder.model_builder(config.model)   # PointNetPP_Frame, 从零训练
    base_model.to(0)
    base_model = nn.DataParallel(base_model).cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    base_model.zero_grad()

    best_val = Acc_Metric(0.)
    best_state = None
    patience = 0

    for epoch in range(MAX_EPOCH):
        base_model.train()
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dl):
            points, label = data[0].cuda(), data[1].cuda()   # (B, N, 3), (B,)
            ret = base_model(points)
            loss = criterion(ret, label.long())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
            optimizer.step()
            base_model.zero_grad()

        if isinstance(scheduler, list):
            for s in scheduler: s.step(epoch)
        else:
            scheduler.step(epoch)

        if epoch == 0 or epoch % 10 == 0:
            print(f'  [fold S{test_subj}] epoch {epoch:3d} loss {loss.item():.4f}', flush=True)

        val_metrics = validate_frames(base_model, val_dl, epoch, args, config)
        if val_metrics.better_than(best_val):
            best_val = val_metrics
            best_state = copy.deepcopy(base_model.module.state_dict())
            patience = 0
        else:
            patience += 1
        if patience >= EARLY_STOP_PATIENCE:
            break

    base_model.module.load_state_dict(best_state)
    base_model.eval()

    all_preds, all_labels = predict_all_frames(base_model, test_dl)

    from sklearn.metrics import classification_report
    r = classification_report(all_labels, all_preds,
                              target_names=['Daily', 'Fall'],
                              digits=4, output_dict=True, zero_division=0)

    n_train = len(train_dl.dataset)
    del base_model, optimizer, scheduler, train_dl, val_dl, test_dl
    torch.cuda.empty_cache()
    return {
        'test_subj': test_subj,
        'accuracy':      r['accuracy'] * 100,
        'fall_recall':    r['Fall']['recall'] * 100,
        'fall_precision': r['Fall']['precision'] * 100,
        'fall_f1':        r['Fall']['f1-score'] * 100,
        'n_train':        n_train,
    }


def main():
    sys.argv = ['loso_pointnetpp', '--config', CFG_PATH, '--ckpts', 'None']
    args = parser.get_args()
    args.use_gpu = True; args.distributed = False; args.local_rank = 0
    args.resume = False; args.finetune_model = False; args.scratch_model = True

    config = get_config(args, logger=get_root_logger())

    print(f'\n{"="*70}', flush=True)
    print('PointNet++ 帧级分类器 (外部基线) — LOSO-CV', flush=True)
    print(f'单帧输入 (每帧=一个样本, 原论文逐帧分类) + 逐帧评估 (LOSO 仅按志愿者划分, 1024 点)', flush=True)
    print(f'{"="*70}', flush=True)

    all_results = []
    for test_subject_id in range(1, TOTAL_SUBJECTS + 1):
        val_subject_id = (test_subject_id % TOTAL_SUBJECTS) + 1
        print(f'\n{"─"*60}', flush=True)
        print(f'LOSO FOLD {test_subject_id}/{TOTAL_SUBJECTS}: test=S{test_subject_id}, val=S{val_subject_id}', flush=True)
        m = train_fold(args, config, test_subject_id, val_subject_id)
        all_results.append(m)
        print(f'  ✓ acc={m["accuracy"]:.2f}%  recall={m["fall_recall"]:.2f}%  '
              f'precision={m["fall_precision"]:.2f}%  f1={m["fall_f1"]:.2f}%', flush=True)

    print(f'\n{"="*70}', flush=True)
    print('LOSO-CV FINAL RESULT (PointNet++ Frame baseline)', flush=True)
    print(f'{"Subj":>5} {"#Train":>8} {"Acc":>8} {"Recall":>8} {"Prec":>8} {"F1":>8}', flush=True)
    print(f'{"─"*56}', flush=True)
    for m in all_results:
        print(f'S{m["test_subj"]:>2} {m["n_train"]:>8} {m["accuracy"]:>7.2f}% '
              f'{m["fall_recall"]:>7.2f}% {m["fall_precision"]:>7.2f}% {m["fall_f1"]:>7.2f}%', flush=True)

    import pandas as pd
    df = pd.DataFrame(all_results)
    for metric in ['accuracy', 'fall_recall', 'fall_precision', 'fall_f1']:
        print(f'{metric:20s}: {df[metric].mean():.2f} ± {df[metric].std():.2f}', flush=True)
    print(f'{"="*70}', flush=True)
    print(f'\nDone.', flush=True)


if __name__ == '__main__':
    main()
