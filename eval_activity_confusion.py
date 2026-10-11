"""对 STeP-Net (GRU/其他模块) 测生成 per-activity 混淆矩阵。

审稿人 R1-8 / R2-1 要求：
  将测试预测按底层活动 (run / walk / sit / squat / fall) 细分，
  而非仅二分类 (daily vs fall)，以说明误差发生在哪些活动上。

用法（在服务器/models所在机器上）：
    python eval_activity_confusion.py --config cfgs/finetune_sequence_gru.yaml \
        --ckpts ./experiments/finetune_sequence_gru/cfgs/loso_gru_final/fold_test_subj_1/ckpt-best.pth \
        --test_subject_id 1 --val_subject_id 2
给出每个 fold 的 ckpt 与下标即可；下面提供汇总所有 9 折的函数。
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse, copy
import torch
import numpy as np
import torch.nn as nn

from utils import parser, misc
from utils.config import get_config
from utils.logger import get_root_logger, print_log
from tools import builder

ACTIVITIES = ['run', 'walk', 'sit', 'squat', 'fall']
# 细粒度活动 -> 二分类标签
ACTIVITY2BIN = {'run': 0, 'walk': 0, 'sit': 0, 'squat': 0, 'fall': 1}


def build_test_loader(args, config, test_subj, val_subj):
    cfg = copy.deepcopy(config)
    cfg.dataset.test._base_.NAME = 'SubjectSplitSequenceDataset'
    cfg.dataset.test.others['test_subject_id'] = test_subj
    cfg.dataset.test.others['val_subject_id'] = val_subj
    _, test_dl = builder.dataset_builder(args, cfg.dataset.test)
    return test_dl


@torch.no_grad()
def collect_predictions(base_model, test_dl):
    """返回 (pred_bin, true_activity_name) 逐样本列表。"""
    base_model.eval()
    preds, acts = [], []
    for _, _, data in test_dl:
        points, label = data[0].cuda(), data[1].cuda()
        activity_name = data[2] if len(data) > 2 else None  # list of str per sample
        logits = base_model(points)  # (B,) or (B, 2)
        if logits.dim() == 2:
            pred = logits.argmax(-1).view(-1)
        else:
            pred = logits.view(-1)
        preds.append(pred.detach().cpu())
        if activity_name is not None:
            acts.extend(activity_name)
    preds = torch.cat(preds).numpy()
    return preds, (acts if acts else None)


def build_activity_confusion(preds, acts, test_subj):
    """构造 真实活动(run/walk/sit/squat/fall) x 预测二分类(daily/fall) 的混淆矩阵。"""
    from sklearn.metrics import confusion_matrix, classification_report
    if acts is None:
        print(f'Fold S{test_subj}: activity names not provided; skip per-activity matrix.')
        return
    true_activity = np.array(acts)
    # 二分类真值
    true_bin = np.array([ACTIVITY2BIN[a] for a in true_activity])

    print(f'\n===== Fold S{test_subj} — Per-Activity Confusion Matrix =====')
    print('真实活动 (row) x 预测二分类 (col: 0=daily, 1=fall)')
    for act in ACTIVITIES:
        mask = true_activity == act
        n = mask.sum()
        if n == 0:
            print(f'{act:6s}: (no samples)')
            continue
        # 每个活动被预测为 daily / fall 的数量
        pred_daily = (preds[mask] == 0).sum()
        pred_fall = (preds[mask] == 1).sum()
        # 其中真值确实是 fall 的 (对 fall 活动, 漏检/命中)
        if act == 'fall':
            detected = (preds[mask] == 1).sum()
        print(f'{act:6s}: n={n:5d} | 预测 daily={pred_daily:5d}, 预测 fall={pred_fall:5d} '
              f'{"| fall命中=" + str(detected) if act == "fall" else ""}')

    # 二分类混淆矩阵
    cm = confusion_matrix(true_bin, preds)
    print('\n二分类混淆矩阵 [[daily_true, fall_true];[daily_pred...]]:')
    print(cm)

    # 按活动粒度统计 fall 的 漏检/误报
    fall_mask = true_activity == 'fall'
    total_fall = fall_mask.sum()
    missed = true_activity[fall_mask & (preds == 0)]
    print(f'\nFall 漏检分析: 总 fall={total_fall}, 被误判为 daily(漏检)={len(missed)} '
          f'(漏检率={len(missed) / max(total_fall, 1) * 100:.2f}%)')
    fp_activities = {a: int(((true_activity == a) & (preds == 1)).sum()) for a in ACTIVITIES[:-1]}
    print('各日常活动被误报为 fall 的数量:', fp_activities)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--ckpts', required=True, help="单个fold的best ckpt")
    p.add_argument('--test_subject_id', type=int, required=True)
    p.add_argument('--val_subject_id', type=int, required=True)
    custom, _ = p.parse_known_args()

    # 用项目标准 parser 补齐所有字段 (num_workers/experiment_path 等)，
    # 再合并自定义参数。
    from utils import parser as _parser
    sys.argv = ['eval_activity_confusion', '--config', custom.config,
                '--ckpts', custom.ckpts]
    args = _parser.get_args()
    args.test_subject_id = custom.test_subject_id
    args.val_subject_id = custom.val_subject_id
    args.use_gpu = True
    args.distributed = False
    args.local_rank = 0

    config = get_config(args, logger=get_root_logger())
    test_dl = build_test_loader(args, config, args.test_subject_id, args.val_subject_id)

    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.ckpts, logger=get_root_logger())
    base_model.to(0)
    base_model = nn.DataParallel(base_model).cuda()

    preds, acts = collect_predictions(base_model, test_dl)
    build_activity_confusion(preds, acts, args.test_subject_id)


if __name__ == '__main__':
    main()