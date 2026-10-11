"""Per-activity confusion matrix for LOSO evaluation.
Usage: python confusion_matrix.py --config cfgs/finetune_sequence_gru.yaml \
       --ckpts <best_model_per_fold_or_pretrain> --exp_name confusion_gru

If ckpts points to a single pretrain, LOSO folds are trained from scratch.
If ckpts points to an experiment dir with per-fold checkpoints, only eval is run.
"""
import os, sys, copy, torch
sys.path.insert(0, os.path.dirname(__file__))
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tools.runner_finetune import run_net as finetune
from utils import parser, misc
from utils.config import get_config
from utils.logger import get_root_logger
from tools import builder

ACTIVITIES = ['run', 'walk', 'squat', 'sit', 'fall']
NUM_ACTIVITIES = len(ACTIVITIES)
TOTAL_SUBJECTS = 9


def evaluate_subject(args, config, test_subject_id, val_subject_id, ckpt_path):
    """Run inference on one test subject, return (y_true_activities, y_pred_labels)."""
    from tools.runner_finetune import test_net

    orig_ckpts = args.ckpts
    args.ckpts = ckpt_path
    config = copy.deepcopy(config)

    for subset in ['test']:
        config.dataset[subset]._base_.NAME = 'SubjectSplitSequenceDataset'
        config.dataset[subset].others['test_subject_id'] = test_subject_id
        config.dataset[subset].others['val_subject_id'] = val_subject_id

    # Build test dataloader
    from tools import builder as bd
    _, test_dataloader = bd.dataset_builder(args, config.dataset.test)

    # Build model and load checkpoint
    base_model = bd.model_builder(config.model)
    bd.load_model(base_model, args.ckpts, logger=get_root_logger())
    base_model.to(args.local_rank)
    base_model = nn.DataParallel(base_model).cuda()
    base_model.eval()

    all_true_acts = []
    all_preds = []
    with torch.no_grad():
        for _, _, data in test_dataloader:
            points = data[0].cuda()
            labels_bin = data[1]
            activities = data[2]  # list of strings
            logits = base_model(points)
            preds = logits.argmax(-1).cpu()

            all_true_acts.extend(activities)
            all_preds.extend(preds.tolist())

    args.ckpts = orig_ckpts
    del base_model, test_dataloader
    torch.cuda.empty_cache()
    return all_true_acts, all_preds


def main():
    args = parser.get_args()
    args.use_gpu = torch.cuda.is_available()
    args.distributed = False
    args.local_rank = 0

    original_ckpts = args.ckpts
    original_exp_path = args.experiment_path

    config = get_config(args, logger=get_root_logger())

    all_true_acts, all_preds_bin = [], []
    all_subject_ids = []

    for test_subject_id in range(1, TOTAL_SUBJECTS + 1):
        val_subject_id = (test_subject_id % TOTAL_SUBJECTS) + 1

        fold_ckpt = os.path.join(original_exp_path, f'fold_test_subj_{test_subject_id}', 'ckpt-best.pth')
        if os.path.exists(fold_ckpt):
            ckpt_to_use = fold_ckpt
            print(f'Subject {test_subject_id}: using fold checkpoint {fold_ckpt}', flush=True)
        else:
            ckpt_to_use = original_ckpts
            print(f'Subject {test_subject_id}: fold ckpt not found, pretrain + train from scratch', flush=True)
            finetune(args, config, None, None, early_stop_patience=15)
            ckpt_to_use = os.path.join(args.experiment_path, f'fold_test_subj_{test_subject_id}', 'ckpt-best.pth')
            if not os.path.exists(ckpt_to_use):
                print(f'  No checkpoint found for subject {test_subject_id}, skipping.', flush=True)
                continue

        true_acts, preds_bin = evaluate_subject(args, config, test_subject_id, val_subject_id, ckpt_to_use)
        all_true_acts.extend(true_acts)
        all_preds_bin.extend(preds_bin)

        # Per-subject per-activity breakdown
        true_bin = [1 if a == 'fall' else 0 for a in true_acts]
        print(f'\n--- Subject {test_subject_id} ---', flush=True)
        print(classification_report(true_bin, preds_bin,
                                    target_names=['Daily', 'Fall'], digits=4, zero_division=0), flush=True)

    # --- Full per-activity confusion matrix ---
    print(f'\n{"="*80}', flush=True)
    print('PER-ACTIVITY BINARY CONFUSION MATRIX (All Subjects)', flush=True)
    print(f'Rows=true activity, Cols=predicted binary label', flush=True)
    print(f'{"="*80}', flush=True)

    y_pred_bin = np.array(all_preds_bin)       # predicted binary (0/1)
    y_true_acts = all_true_acts                 # true activity names

    rows = []
    for act in ACTIVITIES:
        mask = [a == act for a in y_true_acts]
        total = sum(mask)
        if total == 0:
            continue
        pred_daily = sum(1 for i, m in enumerate(mask) if m and y_pred_bin[i] == 0)
        pred_fall  = sum(1 for i, m in enumerate(mask) if m and y_pred_bin[i] == 1)
        rows.append({
            'Activity': act,
            'Total': total,
            'Pred Daily': pred_daily,
            'Pred Fall': pred_fall,
        })
    cm_df = pd.DataFrame(rows)
    print(cm_df.to_string(index=False), flush=True)

    print(f'\n{"="*80}', flush=True)
    print('Fall Detection Rate by Activity', flush=True)
    print(f'{"="*80}', flush=True)
    for r in rows:
        total = r['Total']
        pred_fall = r['Pred Fall']
        pct = pred_fall / total * 100
        if r['Activity'] == 'fall':
            print(f'  Fall:     {pred_fall:4d}/{total:4d} correctly detected  ({pct:.2f}%)', flush=True)
        else:
            print(f'  {r["Activity"]:>6}: {pred_fall:4d}/{total:4d} misclassified as fall  ({pct:.2f}%)', flush=True)


if __name__ == '__main__':
    main()
