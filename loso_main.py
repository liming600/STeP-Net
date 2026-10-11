# loso_main.py (Final Robust Version)

import numpy as np
import pandas as pd
import torch
import os

from tools.runner_finetune import run_net as finetune, test_net
from utils import parser, dist_utils, misc
from utils.logger import *
from utils.config import *
from tools import builder


def main():
    args = parser.get_args()
    args.use_gpu = torch.cuda.is_available()

    original_pretrain_ckpt_path = args.ckpts
    original_exp_path = args.experiment_path

    if not hasattr(args, 'launcher') or args.launcher == 'none':
        args.distributed = False
        args.local_rank = 0
    else:
        args.distributed = True
        args.local_rank = int(os.environ.get('LOCAL_RANK', 0))

    logger = get_logger(args.log_name)
    base_dataset_name = 'SubjectSplitSequenceDataset'
    print_log('Using the sequence-based LOSO protocol.', logger=logger)

    TOTAL_SUBJECTS = 9
    all_fold_metrics = []

    for test_subject_id in range(1, TOTAL_SUBJECTS + 1):
        val_subject_id = (test_subject_id % TOTAL_SUBJECTS) + 1

        fold_exp_name = f'fold_test_subj_{test_subject_id}'
        args.experiment_path = os.path.join(original_exp_path, fold_exp_name)
        if not os.path.exists(args.experiment_path):
            os.makedirs(args.experiment_path)

        print_log(
            f'{"=" * 20} LOSO FOLD {test_subject_id}/{TOTAL_SUBJECTS}: Testing on Subject {test_subject_id}, Val on {val_subject_id} {"=" * 20}',
            logger=logger)
        print_log(f'Results for this fold will be saved in: {args.experiment_path}', logger=logger)

        args.ckpts = original_pretrain_ckpt_path

        config = get_config(args, logger=get_root_logger())

        for subset in ['train', 'val', 'test']:
            if subset in config.dataset:
                config.dataset[subset]._base_.NAME = base_dataset_name
                config.dataset[subset].others['test_subject_id'] = test_subject_id
                config.dataset[subset].others['val_subject_id'] = val_subject_id

        misc.set_random_seed(args.seed + test_subject_id)

        finetune(args, config, None, None, early_stop_patience=15)

        print_log(f'--- Testing on Subject {test_subject_id} using best model from this fold ---', logger=logger)

        best_ckpt_path = os.path.join(args.experiment_path, 'ckpt-best.pth')

        if not os.path.exists(best_ckpt_path):
            print_log(f'[Error] ckpt-best.pth not found for this fold. Skipping test.', logger=logger)
            continue

        args.ckpts = best_ckpt_path

        test_metrics_dict = test_net(args, config)
        if test_metrics_dict:
            all_fold_metrics.append(test_metrics_dict)
            print_log(
                f"Test metrics on Subject {test_subject_id}: { {k: f'{v * 100:.2f}%' for k, v in test_metrics_dict.items()} }",
                logger=logger)

        args.experiment_path = original_exp_path

    if not all_fold_metrics:
        print_log('LOSO-CV finished, but no metrics were collected.', logger=logger)
        return

    df_metrics = pd.DataFrame(all_fold_metrics)
    mean_metrics = df_metrics.mean()
    std_metrics = df_metrics.std()

    summary_df = pd.DataFrame({'Mean': mean_metrics, 'Std': std_metrics}) * 100

    print_log(f'{"=" * 20} LOSO-CV FINAL RESULT {"=" * 20}', logger=logger)
    print_log(f'Evaluated on {len(df_metrics)} folds.', logger=logger)
    print_log("--- Performance Summary (in %) ---", logger=logger)
    print_log(summary_df.to_string(float_format="%.2f"), logger=logger)

    print_log("--- Per-Metric Summary ---", logger=logger)
    for metric in df_metrics.columns:
        m = mean_metrics[metric] * 100
        s = std_metrics[metric] * 100
        print_log(f'{metric:20s}: {m:.2f} ± {s:.2f}', logger=logger)

    print_log("--- Per-Fold Details ---", logger=logger)
    for i, row in df_metrics.iterrows():
        fold_str = ', '.join([f'{k}={v * 100:.2f}' for k, v in row.items()])
        print_log(f'Fold {i + 1}: {fold_str}', logger=logger)


if __name__ == '__main__':
    main()
