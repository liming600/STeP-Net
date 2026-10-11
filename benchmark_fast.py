import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
import pandas as pd
from torch.utils.data import Dataset
from plyfile import PlyData
from tqdm import tqdm

from tools import builder
from tools.runner_finetune import run_net as finetune, test_net
from utils.config import get_config
from utils.logger import get_logger, print_log

# --- [修复] 正确的导入路径 ---
from datasets.build import DATASETS

# ================= 配置区域 =================
CONFIG_MAP = {
    8: 'cfgs/finetune_sequence_gru_len8.yaml',
    12: 'cfgs/finetune_sequence_gru_len12.yaml',
    16: 'cfgs/finetune_sequence_gru.yaml',
    20: 'cfgs/finetune_sequence_gru_len20.yaml'
}

PRETRAINED_PATH = '/workspace/ZLL/point-MAE/experiments/pretrain_lidarhuman/cfgs/pretrain_on_lidarhuman1/ckpt-epoch-300.pth'
QUICK_EPOCHS = 40
TOTAL_SUBJECTS = 9


@DATASETS.register_module()
class ExplicitSubjectSequenceDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.root = config.DATA_PATH
        self.npoints = config.N_POINTS
        self.mode = config.subset
        self.subject_ids = config.subject_ids
        self.sequence_length = config.sequence_length
        self.stride = config.stride
        self.sequences = []
        self.class_map = {'run': 0, 'walk': 0, 'squat': 0, 'sit': 0, 'fall': 1}

        subjects_to_load = [f'subject_{i:02d}' for i in self.subject_ids]

        for subject_dir in subjects_to_load:
            subject_path = os.path.join(self.root, subject_dir)
            if not os.path.isdir(subject_path): continue
            for activity_dir in os.listdir(subject_path):
                activity_path = os.path.join(subject_path, activity_dir)
                if not os.path.isdir(activity_path) or activity_dir not in self.class_map: continue
                label = self.class_map[activity_dir]
                for group_dir in os.listdir(activity_path):
                    group_path = os.path.join(activity_path, group_dir)
                    if not os.path.isdir(group_path): continue
                    frames = sorted([os.path.join(group_path, f) for f in os.listdir(group_path) if f.endswith('.ply')])
                    if len(frames) >= self.sequence_length:
                        for i in range(0, len(frames) - self.sequence_length + 1, self.stride):
                            self.sequences.append((frames[i: i + self.sequence_length], label))

    def __getitem__(self, idx):
        frame_paths, label = self.sequences[idx]
        point_sequence = []
        for file_path in frame_paths:
            points = self._load_ply(file_path)
            point_sequence.append(points)
        points = torch.stack(point_sequence, dim=0)
        label = torch.tensor(label).long()
        return 'sequence_sample', 'subject_split', (points, label)

    def _load_ply(self, file_path):
        try:
            with open(file_path, 'rb') as f:
                plydata = PlyData.read(f)
                points = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
        except:
            points = np.zeros((self.npoints, 3))
        if points.shape[0] > self.npoints:
            indices = np.random.choice(points.shape[0], self.npoints, replace=False)
            points = points[indices]
        elif points.shape[0] < self.npoints:
            padding = np.zeros((self.npoints - points.shape[0], 3))
            points = np.vstack([points, padding])
        return torch.from_numpy(points).float()

    def __len__(self):
        return len(self.sequences)


def run_benchmark_single_len(target_len):
    EXP_ROOT = 'experiments/benchmark_fast_cv'
    if not os.path.exists(EXP_ROOT): os.makedirs(EXP_ROOT)

    # 3-Fold 策略
    FOLDS = [
        (1, [1, 2, 3], [4]),
        (2, [4, 5, 6], [7]),
        (3, [7, 8, 9], [1]),
    ]

    config_file = CONFIG_MAP[target_len]
    print(f"\n{'#' * 30}\n### Benchmarking Sequence Length: {target_len} (3-Fold CV)\n{'#' * 30}")

    fold_accuracies = []

    for fold_idx, test_ids, val_ids in FOLDS:
        train_ids = [i for i in range(1, TOTAL_SUBJECTS + 1) if i not in test_ids and i not in val_ids]

        print(f"\n>>> Fold {fold_idx}: Test={test_ids}, Val={val_ids}")

        exp_name = f"len{target_len}_fold{fold_idx}"
        current_exp_path = os.path.join(EXP_ROOT, exp_name)
        if not os.path.exists(current_exp_path): os.makedirs(current_exp_path)

        args = argparse.Namespace(
            config=config_file,
            use_gpu=True, local_rank=0,
            experiment_path=current_exp_path,
            seed=42 + fold_idx,
            launcher='none', resume=False, distributed=False,
            ckpts=PRETRAINED_PATH, val=True, log_name=f'log_len{target_len}_fold{fold_idx}',
            num_workers=4, finetune_model=True, scratch_model=False,
            val_freq=1, print_freq=10, start_epoch=0, test=False,
            tfboard_path=os.path.join(current_exp_path, 'tfboard')
        )

        config = get_config(args, logger=None)
        config.max_epoch = QUICK_EPOCHS

        # 动态替换 Dataset
        for subset, ids in zip(['train', 'val', 'test'], [train_ids, val_ids, test_ids]):
            config.dataset[subset]._base_.NAME = 'ExplicitSubjectSequenceDataset'
            config.dataset[subset].others.subset = subset
            config.dataset[subset].others.subject_ids = ids
            if 'test_subject_id' in config.dataset[subset].others:
                del config.dataset[subset].others['test_subject_id']

        # 显存优化
        if target_len >= 20:
            for subset in ['train', 'val', 'test']:
                config.dataset[subset].others.bs = min(config.dataset[subset].others.bs, 10)

        # 运行
        # 这里的 None 对应 runner 里的 writer 参数
        finetune(args, config, train_writer=None, val_writer=None)

        # 测试
        best_ckpt = os.path.join(current_exp_path, 'ckpt-best.pth')
        if os.path.exists(best_ckpt):
            args.ckpts = best_ckpt
            metrics = test_net(args, config)
            acc = float(metrics.get('acc', 0.0).replace('%', '')) if metrics else 0.0
        else:
            acc = 0.0

        fold_accuracies.append(acc)
        print(f"Fold {fold_idx} Result: {acc:.2f}%")

    avg_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)

    with open(os.path.join(EXP_ROOT, 'results_summary.txt'), 'a') as f:
        f.write(f"Length {target_len}: {avg_acc:.2f}% (±{std_acc:.2f})\n")

    print(f"\n===> Sequence Length {target_len} Finished. Average Accuracy: {avg_acc:.2f}% (±{std_acc:.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--len', type=int, required=True, help='Specify sequence length to run (8, 12, 16, 20)')
    args = parser.parse_args()

    run_benchmark_single_len(args.len)