# datasets/SubjectSplitSequenceDataset.py

import os
import torch
from torch.utils.data import Dataset
from plyfile import PlyData
import numpy as np
from tqdm import tqdm
from .build import DATASETS
from utils.logger import print_log


@DATASETS.register_module()
class SubjectSplitSequenceDataset(Dataset):
    """
    一个专门为【时序模型】进行【留一主体交叉验证 (LOSO-CV)】设计的数据集类。
    它会根据传入的 subject ID 动态加载数据，并自动执行滑窗操作。
    """

    def __init__(self, config):
        super().__init__()
        # 从 config 对象中读取所有需要的参数
        self.root = config.DATA_PATH
        self.npoints = config.N_POINTS

        self.mode = config.subset
        self.test_subject_id = config.test_subject_id
        self.val_subject_id = config.val_subject_id

        self.sequence_length = config.sequence_length
        self.stride = config.stride

        self.sequences = []  # 最终存储 ( [path_list], label, activity_name ) 格式的样本
        self.class_map = {'run': 0, 'walk': 0, 'squat': 0, 'sit': 0, 'fall': 1}

        print_log(
            f"Initializing SEQUENCE dataset in '{self.mode}' mode. Test Subj: {self.test_subject_id}, Val Subj: {self.val_subject_id}",
            logger='Dataset')

        # 1. 确定要加载哪些 subject 的文件夹
        all_subjects = sorted([d for d in os.listdir(self.root) if d.startswith('subject_')])
        subjects_to_load = []
        if self.mode == 'train':
            subjects_to_load = [s for s in all_subjects if
                                int(s.split('_')[-1]) not in [self.test_subject_id, self.val_subject_id]]
        elif self.mode == 'val':
            subjects_to_load = [f'subject_{self.val_subject_id:02d}']
        elif self.mode == 'test':
            subjects_to_load = [f'subject_{self.test_subject_id:02d}']

        print_log(f"Loading data for subjects: {subjects_to_load}", logger='Dataset')

        # 2. 遍历文件夹，按动作组收集所有帧路径
        pbar = tqdm(subjects_to_load, desc=f'Processing {self.mode} set')
        for subject_dir in pbar:
            subject_path = os.path.join(self.root, subject_dir)
            if not os.path.isdir(subject_path): continue

            for activity_dir in os.listdir(subject_path):
                activity_path = os.path.join(subject_path, activity_dir)
                if not os.path.isdir(activity_path) or activity_dir not in self.class_map: continue

                label = self.class_map[activity_dir]

                for group_dir in os.listdir(activity_path):
                    group_path = os.path.join(activity_path, group_dir)
                    if not os.path.isdir(group_path): continue

                    # 读取并排序一个动作组内的所有帧
                    frames = sorted([os.path.join(group_path, f) for f in os.listdir(group_path) if f.endswith('.ply')])

                    # 3. 在该动作组上执行滑窗操作
                    if len(frames) >= self.sequence_length:
                        for i in range(0, len(frames) - self.sequence_length + 1, self.stride):
                            sequence_paths = frames[i: i + self.sequence_length]
                            self.sequences.append((sequence_paths, label, activity_dir))

        print_log(f"Successfully generated {len(self.sequences)} sequence samples for '{self.mode}' set.",
                  logger='Dataset')

    def __getitem__(self, idx):
        frame_paths, label, activity_name = self.sequences[idx]

        point_sequence = []
        for file_path in frame_paths:
            # 调用辅助函数加载并处理单个 ply 文件
            points = self._load_ply(file_path)
            point_sequence.append(points)

        # 将帧列表堆叠成一个 (T, N, 3) 的张量
        points = torch.stack(point_sequence, dim=0)
        label = torch.tensor(label).long()

        # 为了兼容 runner 的数据解包 (data[0]=points, data[1]=label, data[2]=activity_name)
        return 'sequence_sample', 'subject_split', (points, label, activity_name)

    def _load_ply(self, file_path):
        """辅助函数，用于加载和预处理单个 PLY 文件"""
        try:
            with open(file_path, 'rb') as f:
                plydata = PlyData.read(f)
                points = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
        except Exception as e:
            print_log(f"Warning: Failed to load {file_path}. Reason: {e}. Returning zeros.", logger='Dataset')
            points = np.zeros((self.npoints, 3))

        # 采样或填充点云到 npoints
        if points.shape[0] > self.npoints:
            indices = np.random.choice(points.shape[0], self.npoints, replace=False)
            points = points[indices]
        elif points.shape[0] < self.npoints:
            padding = np.zeros((self.npoints - points.shape[0], 3))
            points = np.vstack([points, padding])

        return torch.from_numpy(points).float()

    def __len__(self):
        return len(self.sequences)