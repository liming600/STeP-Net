# datasets/SubjectSplitDataset.py (修正版)

import os
import torch
from torch.utils.data import Dataset
from plyfile import PlyData
import numpy as np
from tqdm import tqdm
from .build import DATASETS


@DATASETS.register_module()
class SubjectSplitDataset(Dataset):
    """
    一个专门为 LOSO-CV 设计的数据集类。
    """

    # --- [核心修复] ---
    # __init__ 函数签名改变，不再接收 mode, test_subject_id, val_subject_id
    # 直接从 config 对象中读取所有需要的信息
    def __init__(self, config):
        super().__init__()
        self.root = config.DATA_PATH
        self.npoints = getattr(config, 'N_POINTS', 1024)

        # 从 config.others (即 default_args) 中获取参数
        self.mode = config.subset  # 'train', 'val', or 'test'
        self.test_subject_id = config.test_subject_id
        self.val_subject_id = config.val_subject_id

        # 数据增强开关 (忠实复现 Ayyagari et al., 2025 的训练流程)：
        #   random rotation / translation / scaling / jitter
        # 仅训练集启用，且仅在 use_augmentation=True 时生效 (默认关闭，不影响主实验)。
        self.use_augmentation = getattr(config, 'use_augmentation', False) and self.mode == 'train'

        self.datapath = []
        self.labels = []
        # 假设 class_map，如果你的 FallDataset 有更复杂的逻辑，需要相应调整
        self.class_map = {'run': 0, 'walk': 0, 'squat': 0, 'sit': 0, 'fall': 1}

        print(
            f"Initializing dataset in '{self.mode}' mode. Test Subject: {self.test_subject_id}, Val Subject: {self.val_subject_id}")

        all_subjects = sorted([d for d in os.listdir(self.root) if d.startswith('subject_')])

        subjects_to_load = []
        if self.mode == 'train':
            subjects_to_load = [s for s in all_subjects if
                                int(s.split('_')[-1]) not in [self.test_subject_id, self.val_subject_id]]
        elif self.mode == 'val':
            subjects_to_load = [f'subject_{self.val_subject_id:02d}']
        elif self.mode == 'test':
            subjects_to_load = [f'subject_{self.test_subject_id:02d}']

        print(f"Loading data for subjects: {subjects_to_load}")

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

                    frames = sorted(os.listdir(group_path))
                    for frame_file in frames:
                        if frame_file.endswith('.ply'):
                            file_path = os.path.join(group_path, frame_file)
                            self.datapath.append(file_path)
                            self.labels.append(label)

        print(f"Successfully loaded {len(self.datapath)} frames for '{self.mode}' set.")

    def __getitem__(self, idx):
        file_path = self.datapath[idx]
        label = self.labels[idx]

        try:
            with open(file_path, 'rb') as f:
                plydata = PlyData.read(f)
                points = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
        except Exception as e:
            print(f"Error loading PLY file '{file_path}': {e}")
            points = np.zeros((self.npoints, 3))

        if points.shape[0] > self.npoints:
            indices = np.random.choice(points.shape[0], self.npoints, replace=False)
            points = points[indices]
        elif points.shape[0] < self.npoints:
            padding = np.zeros((self.npoints - points.shape[0], 3))
            points = np.vstack([points, padding])

        if self.use_augmentation:
            points = self._augment(points)

        points = torch.from_numpy(points).float()
        label = torch.tensor(label).long()

        taxonomy_id = 'subject_split'
        model_id = 'frame'

        return taxonomy_id, model_id, (points, label)

    def _augment(self, points):
        """点云数据增强 (random rotation / translation / scaling / jitter)。

        与 Ayyagari et al. (2025) 的训练流程一致。
        """
        # random rotation (绕 Z 轴 + 任意轴)
        theta = np.random.uniform(0, 2 * np.pi)
        rot_z = np.array([[np.cos(theta), -np.sin(theta), 0],
                          [np.sin(theta), np.cos(theta), 0],
                          [0, 0, 1]], dtype=np.float64)
        axis = np.random.randn(3)
        axis = axis / (np.linalg.norm(axis) + 1e-8)
        phi = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(phi), np.sin(phi)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]], dtype=np.float64)
        rot_axis = np.eye(3) + s * K + (1 - c) * (K @ K)
        rot = rot_axis @ rot_z
        points = points @ rot.T

        # random translation
        shift = np.random.uniform(-0.2, 0.2, size=(1, 3))
        points = points + shift

        # random scaling
        scale = np.random.uniform(0.8, 1.2)
        points = points * scale

        # jitter
        points = points + np.clip(np.random.normal(0, 0.02, size=points.shape), -0.05, 0.05)
        return points

    def __len__(self):
        return len(self.datapath)