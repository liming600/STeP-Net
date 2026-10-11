import os
import torch
from torch.utils.data import Dataset
import numpy as np
from plyfile import PlyData
from .build import DATASETS
from tqdm import tqdm


@DATASETS.register_module()
class FallSequenceDataset(Dataset):
    # ====================================================================
    # ---                这是最终修正后的 __init__ 函数                 ---
    # ====================================================================
    def __init__(self, config):
        super().__init__()
        self.root = config.DATA_PATH
        self.npoints = config.N_POINTS

        # --- [核心修改] ---
        # 所有参数都直接从合并后的 config 对象中获取
        self.subset = config.subset
        self.sequence_length = config.sequence_length
        self.stride = config.stride

        # 1. 加载所有帧
        all_frames = []
        split_file = os.path.join(self.root, f'{self.subset}_split.txt')
        try:
            with open(split_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    rel_path, label_str = line.split()
                    full_path = os.path.join(self.root, rel_path)
                    all_frames.append((full_path, int(label_str)))
        except FileNotFoundError:
            print(f"错误: 划分文件 '{split_file}' 未找到!")
            raise

        # 2. 按序列分组
        sequences_dict = {}
        print(f"在 '{self.subset}' 集中，按序列分组 {len(all_frames)} 帧...")
        for path, label in all_frames:
            seq_name = os.path.dirname(path)
            if seq_name not in sequences_dict:
                sequences_dict[seq_name] = []
            sequences_dict[seq_name].append((path, label))

        # 3. 滑窗生成样本
        self.sequences = []
        print(f"为 {len(sequences_dict)} 个序列生成滑窗样本 (长度: {self.sequence_length}, 步长: {self.stride})...")
        for seq_name, frames in tqdm(sequences_dict.items(), desc=f"Processing {self.subset} set"):
            frames.sort(key=lambda x: x[0])
            if len(frames) < self.sequence_length:
                continue
            for i in range(0, len(frames) - self.sequence_length + 1, self.stride):
                sequence_frames_paths = [f[0] for f in frames[i: i + self.sequence_length]]
                sequence_labels = [f[1] for f in frames[i: i + self.sequence_length]]
                sequence_label = 1 if 1 in sequence_labels else 0
                self.sequences.append((sequence_frames_paths, sequence_label))

        print(f"在 '{self.subset}' 集中，最终生成了 {len(self.sequences)} 个序列样本。")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence_paths, label = self.sequences[idx]

        points_list = []
        for frame_path in sequence_paths:
            try:
                with open(frame_path, 'rb') as f:
                    plydata = PlyData.read(f)
                    points = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
                points_list.append(torch.from_numpy(points).float())
            except Exception as e:
                print(f"错误: 无法加载PLY文件 '{frame_path}': {e}")
                return None

        points_sequence = torch.stack(points_list, dim=0)
        label = torch.tensor(label).long()

        taxonomy_id = 'fall_sequence'
        model_id = 'sample'
        return taxonomy_id, model_id, (points_sequence, label)