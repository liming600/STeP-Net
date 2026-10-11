import os
import torch
from torch.utils.data import Dataset
import numpy as np
from plyfile import PlyData
from .build import DATASETS


@DATASETS.register_module()
class LIDARHuman26MDataset(Dataset):
    # ====================================================================
    # ---                这是最终修正后的 __init__ 函数                 ---
    # ====================================================================
    def __init__(self, config):
        super().__init__()
        self.root = config.DATA_PATH
        self.npoints = config.N_POINTS

        # --- [核心修改] ---
        # 直接从合并后的 config 对象中获取 subset
        self.subset = config.subset

        self.datapath = []
        self.label = 0  # 预训练时标签无意义

        segment_dir = os.path.join(self.root, 'segment')
        # 根据 self.subset 的值 (将是 'train' 或 'test') 构建文件名
        split_file = os.path.join(self.root, f'{self.subset}.txt')

        try:
            with open(split_file, 'r') as f:
                folders_to_load = [line.strip() for line in f if line.strip()]

            for folder_name in folders_to_load:
                folder_path = os.path.join(segment_dir, folder_name)

                if not os.path.isdir(folder_path):
                    print(f"警告: 在 {segment_dir} 中找不到文件夹 '{folder_name}'，已跳过。")
                    continue

                for filename in os.listdir(folder_path):
                    if filename.endswith('.ply'):
                        full_path = os.path.join(folder_path, filename)
                        self.datapath.append((full_path, self.label))
        except FileNotFoundError:
            print(f"错误: 划分文件 '{split_file}' 未找到!")
            raise

        print(f"在 '{self.subset}' 集中，成功加载了 {len(self.datapath)} 个点云文件。")

    def __getitem__(self, idx):
        file_path, label = self.datapath[idx]

        try:
            with open(file_path, 'rb') as f:
                plydata = PlyData.read(f)
                points = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
        except Exception as e:
            print(f"错误: 无法加载PLY文件 '{file_path}': {e}")
            return None

        points = torch.from_numpy(points).float()
        label = torch.tensor(label).long()

        taxonomy_id = 'lidarhuman'
        model_id = 'sample'

        return taxonomy_id, model_id, (points, label)

    def __len__(self):
        return len(self.datapath)