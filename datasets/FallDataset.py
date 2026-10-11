import os
import torch
from torch.utils.data import Dataset
import numpy as np
from plyfile import PlyData
from .build import DATASETS


@DATASETS.register_module()
class FallDataset(Dataset):
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

        # 根据 self.subset 的值 (将是 'train', 'val', 或 'test') 构建文件名
        split_file = os.path.join(self.root, f'{self.subset}_split.txt')

        try:
            with open(split_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rel_path, label_str = line.split()
                    full_path = os.path.join(self.root, rel_path)
                    self.datapath.append((full_path, int(label_str)))
        except FileNotFoundError:
            print(f"错误: 划分文件 '{split_file}' 未找到! 请检查您的DATA_PATH和划分文件名。")
            raise

        print(f"在 '{self.subset}' 集中，成功加载了 {len(self.datapath)} 个跌倒检测点云文件。")

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

        taxonomy_id = 'falldataset'
        model_id = 'sample'

        return taxonomy_id, model_id, (points, label)

    def __len__(self):
        return len(self.datapath)