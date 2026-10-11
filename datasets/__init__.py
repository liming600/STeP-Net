# 1. 首先，从干净的 build.py 中导入“工具”，让它们先被创建好。
from .build import build_dataset_from_cfg, DATASETS

# 2. 然后，再依次导入所有需要使用这些“工具”的“工人们”（即各个Dataset类）。
#    这个动作会自动触发每个文件顶部的 @DATASETS.register_module() 装饰器。
from .SubjectSplitDataset import SubjectSplitDataset
from .SubjectSplitSequenceDataset import SubjectSplitSequenceDataset

from .LIDARHuman26MDataset import LIDARHuman26MDataset
from .FallDataset import FallDataset
from .FallSequenceDataset import FallSequenceDataset
# 导入项目自带的其他数据集，确保它们也能被注册和使用
from .ModelNetDataset import ModelNet
from .ScanObjectNNDataset import ScanObjectNN
from .ShapeNet55Dataset import ShapeNet


# 3. 定义 __all__ 列表，这是一个Python模块设计的良好实践。
__all__ = [
    'build_dataset_from_cfg',
    'DATASETS',
    'SubjectSplitDataset',
    'SubjectSplitSequenceDataset',
    'LIDARHuman26MDataset',
    'FallDataset',
    'FallSequenceDataset',
    'ModelNet',
    'ScanObjectNN',
    'ShapeNet',
]