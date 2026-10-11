# 1. 导入“模型建造工厂”和“模型注册表”
from .build import build_model_from_cfg, MODELS

# 2. 导入所有具体的模型类，这个动作会自动触发它们的注册

# 从 Point_MAE.py 文件中，同时导入 Point_MAE 和 PointTransformer 两个类
# 这是必需的，因为预训练、单帧微调、以及时序模型的内部骨干都需要它们
from .Point_MAE import Point_MAE, PointTransformer

# 导入我们为时序任务创建的4个新模型
from .PointMAE_LSTM import PointMAE_LSTM
from .PointMAE_Transformer import PointMAE_Transformer
from .PointMAE_GRU import PointMAE_GRU
from .PointMAE_1DCNN import PointMAE_1DCNN

# 导入外部基线模型 (PointNet++ 帧级分类器)
from .PointNetPP_Frame import PointNetPP_Frame

# 3. 定义 __all__ 列表，明确暴露所有可用的模型类
__all__ = [
    'build_model_from_cfg',
    'MODELS',
    'Point_MAE',
    'PointTransformer',
    'PointMAE_LSTM',
    'PointMAE_Transformer',
    'PointMAE_GRU',
    'PointMAE_1DCNN',
    'PointNetPP_Frame',
]