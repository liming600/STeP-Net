import torch
import numpy as np
import yaml
import time
from tools import builder


# ================= 修正后的 Config 类 =================
class Config(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"No attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value

    @classmethod
    def from_yaml(cls, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls._convert(data)

    @classmethod
    def _convert(cls, d):
        if isinstance(d, list):
            return [cls._convert(v) for v in d]
        elif isinstance(d, dict):
            obj = cls()
            for k, v in d.items():
                obj[k] = cls._convert(v)
            return obj
        else:
            return d


# ================= 配置路径 =================
# 更新了命名规范，并添加了 Scratch 版本的 GRU 配置
CONFIG_FILES = {
    'STeP-SF': 'cfgs/finetune_fall_detection.yaml',  # 原 Single Frame
    'STeP-TC': 'cfgs/finetune_sequence_1dcnn.yaml',  # 原 1D-CNN
    'STeP-TL': 'cfgs/finetune_sequence_lstm.yaml',  # 原 LSTM
    'STeP-TG': 'cfgs/finetune_sequence_gru.yaml',  # 原 GRU
    'STeP-TTr': 'cfgs/finetune_sequence_transformer.yaml',  # 原 Transformer
    'STeP-TG-Scratch': 'cfgs/finetune_overfit.yaml'# 新增：从零训练的 GRU
}


def analyze_model(name, config_path):
    print(f"--- Analyzing {name} ---")
    try:
        config = Config.from_yaml(config_path)
        # 尝试构建模型
        model = builder.model_builder(config.model)
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
    except Exception as e:
        print(f"Error loading {name} from {config_path}: {e}")
        return

    # 1. 参数量分析
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    # 2. 推理速度 (End-to-End Latency)
    # 构造输入: 序列模型通常输入 (B, Frames, Points, C)
    B, N, C = 1, 1024, 3

    # 判断是否为序列模型：
    # 逻辑：只要名字不是 STeP-SF (Single Frame)，或者配置文件路径包含 sequence/lstm/gru 等关键词
    is_sequence_name = name != 'STeP-SF'
    is_sequence_file = any(x in config_path for x in ['sequence', 'lstm', 'gru', 'transformer', 'overfit'])

    is_sequence = is_sequence_name or is_sequence_file

    if is_sequence:
        # 尝试从配置读取帧数，优先读取 sequence_length，其次 nframes，默认 16
        try:
            others = config.dataset.train.others
            if 'sequence_length' in others:
                n_frames = others.sequence_length
            elif 'nframes' in others:
                n_frames = others.nframes
            else:
                n_frames = 16
        except:
            n_frames = 16

        dummy_input = torch.randn(B, n_frames, N, C).cuda()
        input_type = f"Sequence ({n_frames} frames)"
    else:
        dummy_input = torch.randn(B, N, C).cuda()
        input_type = "Single Frame"

    # 预热 GPU
    try:
        with torch.no_grad():
            for _ in range(10):
                model(dummy_input)
    except Exception as e:
        print(f"Warning: Warmup failed. Error: {e}")
        return

    # 测速
    times = []
    with torch.no_grad():
        # 跑 100 次以获得更稳定的平均值
        for _ in range(100):
            torch.cuda.synchronize()
            start = time.time()
            model(dummy_input)
            torch.cuda.synchronize()
            times.append((time.time() - start) * 1000)

    avg_latency = np.mean(times)
    fps = 1000 / avg_latency if avg_latency > 0 else 0

    # 打印详细报告
    print(f"  > Input Type:                {input_type}")
    print(f"  > Total Params (Overall):    {total_params / 1e6:.2f} M")
    print(f"  > Frozen Params:             {frozen_params / 1e6:.2f} M  ({frozen_params / total_params * 100:.1f}%)")
    print(
        f"  > Trainable Params:          {trainable_params / 1e6:.2f} M  ({trainable_params / total_params * 100:.1f}%)")
    print(f"  > Inference Latency:         {avg_latency:.2f} ms")
    print(f"  > FPS:                       {fps:.1f}")
    print("")


if __name__ == "__main__":
    for name, path in CONFIG_FILES.items():
        analyze_model(name, path)