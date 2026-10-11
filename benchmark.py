import torch
import time
import argparse
import numpy as np

# 导入项目核心组件
from utils.config import get_config
from tools import builder
from utils.logger import get_root_logger


def benchmark_model(config_path, ckpt_path, batch_size=1):
    """
    对指定的模型进行基准测试，测量其延迟和FPS。
    """

    # --- 1. 环境和配置准备 ---
    # 模拟一个 argparse 对象
    class Args:
        def __init__(self):
            self.config = config_path
            self.ckpts = ckpt_path
            self.local_rank = 0
            self.launcher = 'none'
            self.exp_name = 'benchmark'
            self.log_name = 'benchmark_logger'
            self.use_gpu = torch.cuda.is_available()

    args = Args()
    logger = get_root_logger(name=args.log_name)
    config = get_config(args, logger=logger)

    # --- 2. 构建模型并加载权重 ---
    model = builder.model_builder(config.model)
    builder.load_model(model, args.ckpts, logger=logger)
    if args.use_gpu:
        model.cuda()
    model.eval()  # 关键：设置为评估模式

    # --- 3. 准备输入数据 ---
    # 判断是单帧还是时序模型
    is_sequence_model = config.model.NAME in ['PointMAE_LSTM', 'PointMAE_GRU', 'PointMAE_1DCNN', 'PointMAE_Transformer']

    if is_sequence_model:
        T = config.dataset.train.others.sequence_length
        N = config.npoints
        dummy_input = torch.randn(batch_size, T, N, 3)
        model_type_str = f"Sequence Model (T={T})"
    else:  # 单帧模型
        N = config.npoints
        dummy_input = torch.randn(batch_size, N, 3)
        model_type_str = "Single-Frame Model"

    if args.use_gpu:
        dummy_input = dummy_input.cuda()

    print("\n" + "=" * 80)
    print(f"开始基准测试: {config.model.NAME}")
    print(f"模型类型: {model_type_str}")
    print(f"输入张量形状: {list(dummy_input.shape)}")
    print("=" * 80)

    # --- 4. 执行基准测试 ---
    with torch.no_grad():
        # a. GPU预热
        print("正在预热GPU...")
        for _ in range(50):
            _ = model(dummy_input)

        # b. 开始计时
        torch.cuda.synchronize()  # 确保之前的CUDA操作全部完成
        start_time = time.time()

        # c. 运行多次以获得稳定平均值
        num_runs = 200
        print(f"正在执行 {num_runs} 次推理...")
        for _ in range(num_runs):
            _ = model(dummy_input)

        # d. 结束计时
        torch.cuda.synchronize()
        end_time = time.time()

    # --- 5. 计算并打印结果 ---
    total_time = end_time - start_time
    avg_latency_ms = (total_time / num_runs) * 1000
    # FPS的计算需要考虑batch size
    fps = (num_runs * batch_size) / total_time

    print("\n--- 基准测试结果 ---")
    print(f"模型名称: {config.model.NAME}")
    print(f"平均延迟 (Latency): {avg_latency_ms:.3f} ms")
    print(f"吞吐率 (Throughput): {fps:.2f} samples/sec")
    print("-" * 20 + "\n")


if __name__ == '__main__':
    # ==============================================================================
    # ---                         用户配置区域                         ---
    # ==============================================================================

    # --- 示例1: 测试 GRU (T=16) 模型 ---
    print("--- Running Benchmark for GRU (T=16) ---")
    gru_config = 'cfgs/finetune_sequence_gru.yaml'
    gru_ckpt = '/workspace/ZLL/point-MAE/experiments/finetune_sequence_gru/cfgs/finetune_sequence_gru_exp1/ckpt-best.pth'  # <-- 替换为您的真实路径
    benchmark_model(gru_config, gru_ckpt, batch_size=1)

    # --- 示例2: 测试 1D-CNN (T=16) 模型 ---
    # print("\n--- Running Benchmark for 1D-CNN (T=16) ---")
    # cnn_config = 'cfgs/finetune_sequence_1dcnn.yaml'
    # cnn_ckpt = '/path/to/your/1dcnn/ckpt-best.pth' # <-- 替换为您的真实路径
    # benchmark_model(cnn_config, cnn_ckpt, batch_size=1)

    # --- 示例3: 测试 单帧模型 ---
    # print("\n--- Running Benchmark for Single-Frame Model ---")
    # single_frame_config = 'cfgs/finetune_fall_detection.yaml'
    # single_frame_ckpt = '/path/to/your/single_frame/ckpt-best.pth' # <-- 替换为您的真实路径
    # benchmark_model(single_frame_config, single_frame_ckpt, batch_size=1)

    # ==============================================================================