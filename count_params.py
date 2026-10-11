import argparse
from utils.config import get_config
from tools import builder


def count_parameters(model):
    """计算模型的可训练参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(description='Count model parameters')
    parser.add_argument('--config', type=str, required=True, help='Path to the model config file')
    args = parser.parse_args()

    # 模拟一个用于 get_config 的最小化 args 对象
    class DummyArgs:
        def __init__(self):
            self.config = args.config
            self.launcher = 'none'

    config = get_config(DummyArgs())

    # 构建模型
    model = builder.model_builder(config.model)

    # 计算参数量
    num_params = count_parameters(model)

    print("\n" + "=" * 50)
    print(f"模型名称: {config.model.NAME}")
    print(f"可训练参数量: {num_params:,} ({num_params / 1e6:.2f} M)")
    print("=" * 50 + "\n")


if __name__ == '__main__':
    main()