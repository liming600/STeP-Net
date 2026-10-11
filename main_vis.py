from utils import parser, dist_utils, misc
from utils.logger import get_root_logger
from utils.config import get_config
import time
import os
import torch


# 导入我们新创建的 run_vis 函数
from tools.runner_vis import run_vis


def main():
    # 1. 解析命令行参数
    args = parser.get_args()

    # 2. 设置GPU和分布式 (对于可视化，我们简化它)
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True
    args.distributed = False  # 可视化不需要分布式

    # 3. 创建日志记录器 (Logger)
    # 我们将日志直接输出到终端，也可以指定文件
    logger = get_root_logger()

    # 4. 调用 get_config，并将创建好的 logger 传递给它
    #    get_config 函数需要一个名为 'logger' 的参数
    config = get_config(args, logger=logger)

    # 5. 设置随机种子
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: {args.deterministic}')
        misc.set_random_seed(args.seed, deterministic=args.deterministic)

    # 6. 直接调用我们自己的可视化执行器
    run_vis(args, config)


if __name__ == '__main__':
    main()