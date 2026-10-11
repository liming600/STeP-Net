# 导入各个runner模块中的 run_net 函数，并赋予清晰的别名
from tools.runner_pretrain import run_net as pretrain
from tools.runner_finetune import run_net as finetune
from tools.runner_finetune import test_net
# (注意：我们不再需要 runner_finetune_sequence 了，因为功能已合并)

from utils import parser, dist_utils, misc
from utils.logger import *
from utils.config import *
import time
import os
import torch
from tensorboardX import SummaryWriter


def main():
    args = parser.get_args()
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True
    if args.launcher == 'none':
        args.distributed = False
    else:
        # (分布式逻辑)
        pass

    config = get_config(args, logger=get_root_logger())
    logger = get_logger(args.log_name)

    train_writer = val_writer = None
    if not args.test and args.local_rank == 0:
        train_writer = SummaryWriter(os.path.join(args.tfboard_path, 'train'))
        val_writer = SummaryWriter(os.path.join(args.tfboard_path, 'test'))

    logger.info(f'Distributed training: {args.distributed}')

    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: {args.deterministic}')
        misc.set_random_seed(args.seed + args.local_rank, deterministic=args.deterministic)

    # --- [核心修改：统一调用方式] ---
    if args.test:
        # 直接调用 test_net 函数
        test_net(args, config)
    else:
        if args.finetune_model or args.scratch_model:
            # 直接调用 finetune 函数
            finetune(args, config, train_writer, val_writer)
        else:
            # 直接调用 pretrain 函数
            pretrain(args, config, train_writer, val_writer)


if __name__ == '__main__':
    main()