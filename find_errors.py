# find_errors.py (Elegant version, fully compatible with main.py and loso_main.py)

import torch
import torch.nn as nn
import os
import argparse
import re  # 引入正则表达式库用于解析路径

from utils import parser as project_parser
from utils.config import get_config
from utils.logger import get_logger
from tools import builder


def find_misclassified_samples(args):
    # 智能判断任务类型
    config = get_config(args, logger=get_logger(args.log_name))
    logger = get_logger(args.log_name)

    model_name = config.model.NAME
    is_sequence_task = 'sequence_length' in config.dataset.get('train', config.dataset.get('test')).others

    if is_sequence_task:
        logger.info('Task type detected: Sequence-based')
    else:
        logger.info('Task type detected: Single-frame-based')

    # --- [核心优雅逻辑] 自动从路径中解析 subject ID ---
    is_loso_run = 'fold_test_subj_' in args.ckpts
    if is_loso_run:
        logger.info("LOSO checkpoint detected. Automatically configuring for subject-specific testing.")
        # 使用正则表达式从路径中提取数字
        match = re.search(r'fold_test_subj_(\d+)', args.ckpts)
        if match:
            test_subject_id = int(match.group(1))
            val_subject_id = (test_subject_id % 7) + 1  # 保持与 loso_main.py 一致

            # 动态修改 config
            base_dataset_name = 'SubjectSplitSequenceDataset' if is_sequence_task else 'SubjectSplitDataset'
            config.dataset.test._base_.NAME = base_dataset_name
            config.dataset.test.others['test_subject_id'] = test_subject_id
            config.dataset.test.others['val_subject_id'] = val_subject_id
            logger.info(f"Inferred Test Subject ID: {test_subject_id}")
        else:
            logger.error("Could not parse subject ID from checkpoint path. Please check the folder naming convention.")
            return
    else:
        logger.info("Standard (non-LOSO) checkpoint detected.")

    # --- 后续流程几乎不变 ---
    config.dataset.test.others.bs = 1

    original_distributed = args.distributed
    args.distributed = False

    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)

    args.distributed = original_distributed
    logger.info(f"Loaded {len(test_dataloader.dataset)} samples for the test set.")

    model = builder.model_builder(config.model)
    builder.load_model(model, args.ckpts, logger=logger)

    device = torch.device(f"cuda:{args.local_rank}" if args.use_gpu else "cpu")
    model.to(device)
    model.eval()

    misclassified_count = 0
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points, label = data
            points = points.to(device)
            label = label.to(device)
            logits = model(points)
            pred = logits.argmax(-1)

            if pred.item() != label.item():
                misclassified_count += 1

                dataset_obj = test_dataloader.dataset
                log_message = "  File Path Info: Not Available"  # 默认值

                # 智能探测路径属性
                possible_attrs = ['sequences', 'datapath']
                for attr in possible_attrs:
                    if hasattr(dataset_obj, attr):
                        path_data = getattr(dataset_obj, attr)[idx]
                        if is_sequence_task:
                            log_message = f"  Sequence starts with file: {path_data[0][0]}"
                        else:
                            log_message = f"  File Path: {path_data[0]}"
                        break

                class_names = ['Daily Activity (0)', 'Fall (1)']
                true_label_name = class_names[label.item()]
                pred_label_name = class_names[pred.item()]

                logger.warning(f"--- Misclassified Sample #{misclassified_count} ---")
                logger.warning(log_message)
                logger.warning(f"  True Label: {true_label_name}")
                logger.warning(f"  Predicted Label: {pred_label_name}")
                logger.warning(f"  Logits: {logits.cpu().numpy().flatten()}")
                logger.warning("------------------------------------")

    if misclassified_count == 0:
        logger.info("Congratulations! No misclassified samples were found.")
    else:
        logger.info(f"Diagnostic run finished. Found {misclassified_count} misclassified samples.")


if __name__ == '__main__':
    args = project_parser.get_args()
    args.use_gpu = torch.cuda.is_available()
    if not hasattr(args, 'launcher') or args.launcher == 'none':
        args.distributed = False
    else:
        args.distributed = True
    args.test = True
    find_misclassified_samples(args)