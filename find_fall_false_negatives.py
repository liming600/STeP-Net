"""Map LOSO false-negative fall windows back to their original PLY sequences.

This is an inference-only audit for the already trained main STeP-TG LOSO models.
It writes all tested fall windows and the false-negative subset to CSV, preserving the
subject, original fall-event directory, and first/last PLY file of every window.

Run after the GPU is free from training:
    python -u find_fall_false_negatives.py \
      --checkpoint-root ./experiments/finetune_sequence_gru/cfgs/default
"""
import argparse
import copy
import csv
import os
import sys
from functools import lru_cache

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
os.environ['PYTHONUNBUFFERED'] = '1'

from tools import builder
from utils import misc, parser as project_parser
from utils.config import get_config
from utils.logger import get_root_logger


TOTAL_SUBJECTS = 9


def parse_options():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument('--config', default='cfgs/finetune_sequence_gru.yaml')
    command.add_argument(
        '--checkpoint-root',
        default='./experiments/finetune_sequence_gru/cfgs/default',
        help='Directory containing fold_test_subj_<id>/ckpt-best.pth.',
    )
    command.add_argument('--output-dir', default='results/fall_false_negative_audit')
    command.add_argument('--num-workers', type=int, default=8)
    command.add_argument('--seed', type=int, default=42)
    return command.parse_args()


def make_project_args(config_path, num_workers):
    saved_argv = sys.argv
    try:
        sys.argv = [
            'find_fall_false_negatives', '--config', config_path,
            '--num_workers', str(num_workers), '--exp_name', 'fall_error_audit',
        ]
        args = project_parser.get_args()
    finally:
        sys.argv = saved_argv
    args.use_gpu = torch.cuda.is_available()
    args.distributed = False
    args.local_rank = 0
    return args


def build_test_loader(args, config, test_subject_id, val_subject_id):
    cfg = copy.deepcopy(config)
    cfg.dataset.test._base_.NAME = 'SubjectSplitSequenceDataset'
    cfg.dataset.test.others['test_subject_id'] = test_subject_id
    cfg.dataset.test.others['val_subject_id'] = val_subject_id
    # Batch size 1 keeps output rows in one-to-one order with dataset.sequences.
    cfg.dataset.test.others['bs'] = 1
    _, test_loader = builder.dataset_builder(args, cfg.dataset.test)
    return test_loader


def checkpoint_path(root, subject_id):
    return os.path.join(root, f'fold_test_subj_{subject_id}', 'ckpt-best.pth')


@lru_cache(maxsize=None)
def frame_positions(event_dir):
    """Return zero-based PLY positions using the same lexical order as the dataset."""
    files = sorted(
        os.path.join(event_dir, name)
        for name in os.listdir(event_dir)
        if name.endswith('.ply')
    )
    return {path: index for index, path in enumerate(files)}


def row_for_window(subject_id, val_subject_id, index, sequence, prediction, logits):
    paths, label, activity = sequence
    event_dir = os.path.dirname(paths[0])
    positions = frame_positions(event_dir)
    start_index = positions[paths[0]]
    end_index = positions[paths[-1]]
    return {
        'test_subject': subject_id,
        'val_subject': val_subject_id,
        'dataset_index': index,
        'activity': activity,
        'true_label': int(label),
        'prediction': int(prediction),
        'is_false_negative': int(label == 1 and prediction == 0),
        'event_directory': event_dir,
        'event_id': os.path.basename(event_dir),
        'first_ply': paths[0],
        'last_ply': paths[-1],
        'window_start_frame_index_0based': start_index,
        'window_end_frame_index_0based': end_index,
        'window_start_frame_number_1based': start_index + 1,
        'window_end_frame_number_1based': end_index + 1,
        'sequence_length': len(paths),
        'logit_daily': float(logits[0]),
        'logit_fall': float(logits[1]),
    }


@torch.no_grad()
def evaluate_fold(args, config, options, test_subject_id):
    val_subject_id = (test_subject_id % TOTAL_SUBJECTS) + 1
    ckpt_path = checkpoint_path(options.checkpoint_root, test_subject_id)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Missing checkpoint for fold {test_subject_id}: {ckpt_path}')

    misc.set_random_seed(options.seed + test_subject_id - 1)
    test_loader = build_test_loader(args, config, test_subject_id, val_subject_id)
    dataset = test_loader.dataset
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, ckpt_path, logger=get_root_logger())
    base_model.to(args.local_rank)
    base_model = nn.DataParallel(base_model).cuda()
    base_model.eval()

    rows = []
    for index, (_, _, data) in enumerate(test_loader):
        points = data[0].cuda(non_blocking=True)
        logits = base_model(points)
        prediction = int(logits.argmax(-1).item())
        sequence = dataset.sequences[index]
        _, label, activity = sequence
        if activity == 'fall' or label == 1:
            rows.append(
                row_for_window(
                    test_subject_id, val_subject_id, index, sequence, prediction,
                    logits.detach().cpu().view(-1).tolist(),
                )
            )

    del base_model, test_loader
    torch.cuda.empty_cache()
    return rows


def write_csv(path, rows):
    fields = [
        'test_subject', 'val_subject', 'dataset_index', 'activity', 'true_label',
        'prediction', 'is_false_negative', 'event_directory', 'event_id',
        'first_ply', 'last_ply', 'window_start_frame_index_0based',
        'window_end_frame_index_0based', 'window_start_frame_number_1based',
        'window_end_frame_number_1based', 'sequence_length', 'logit_daily', 'logit_fall',
    ]
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_events(rows):
    """Collapse overlapping test windows into their original fall-event directory."""
    events = {}
    for row in rows:
        key = row['event_directory']
        if key not in events:
            events[key] = {
                'test_subject': row['test_subject'],
                'val_subject': row['val_subject'],
                'event_id': row['event_id'],
                'event_directory': key,
                'first_ply': row['first_ply'],
                'last_ply': row['last_ply'],
                'first_window_start_frame_number_1based': row['window_start_frame_number_1based'],
                'last_window_end_frame_number_1based': row['window_end_frame_number_1based'],
                'n_test_windows': 0,
                'n_detected_windows': 0,
                'n_false_negative_windows': 0,
                'has_false_negative_window': 0,
            }
        event = events[key]
        if row['window_start_frame_number_1based'] < event['first_window_start_frame_number_1based']:
            event['first_window_start_frame_number_1based'] = row['window_start_frame_number_1based']
            event['first_ply'] = row['first_ply']
        if row['window_end_frame_number_1based'] > event['last_window_end_frame_number_1based']:
            event['last_window_end_frame_number_1based'] = row['window_end_frame_number_1based']
            event['last_ply'] = row['last_ply']
        event['n_test_windows'] += 1
        event['n_detected_windows'] += int(row['prediction'] == 1)
        event['n_false_negative_windows'] += row['is_false_negative']
        event['has_false_negative_window'] = int(event['n_false_negative_windows'] > 0)
    return list(events.values())


def write_event_csv(path, rows):
    fields = [
        'test_subject', 'val_subject', 'event_id', 'event_directory', 'first_ply',
        'last_ply', 'first_window_start_frame_number_1based',
        'last_window_end_frame_number_1based', 'n_test_windows', 'n_detected_windows', 'n_false_negative_windows',
        'has_false_negative_window',
    ]
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    options = parse_options()
    if not torch.cuda.is_available():
        raise RuntimeError('This audit requires a CUDA GPU.')
    os.makedirs(options.output_dir, exist_ok=True)
    args = make_project_args(options.config, options.num_workers)
    config = get_config(args, logger=get_root_logger())

    all_fall_windows = []
    for test_subject_id in range(1, TOTAL_SUBJECTS + 1):
        print(f'\n===== Auditing LOSO fold {test_subject_id}/{TOTAL_SUBJECTS} =====', flush=True)
        rows = evaluate_fold(args, config, options, test_subject_id)
        false_negatives = sum(row['is_false_negative'] for row in rows)
        print(f'Fall windows: {len(rows)} | false negatives: {false_negatives}', flush=True)
        all_fall_windows.extend(rows)

    false_negative_windows = [row for row in all_fall_windows if row['is_false_negative']]
    all_events = aggregate_events(all_fall_windows)
    events_with_false_negatives = [
        event for event in all_events if event['has_false_negative_window']
    ]
    all_path = os.path.join(options.output_dir, 'all_fall_test_windows.csv')
    fn_path = os.path.join(options.output_dir, 'false_negative_fall_windows.csv')
    all_events_path = os.path.join(options.output_dir, 'all_fall_test_events.csv')
    fn_events_path = os.path.join(options.output_dir, 'fall_events_with_false_negative_windows.csv')
    write_csv(all_path, all_fall_windows)
    write_csv(fn_path, false_negative_windows)
    write_event_csv(all_events_path, all_events)
    write_event_csv(fn_events_path, events_with_false_negatives)
    print('\n===== Audit complete =====')
    print(f'All fall test windows: {len(all_fall_windows)} -> {all_path}')
    print(f'False-negative windows: {len(false_negative_windows)} -> {fn_path}')
    print(f'All fall test events: {len(all_events)} -> {all_events_path}')
    print(f'Events with at least one false-negative window: {len(events_with_false_negatives)} -> {fn_events_path}')


if __name__ == '__main__':
    main()
