"""Nine-fold LOSO fall-ratio ablation for the STeP-TG model.

For every held-out subject, only fall windows in the training split are
subsampled. Validation and test windows are never changed. Each ratio/fold
combination is trained independently and results are written to a CSV file.

Example:
    python -u fall_ratio_ablation.py --loss ce --output results/fall_ratio_9fold.csv
    python -u fall_ratio_ablation.py --loss ce --subjects 1 --ratios 8
"""
import argparse
import copy
import csv
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(__file__))
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from tools import builder
from tools.runner_finetune import Acc_Metric, train_transforms, validate
from utils import misc, parser
from utils.config import get_config
from utils.logger import get_root_logger


CKPT = './experiments/pretrain_lidarhuman/cfgs/pretrain_on_lidarhuman1/ckpt-epoch-300.pth'
CFG_PATH = 'cfgs/finetune_sequence_gru.yaml'
TOTAL_SUBJECTS = 9
CLASS_WEIGHTS = torch.tensor([0.56, 4.65])
FOCAL_ALPHA = 0.89
FOCAL_GAMMA = 1.5


class FocalLoss(nn.Module):
    def __init__(self, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - pt) ** self.gamma * ce_loss).mean()


def parse_csv_values(value, value_type, argument_name):
    try:
        values = [value_type(item.strip()) for item in value.split(',') if item.strip()]
    except ValueError as error:
        raise ValueError(f'Invalid {argument_name}: {value!r}') from error
    if not values:
        raise ValueError(f'{argument_name} cannot be empty.')
    return values


def build_train_val_test(args, config, test_subject_id, val_subject_id, seed):
    """Build one LOSO split using the same subject rotation as loso_main.py."""
    cfg = copy.deepcopy(config)
    for subset in ('train', 'val', 'test'):
        cfg.dataset[subset]._base_.NAME = 'SubjectSplitSequenceDataset'
        cfg.dataset[subset].others['test_subject_id'] = test_subject_id
        cfg.dataset[subset].others['val_subject_id'] = val_subject_id
    misc.set_random_seed(seed)
    train_sampler, train_dl = builder.dataset_builder(args, cfg.dataset.train)
    _, val_dl = builder.dataset_builder(args, cfg.dataset.val)
    _, test_dl = builder.dataset_builder(args, cfg.dataset.test)
    return train_sampler, train_dl, val_dl, test_dl


def collect_labels(dataset):
    """Read cached labels without loading PLY frames solely to inspect labels."""
    sequences = getattr(dataset, 'sequences', None)
    if sequences is not None:
        return [int(sequence[1]) for sequence in sequences]
    labels = []
    for index in range(len(dataset)):
        _, _, data = dataset[index]
        labels.append(int(data[1].item()))
    return labels


def subsample_train_dl(train_dl, target_fall_pct, seed):
    """Keep all daily windows and randomly retain falls at the target ratio."""
    if not 0 < target_fall_pct < 100:
        raise ValueError(f'Fall ratio must be in (0, 100), received {target_fall_pct}.')
    dataset = train_dl.dataset
    labels = collect_labels(dataset)
    fall_indices = [index for index, label in enumerate(labels) if label == 1]
    daily_indices = [index for index, label in enumerate(labels) if label == 0]
    if not fall_indices or not daily_indices:
        raise RuntimeError('Training split must contain both daily and fall windows.')

    desired_fall = round(len(daily_indices) * target_fall_pct / (100.0 - target_fall_pct))
    n_keep = min(max(desired_fall, 1), len(fall_indices))
    selected_fall = set(random.Random(seed).sample(fall_indices, n_keep))
    kept_indices = [
        index for index, label in enumerate(labels)
        if label == 0 or index in selected_fall
    ]
    actual_pct = 100.0 * n_keep / len(kept_indices)
    print(
        f'  Training windows: {len(kept_indices)} total; fall {n_keep}/{len(kept_indices)} '
        f'({actual_pct:.2f}%; target {target_fall_pct:.1f}%).',
        flush=True,
    )
    return DataLoader(
        Subset(dataset, kept_indices), batch_size=train_dl.batch_size, shuffle=True,
        num_workers=train_dl.num_workers, pin_memory=train_dl.pin_memory,
        drop_last=train_dl.drop_last, worker_init_fn=train_dl.worker_init_fn,
    ), len(kept_indices), n_keep, actual_pct


def get_loss_fn(loss_type, device):
    if loss_type == 'ce':
        return nn.CrossEntropyLoss()
    if loss_type == 'weighted':
        return nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(device))
    if loss_type == 'focal':
        return FocalLoss()
    raise ValueError(f'Unknown loss: {loss_type}')


def make_runner_args(num_workers, seed):
    """Construct the repository's normal argument object without CLI collisions."""
    saved_argv = sys.argv
    try:
        sys.argv = [
            'fall_ratio_ablation', '--config', CFG_PATH, '--ckpts', CKPT,
            '--num_workers', str(num_workers), '--seed', str(seed),
            '--exp_name', 'fall_ratio_ablation',
        ]
        args = parser.get_args()
    finally:
        sys.argv = saved_argv
    args.use_gpu = True
    args.distributed = False
    args.local_rank = 0
    args.resume = False
    args.finetune_model = True
    args.scratch_model = False
    return args


def run_trial(target_fall_pct, loss_type, test_subject_id, num_workers, seed):
    val_subject_id = (test_subject_id % TOTAL_SUBJECTS) + 1
    args = make_runner_args(num_workers, seed)
    config = get_config(args, logger=get_root_logger())
    _, train_dl, val_dl, test_dl = build_train_val_test(
        args, config, test_subject_id, val_subject_id, seed,
    )
    train_dl, n_train, n_fall, actual_pct = subsample_train_dl(train_dl, target_fall_pct, seed)

    base_model = builder.model_builder(config.model)
    base_model.load_model_from_ckpt(args.ckpts)
    base_model.to(0)
    base_model = nn.DataParallel(base_model).cuda()
    if config.model.get('freeze_encoder', False):
        for name, parameter in base_model.named_parameters():
            if 'point_mae_encoder' in name or 'MAE_encoder' in name:
                parameter.requires_grad = False

    criterion = get_loss_fn(loss_type, torch.device('cuda:0'))
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    base_model.zero_grad()
    best_val = Acc_Metric(0.0)
    best_state = None
    best_epoch = None
    patience = 0

    for epoch in range(101):
        base_model.train()
        if config.model.get('freeze_encoder', False) and hasattr(base_model.module, 'point_mae_encoder'):
            base_model.module.point_mae_encoder.eval()
        for _, _, data in train_dl:
            points, label = data[0].cuda(), data[1].cuda()
            points = train_transforms(points)
            loss = criterion(base_model(points), label.long())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
            optimizer.step()
            base_model.zero_grad()

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        # Keep validation and early stopping aligned with runner_finetune.run_net.
        if epoch == 0:
            continue
        val_metrics = validate(base_model, val_dl, epoch, None, args, config)
        if val_metrics.better_than(best_val):
            best_val = val_metrics
            best_state = copy.deepcopy(base_model.module.state_dict())
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
        if patience >= 15:
            break

    if best_state is None:
        raise RuntimeError('No validation checkpoint was selected.')
    base_model.module.load_state_dict(best_state)
    base_model.eval()
    all_predictions, all_labels = [], []
    with torch.no_grad():
        for _, _, data in test_dl:
            points, labels = data[0].cuda(), data[1].cuda()
            all_predictions.append(base_model(points).argmax(-1).cpu())
            all_labels.append(labels.cpu())
    predictions = torch.cat(all_predictions).numpy()
    labels = torch.cat(all_labels).numpy()
    from sklearn.metrics import classification_report
    report = classification_report(
        labels, predictions, target_names=['Daily', 'Fall'], digits=4,
        output_dict=True, zero_division=0,
    )
    result = {
        'loss_type': loss_type,
        'fold': test_subject_id,
        'val_subject': val_subject_id,
        'seed': seed,
        'target_fall_ratio_pct': target_fall_pct,
        'actual_fall_ratio_pct': actual_pct,
        'n_train_windows': n_train,
        'n_train_fall_windows': n_fall,
        'best_epoch': best_epoch,
        'accuracy_pct': report['accuracy'] * 100,
        'fall_precision_pct': report['Fall']['precision'] * 100,
        'fall_recall_pct': report['Fall']['recall'] * 100,
        'fall_f1_pct': report['Fall']['f1-score'] * 100,
    }
    del base_model, optimizer, scheduler, train_dl, val_dl, test_dl, config
    torch.cuda.empty_cache()
    return result


METRIC_FIELDS = ('accuracy_pct', 'fall_precision_pct', 'fall_recall_pct', 'fall_f1_pct')


def summarise_results(results):
    summaries = []
    for ratio in sorted({row['target_fall_ratio_pct'] for row in results}, reverse=True):
        rows = [row for row in results if row['target_fall_ratio_pct'] == ratio]
        summary = {'target_fall_ratio_pct': ratio, 'n_folds': len(rows)}
        for field in METRIC_FIELDS:
            values = np.asarray([row[field] for row in rows], dtype=float)
            summary[f'{field}_mean'] = float(values.mean())
            summary[f'{field}_std'] = float(values.std(ddof=1)) if len(values) > 1 else float('nan')
        summaries.append(summary)
    return summaries


def write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_existing_results(path, loss_type, subjects, ratios, base_seed):
    """Load validated completed trials for an explicit --resume invocation."""
    if not os.path.exists(path):
        raise FileNotFoundError(f'Cannot resume: per-fold result file does not exist: {path}')

    required_fields = {
        'loss_type', 'fold', 'target_fall_ratio_pct', 'actual_fall_ratio_pct',
        'val_subject', 'seed', 'n_train_windows', 'n_train_fall_windows',
        'best_epoch', *METRIC_FIELDS,
    }
    with open(path, newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required_fields.issubset(reader.fieldnames):
            raise ValueError('Cannot resume: CSV schema does not match the current script.')
        rows = list(reader)

    numeric_fields = required_fields - {'loss_type'}
    results = []
    valid_pairs = set()
    for row in rows:
        if row['loss_type'] != loss_type:
            raise ValueError(
                f"Cannot resume: CSV contains loss_type={row['loss_type']!r}, expected {loss_type!r}."
            )
        parsed = {'loss_type': row['loss_type']}
        for field in numeric_fields:
            value = row[field]
            parsed[field] = int(float(value)) if field in {
                'fold', 'val_subject', 'seed', 'n_train_windows',
                'n_train_fall_windows', 'best_epoch',
            } else float(value)

        pair = (parsed['target_fall_ratio_pct'], parsed['fold'])
        if pair in valid_pairs:
            raise ValueError(f'Cannot resume: duplicate completed trial in CSV: {pair}.')
        expected_seed = base_seed + parsed['fold'] - 1
        if parsed['seed'] != expected_seed:
            raise ValueError(
                f"Cannot resume: trial {pair} has seed={parsed['seed']}, expected {expected_seed}."
            )
        valid_pairs.add(pair)
        results.append(parsed)

    requested_pairs = {(ratio, subject) for ratio in ratios for subject in subjects}
    unexpected_pairs = valid_pairs - requested_pairs
    if unexpected_pairs:
        raise ValueError(
            'Cannot resume: CSV contains trials outside this command\'s --subjects/--ratios scope: '
            f'{sorted(unexpected_pairs)}'
        )
    return results, valid_pairs


def print_summary(summaries):
    print('\n' + '=' * 88)
    print('Nine-fold LOSO fall-ratio ablation (mean ± sample standard deviation, %)')
    print(f"{'Fall ratio':>11} {'Acc':>18} {'Precision':>18} {'Recall':>18} {'F1':>18}")
    print('-' * 88)
    for summary in summaries:
        values = [
            f"{summary[f'{field}_mean']:.2f} ± {summary[f'{field}_std']:.2f}"
            for field in METRIC_FIELDS
        ]
        print(f"{summary['target_fall_ratio_pct']:>10.1f}% " + ''.join(f'{value:>18}' for value in values))
    print('=' * 88)


def main():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument('--loss', required=True, choices=('ce', 'weighted', 'focal'))
    command.add_argument('--subjects', default='1,2,3,4,5,6,7,8,9',
                         help='Comma-separated held-out subject IDs (default: all 9).')
    command.add_argument('--ratios', default='8,5,2,1',
                         help='Comma-separated target fall ratios in percent (default: 8,5,2,1).')
    command.add_argument('--seed', type=int, default=42,
                         help='Base seed; fold i uses seed + i - 1 (default: 42).')
    command.add_argument('--num-workers', type=int, default=8)
    command.add_argument('--output', default='results/fall_ratio_9fold_per_fold.csv')
    command.add_argument('--summary-output', default='results/fall_ratio_9fold_summary.csv')
    command.add_argument('--resume', action='store_true',
                         help='Load --output and skip already completed ratio/fold trials.')
    options = command.parse_args()

    subjects = parse_csv_values(options.subjects, int, '--subjects')
    ratios = parse_csv_values(options.ratios, float, '--ratios')
    if len(set(subjects)) != len(subjects) or any(subject < 1 or subject > TOTAL_SUBJECTS for subject in subjects):
        raise ValueError('--subjects must contain unique IDs in 1..9.')
    if len(set(ratios)) != len(ratios) or any(not 0 < ratio < 100 for ratio in ratios):
        raise ValueError('--ratios must contain unique values in (0, 100).')
    if not torch.cuda.is_available():
        raise RuntimeError('This experiment requires an available CUDA GPU.')

    print('=' * 88)
    print(f'Fall-ratio ablation: loss={options.loss}, subjects={subjects}, ratios={ratios}')
    print('Protocol: only training fall windows are subsampled; validation/test splits are unchanged.')
    print('=' * 88, flush=True)
    results, completed_pairs = ([], set())
    if options.resume:
        results, completed_pairs = load_existing_results(
            options.output, options.loss, subjects, ratios, options.seed,
        )
        print(f'Resuming: found {len(completed_pairs)} completed trial(s) in {options.output}.', flush=True)
    for ratio in ratios:
        for test_subject_id in subjects:
            if (ratio, test_subject_id) in completed_pairs:
                print(f'Skipping completed trial: ratio={ratio:.1f}%, fold={test_subject_id}.', flush=True)
                continue
            seed = options.seed + test_subject_id - 1
            print(f'\n--- ratio={ratio:.1f}% | fold={test_subject_id}/{TOTAL_SUBJECTS} | '
                  f'val={(test_subject_id % TOTAL_SUBJECTS) + 1} | seed={seed} ---', flush=True)
            result = run_trial(ratio, options.loss, test_subject_id, options.num_workers, seed)
            results.append(result)
            write_csv(options.output, results)
            write_csv(options.summary_output, summarise_results(results))
            print(
                f"Completed: Acc={result['accuracy_pct']:.2f}  "
                f"Precision={result['fall_precision_pct']:.2f}  "
                f"Recall={result['fall_recall_pct']:.2f}  F1={result['fall_f1_pct']:.2f}",
                flush=True,
            )
    summaries = summarise_results(results)
    write_csv(options.summary_output, summaries)
    print_summary(summaries)
    print(f'Per-fold results: {options.output}')
    print(f'Summary results:  {options.summary_output}')


if __name__ == '__main__':
    main()
