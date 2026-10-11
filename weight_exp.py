"""CE + class‑weight comparison: 4 models × 3 weight configs on Fold 1."""
import os, sys, copy, torch, time
sys.path.insert(0, os.path.dirname(__file__))
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import torch.nn as nn
from tools.runner_finetune import validate
from utils import parser, misc
from utils.config import get_config
from utils.logger import get_root_logger
from tools import builder

CKPT = './experiments/pretrain_lidarhuman/cfgs/pretrain_on_lidarhuman1/ckpt-epoch-300.pth'
TEST_SUBJ = 1
VAL_SUBJ = 2

CONFIGS = [
    ("1DCNN",       "cfgs/finetune_sequence_1dcnn.yaml"),
    ("GRU",         "cfgs/finetune_sequence_gru.yaml"),
    ("LSTM",        "cfgs/finetune_sequence_lstm.yaml"),
    ("Transformer", "cfgs/finetune_sequence_transformer.yaml"),
]

WEIGHTS = [
    ([1.0, 2.0],   "w2x"),
    ([1.0, 4.0],   "w4x"),
    ([0.56, 4.65], "w_bal"),
]


def run_fold1(cw, cfg_path):
    sys.argv = ['weight_exp', '--config', cfg_path, '--ckpts', CKPT]
    args = parser.get_args()
    args.use_gpu = True; args.distributed = False; args.local_rank = 0

    config = get_config(args, logger=get_root_logger())
    for subset in ['train', 'val', 'test']:
        config.dataset[subset]._base_.NAME = 'SubjectSplitSequenceDataset'
        config.dataset[subset].others['test_subject_id'] = TEST_SUBJ
        config.dataset[subset].others['val_subject_id'] = VAL_SUBJ

    misc.set_random_seed(args.seed + TEST_SUBJ)

    (_, train_dataloader), (_, val_dataloader) = \
        builder.dataset_builder(args, config.dataset.train), \
        builder.dataset_builder(args, config.dataset.val)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)

    base_model = builder.model_builder(config.model)
    base_model.load_model_from_ckpt(args.ckpts)
    base_model.to(0)
    base_model = nn.DataParallel(base_model).cuda()

    cw_tensor = torch.tensor(cw, device='cuda')
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    base_model.zero_grad()

    best_val_acc = 0
    best_state = None
    patience = 0

    for epoch in range(30):
        base_model.train()
        if config.model.get('freeze_encoder', False):
            if hasattr(base_model.module, 'point_mae_encoder'):
                base_model.module.point_mae_encoder.eval()

        for _, _, data in train_dataloader:
            points, label = data[0].cuda(), data[1].cuda()
            ret = base_model(points)
            loss = torch.nn.functional.cross_entropy(ret, label.long(), weight=cw_tensor)
            _loss = loss / config.get('step_per_update', 1)
            _loss.backward()
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
            optimizer.step()
            base_model.zero_grad()

        if isinstance(scheduler, list):
            for s in scheduler: s.step(epoch)
        else:
            scheduler.step(epoch)

        metrics = validate(base_model, val_dataloader, epoch, None, args, config)
        if metrics.acc > best_val_acc:
            best_val_acc = metrics.acc
            best_state = copy.deepcopy(base_model.module.state_dict())
            patience = 0
        else:
            patience += 1
        if patience >= 10:
            print(f'  Early stop at epoch {epoch}', flush=True)
            break

    base_model.module.load_state_dict(best_state)
    base_model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for _, _, data in test_dataloader:
            points, label = data[0].cuda(), data[1].cuda()
            logits = base_model(points)
            all_preds.append(logits.argmax(-1).detach())
            all_labels.append(label.detach())

    all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
    all_labels = torch.cat(all_labels, dim=0).cpu().numpy()

    from sklearn.metrics import classification_report
    r = classification_report(all_labels, all_preds, target_names=['Daily', 'Fall'],
                              digits=4, output_dict=True, zero_division=0)

    del base_model, optimizer, scheduler, train_dataloader, val_dataloader, test_dataloader, config
    torch.cuda.empty_cache()

    return {
        'accuracy': r['accuracy'] * 100,
        'fall_recall': r['Fall']['recall'] * 100,
        'fall_precision': r['Fall']['precision'] * 100,
        'fall_f1': r['Fall']['f1-score'] * 100,
    }


def main():
    all_results = []
    for cw, wname in WEIGHTS:
        print(f"\n{'#'*75}", flush=True)
        print(f"# CE + Class‑Weight: {cw}", flush=True)
        print(f"{'#'*75}", flush=True)
        for model_name, cfg_path in CONFIGS:
            print(f"\n{'='*50} {model_name} weight={wname} {cw} {'='*50}", flush=True)
            t0 = time.time()
            m = run_fold1(cw, cfg_path)
            elapsed = time.time() - t0
            all_results.append((wname, str(cw), model_name, m))
            print(f"  RESULT [{elapsed:.0f}s]: acc={m['accuracy']:.2f}% "
                  f"fall_recall={m['fall_recall']:.2f}% "
                  f"fall_precision={m['fall_precision']:.2f}% "
                  f"fall_f1={m['fall_f1']:.2f}%", flush=True)

    print(f"\n{'='*90}", flush=True)
    print(f"{'Weight':<8} {'cw':<16} {'Model':<14} {'Acc':>8} {'F_Rcl':>8} {'F_Pre':>8} {'F_F1':>8}", flush=True)
    print(f"{'-'*90}", flush=True)
    for wname, cw_str, name, m in all_results:
        print(f"{wname:<8} {cw_str:<16} {name:<14} {m['accuracy']:>7.2f}% "
              f"{m['fall_recall']:>7.2f}% {m['fall_precision']:>7.2f}% {m['fall_f1']:>7.2f}%", flush=True)
    print(f"{'='*90}", flush=True)


if __name__ == '__main__':
    main()
