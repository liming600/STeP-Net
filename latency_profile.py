"""Latency breakdown: spatial encoder vs temporal module (direct timing).
Measures per-sequence inference time for all 4 temporal variants.
Temporal module is timed DIRECTLY (not by subtraction) to avoid noise.
"""
import sys, os, torch, time
sys.path.insert(0, os.path.dirname(__file__))
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from utils import parser
from utils.config import get_config
from utils.logger import get_root_logger
from tools import builder

BATCH_SIZE = 1
SEQ_LEN = 16
N_POINTS = 1024
TRANS_DIM = 384
WARMUP = 100
ITERATIONS = 1000

CONFIGS = [
    ("1DCNN",       "cfgs/finetune_sequence_1dcnn.yaml"),
    ("GRU",         "cfgs/finetune_sequence_gru.yaml"),
    ("LSTM",        "cfgs/finetune_sequence_lstm.yaml"),
    ("Transformer", "cfgs/finetune_sequence_transformer.yaml"),
]

CKPT = './experiments/pretrain_lidarhuman/cfgs/pretrain_on_lidarhuman1/ckpt-epoch-300.pth'


def build_temporal_forward(model):
    """Return a callable that runs only the temporal part given encoder features."""
    m = model.module if hasattr(model, 'module') else model

    if hasattr(m, 'gru'):
        def temporal(features):
            out, _ = m.gru(features)
            logits = m.classifier(out[:, -1, :])
            return logits
    elif hasattr(m, 'lstm'):
        def temporal(features):
            out, _ = m.lstm(features)
            logits = m.classifier(out[:, -1, :])
            return logits
    elif hasattr(m, 'cnn_layers'):
        def temporal(features):
            cnn_input = features.permute(0, 2, 1)
            cnn_out = cnn_input
            for layer in m.cnn_layers:
                cnn_out = layer(cnn_out)
            pooled = m.global_avg_pooling(cnn_out)
            aggregated = pooled.squeeze(-1)
            logits = m.classifier(aggregated)
            return logits
    elif hasattr(m, 'temporal_transformer'):
        def temporal(features):
            seq_len = features.size(1)
            if seq_len <= m.positional_encoding.size(1):
                pos = m.positional_encoding[:, :seq_len, :]
            else:
                pos = m.positional_encoding[:, :m.positional_encoding.size(1), :]
            out = m.temporal_transformer(features + pos)
            aggregated = out.mean(dim=1)
            logits = m.classifier(aggregated)
            return logits
    else:
        raise RuntimeError(f'Unknown temporal structure for {type(m).__name__}')
    return temporal


def measure_time(fn, *args):
    for _ in range(WARMUP):
        _ = fn(*args)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(ITERATIONS):
        _ = fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / ITERATIONS * 1000  # ms


def main():
    import sys as _sys
    if '--config' not in _sys.argv:
        _sys.argv.extend(['--config', 'cfgs/finetune_sequence_1dcnn.yaml'])
    args = parser.get_args()
    args.use_gpu = torch.cuda.is_available()
    args.distributed = False
    args.local_rank = 0

    dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, N_POINTS, 3).cuda()
    dummy_flat = dummy_input.view(-1, N_POINTS, 3)
    dummy_feat = torch.randn(BATCH_SIZE, SEQ_LEN, TRANS_DIM).cuda()

    print(f"{'Model':<14} {'Total(ms)':>10} {'Encoder(ms)':>12} {'Temporal(ms)':>13} {'Encoder%':>9} {'Temporal%':>10}")
    print('-' * 70)

    for name, cfg_path in CONFIGS:
        args.config = cfg_path
        config = get_config(args, logger=get_root_logger())
        base_model = builder.model_builder(config.model)
        base_model.load_model_from_ckpt(CKPT)
        base_model.to(0)
        import torch.nn as nn
        model = nn.DataParallel(base_model).cuda()
        temporal_fn = build_temporal_forward(model)

        with torch.no_grad():
            total_ms = measure_time(model, dummy_input)
            encoder = model.module.point_mae_encoder
            encoder_ms = measure_time(lambda x: encoder(x, feature_only=True), dummy_flat)
            temporal_ms = measure_time(temporal_fn, dummy_feat)

        print(f"{name:<14} {total_ms:>9.2f} {encoder_ms:>11.2f} {temporal_ms:>12.2f} "
              f"{encoder_ms/total_ms*100:>8.1f}% {temporal_ms/total_ms*100:>9.1f}%", flush=True)

        del model, base_model, config
        torch.cuda.empty_cache()

    print('-' * 70)
    print('Latency breakdown done.')


if __name__ == '__main__':
    main()
