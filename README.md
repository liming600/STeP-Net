# STeP-Net: Spatio-Temporal Point-Cloud Network for Fall Detection

STeP-Net extends Point-MAE self-supervised point-cloud learning with frozen Point-MAE encoders and temporal classifiers for fall detection. The release contains Point-MAE pre-training on LIDARHuman26M, four temporal fine-tuning variants (1D-CNN, LSTM, GRU, and Transformer), leave-one-subject-out (LOSO) evaluation, and per-activity analysis.

The code is released under the [MIT License](LICENSE). It includes adaptations of [Point-MAE](https://github.com/Pang-Yatian/Point-MAE), [Point-BERT](https://github.com/lulutang0608/Point-BERT), [Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch), and [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch); retain their notices when redistributing derived work.

## Release contents

| Commitment | Location |
| --- | --- |
| Point-MAE self-supervised pre-training on LIDARHuman26M | `main.py`, `tools/runner_pretrain.py`, `cfgs/pretrain_lidarhuman.yaml`, `datasets/LIDARHuman26MDataset.py` |
| STeP-Net temporal variants with encoder freezing | `models/PointMAE_{1DCNN,LSTM,GRU,Transformer}.py` and `cfgs/finetune_sequence_*.yaml` |
| Training, LOSO 9-fold CV, and evaluation | `main.py`, `loso_main.py`, `confusion_matrix.py`, `eval_activity_confusion.py` |
| Pre-trained LIDARHuman26M encoder | GitHub Release asset; see [weights/README.md](weights/README.md) |
| Dataset loaders and format documentation | `datasets/` and [docs/dataset.md](docs/dataset.md) |

## Setup

The project was developed with Python 3.8, PyTorch 1.x, CUDA, and Linux. Install a CUDA-enabled PyTorch build compatible with the local driver, then create the environment and install the remaining packages:

```bash
conda create -n step-net python=3.8 -y
conda activate step-net
# Install the PyTorch build matching your CUDA driver from pytorch.org.
pip install -r requirements.txt
```

Build the required CUDA extensions from the repository root:

```bash
cd extensions/chamfer_dist && python setup.py install --user
cd ../..
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```

## Data preparation

No participant-derived data is distributed with this repository. Configure the dataset root in the corresponding file under `cfgs/dataset_configs/` before running an experiment:

- `LIDARHuman26MDataset.yaml` for Point-MAE pre-training;
- `FallSequenceDataset.yaml` for fine-tuning with split files; or
- `SubjectSplitSequenceDataset.py` through `loso_main.py` for subject-wise LOSO evaluation.

Required directory layouts, labels, and PLY fields are documented in [docs/dataset.md](docs/dataset.md).

## Pre-train the Point-MAE encoder

```bash
python main.py --config cfgs/pretrain_lidarhuman.yaml --exp_name pretrain_lidarhuman
```

This command uses `LIDARHuman26MDataset` and writes checkpoints under `experiments/`. For the released encoder used to reproduce the temporal experiments, download the asset described in [weights/README.md](weights/README.md).

## Fine-tune the four STeP-Net variants

All four published temporal configurations freeze the Point-MAE encoder (`freeze_encoder: true`). Replace `/path/to/stepnet_lidarhuman26m_encoder_epoch300.pth` with the downloaded release asset.

```bash
# 1D-CNN
python main.py --config cfgs/finetune_sequence_1dcnn.yaml --finetune_model \
  --ckpts /path/to/stepnet_lidarhuman26m_encoder_epoch300.pth --exp_name stepnet_1dcnn

# LSTM
python main.py --config cfgs/finetune_sequence_lstm.yaml --finetune_model \
  --ckpts /path/to/stepnet_lidarhuman26m_encoder_epoch300.pth --exp_name stepnet_lstm

# GRU
python main.py --config cfgs/finetune_sequence_gru.yaml --finetune_model \
  --ckpts /path/to/stepnet_lidarhuman26m_encoder_epoch300.pth --exp_name stepnet_gru

# Transformer
python main.py --config cfgs/finetune_sequence_transformer.yaml --finetune_model \
  --ckpts /path/to/stepnet_lidarhuman26m_encoder_epoch300.pth --exp_name stepnet_transformer
```

## Reproduce LOSO 9-fold results

Run the command below once for each temporal configuration above. In fold *i*, subject *i* is held out for testing and the following subject is used for validation.

```bash
python loso_main.py --config cfgs/finetune_sequence_gru.yaml --finetune_model \
  --ckpts /path/to/stepnet_lidarhuman26m_encoder_epoch300.pth --exp_name stepnet_gru_loso
```

The other variants use the same command with their respective configuration files. Results, per-fold checkpoints, logs, and TensorBoard events are written under `experiments/`.

## Evaluation

Generate the aggregate per-activity LOSO confusion matrix after a LOSO run:

```bash
python confusion_matrix.py --config cfgs/finetune_sequence_gru.yaml \
  --ckpts /path/to/stepnet_lidarhuman26m_encoder_epoch300.pth --exp_name stepnet_gru_loso
```

Evaluate one saved fold with the per-activity diagnostic:

```bash
python eval_activity_confusion.py --config cfgs/finetune_sequence_gru.yaml \
  --ckpts experiments/finetune_sequence_gru/cfgs/stepnet_gru_loso/fold_test_subj_1/ckpt-best.pth \
  --test_subject_id 1 --val_subject_id 2
```

## Reproducibility and release checks

See [docs/reproducibility.md](docs/reproducibility.md) for the reporting protocol and [docs/release-checklist.md](docs/release-checklist.md) before publishing. The repository intentionally contains only the STeP-Net training and evaluation path; generated data, logs, checkpoints, figures, and local configurations are excluded from Git.
