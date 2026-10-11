# Dataset preparation

The fall-detection dataset is intentionally excluded from version control. It may contain participant-derived data and must only be shared under the permissions and consent conditions that apply to the collection.

## Configuration

The temporal experiments load their root path from `cfgs/dataset_configs/FallSequenceDataset.yaml`:

```yaml
DATA_PATH: /absolute/path/to/fall_detection_dataset
N_POINTS: 1024
NUM_CATEGORY: 2
```

Use an absolute path on each machine. Keep personal or server-specific settings in an ignored `*.local.yaml` file rather than committing them.

## LIDARHuman26M pre-training layout

`datasets/LIDARHuman26MDataset.py` expects the following layout for self-supervised Point-MAE pre-training:

```text
LIDARHuman26M/
├── train.txt
├── test.txt
└── segment/
    ├── <segment_id_from_train.txt>/*.ply
    └── <segment_id_from_test.txt>/*.ply
```

Each line in `train.txt` or `test.txt` contains one segment directory name. Each PLY frame must provide `vertex.x`, `vertex.y`, and `vertex.z` fields. Labels are intentionally ignored during self-supervised pre-training.

## Expected sample contract

For the fall-detection LOSO protocol, `datasets/SubjectSplitSequenceDataset.py` expects this layout:

```text
fall_detection_dataset/
├── subject_01/
│   ├── walk/<recording_id>/*.ply
│   ├── run/<recording_id>/*.ply
│   ├── squat/<recording_id>/*.ply
│   ├── sit/<recording_id>/*.ply
│   └── fall/<recording_id>/*.ply
├── subject_02/
└── ...
```

Subject directory names must be zero-padded (`subject_01` through `subject_09`). Activity names map to labels as follows: `run`, `walk`, `squat`, and `sit` are class 0; `fall` is class 1. Each `.ply` frame must provide `vertex.x`, `vertex.y`, and `vertex.z` fields. Frames within a recording are lexicographically sorted and converted to sliding windows.

The alternative `FallSequenceDataset` uses `train_split.txt`, `val_split.txt`, and `test_split.txt` at the dataset root. Each non-empty line contains a relative PLY path and an integer label, separated by whitespace.

Each point cloud is sampled or zero-padded to `N_POINTS` points. The temporal configurations use a sequence length and stride defined in the corresponding file under `cfgs/`.

## Distribution checklist

If a distributable version of the data becomes available, add a separate dataset card that states:

- collection protocol and task definition;
- participant consent, privacy handling, and redistribution terms;
- train/validation/test or subject split identifiers;
- file naming convention, point representation, coordinate system, and labels;
- preprocessing, sampling, and any excluded samples; and
- a checksum and immutable version identifier.

Do not publish raw participant data, identifiers, or derived annotations until the relevant approval and data-use terms have been confirmed.
