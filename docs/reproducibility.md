# Reproducibility notes

## Protocol

`loso_main.py` evaluates nine folds. In fold *i*, subject *i* is the test subject and the following subject (wrapping after 9) is the validation subject. The script writes a best checkpoint per fold and reports both per-fold values and mean ± standard deviation.

## Recommended reporting

For every reported result, record:

1. Git commit hash and the exact configuration file.
2. Dataset version, split identifiers, and preprocessing version.
3. Pretraining checkpoint source or a statement that training started from scratch.
4. Random seed, GPU model, CUDA version, PyTorch version, and command line.
5. Per-fold accuracy, precision, recall, F1, and the aggregate mean ± standard deviation.

## Sanity checks

Before launching LOSO, check that the dataset configuration points at the intended data root, that all nine subjects are present, and that the checkpoint architecture matches the selected temporal head. Keep logs and generated figures outside version control; the repository `.gitignore` already excludes the standard output locations.
