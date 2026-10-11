# Contributing

Please keep changes focused, reproducible, and free of participant data, checkpoints, and machine-specific paths.

1. Create a branch and describe the purpose of the change.
2. Run `python -m compileall models datasets tools utils` after Python changes.
3. For experiment changes, add or update a configuration file and document the command used.
4. Do not commit `experiments/`, `data/`, CUDA build products, logs, or generated figures.
5. Preserve attribution and licenses for upstream code and dependencies.

For bug reports, include the command, configuration, environment versions, complete traceback, and a minimal non-sensitive reproduction when possible.
