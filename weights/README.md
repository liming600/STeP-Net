# Released pre-trained encoder

The reproducibility checkpoint is distributed as a GitHub Release asset rather than a Git-tracked file. This keeps the source repository lightweight and avoids GitHub's 100 MiB regular-file limit.

## Official asset

| Field | Value |
| --- | --- |
| Asset name | `stepnet_lidarhuman26m_encoder_epoch300.pth` |
| Source training configuration | `cfgs/pretrain_lidarhuman.yaml` |
| Training epoch | 300 |
| Size | 348,288,755 bytes (332.15 MiB) |
| SHA-256 | `DD52E04C5ECBA347F3B4EBE080C1FCBEF89C095E55B9EA1AF9A0B58F15FB245F` |

Before the public release, upload this exact file to the first GitHub Release (for example, tag `v1.0.0`) and replace `<OWNER>` and `<REPOSITORY>` below with the public repository coordinates:

```bash
mkdir -p checkpoints
curl -L -o checkpoints/stepnet_lidarhuman26m_encoder_epoch300.pth \
  https://github.com/<OWNER>/<REPOSITORY>/releases/download/v1.0.0/stepnet_lidarhuman26m_encoder_epoch300.pth
sha256sum checkpoints/stepnet_lidarhuman26m_encoder_epoch300.pth
```

The checksum must match the value in the table. Pass the downloaded path to `--ckpts` for every temporal fine-tuning or LOSO command.
