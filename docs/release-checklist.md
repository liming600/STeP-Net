# Public release checklist

Run this checklist before creating the public repository.

- [ ] Confirm that the chosen repository name, authors, and paper citation are final; replace the generic metadata in `CITATION.cff`.
- [ ] Confirm the MIT license is compatible with every retained upstream component and preserve third-party notices.
- [ ] Confirm the dataset, participant-derived annotations, checkpoints, and figures may be redistributed; remove them if not.
- [ ] Search the staged files for personal paths, API keys, emails, institution-only URLs, and review material.
- [ ] Run `git status --ignored` and verify that only source, configuration, documentation, and intentionally distributable assets will be tracked.
- [ ] Upload `stepnet_lidarhuman26m_encoder_epoch300.pth` to the GitHub Release, verify its SHA-256 against `weights/README.md`, and replace the release URL placeholders.
- [ ] Create a clean environment, install dependencies, build extensions, and run a documented smoke test.
- [ ] Record the exact dataset version and checkpoint provenance for every result claimed in the paper.

Suggested first commit:

```bash
git init
git add README.md LICENSE CITATION.cff CONTRIBUTING.md .gitignore \
  main.py loso_main.py models datasets tools utils cfgs extensions segmentation docs requirements.txt point-mae-environment.yml
git status
git commit -m "Initial public release"
```

Review `git status` before the commit and add any intentionally public assets explicitly; do not use a blind `git add .` for the first release.
