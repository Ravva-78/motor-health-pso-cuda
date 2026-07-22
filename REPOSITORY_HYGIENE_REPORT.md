# Repository Hygiene Report
**Date:** 2026-07-23
**Version:** 1.0.0

## Verified Removals
- `__pycache__` and `.pytest_cache` directories successfully removed from all packages.
- `CACHEDIR.TAG`, `.coverage`, and `.v/` cache files removed.
- `app_backup.py` deleted as it is a redundant backup file.
- `results/app.log` and `results/training.log` temporary text logs cleared.

## Verified Intentional Inclusions
- `data/predictive_maintenance.csv`: Included intentionally as the primary dataset for simulation and model retraining.
- `backend/model.onnx`, `backend/saved_model.pkl`, `backend/scaler.pkl`: Kept explicitly as the finalized trained ML models for the `v1.0.0` release.
- `.gitignore`: Verified that large files/OS artifacts are correctly excluded.

## Secrets & Vulnerability Check
- No `.env` files committed.
- No exposed passwords or API keys found (project architecture utilizes local IPC/MQTT).
- Local absolute Windows paths removed where applicable during documentation scrub.

*Verdict: Clean and ready for public publication.*
