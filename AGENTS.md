# Repository Guide

## Architecture

- Keep command parsing lightweight in `csd_image2embedding/cli.py`.
- Keep orchestration in `workflow.py`; model implementations must not own data or UI.
- Keep Lance access in `data/lance.py` and generated-artifact publishing in
  `artifacts.py`.
- Keep projection and clustering independent from Dash.
- Implement model backends through the contract in `models/base.py`.
- Use explicit relative imports inside the package and `pathlib.Path` for new
  filesystem code.

## Model Rules

- Default to `image-only`; do not infer text mode from sidecar files.
- CSD supports only `image-only`.
- DINOv3 production code must not import or construct `Dinov2Model`.
- Raw DINOv3 checkpoints load through the pinned official factory with strict
  state-dict validation.
- Do not modify the external `D:\styledecouple_dinov3` checkout.

## Verification

Run from the repository root:

```powershell
python -m pytest -q
python -m ruff format --check .
python -m ruff check .
```

Use the opt-in model smoke test only when local weights and a suitable device are
available.
