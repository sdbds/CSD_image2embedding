# CSD Image to Embedding

Generate style and content embeddings from images, project them to 2D, cluster
them, and inspect the results in a Dash application. The package supports the
original CSD-CLIP backend and a strict SigLIP2-DINOv3 backend.

## Installation

Python 3.11 is supported. From PowerShell on Windows:

```powershell
./Step1_install-uv.ps1
```

On Linux, install PowerShell once and run the same setup script:

```bash
bash Step0_for_linux_install_pwsh.bash
pwsh ./Step1_install-uv.ps1
```

Activate `.venv` before entering the direct `python` commands below. The two
PowerShell workflow scripts locate the environment automatically.

The accelerated `flash-kmeans` backend is optional. When it cannot be installed
or initialized, clustering falls back to scikit-learn KMeans.

## Image-Only Workflow

Place images under `datasets/`, including nested directories, then run:

```powershell
./Step2_embedding.ps1
```

The equivalent CSD command is:

```powershell
python -m csd_image2embedding --backend csd --text-mode image-only --train-data-dir datasets
```

Image-only is always the default. Sidecar `.txt` files do not change or
invalidate image-only embeddings. CSD supports image-only mode only.

## SigLIP2-DINOv3

The default configuration is
`configs/siglip_dinov3.yaml`. It currently points to the locally reviewed model
assets under `D:/styledecouple_dinov3`; update `model_base_path` when those
assets live elsewhere.

```powershell
python -m csd_image2embedding --backend siglip-dinov3 --text-mode image-only --style-model-config configs/siglip_dinov3.yaml
```

Caption-guided mode is experimental. For every image, create a same-name `.txt`
sidecar containing one ordinary caption or prompt. The caption is treated only
as content guidance; missing, empty, or unreadable captions reject the entire
run before model loading.

```powershell
python -m csd_image2embedding --backend siglip-dinov3 --text-mode caption-guided --style-model-config configs/siglip_dinov3.yaml
```

## Input Ownership

Without `--dataset-path`, the application discovers `--train-data-dir` and
creates a deterministic source snapshot. Passing `--dataset-path some.lance`
makes that Lance dataset authoritative and bypasses directory discovery.

To create a standalone Lance snapshot without running inference:

```powershell
python -m csd_image2embedding.data datasets --output datasets.lance
```

The command refuses to replace an existing destination. Delete or rename an old
snapshot explicitly when replacement is intentional.

## Generated Artifacts

New source snapshots, embeddings, and projections live under `.artifacts/`.
Embedding builds are immutable and published through an atomic `current.json`
pointer. Input bytes, backend, mode, model, preprocessing, schema, projection,
and clustering parameters participate in the relevant identities. Existing
legacy `datasets.lance`, `embeddings_*.lance`, and output directories are never
silently reused or deleted. Cluster exports are separated under
`output/<view>/runs/<identity>/`. Use `--rebuild` to publish a fresh compatible
build.

`python main.py` was intentionally removed. Use
`python -m csd_image2embedding`; the hidden legacy model option remains only for
one migration cycle. The old mutable `--embeddings-path` option now fails with a
migration hint.

## Development

Install the small development tool set and run verification:

```powershell
uv pip install -r requirements-dev.txt --python .venv
.venv/Scripts/python.exe -m ruff format --check .
.venv/Scripts/python.exe -m ruff check .
.venv/Scripts/python.exe -m pytest -q
```

`requirements.txt` lists direct runtime dependencies. Regenerate the resolved
lock instead of hand-editing transitive versions:

```powershell
uv pip compile requirements.txt --output-file requirements-uv.txt
```

The original CSD model is documented at
[Hugging Face](https://huggingface.co/yuxi-liu-wired/CSD).
