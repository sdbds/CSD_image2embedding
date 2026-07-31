# Package Refactor and SigLIP2-DINOv3 Backend Design

**Date:** 2026-07-31

## Context

The repository is currently a flat collection of Python modules. The command-line
workflow, image ingestion, model inference, clustering, projection, export, and
Dash UI are coupled through top-level imports. Several module names are generic
(`model.py`, `pipeline.py`, and multiple `*_utils.py` files), and `dash_page.py`
contains model-independent clustering logic alongside UI code.

The existing `sd` backend also contains two inference correctness problems that
were fixed in the upstream `D:\styledecouple_dinov3` repository:

1. A raw DINOv3 checkpoint is remapped into a Hugging Face `Dinov2Model` with
   `strict=False`. This drops DINOv3 register tokens and RoPE state and leaves an
   unrelated absolute position embedding initialized outside the checkpoint.
2. Image preprocessing uses aspect-preserving resize plus center crop, while the
   model processors use direct square resize with bilinear interpolation and
   antialiasing.

The current image-only wrapper additionally labels an orthogonal projection of
SigLIP2 image features against projected DINOv3 features as pure style. The
upstream model defines full `s_pure` using separate style and content text. This
application normally receives only images, with an optional single sidecar
caption, so it needs an explicit two-mode contract rather than pretending the
inputs are equivalent.

## Goals

1. Move every top-level Python module into a real `csd_image2embedding` package.
2. Separate data, model, clustering, projection, dashboard, and orchestration
   responsibilities behind testable interfaces.
3. Replace generic and inconsistent names with domain-specific PEP 8 names.
4. Preserve the existing one-command PowerShell workflow.
5. Keep the CSD-CLIP backend and add a corrected SigLIP2-DINOv3 backend.
6. Make SigLIP2-DINOv3 image-only inference the default text mode and use a
   single sidecar caption as content guidance only when the entire dataset has
   valid captions.
7. Fail closed when model architecture, weights, preprocessing, or cached
   embeddings are incompatible.
8. Add executable formatting, linting, and test conventions suitable for a
   single-maintainer repository.

## Non-Goals

- Modify the external `D:\styledecouple_dinov3` working tree.
- Add ConvRot8 bundle inference in this change.
- Train or fine-tune DINOv3, SigLIP2, CSD-CLIP, or the alignment projector.
- Automatically split a generic prompt into style and content descriptions.
- Mix image-only and caption-guided embeddings within one dataset run.
- Redesign the visual appearance or feature set of the Dash application.
- Change clustering or projection mathematics except where required to isolate
  their modules and validate inputs.

## Architecture Options

### Selected: Standalone Modular Package

Keep the inference runtime needed by this application inside an isolated
`models/siglip_dinov3` package. Port the corrected upstream loader,
preprocessing, projector, and model math together with focused regression tests.
The application remains runnable without importing an external source checkout.

### Rejected: Install the Upstream Repository

Making `styledecouple_dinov3` an installable dependency would establish one
source of truth, but it requires a coordinated refactor of a second repository
whose working tree currently contains substantial uncommitted work. That scope
is not necessary for this application refactor.

### Rejected: Dynamic External-Path Import

Adding an external source directory to `sys.path` is the shortest implementation
but makes the application machine-specific, weakens reproducibility, and turns
the current absolute path into an undocumented runtime dependency.

## Package Layout

```text
csd_image2embedding/
|-- __init__.py
|-- __main__.py
|-- cli.py
|-- settings.py
|-- workflow.py
|-- data/
|   |-- __init__.py
|   |-- discovery.py
|   |-- image_dataset.py
|   |-- lance_store.py
|   |-- manifests.py
|   `-- export.py
|-- models/
|   |-- __init__.py
|   |-- base.py
|   |-- registry.py
|   |-- csd_clip/
|   |   |-- __init__.py
|   |   |-- backend.py
|   |   `-- model.py
|   `-- siglip_dinov3/
|       |-- __init__.py
|       |-- backend.py
|       |-- feature_extractors.py
|       |-- projector.py
|       |-- style_decoupler.py
|       `-- transforms.py
|-- clustering/
|   |-- __init__.py
|   |-- algorithms.py
|   |-- coordinates.py
|   |-- results.py
|   `-- export.py
|-- projection/
|   |-- __init__.py
|   |-- algorithms.py
|   |-- cache.py
|   `-- manager.py
`-- dashboard/
    |-- __init__.py
    |-- app.py
    |-- callbacks.py
    |-- figures.py
    |-- layout.py
    `-- views.py
```

The package remains directly below the repository root instead of using a
`src/` directory. This keeps `python -m csd_image2embedding` runnable from a
fresh checkout without an editable install or `PYTHONPATH` mutation.

## Module Boundaries

### CLI and Workflow

`__main__.py` only calls `cli.main()`. `cli.py` owns argument parsing and maps
arguments into typed settings. `workflow.py` owns the high-level sequence:

1. discover or load the input dataset;
2. resolve one backend and one dataset-wide text mode;
3. validate or generate embeddings;
4. launch projections, clustering, export, and the dashboard.

No model, Dash, or large optional dependency is imported merely to parse
`--help`.

The public options become:

- `--backend csd|siglip-dinov3`;
- `--text-mode auto|image-only|caption-guided`;
- `--style-model-config <path>`;
- `--rebuild` to replace incompatible generated artifacts explicitly.

The existing `--model_type csd|sd` spelling remains as a temporary hidden alias
so old scripts fail gracefully during migration. `Step2_embedding.ps1` and the
README use the new names.

### Data

`discovery.py` recursively discovers supported image files in deterministic
path order. A sidecar caption is the same path with a `.txt` suffix replacing
the image suffix. Captions are stripped and considered valid only when nonempty.
Readers try UTF-8 with BOM handling first and GB18030 only after a Unicode decode
failure.

`ImageRecord` contains the canonical path, image bytes or load handle, optional
caption, content hash, and source metadata. Lance serialization is owned only by
`lance_store.py`; model and dashboard modules never call Lance directly.

The dataset fingerprint is derived from sorted relative paths, image content
hashes, and caption content hashes. Adding, deleting, replacing, or editing a
caption therefore invalidates dependent artifacts.

### Model Backends

All inference implementations satisfy one protocol:

```python
class EmbeddingBackend(Protocol):
    name: str
    fingerprint: str

    def encode(self, batch: ImageBatch) -> EmbeddingBatch:
        ...
```

`EmbeddingBatch` contains float32 NumPy arrays for style and content embeddings,
the resolved text mode, backend name, and model fingerprint. Backends return
normalized two-dimensional arrays with one row per input record. The workflow
validates row counts, dimensions, finite values, and nonzero norms before
storage.

`CSDClipBackend` owns the renamed `CSDClip` model and all CLIP preprocessing.
`SiglipDinoBackend` owns the corrected dual-encoder runtime and text-mode math.
The registry maps stable CLI names to backend factories; it does not discover
arbitrary Python plugins.

## SigLIP2-DINOv3 Semantics

Let:

- `b_bar` be the normalized SigLIP2 image embedding;
- `c_bar` be the normalized aligned DINOv3 embedding;
- `d_bar` be the normalized SigLIP2 text embedding of a generic caption.

### Image-Only Mode

```text
style_embedding   = b_bar
content_embedding = c_bar
```

This is the default when there are no captions or caption coverage is partial.
It is a visual dual-encoder representation and is not labeled `s_pure`.

### Caption-Guided Mode

The single generic caption is treated as content guidance, never as a style
description:

```text
content_reference = normalize(c_bar + d_bar)
similarity         = dot(b_bar, content_reference)
alpha              = max(0, 1 - similarity)
style_embedding    = normalize(
    b_bar - alpha * similarity * content_reference
)
content_embedding  = content_reference
```

The output is named `caption_guided_style`, not the upstream model's full
`s_pure`. A future dual-caption mode can add the upstream style-text reference
without changing the backend protocol or stored manifest schema.

### Dataset-Wide Mode Resolution

`auto` resolves exactly once before model loading:

- zero valid captions: `image-only`;
- valid caption count equals image count: `caption-guided`;
- partial coverage: `image-only`, with a summary of valid, missing, empty, and
  unreadable sidecars.

Explicit `image-only` ignores captions. Explicit `caption-guided` fails before
inference if any record lacks a valid caption. Different modes are never mixed
in one embedding table.

## Corrected Model Loading

A raw `.pth` DINOv3 checkpoint is loaded by the pinned Meta DINOv3 backbone
factory and applied with `strict=True`. The loader does not construct or import
`Dinov2Model`. It verifies:

- the expected feature dimension;
- four register tokens;
- RoPE state;
- finite rank-two CLS output;
- the configured checkpoint SHA256 when present.

Native Hugging Face DINOv3 directories or model IDs remain supported only when
`config.model_type == "dinov3_vit"` and the expected dimensions and register
tokens match.

Both DINOv3 and SigLIP2 stay frozen in evaluation mode. The SigLIP runtime loads
only the model and tokenizer; image preprocessing is defined by this project.
Transforms use direct square bilinear resize with antialiasing followed by the
configured normalization values.

The model configuration moves to `configs/siglip_dinov3.yaml`. Paths expand
environment variables and `~`, and relative paths resolve from the config file.
The default checkpoint is `checkpoint_best.safetensors`, not a numbered epoch.
Checkpoint provenance is compared by architecture, model/checkpoint hashes,
pinned Meta source revision, projector dimensions, and preprocessing settings.
Machine-specific absolute paths recorded by the training host are not compared.

## Artifact and Cache Compatibility

Each generated Lance embedding dataset has an adjacent JSON manifest containing:

- artifact schema version;
- dataset fingerprint;
- backend name and model fingerprint;
- resolved text mode;
- preprocessing fingerprint;
- embedding dimensions and row count;
- creation command and relevant settings.

An old artifact without a manifest is incompatible. A mismatch in any identity
field is never silently reused. Default generated paths are regenerated when
`--rebuild` is supplied; an explicitly named incompatible path otherwise fails
with an actionable error.

Projection cache keys use the embedding manifest digest rather than a weak
sample of vector rows. Export directories include backend and text mode so
classification results from different feature spaces cannot overwrite one
another.

## Dashboard Refactor

Clustering algorithms move out of `dash_page.py` into
`clustering/algorithms.py`. Coordinate selection and representative-image
selection live in `clustering/coordinates.py`; result adaptation lives in
`clustering/results.py`.

The dashboard package has narrow roles:

- `layout.py`: component tree and stable component IDs;
- `figures.py`: Plotly figure and tooltip construction;
- `callbacks.py`: callback registration and input validation;
- `views.py`: view configuration and cached view computation;
- `app.py`: Dash application construction and server startup.

Dashboard callbacks receive services or callables instead of closing over raw
Lance datasets and model state. Importing clustering or projection code does not
import Dash.

## Naming and Code Standards

- Modules, functions, variables, and CLI implementation fields use
  `snake_case`.
- Classes use `CapWords`; `CSD_CLIP` becomes `CSDClip` and
  `CSDCLIPPipeline` is replaced by `CSDClipBackend`.
- Constants use `UPPER_SNAKE_CASE`.
- Generic `*_utils.py` names are removed in favor of domain nouns.
- Package-internal imports are explicit relative imports.
- Public boundaries and non-obvious tensor contracts have type annotations.
- Optional dependencies catch `ImportError`, not bare `except` or arbitrary
  runtime failures.
- New filesystem code uses `pathlib.Path`.

`pyproject.toml` is the executable source of formatting and lint policy. Ruff
owns import sorting, formatting, common correctness checks, and naming checks.
`requirements-dev.txt` contains the small developer-only tool set. A short root
`AGENTS.md` records architecture boundaries and verification commands; a
single-maintainer repository does not need `CONTRIBUTING.md`.

## Error Handling

- Empty datasets fail before model loading.
- Corrupt images are reported with paths and skipped only during ingestion; a
  run fails if no valid records remain.
- Explicit caption-guided mode reports the first missing or invalid sidecars and
  the total count, then exits before inference.
- Wrong DINOv3 architecture, incomplete state dictionaries, non-finite outputs,
  and provenance mismatches are fatal.
- Missing optional reducers appear disabled in the UI with their import reason.
- Backend inference errors identify the backend, batch range, and input paths
  while preserving the original exception as the cause.
- Dashboard rendering may show a scoped unavailable view, but it must not turn a
  model or cache validation failure into an empty chart.

## Testing Strategy

### Unit Tests

- deterministic image and sidecar discovery;
- caption decoding, coverage accounting, and dataset-wide mode resolution;
- image-only and caption-guided tensor math with small fake encoders;
- strict Meta DINOv3 and native Transformers loader dispatch;
- rejection of the legacy DINOv2-container path;
- square transform parity on a non-square synthetic image;
- model and artifact fingerprint stability and mismatch rejection;
- clustering and projection behavior without importing Dash;
- dashboard layout, callback, figure, and cache behavior with fakes.

### Integration Tests

- build a tiny Lance input and embedding artifact with a fake backend;
- run the CLI workflow through embedding storage and projection generation;
- verify `python -m csd_image2embedding --help` without model loading;
- verify the PowerShell command construction independently of model weights.

### Model Smoke Test

An opt-in test loads the local corrected DINOv3 checkpoint and best projector,
runs one non-square image through both encoders, and verifies deterministic,
finite `[1, 1024]` outputs. It is not part of the fast default suite because the
model assets are large.

## Migration Sequence

1. Add quality configuration, package skeleton, and import-boundary tests.
2. Move data, CSD model, clustering, projection, and dashboard modules without
   changing behavior.
3. Split the dashboard and replace old imports and entry points.
4. Add manifests and reject legacy caches.
5. Port the corrected SigLIP2-DINOv3 runtime with upstream regression tests.
6. Add dataset-wide image-only and caption-guided modes.
7. Update scripts, configuration, README, ignored artifacts, and naming.
8. Remove compatibility modules only after all internal imports and tests use
   the new package.
9. Reformat, lint, run the full fast suite, then run the opt-in real-model smoke
   test when local assets and a suitable device are available.

At every migration step, tests must pass before the next responsibility moves.
Structure changes and model-semantic changes are not combined in one debugging
step.

## Acceptance Criteria

- No Python file remains at the repository root.
- `python -m csd_image2embedding --help` and `Step2_embedding.ps1` use the new
  package entry point.
- No production DINOv3 path imports or constructs `Dinov2Model`.
- DINOv3 raw checkpoints load strictly through the pinned official factory.
- Preprocessing matches the corrected square-resize contract.
- Image-only is used for absent or partial caption coverage; caption-guided mode
  is used only for complete coverage or an explicit valid request.
- One embedding artifact contains exactly one backend, model fingerprint, and
  text mode.
- Legacy or incompatible embeddings and projections cannot be silently reused.
- Clustering and projection tests run without importing Dash.
- Ruff formatting and checks pass.
- The complete fast test suite passes; any unavailable large-model smoke test is
  reported explicitly rather than claimed as run.

