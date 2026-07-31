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
6. Make image-only inference the default for every backend. Allow the
   SigLIP2-DINOv3 backend to use a single sidecar caption as content guidance
   only when the user explicitly selects caption-guided mode and every record
   has a valid caption.
7. Fail closed when model architecture, weights, preprocessing, or cached
   embeddings are incompatible.
8. Add executable formatting, linting, and test conventions suitable for a
   single-maintainer repository.

## Non-Goals

- Modify the external `D:\styledecouple_dinov3` working tree.
- Add ConvRot8 bundle inference in this change.
- Train or fine-tune DINOv3, SigLIP2, CSD-CLIP, or the alignment projector.
- Infer the semantic role of a generic prompt from the presence of a `.txt`
  file, or automatically split a prompt into style and content descriptions.
- Mix image-only and caption-guided embeddings within one dataset run.
- Pin or fingerprint mutable Hugging Face revisions, tokenizer assets, or the
  tokenization policy in this change. Reproducible production runs use local
  model assets; remote IDs remain a best-effort convenience.
- Redesign the visual appearance or feature set of the Dash application.
- Change clustering or projection mathematics except where required to isolate
  their modules and validate inputs.

## Architecture Options

### Selected: Standalone Modular Package

Keep the inference runtime needed by this application inside an isolated
`models/siglip_dinov3` package. Port the corrected upstream loader,
preprocessing, projector, and model math together with focused regression tests.
The application remains runnable without importing an external source checkout.
Because the upstream working tree is not clean, the port also records its source
HEAD, a digest of the source diff, per-file source and destination hashes,
license, and a short synchronization procedure. The vendored snapshot, not the
mutable external directory, is the implementation reference after the port.

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
|-- workflow.py
|-- artifacts.py
|-- data/
|   |-- __init__.py
|   |-- discovery.py
|   `-- lance.py
|-- models/
|   |-- __init__.py
|   |-- base.py
|   |-- csd.py
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
|   |-- analysis.py
|   `-- export.py
|-- projection/
|   |-- __init__.py
|   |-- algorithms.py
|   `-- manager.py
`-- dashboard/
    |-- __init__.py
    |-- app.py
    `-- figures.py
```

The package remains directly below the repository root instead of using a
`src/` directory. This keeps `python -m csd_image2embedding` runnable from a
fresh checkout without an editable install or `PYTHONPATH` mutation.

This layout is an upper bound, not a demand to create empty abstractions.
Modules are created only when the corresponding responsibility moves. Small
types and helpers stay with their owner until a second real consumer justifies
another boundary.

## Module Boundaries

### CLI and Workflow

`__main__.py` only calls `cli.main()`. `cli.py` owns argument parsing and maps
arguments into typed settings. `workflow.py` owns the high-level sequence:

1. discover or load the input dataset;
2. validate the requested text mode against the selected backend's capabilities;
3. validate or generate embeddings;
4. launch projections, clustering, export, and the dashboard.

No model, Dash, or large optional dependency is imported merely to parse
`--help`.

The public options become:

- `--backend csd|siglip-dinov3`;
- `--text-mode image-only|caption-guided`, defaulting to `image-only`;
- `--style-model-config <path>`;
- `--rebuild` to replace incompatible generated artifacts explicitly.

The existing `--model_type csd|sd` spelling remains as a temporary hidden alias
so old scripts fail gracefully during migration. `Step2_embedding.ps1` and the
README use the new names. CSD supports only `image-only`; requesting
`caption-guided` with CSD fails before importing or loading a model.

The explicit requirement that no Python file remain at the repository root
means `python main.py` is an intentional entry-point break. The supported
one-command PowerShell workflow remains compatible, and the old option spelling
continues to work through the package entry point for one migration cycle.

### Data

`discovery.py` recursively discovers supported image files in deterministic
path order. A sidecar caption is the same path with a `.txt` suffix replacing
the image suffix. Captions are stripped and considered valid only when nonempty.
Readers try UTF-8 with BOM handling first and GB18030 only after a Unicode decode
failure.

`ImageRecord` contains the canonical path, image bytes or load handle, optional
caption, content hash, and source metadata. Lance serialization is owned only by
`data/lance.py`; model and dashboard modules never call Lance directly.

Directory input and explicit Lance input have different source-of-truth rules:

- For a directory input, the directory is authoritative. Every run performs
  deterministic discovery and hashes the sorted image files before reusing a
  generated Lance snapshot. The snapshot manifest is compared with the active
  source before any embedding cache is accepted. Adding, deleting, or replacing
  an image therefore invalidates the snapshot.
- For an explicitly supplied external Lance path, the Lance rows and schema are
  authoritative. The application fingerprints that immutable input and does not
  claim to detect later changes in an unrelated source directory.

Input identity is split by dependency instead of using one coarse dataset hash:

- `image_digest` covers ordered relative paths and image content hashes;
- `caption_digest` covers sidecar presence and caption content;
- image-only artifacts depend only on `image_digest`;
- caption-guided artifacts depend on both digests.

Editing a caption cannot force an expensive CSD or SigLIP image-only embedding
rebuild, but it always invalidates a caption-guided artifact.

Discovery decodes candidate images before they enter the accepted record set.
Unreadable files are reported with their relative paths and reasons, but do not
participate in image digests, caption coverage, or row counts. Embedded Lance
rows are fingerprinted from their actual image bytes. Path-only rows are read
and checked against their declared hashes both during fingerprinting and again
when consumed.

### Model Backends

All inference implementations satisfy one protocol:

```python
class EmbeddingBackend(Protocol):
    name: str
    fingerprint: str
    supported_text_modes: frozenset[str]

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
A small explicit factory in `models/__init__.py` maps stable CLI names to those
two backends; there is no plugin discovery layer. Mode validation happens
against `supported_text_modes` before the factory loads heavy model assets.

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

This is the default regardless of sidecar coverage. It is a visual dual-encoder
representation and is not labeled `s_pure`.

### Caption-Guided Mode

Caption-guided mode is an explicit experimental request. Selecting it declares
that every generic sidecar caption should be treated as content guidance, never
as a style description:

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

### Backend and Dataset Mode Resolution

Mode resolution happens exactly once before model loading:

- CSD plus `image-only`: accepted, captions ignored;
- CSD plus `caption-guided`: rejected as an unsupported backend capability;
- SigLIP2-DINOv3 plus `image-only`: accepted, captions ignored;
- SigLIP2-DINOv3 plus `caption-guided`: accepted only when every record has a
  valid nonempty caption.

An explicit caption-guided request with partial coverage reports valid, missing,
empty, and unreadable sidecars, then fails before inference. The presence of a
`.txt` file never changes the default mode. Different modes are never mixed in
one embedding table.

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

Pinning mutable remote Hugging Face revisions and fingerprinting tokenizer files
or tokenization policy are intentionally outside this change. A remote ID is a
best-effort convenience, not a reproducible artifact source. Local model and
tokenizer directories are the documented path for reproducible runs.

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

The port includes `models/siglip_dinov3/UPSTREAM.md` with the external repository
path, source HEAD, source working-diff digest, copied file list, per-file hashes,
license reference, and synchronization steps. Tests verify the committed
vendored hashes. This records the exact corrected runtime even though the source
working tree was dirty when reviewed.

## Artifact and Cache Compatibility

Each generated artifact is a versioned container directory with its data and
manifest owned together:

```text
.artifacts/embeddings/v3/<input-digest>/<backend>/<mode>/<model-digest>/
|-- current.json
`-- builds/
    `-- <build-digest>/
        |-- data.lance/
        `-- manifest.json
```

The application builds a temporary directory under `builds/`, closes and
validates the Lance dataset, writes the manifest, and only then renames it to its
content digest. It finally replaces the small `current.json` pointer with
`os.replace`. Rebuilds never replace a nonempty directory in place, which keeps
the publish operation reliable on Windows. A process interruption can leave an
unreferenced build for later cleanup, but cannot make readers observe a
valid-looking dataset paired with a stale or missing manifest.

The embedding manifest contains:

- artifact schema version;
- input kind and the mode-specific image/caption digests;
- backend name and model fingerprint;
- resolved text mode;
- preprocessing fingerprint;
- embedding dimensions and row count;
- creation command and relevant settings.

An old artifact without a manifest is incompatible but remains untouched. The
new default path is derived from schema version and identity, so the first new
run automatically creates a separate artifact and preserves legacy
`embeddings_*.lance` data. An explicitly named path with incompatible contents
fails with an actionable error. `--rebuild` publishes a new immutable build and
atomically changes `current.json`; it never mutates a live build in place.

Projection cache identity hashes all inputs that can change its result:

- embedding manifest digest;
- reducer name and complete canonical parameter mapping;
- random seed;
- reducer package and implementation version;
- projection cache schema version.

Export paths use a run identity containing the embedding digest, projection
identity, clustering algorithm and parameters, actual implementation version
and source digest, seed, and export schema version. Files are written to a
sibling temporary directory from authoritative source-snapshot bytes; the
complete inventory manifest is written last, then the directory is atomically
renamed. A nonempty, partial, or differently identified directory is rejected
rather than skipped or mixed with new images. Symlink exports revalidate the
active file against the authoritative source hash before linking.

## Dashboard Refactor

Clustering algorithms move out of `dash_page.py` into
`clustering/algorithms.py`. Coordinate selection, result adaptation, and
representative-image selection move together to `clustering/analysis.py` because
they operate on the same labels and coordinate arrays. Plot construction moves
to `dashboard/figures.py`; layout, callbacks, view caching, and server startup
remain in `dashboard/app.py` until another independent responsibility is proven
large enough to extract.

Dashboard callbacks receive services or callables instead of raw Lance datasets
and model state. Importing clustering or projection code does not import Dash.
The refactor removes real coupling first instead of pre-creating separate layout,
callback, and view modules.

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
- A backend/text-mode combination outside `supported_text_modes` fails before
  importing the backend's heavy dependencies.
- A directory-source snapshot mismatch creates a new versioned input artifact;
  an explicit external Lance input is never reconciled against a directory.
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
- caption decoding and coverage accounting;
- the complete backend/mode compatibility matrix;
- image-only and caption-guided tensor math with small fake encoders;
- strict Meta DINOv3 and native Transformers loader dispatch;
- rejection of the legacy DINOv2-container path;
- square transform parity on a non-square synthetic image;
- separate image and caption digest dependency tests;
- source-directory change detection and explicit-Lance source behavior;
- model and artifact fingerprint stability and mismatch rejection;
- atomic artifact interruption recovery and legacy-cache preservation;
- projection cache misses for reducer parameter, seed, and version changes;
- export run-identity mismatch rejection;
- vendored upstream snapshot hash verification;
- clustering and projection behavior without importing Dash;
- dashboard layout, callback, figure, and cache behavior with fakes.

### Integration Tests

- build a tiny Lance input and embedding artifact with a fake backend;
- run the CLI workflow through embedding storage and projection generation;
- edit an image and caption independently and verify only dependent artifacts
  are invalidated;
- start from legacy `datasets.lance` and `embeddings_*.lance` paths and verify the
  default command creates versioned artifacts without overwriting them;
- verify `python -m csd_image2embedding --help` without model loading;
- verify the PowerShell command construction independently of model weights.

### Model Smoke Test

An opt-in test loads the local corrected DINOv3 checkpoint and best projector,
runs one non-square image through both encoders, and verifies deterministic,
finite `[1, 1024]` outputs. It is not part of the fast default suite because the
model assets are large. The transform output is also compared pixel-for-pixel
with the corrected reference processor contract.

### Caption-Guided Evaluation Gate

Caption-guided inference remains explicitly experimental until a labeled
retrieval or clustering comparison demonstrates that it improves the target
metric over image-only embeddings on representative data. Tensor correctness
tests establish implementation fidelity, not modeling benefit. The CLI may
expose the explicit mode before that evidence exists, but documentation and
defaults must not describe it as the preferred or automatically selected mode.

## Migration Sequence

1. Add characterization tests for current CLI, CSD embeddings, cache paths,
   clustering, projection, export, and the one-command PowerShell workflow.
2. Add quality configuration and the minimal package skeleton, then move modules
   without changing model or cache semantics.
3. Establish directory-versus-Lance source ownership, dependency-specific
   digests, atomic versioned artifacts, and complete projection/export identity.
4. Port the corrected SigLIP2-DINOv3 runtime as a recorded vendored snapshot and
   stabilize the image-only backend with upstream regression tests.
5. Move model-independent clustering out of Dash and split only figure creation
   from the remaining application code.
6. Add explicit caption-guided mode behind its backend capability and
   experimental documentation, then run the evaluation gate separately.
7. Update scripts, configuration, README, ignored artifacts, and naming; remove
   old internal modules only after all imports and tests use the package.
8. Reformat, lint, run the full fast suite, then run the opt-in real-model smoke
   test when local assets and a suitable device are available.

At every migration step, tests must pass before the next responsibility moves.
Structure changes and model-semantic changes are not combined in one debugging
step.

## Acceptance Criteria

- No Python file remains at the repository root.
- `python -m csd_image2embedding --help` and `Step2_embedding.ps1` use the new
  package entry point.
- The old option spelling remains accepted through the package entry point for
  one migration cycle; the intentional removal of `python main.py` is documented.
- No production DINOv3 path imports or constructs `Dinov2Model`.
- DINOv3 raw checkpoints load strictly through the pinned official factory.
- Preprocessing matches the corrected square-resize contract.
- CSD accepts only image-only mode. SigLIP2-DINOv3 defaults to image-only even
  when sidecars exist. Caption-guided mode requires an explicit request and
  complete valid coverage.
- One embedding artifact contains exactly one backend, model fingerprint, and
  text mode.
- Image-only artifact identity excludes caption changes; caption-guided identity
  includes them.
- Directory input changes invalidate generated snapshots, while explicit Lance
  inputs are treated as their own source of truth.
- Legacy artifacts remain untouched and the default command creates a new
  versioned artifact without requiring `--rebuild`.
- Projection and export identities cover all result-affecting parameters,
  seeds, versions, and schema numbers.
- The copied SigLIP2-DINOv3 runtime has a committed, hash-verifiable upstream
  snapshot record.
- Clustering and projection tests run without importing Dash.
- Ruff formatting and checks pass.
- The complete fast test suite passes; any unavailable large-model smoke test is
  reported explicitly rather than claimed as run.
