# Package Refactor and SigLIP2-DINOv3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the flat scripts into a tested `csd_image2embedding` package, make generated artifacts identity-safe, and provide corrected CSD and SigLIP2-DINOv3 image embedding backends with an explicit experimental caption-guided mode.

**Architecture:** Keep one directly runnable package at the repository root. Backends implement one small embedding protocol; directory discovery and Lance input have explicit ownership rules; immutable artifact builds are published through an atomic pointer; model-independent projection and clustering stay outside Dash. Port only the corrected upstream SigLIP2-DINOv3 inference runtime and record its exact dirty-tree snapshot.

**Tech Stack:** Python 3.11, PyTorch, Transformers, safetensors, PyArrow/Lance, NumPy, scikit-learn, Plotly/Dash, pytest, Ruff, PowerShell.

## Global Constraints

- Keep `Step2_embedding.ps1` as a working one-command entry point.
- Remove every Python file from the repository root by the end of the plan.
- Keep the CSD backend image-only and preserve its embedding math.
- Default every backend to `image-only`; never infer caption semantics from `.txt` presence.
- Expose `caption-guided` only for SigLIP2-DINOv3 and require complete valid sidecars.
- Do not mix embedding modes within one artifact.
- Do not construct or import `Dinov2Model` in any DINOv3 production path.
- Do not modify `D:\styledecouple_dinov3`.
- Do not add ConvRot8 support.
- Do not pin or fingerprint mutable Hugging Face revisions, tokenizer files, or tokenization policy.
- Use local model assets for reproducible runs; remote Hugging Face IDs are best-effort only.
- Use relative imports inside `csd_image2embedding`.
- Use `pathlib.Path` for new filesystem code.
- Use TDD for every behavior change and commit after every task.

## Locked File Map

```text
csd_image2embedding/
|-- __init__.py                 # package version only
|-- __main__.py                 # calls cli.main
|-- cli.py                      # parser, legacy aliases, lightweight entry
|-- workflow.py                 # top-level orchestration and runtime env
|-- artifacts.py                # immutable builds, manifests, atomic pointer
|-- data/
|   |-- __init__.py
|   |-- discovery.py            # image/sidecar scan and dependency digests
|   `-- lance.py                # source snapshots and embedding tables
|-- models/
|   |-- __init__.py             # explicit backend factory
|   |-- base.py                 # protocol, modes, validated output type
|   |-- csd.py                  # CSDClip model and backend
|   `-- siglip_dinov3/
|       |-- __init__.py
|       |-- UPSTREAM.md         # exact vendored-source record
|       |-- backend.py          # config loading and two inference modes
|       |-- feature_extractors.py
|       |-- projector.py
|       |-- style_decoupler.py
|       `-- transforms.py
|-- projection/
|   |-- __init__.py
|   |-- algorithms.py
|   `-- manager.py              # cache identity, persistence, dataframes
|-- clustering/
|   |-- __init__.py
|   |-- algorithms.py
|   |-- analysis.py
|   `-- export.py
`-- dashboard/
    |-- __init__.py
    |-- app.py                  # layout, callbacks, view cache, server
    `-- figures.py              # pure Plotly/tooltip construction

configs/siglip_dinov3.yaml
tests/data/
tests/models/
tests/projection/
tests/clustering/
tests/dashboard/
```

Existing root modules remain temporary compatibility sources until their owner task migrates all imports. Task 10 removes them together; do not create permanent wrapper modules at the root.

---

### Task 1: Quality Baseline and Backend Contract

**Files:**
- Create: `pyproject.toml`
- Create: `requirements-dev.txt`
- Create: `AGENTS.md`
- Create: `csd_image2embedding/__init__.py`
- Create: `csd_image2embedding/models/__init__.py`
- Create: `csd_image2embedding/models/base.py`
- Create: `tests/models/test_base.py`

**Interfaces:**
- Produces: `TextMode = Literal["image-only", "caption-guided"]`
- Produces: `EmbeddingBatch.validate(expected_rows: int) -> None`
- Produces: `validate_backend_mode(backend_name, supported_modes, requested_mode) -> str`
- Produces: `EmbeddingBackend` protocol with `name`, `fingerprint`, `supported_text_modes`, and `encode`

- [ ] **Step 1: Write failing backend-contract tests**

```python
import numpy as np
import pytest

from csd_image2embedding.models.base import EmbeddingBatch, validate_backend_mode


def test_validate_backend_mode_rejects_caption_mode_for_csd():
    with pytest.raises(ValueError, match="csd.*caption-guided"):
        validate_backend_mode("csd", frozenset({"image-only"}), "caption-guided")


def test_embedding_batch_rejects_non_finite_or_wrong_row_count():
    batch = EmbeddingBatch(
        style_embeddings=np.array([[np.nan, 0.0]], dtype=np.float32),
        content_embeddings=np.ones((1, 2), dtype=np.float32),
        mode="image-only",
        backend="fake",
        model_fingerprint="model-1",
    )
    with pytest.raises(ValueError, match="finite"):
        batch.validate(expected_rows=1)
```

- [ ] **Step 2: Run the focused tests and verify import failure**

Run: `python -m pytest tests/models/test_base.py -v`

Expected: FAIL because `csd_image2embedding.models.base` does not exist.

- [ ] **Step 3: Add the minimal contract implementation**

```python
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

TextMode = Literal["image-only", "caption-guided"]


@dataclass(frozen=True)
class EmbeddingBatch:
    style_embeddings: np.ndarray
    content_embeddings: np.ndarray
    mode: TextMode
    backend: str
    model_fingerprint: str

    def validate(self, expected_rows: int) -> None:
        arrays = (self.style_embeddings, self.content_embeddings)
        if any(array.ndim != 2 for array in arrays):
            raise ValueError("Embedding arrays must have rank 2")
        if any(len(array) != expected_rows for array in arrays):
            raise ValueError("Embedding row count does not match the input batch")
        if any(not np.isfinite(array).all() for array in arrays):
            raise ValueError("Embedding arrays must contain only finite values")
        if any(np.any(np.linalg.norm(array, axis=1) == 0) for array in arrays):
            raise ValueError("Embedding rows must have nonzero norms")


def validate_backend_mode(
    backend_name: str,
    supported_modes: frozenset[str],
    requested_mode: str,
) -> str:
    if requested_mode not in supported_modes:
        raise ValueError(
            f"Backend '{backend_name}' does not support text mode '{requested_mode}'"
        )
    return requested_mode


class EmbeddingBackend(Protocol):
    name: str
    fingerprint: str
    supported_text_modes: frozenset[str]

    def encode(self, images, captions=None) -> EmbeddingBatch:
        raise NotImplementedError
```

- [ ] **Step 4: Add executable quality configuration**

Configure `pyproject.toml` with Python 3.11, line length 88, pytest test paths, and Ruff rules `E`, `F`, `I`, `B`, `UP`, and `N`. Put `pytest>=8,<10` and `ruff>=0.12,<1` in `requirements-dev.txt`. Add repository-specific commands and module boundaries to `AGENTS.md` without duplicating global response-format instructions.

Run: `python -m pip install -r requirements-dev.txt`

- [ ] **Step 5: Verify and commit**

Run: `python -m pytest tests/models/test_base.py -v`

Expected: PASS.

Run: `python -m ruff check csd_image2embedding/models/base.py tests/models/test_base.py`

Expected: PASS.

Commit:

```powershell
git add pyproject.toml requirements-dev.txt AGENTS.md csd_image2embedding tests/models/test_base.py
git commit -m "chore: establish package quality and backend contracts"
```

---

### Task 2: Directory Discovery and Dependency-Specific Digests

**Files:**
- Create: `csd_image2embedding/data/__init__.py`
- Create: `csd_image2embedding/data/discovery.py`
- Create: `tests/data/test_discovery.py`

**Interfaces:**
- Produces: `SourceRecord(relative_path, image_path, image_sha256, caption, caption_sha256, caption_status)`
- Produces: `DirectorySnapshot(records, image_digest, caption_digest, caption_counts)`
- Produces: `discover_directory(root: Path) -> DirectorySnapshot`

- [ ] **Step 1: Write failing discovery and digest tests**

```python
def test_caption_change_only_changes_caption_digest(tmp_path):
    image = tmp_path / "nested" / "a.jpg"
    image.parent.mkdir()
    image.write_bytes(b"fake-image")
    sidecar = image.with_suffix(".txt")
    sidecar.write_text("a lake", encoding="utf-8")

    first = discover_directory(tmp_path)
    sidecar.write_text("a mountain", encoding="utf-8")
    second = discover_directory(tmp_path)

    assert first.image_digest == second.image_digest
    assert first.caption_digest != second.caption_digest


def test_discovery_reports_missing_empty_and_invalid_sidecars(tmp_path):
    for name in ("valid.jpg", "missing.jpg", "empty.jpg", "invalid.jpg"):
        (tmp_path / name).write_bytes(name.encode("ascii"))
    (tmp_path / "valid.txt").write_text("a lake", encoding="utf-8")
    (tmp_path / "empty.txt").write_text("  ", encoding="utf-8")
    (tmp_path / "invalid.txt").write_bytes(b"\xff\xfe\x00")

    snapshot = discover_directory(tmp_path)

    assert [record.relative_path for record in snapshot.records] == [
        "empty.jpg", "invalid.jpg", "missing.jpg", "valid.jpg"
    ]
    assert snapshot.caption_counts == {
        "valid": 1,
        "missing": 1,
        "empty": 1,
        "unreadable": 1,
    }
```

Use real one-pixel PNG bytes for tests that later open images; use arbitrary
bytes only in tests that exercise hashing without decoding.

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `python -m pytest tests/data/test_discovery.py -v`

Expected: FAIL because `discover_directory` does not exist.

- [ ] **Step 3: Implement deterministic discovery**

Implement these exact rules:

```python
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif", ".jxl"})

def read_caption(path: Path) -> tuple[str | None, str]:
    if not path.exists():
        return None, "missing"
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            text = raw.decode(encoding).strip()
            return (text, "valid") if text else (None, "empty")
        except UnicodeDecodeError:
            continue
    return None, "unreadable"
```

Sort by POSIX-form relative path. Compute SHA256 over image bytes. Compute
`image_digest` from canonical JSON rows `[relative_path, image_sha256]` and
`caption_digest` from `[relative_path, caption_status, caption_sha256]`.

- [ ] **Step 4: Verify dependency behavior and commit**

Run: `python -m pytest tests/data/test_discovery.py -v`

Expected: PASS.

Commit:

```powershell
git add csd_image2embedding/data tests/data/test_discovery.py
git commit -m "feat: add deterministic image and caption discovery"
```

---

### Task 3: Immutable Artifact Builds and Legacy Preservation

**Files:**
- Create: `csd_image2embedding/artifacts.py`
- Create: `tests/test_artifacts.py`

**Interfaces:**
- Produces: `ArtifactIdentity.digest() -> str`
- Produces: `ArtifactStore.stage(identity) -> ContextManager[Path]`
- Produces: `ArtifactStore.publish(identity, staged_path, manifest) -> Path`
- Produces: `ArtifactStore.resolve(identity) -> Path | None`
- Produces: `validate_manifest(expected, actual) -> None`

- [ ] **Step 1: Write failing publication and migration tests**

```python
def fake_identity():
    return ArtifactIdentity(
        input_digest="input-a",
        backend="fake",
        mode="image-only",
        model_digest="model-a",
        preprocessing_digest="prep-a",
        schema_version=2,
    )


def test_publish_switches_pointer_only_after_complete_build(tmp_path):
    store = ArtifactStore(tmp_path)
    identity = fake_identity()
    with store.stage(identity) as staged:
        (staged / "data.lance").mkdir()
        (staged / "data.lance" / "data.bin").write_bytes(b"complete")
        published = store.publish(identity, staged, {"data_digest": "abc"})

    assert store.resolve(identity) == published
    assert (published / "manifest.json").is_file()


def test_failed_stage_never_changes_current_pointer(tmp_path):
    store = ArtifactStore(tmp_path)
    identity = fake_identity()
    with store.stage(identity) as staged:
        (staged / "data.lance").mkdir()
        first = store.publish(identity, staged, {"data_digest": "first"})

    with pytest.raises(RuntimeError, match="stop"):
        with store.stage(identity) as staged:
            (staged / "partial").write_text("partial", encoding="utf-8")
            raise RuntimeError("stop")

    assert store.resolve(identity) == first


def test_legacy_embedding_path_is_not_removed_or_reused(tmp_path):
    legacy = tmp_path / "embeddings_csd.lance"
    legacy.mkdir()
    store = ArtifactStore(tmp_path / ".artifacts")
    assert store.resolve(fake_identity()) is None
    assert legacy.exists()
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `python -m pytest tests/test_artifacts.py -v`

Expected: FAIL because the artifact API does not exist.

- [ ] **Step 3: Implement canonical identities and Windows-safe publication**

Use dataclasses and canonical JSON (`sort_keys=True`, compact separators). The
logical path is:

```text
<root>/embeddings/v2/<input_digest>/<backend>/<mode>/<model_digest>/
```

Each logical directory contains `builds/<manifest_digest>/` and `current.json`.
Build under `builds/.tmp-<uuid>`, write and fsync `manifest.json`, rename the
temporary directory to its digest when the destination is absent, write a
temporary pointer in the logical directory, and publish it with
`os.replace(pointer_tmp, current.json)`. Clean a failed temporary directory in
the context manager without touching an existing pointer.

- [ ] **Step 4: Add mismatch validation**

`validate_manifest` must report the exact differing identity keys. It must not
compare creation timestamps or commands. `--rebuild` behavior is represented by
publishing a new immutable build and switching the pointer, not overwriting an
existing Lance directory.

- [ ] **Step 5: Verify and commit**

Run: `python -m pytest tests/test_artifacts.py -v`

Expected: PASS.

Commit:

```powershell
git add csd_image2embedding/artifacts.py tests/test_artifacts.py
git commit -m "feat: publish versioned artifacts atomically"
```

---

### Task 4: Lance Data Ownership, CSD Backend, and Package Entry Point

**Files:**
- Create: `csd_image2embedding/data/lance.py`
- Create: `csd_image2embedding/models/csd.py`
- Create: `csd_image2embedding/cli.py`
- Create: `csd_image2embedding/workflow.py`
- Create: `csd_image2embedding/__main__.py`
- Create: `tests/data/test_lance.py`
- Create: `tests/models/test_csd.py`
- Create: `tests/test_cli.py`
- Modify: `Step2_embedding.ps1`
- Modify: `tests/test_embedding_storage.py`
- Modify: `tests/test_pipeline_batching.py`
- Modify: `tests/test_precision_utils.py`
- Modify: `tests/test_runtime_env.py`

**Interfaces:**
- Produces: `LanceImageDataset`
- Produces: `write_source_snapshot(snapshot, output_path) -> lance.LanceDataset`
- Produces: `fingerprint_external_lance(dataset) -> str`
- Produces: `build_embedding_table(paths, previews, style_embeddings, content_embeddings, style_projection, content_projection) -> pyarrow.Table`
- Produces: `CSDClipBackend.encode(images, captions=None) -> EmbeddingBatch`
- Produces: `build_parser()`, `parse_args(argv)`, and `main(argv=None) -> int`

- [ ] **Step 1: Write failing CLI compatibility tests**

```python
def test_cli_defaults_to_csd_image_only():
    args = parse_args([])
    assert args.backend == "csd"
    assert args.text_mode == "image-only"


def test_legacy_sd_option_maps_to_siglip_backend():
    args = parse_args(["--model_type", "sd"])
    assert args.backend == "siglip-dinov3"


def test_module_help_does_not_import_dash_or_models():
    result = subprocess.run(
        [sys.executable, "-m", "csd_image2embedding", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--backend" in result.stdout
```

- [ ] **Step 2: Write failing Lance source tests**

Cover generated directory snapshots and explicit Lance inputs separately. A
directory snapshot manifest must carry `image_digest` and `caption_digest`; an
external Lance fingerprint must depend on canonical schema, row count, paths,
and stored content hashes but never inspect a source directory.

Run: `python -m pytest tests/test_cli.py tests/data/test_lance.py -v`

Expected: FAIL because package entry and Lance APIs do not exist.

- [ ] **Step 3: Consolidate existing data code**

Move and clean the behavior from `datasets.py`, `lancedatasets.py`, and
`embedding_storage.py` into `data/lance.py`. Fix the undefined `root` in the
separate-caption branch, replace bare `except` with `ImportError`, use lowercase
suffix matching, and rename `transform2lance` to `write_source_snapshot`.

Keep the existing embedding columns (`path`, `image`, `style_embedding`,
`content_embedding`, `x1`, `y1`, `x2`, `y2`) so the dashboard contract remains
stable during migration.

- [ ] **Step 4: Consolidate CSD model and inference behavior**

Move `model.py`, `pipeline.py`, `inference_utils.py`, and `precision_utils.py`
into `models/csd.py`. Rename `CSD_CLIP` to `CSDClip`; keep the same backbone,
projection heads, preprocessing, precision selection, and normalized outputs.
Return an `EmbeddingBatch` containing the CSD pipeline's normalized style and
content arrays, `mode="image-only"`, `backend="csd"`, and the deterministic CSD
model fingerprint. Declare:

```python
name = "csd"
supported_text_modes = frozenset({"image-only"})
```

Use a deterministic fingerprint from model name, processor name, dimensions,
precision policy, and backend schema version.

- [ ] **Step 5: Add the lightweight package CLI and workflow boundary**

`cli.py` must parse arguments without importing Torch, Dash, Lance, or model
modules. It maps hidden `--model_type csd|sd` to `--backend`, rejects conflicting
new and legacy selections, and imports `workflow.run` only after parsing.

`workflow.py` owns `configure_runtime_env`, backend factory calls, embedding
generation, and later dashboard launch. At this task, route CSD through the new
backend and call still-unmigrated projection/dashboard modules through narrow
imports so behavior remains runnable.

Change `Step2_embedding.ps1` line 75 to:

```powershell
python -m csd_image2embedding $ext_args
```

- [ ] **Step 6: Update focused tests and verify**

Run:

```powershell
python -m pytest tests/test_cli.py tests/data/test_lance.py tests/models/test_csd.py tests/test_embedding_storage.py tests/test_pipeline_batching.py tests/test_precision_utils.py tests/test_runtime_env.py -v
```

Expected: PASS.

Run: `python -m csd_image2embedding --help`

Expected: exit 0 without loading model weights.

- [ ] **Step 7: Commit**

```powershell
git add csd_image2embedding Step2_embedding.ps1 tests
git commit -m "refactor: add package entry and CSD backend"
```

---

### Task 5: Complete Projection Cache Identity

**Files:**
- Create: `csd_image2embedding/projection/__init__.py`
- Create: `csd_image2embedding/projection/algorithms.py`
- Create: `csd_image2embedding/projection/manager.py`
- Create: `tests/projection/test_identity.py`
- Move/modify: `tests/test_projection_cache.py` -> `tests/projection/test_manager.py`

**Interfaces:**
- Produces: `ProjectionSpec(name, parameters, random_state, implementation_version)`
- Produces: `ProjectionSpec.digest(embedding_digest: str) -> str`
- Produces: `ProjectionManager.get_projected_dataframe(spec) -> pandas.DataFrame`

- [ ] **Step 1: Write failing full-identity tests**

```python
def test_projection_digest_changes_for_seed_parameter_and_version():
    base = ProjectionSpec("umap", {"metric": "cosine", "n_components": 2}, 42, "umap-1")
    assert base.digest("emb") != replace(base, random_state=7).digest("emb")
    assert base.digest("emb") != replace(base, parameters={"metric": "euclidean", "n_components": 2}).digest("emb")
    assert base.digest("emb") != replace(base, implementation_version="umap-2").digest("emb")
```

Also assert dictionary key order does not change the digest and the embedding
manifest digest always participates.

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest tests/projection -v`

Expected: FAIL because the new projection package does not exist.

- [ ] **Step 3: Move algorithms and combine manager/cache ownership**

Move projection computation from `projection_algorithms.py`; merge
`projection_cache.py` persistence into `projection/manager.py`; keep reducers
lazy. Store each bundle under:

```text
.artifacts/projections/v2/<projection-digest>.npz
```

Resolve implementation versions with `importlib.metadata.version` and use
`"builtin"` for the legacy stored projection. The persisted metadata must equal
the complete canonical spec before coordinates are reused.

- [ ] **Step 4: Update existing tests and verify**

Run: `python -m pytest tests/projection -v`

Expected: PASS, including existing round-trip and optional reducer tests.

- [ ] **Step 5: Commit**

```powershell
git add csd_image2embedding/projection tests/projection
git commit -m "refactor: make projection cache identity complete"
```

---

### Task 6: Extract Clustering, Export Identity, and Plot Construction

**Files:**
- Create: `csd_image2embedding/clustering/__init__.py`
- Create: `csd_image2embedding/clustering/algorithms.py`
- Create: `csd_image2embedding/clustering/analysis.py`
- Create: `csd_image2embedding/clustering/export.py`
- Create: `csd_image2embedding/dashboard/__init__.py`
- Create: `csd_image2embedding/dashboard/app.py`
- Create: `csd_image2embedding/dashboard/figures.py`
- Create: `tests/clustering/test_export_identity.py`
- Move/modify: clustering and dashboard tests into `tests/clustering/` and `tests/dashboard/`

**Interfaces:**
- Produces: `perform_kmeans`, `perform_hdbscan`, `perform_finch`
- Produces: `GenericClusteringResult` and coordinate/representative helpers
- Produces: `ExportIdentity.digest() -> str`
- Produces: `export_clustered_images(data, clustering_result, output_root, identity, symlink=False) -> Path`
- Produces: `create_cluster_figure(data, clustering_result, representatives, centers, title, feature_set="1")`
- Produces: `create_dashboard_app(view_service) -> Dash`

- [ ] **Step 1: Write failing import-boundary and export tests**

```python
def test_importing_clustering_does_not_import_dash():
    code = "import sys; import csd_image2embedding.clustering; assert 'dash' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], check=False)
    assert result.returncode == 0


def test_export_rejects_nonempty_directory_with_different_identity(tmp_path):
    first = ExportIdentity("emb-a", "proj-a", "kmeans", {"k": 2}, 42, 1)
    second = replace(first, parameters={"k": 3})
    export_clustered_images(fake_data(), fake_result(), tmp_path, first)
    with pytest.raises(ValueError, match="identity"):
        export_clustered_images(fake_data(), fake_result(), tmp_path, second)
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest tests/clustering tests/dashboard -v`

Expected: FAIL because the new modules do not exist.

- [ ] **Step 3: Move model-independent clustering code**

Move result types and coordinate/representative logic from
`cluster_result_utils.py` and `clustering_utils.py` into `analysis.py`. Move
KMeans, HDBSCAN, FINCH, and flash-kmeans fallback code from `dash_page.py` into
`algorithms.py`. Catch only the documented optional-backend failures; preserve
algorithm labels and noise-label behavior.

- [ ] **Step 4: Implement export run manifests**

Move `process_image.py` into `clustering/export.py`. Build the export digest from
embedding digest, projection digest, clusterer name, canonical parameters, seed,
and export schema version. Publish a canonical `run-manifest.json` before image
copies. Reuse only when the identity matches; reject mismatches instead of
skipping colliding `image_N.jpg` files.

- [ ] **Step 5: Split only pure dashboard figure code**

Move Plotly figure and tooltip construction to `dashboard/figures.py`. Keep
layout, callbacks, view cache, and server startup together in `dashboard/app.py`.
Replace raw Lance closure inputs with a callable view service. Move default view
spec construction into `app.py`; do not create layout/callback/view modules.

- [ ] **Step 6: Verify and commit**

Run: `python -m pytest tests/clustering tests/dashboard -v`

Expected: PASS.

Commit:

```powershell
git add csd_image2embedding/clustering csd_image2embedding/dashboard tests/clustering tests/dashboard
git commit -m "refactor: separate clustering and dashboard concerns"
```

---

### Task 7: Port the Corrected SigLIP2-DINOv3 Runtime

**Files:**
- Create: `csd_image2embedding/models/siglip_dinov3/__init__.py`
- Create: `csd_image2embedding/models/siglip_dinov3/feature_extractors.py`
- Create: `csd_image2embedding/models/siglip_dinov3/projector.py`
- Create: `csd_image2embedding/models/siglip_dinov3/style_decoupler.py`
- Create: `csd_image2embedding/models/siglip_dinov3/transforms.py`
- Create: `csd_image2embedding/models/siglip_dinov3/UPSTREAM.md`
- Create: `tests/models/test_siglip_dinov3_loader.py`
- Create: `tests/models/test_siglip_dinov3_transforms.py`
- Modify: `requirements.txt`
- Modify: `requirements-uv.txt` only through its existing lock-generation workflow

**Interfaces:**
- Produces: `FrozenDINOv3`
- Produces: `FrozenSigLIP2`
- Produces: `AlignmentProjector`
- Produces: `StyleDecoupler`
- Produces: `build_image_transform(image_size, mean, std)`

- [ ] **Step 1: Adapt upstream failing regression tests first**

Port the behavior from:

- `D:\styledecouple_dinov3\tests\models\test_feature_extractors.py`
- the square-transform assertions in the upstream design and tests

Tests must use fake Meta and Transformers models to prove that a local `.pth`
calls `_load_meta_dinov3`, calls `load_state_dict` with `strict=True`, rejects a
non-DINOv3 Transformers config is rejected, exactly four register tokens are
required, wrong or nonfinite CLS output is rejected, a non-square image matches
the square bilinear reference transform, and SigLIP loads `AutoTokenizer`
without constructing an image processor.

- [ ] **Step 2: Run and verify failure**

Run:

```powershell
python -m pytest tests/models/test_siglip_dinov3_loader.py tests/models/test_siglip_dinov3_transforms.py -v
```

Expected: FAIL because the corrected runtime modules do not exist.

- [ ] **Step 3: Port only the inference runtime**

Use the current file contents from:

- `D:\styledecouple_dinov3\src\models\feature_extractors.py`
- `D:\styledecouple_dinov3\src\models\projector.py`
- `D:\styledecouple_dinov3\src\models\style_decoupler.py`
- `D:\styledecouple_dinov3\src\data\transforms.py`

Change imports to package-relative imports. Inline a small SHA256 helper in
`feature_extractors.py` instead of importing the upstream training package. Keep
the pinned Meta factory, `strict=True`, register-token/RoPE checks, native HF
DINOv3 guard, CLS shape/finite checks, frozen eval behavior, tokenizer-only
SigLIP load, and square bilinear antialiased transform.

- [ ] **Step 4: Record the dirty upstream snapshot exactly**

In `UPSTREAM.md`, record:

- source repository path `D:\styledecouple_dinov3`;
- `git rev-parse HEAD` result;
- SHA256 of `git diff --binary HEAD` captured as bytes;
- each source relative path and SHA256;
- each vendored relative path and SHA256 after import-only adaptations;
- license path and the command sequence for refreshing the snapshot.

Add a test that parses `UPSTREAM.md`, hashes every vendored path, and compares
the committed digest. Do not claim the files correspond to a clean commit.

- [ ] **Step 5: Update explicit runtime dependencies**

Add direct requirements for `PyYAML`, `safetensors`, and `torchvision` if absent.
Remove no dependency merely because this task no longer imports OpenCV; first
verify the rest of the repository with `rg -n "import cv2|from cv2"`. When
`requirements.txt` changes, regenerate the lock with:

```powershell
uv pip compile requirements.txt -o requirements-uv.txt --index-strategy unsafe-best-match --no-build-isolation -p 3.11
```

- [ ] **Step 6: Verify and commit**

Run:

```powershell
python -m pytest tests/models/test_siglip_dinov3_loader.py tests/models/test_siglip_dinov3_transforms.py -v
rg -n "Dinov2Model|Dinov2Config|_remap_dinov3" csd_image2embedding
```

Expected: tests PASS and `rg` returns no production matches.

Commit:

```powershell
git add csd_image2embedding/models/siglip_dinov3 tests/models requirements.txt requirements-uv.txt
git commit -m "fix: port strict SigLIP2-DINOv3 inference runtime"
```

---

### Task 8: SigLIP2-DINOv3 Image-Only and Explicit Caption-Guided Backend

**Files:**
- Create: `csd_image2embedding/models/siglip_dinov3/backend.py`
- Create: `configs/siglip_dinov3.yaml`
- Create: `tests/models/test_siglip_dinov3_backend.py`
- Modify: `csd_image2embedding/models/__init__.py`
- Modify: `csd_image2embedding/workflow.py`
- Modify: `csd_image2embedding/cli.py`

**Interfaces:**
- Produces: `SiglipDinoBackend.from_config(path, device, precision)`
- Produces: `caption_guided_embeddings(b_bar, c_bar, d_bar) -> tuple[Tensor, Tensor]`
- Produces: explicit `create_backend(name, settings) -> EmbeddingBackend`

- [ ] **Step 1: Write failing mode and math tests with fake encoders**

```python
def test_image_only_returns_siglip_style_and_projected_dino_content():
    backend = fake_backend(mode="image-only")
    output = backend.encode([fake_image()])
    np.testing.assert_allclose(output.style_embeddings, EXPECTED_SIGLIP)
    np.testing.assert_allclose(output.content_embeddings, EXPECTED_PROJECTED_DINO)


def test_caption_guided_uses_caption_as_content_only():
    style, content = caption_guided_embeddings(B_BAR, C_BAR, D_BAR)
    expected_content = torch.nn.functional.normalize(C_BAR + D_BAR, dim=-1)
    similarity = (B_BAR * expected_content).sum(dim=-1, keepdim=True)
    expected_style = torch.nn.functional.normalize(
        B_BAR - torch.clamp(1.0 - similarity, min=0.0) * similarity * expected_content,
        dim=-1,
    )
    torch.testing.assert_close(content, expected_content)
    torch.testing.assert_close(style, expected_style)


def test_caption_guided_requires_complete_valid_captions():
    backend = fake_backend(mode="caption-guided")
    with pytest.raises(ValueError, match="missing=1"):
        backend.encode([fake_image(), fake_image()], ["a lake", None])
```

Also test the complete matrix: CSD/image accepted, CSD/caption rejected,
SigLIP/image accepted, SigLIP/caption accepted only with full captions.

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest tests/models/test_siglip_dinov3_backend.py -v`

Expected: FAIL because the backend does not exist.

- [ ] **Step 3: Implement config and provenance checks**

Rename and reshape `sd_config.yaml` as `configs/siglip_dinov3.yaml`. Resolve
environment variables, `~`, and relative paths from the config location. Point
the default projector to `checkpoint_best.safetensors`. Compare checkpoint
metadata for DINO architecture/hash, pinned Meta source, projector dimensions,
and preprocessing; ignore training-host absolute paths.

Per scope, do not add resolved Hugging Face revisions, tokenizer file hashes, or
tokenization-policy fields.

- [ ] **Step 4: Implement the backend**

Load both frozen encoders and the projector once. Batch images through DINO and
SigLIP transforms. In image-only mode return normalized `b_bar` and `c_bar`. In
caption-guided mode tokenize every caption, compute `d_bar`, call the exact math
function tested above, and return `EmbeddingBatch` with mode
`"caption-guided"`. Validate outputs before returning.

Declare:

```python
name = "siglip-dinov3"
supported_text_modes = frozenset({"image-only", "caption-guided"})
```

Mark caption-guided as experimental in CLI help and README; never select it from
sidecar presence.

- [ ] **Step 5: Integrate the explicit factory and workflow**

`models.create_backend` uses one dictionary with factories for `csd` and
`siglip-dinov3`. `workflow.run` validates the mode before factory construction,
passes captions only in caption-guided mode, validates the resulting batch, and
includes the resolved mode and mode-specific input digest in artifact identity.

- [ ] **Step 6: Verify and commit**

Run:

```powershell
python -m pytest tests/models/test_base.py tests/models/test_csd.py tests/models/test_siglip_dinov3_backend.py -v
```

Expected: PASS.

Commit:

```powershell
git add csd_image2embedding/models csd_image2embedding/workflow.py csd_image2embedding/cli.py configs tests/models
git commit -m "feat: add explicit SigLIP2-DINOv3 embedding modes"
```

---

### Task 9: End-to-End Artifact and Workflow Integration

**Files:**
- Create: `tests/test_workflow.py`
- Modify: `csd_image2embedding/workflow.py`
- Modify: `csd_image2embedding/data/lance.py`
- Modify: `csd_image2embedding/artifacts.py`
- Modify: `csd_image2embedding/projection/manager.py`
- Modify: `csd_image2embedding/clustering/export.py`

**Interfaces:**
- Consumes: discovery digests, backend outputs, artifact store, projection specs, export identity
- Produces: a complete cached run from directory or explicit Lance source

- [ ] **Step 1: Write failing fake-backend workflow tests**

Create a two-image fixture and a fake normalized backend. Add a small
`run_fixture(source, mode, artifact_root, external_lance=None)` helper that calls
the real workflow and returns its source, embedding, projection, and export
identities. Tests must perform these exact state transitions:

1. Replace one image byte sequence and assert source and embedding identities
   both change.
2. Edit one caption in image-only mode and assert the embedding identity stays
   equal.
3. Edit one caption in caption-guided mode and assert the embedding identity
   changes.
4. Pass an explicit Lance path, monkeypatch `discover_directory` to raise, and
   assert the run succeeds without calling discovery.
5. Create legacy `datasets.lance` and `embeddings_csd.lance` directories, run the
   default workflow, and assert both legacy directories remain byte-for-byte
   unchanged while a versioned artifact is published.
6. Publish one build, force the next staged build to raise, and assert
   `current.json` still resolves the first build.

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest tests/test_workflow.py -v`

Expected: FAIL at the first missing integration behavior.

- [ ] **Step 3: Wire the complete workflow one boundary at a time**

For directory input, discover and validate the source snapshot before resolving
an embedding artifact. For explicit Lance input, fingerprint Lance only. Build
the mode-specific input digest, resolve or generate an immutable embedding
build, load its manifest digest into `ProjectionSpec`, pass projection identity
and cluster configuration into `ExportIdentity`, and launch the dashboard from
the projected dataframe.

Do not catch manifest, model, or backend validation errors in the dashboard.
Only a single reducer/view failure may become an unavailable view.

- [ ] **Step 4: Verify workflow and existing behavior**

Run:

```powershell
python -m pytest tests/test_workflow.py tests/projection tests/clustering tests/dashboard -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add csd_image2embedding tests/test_workflow.py
git commit -m "feat: integrate identity-safe embedding workflow"
```

---

### Task 10: Remove Root Python Modules, Update Documentation, and Verify

**Files:**
- Delete: all root `*.py` files
- Delete: `sd_src/`
- Rename: `transfromer2lance.ps1` -> `transform_to_lance.ps1`
- Modify: `README.md`
- Modify: `.gitignore`
- Modify: `Step1_install-uv.ps1`
- Modify: `Step2_embedding.ps1`
- Modify: all remaining tests importing root modules
- Modify: `docs/superpowers/specs/2026-07-31-package-refactor-and-siglip-dinov3-design.md` only if implementation revealed a necessary factual correction

**Interfaces:**
- Produces: final package-only public entry point and documented commands

- [ ] **Step 1: Update all imports before deleting compatibility sources**

Run: `rg -n "^(from|import) (main|model|pipeline|datasets|lancedatasets|dash_page|projection_|clustering_utils|cluster_result_utils|process_image|sd_)" -g "*.py"`

Replace every match with its final package import. Confirm PowerShell scripts use
`python -m csd_image2embedding` or the package Lance command, not a root Python
path.

- [ ] **Step 2: Remove and rename files**

Use `git rm` for every root Python module and `sd_src`. Use `git mv` for the
misspelled PowerShell file. Do not delete datasets, Lance artifacts, output, or
user model assets.

- [ ] **Step 3: Rewrite the README around actual workflows**

Document:

- installation and supported Python version;
- CSD and SigLIP2-DINOv3 backend commands;
- image-only default;
- explicit experimental caption-guided command and full-caption requirement;
- directory versus explicit Lance ownership;
- versioned `.artifacts` behavior and legacy preservation;
- quality commands;
- intentional removal of `python main.py`.

Fix `Useage`, `right clik`, and the incorrect Linux Step 2 command. Update
`.gitignore` for `.artifacts/`. Keep `requirements-uv.txt` generation documented
instead of hand-editing transitive versions.

- [ ] **Step 4: Run format and static checks**

Run:

```powershell
python -m ruff format .
python -m ruff check . --fix
python -m ruff format --check .
python -m ruff check .
```

Expected: all commands exit 0.

- [ ] **Step 5: Run the full fast suite and structural assertions**

Run:

```powershell
python -m pytest -v
python -m csd_image2embedding --help
Get-ChildItem -File -Filter *.py
rg -n "Dinov2Model|Dinov2Config|_remap_dinov3|except:\s*$" csd_image2embedding
git diff --check
```

Expected:

- pytest passes;
- module help exits 0;
- root Python listing is empty;
- forbidden DINOv2 and bare-except search has no production match;
- diff check is clean.

- [ ] **Step 6: Run the opt-in real-model smoke test when assets are available**

Run with the local corrected configuration:

```powershell
python -m pytest tests/models/test_siglip_dinov3_smoke.py -v --run-model-smoke
```

Expected: one non-square image produces deterministic finite `[1, 1024]` DINO
and SigLIP outputs and matches the corrected transform contract. If the current
Windows environment cannot load the assets, run the same command through the
known WSL environment and record the exact command and result; do not report it
as run otherwise.

- [ ] **Step 7: Commit the final migration**

```powershell
git add -A
git commit -m "refactor: complete package migration and documentation"
```

---

## Final Self-Review Checklist

- [ ] Every design acceptance criterion maps to a task and a verification command.
- [ ] CSD has no caption-guided branch.
- [ ] Caption-guided is explicit, complete-coverage only, and experimental.
- [ ] Image-only identity excludes caption content.
- [ ] Directory and external Lance ownership are tested separately.
- [ ] Legacy artifacts are preserved and default paths are versioned.
- [ ] Artifact publication cannot expose a partial build on Windows.
- [ ] Projection and export keys include all result-affecting parameters.
- [ ] The vendored dirty upstream runtime is hash-verifiable.
- [ ] Remote Hugging Face/tokenizer fingerprinting was not added.
- [ ] No root Python file or old import remains.
- [ ] Fast tests, Ruff, diff checks, and available model smoke checks have recorded evidence.
