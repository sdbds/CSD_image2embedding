# Post-Refactor Integrity Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the five verified identity, source-ownership, compatibility, export-publication, and corrupt-input failures found by the independent audit.

**Architecture:** Lance fingerprints and readers will share one row-resolution contract based on the bytes inference actually consumes. Directory snapshots will contain only decoded, accepted records and use relative record IDs; export will read authoritative source bytes by row and publish a complete staged tree atomically. Alignment checkpoints will validate the local SigLIP training artifact and fixed normalization contract, while the previously excluded remote revision/tokenizer cache fingerprint work remains out of scope.

**Tech Stack:** Python 3.11+, Lance, PyArrow, Pillow, safetensors, NumPy, pytest, Ruff.

## Global Constraints

- Keep image-only as the default text mode.
- Do not add Hugging Face revision, tokenizer asset, or tokenization-policy identity to embedding cache fingerprints.
- Preserve path-only external Lance support by validating the current file bytes against the declared hash.
- Version new source/export semantics instead of reinterpreting existing artifacts in place.

---

### Task 1: Authoritative Lance Rows And Accepted Directory Records

**Files:**
- Modify: `csd_image2embedding/data/discovery.py`
- Modify: `csd_image2embedding/data/lance.py`
- Test: `tests/data/test_discovery.py`
- Test: `tests/data/test_lance.py`

**Interfaces:**
- Produces: `LanceSourceRecord(record_id, image_bytes, image_sha256, suffix, source_path, caption)`
- Produces: `LanceImageDataset.read_source(index, path_root=None) -> LanceSourceRecord`
- Consumes: `DirectorySnapshot.records`, which contains only Pillow-decodable images.

- [ ] Add tests proving embedded bytes determine external Lance identity, path-only rows reject changed files, and corrupt directory images are reported but excluded from digests and caption counts.
- [ ] Run the new tests and confirm they fail for the reviewed behaviors.
- [ ] Resolve each Lance row through one helper used by fingerprinting, inference, and export; validate path-only bytes against the declared hash.
- [ ] Validate images during discovery, retain explicit rejection details, and remove the silent `continue` from snapshot writing.
- [ ] Bump the source schema and run the focused data tests.

### Task 2: Stable Record IDs And Source-Backed Export

**Files:**
- Modify: `csd_image2embedding/workflow.py`
- Modify: `csd_image2embedding/clustering/export.py`
- Test: `tests/test_workflow.py`
- Test: `tests/clustering/test_export_identity.py`

**Interfaces:**
- Consumes: `LanceImageDataset.read_source(index, path_root=None)`.
- Produces: `export_clustered_images(..., source_reader=None, source_root=None) -> Path`.

- [ ] Add a workflow test using equal-content directories under different roots, remove the first root, and prove export still succeeds from snapshot bytes.
- [ ] Run the test and confirm the old absolute-path ownership fails.
- [ ] Return relative record IDs from generated snapshots and pass the authoritative source reader into the view service/export path.
- [ ] For copies, write source bytes; for symlinks, resolve the active root/path and revalidate its hash before linking.
- [ ] Run focused workflow/export tests.

### Task 3: Alignment Checkpoint Compatibility

**Files:**
- Modify: `csd_image2embedding/models/siglip_dinov3/backend.py`
- Test: `tests/models/test_siglip_dinov3_backend.py`

**Interfaces:**
- Produces: local SigLIP artifact descriptor compatible with upstream `fingerprint_model_reference`.
- Extends: `validate_checkpoint_provenance(..., siglip_fingerprint=...)`.

- [ ] Add tests proving a mismatched SigLIP artifact digest and non-training normalization are rejected.
- [ ] Run the tests and confirm both fail against the current validator.
- [ ] Port the upstream local artifact hashing contract for compatibility checking and compare digest, file count, feature dimension, and fixed image normalization.
- [ ] Keep remote IDs best-effort and do not change tokenizer/revision cache identity.
- [ ] Run backend unit tests and the real-model smoke test.

### Task 4: Versioned Atomic Cluster Export

**Files:**
- Modify: `csd_image2embedding/clustering/analysis.py`
- Modify: `csd_image2embedding/clustering/algorithms.py`
- Modify: `csd_image2embedding/clustering/export.py`
- Modify: `csd_image2embedding/workflow.py`
- Test: `tests/clustering/test_algorithms.py`
- Test: `tests/clustering/test_export_identity.py`

**Interfaces:**
- Produces: `GenericClusteringResult.implementation` containing distribution/version and source digest.
- Produces: export schema v2 manifest written only after every output file exists.

- [ ] Add tests proving implementation identity changes the digest and a failed export leaves no published output directory.
- [ ] Run the tests and confirm the current implementation fails.
- [ ] Capture the actual clustering implementation descriptor and include it in `ExportIdentity`.
- [ ] Build all export files and the final inventory manifest in a sibling temporary directory, then atomically rename it into place.
- [ ] Validate complete existing v2 manifests for idempotent reuse and reject partial directories.

### Task 5: Full Verification

**Files:**
- Modify: documentation only if behavior or commands changed.

- [ ] Run focused regression tests for all five findings.
- [ ] Run `python -m pytest -q`.
- [ ] Run `ruff format --check .` and `ruff check .`.
- [ ] Run package/data CLI help and PowerShell syntax checks.
- [ ] Run the opt-in real SigLIP2-DINOv3 smoke test.
- [ ] Inspect `git diff --check`, worktree status, and the final diff before reporting completion.
