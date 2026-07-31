"""Identity-safe orchestration for embedding, projection, export, and Dash."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .data.discovery import DirectorySnapshot, discover_directory


def configure_runtime_env() -> None:
    """Disable unused Transformers TensorFlow imports before model loading."""

    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")


@dataclass(frozen=True)
class WorkflowSettings:
    train_data_dir: Path = Path("datasets")
    dataset_path: Path | None = None
    artifact_root: Path = Path(".artifacts")
    backend: str = "csd"
    text_mode: str = "image-only"
    batch_size: int = 12
    rebuild: bool = False
    reducer: str | None = None
    random_state: int = 42
    clusterer: str = "kmeans"
    k_clusters: int = 40
    min_cluster_size: int = 10
    finch_partition_index: int = 1
    output_dir: Path | None = Path("output")
    symlink: bool = False
    model_name: str = "yuxi-liu-wired/CSD"
    processor_name: str = "openai/clip-vit-large-patch14"
    style_model_config: Path = Path("configs/siglip_dinov3.yaml")
    style_model_checkpoint: Path | None = None
    precision: str = "auto"
    device: str | None = None
    creation_command: str = "python -m csd_image2embedding"

    @classmethod
    def from_namespace(cls, args) -> WorkflowSettings:
        values = {
            field: getattr(args, field, getattr(cls, field, None))
            for field in cls.__dataclass_fields__
            if field != "creation_command"
        }
        return cls(**values)


@dataclass(frozen=True)
class WorkflowResult:
    input_kind: str
    source_identity: str
    source_path: Path
    image_digest: str
    caption_digest: str
    embedding_identity: object
    embedding_path: Path
    embedding_manifest_digest: str
    projection_spec: object
    projection_identity: str
    export_identity: object
    view_service: object


@dataclass(frozen=True)
class _ResolvedSource:
    input_kind: str
    source_identity: str
    dataset_path: Path
    image_digest: str
    caption_digest: str
    caption_counts: dict[str, int]
    source_root: Path | None = None


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _write_synced_json(path: Path, value: object) -> None:
    with path.open("wb") as stream:
        stream.write(_canonical_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def _source_identity(kind: str, image_digest: str, caption_digest: str) -> str:
    return _digest(
        {
            "schema_version": 2,
            "input_kind": kind,
            "image_digest": image_digest,
            "caption_digest": caption_digest,
        }
    )


def _ensure_directory_snapshot(
    snapshot: DirectorySnapshot,
    artifact_root: Path,
) -> Path:
    from .data.lance import write_source_snapshot

    source_root = (
        artifact_root
        / "sources"
        / "v2"
        / snapshot.image_digest
        / snapshot.caption_digest
    )
    data_path = source_root / "data.lance"
    manifest_path = source_root / "manifest.json"
    expected_manifest = {
        "schema_version": 2,
        "input_kind": "directory",
        "image_digest": snapshot.image_digest,
        "caption_digest": snapshot.caption_digest,
    }
    if source_root.exists():
        if not data_path.is_dir() or not manifest_path.is_file():
            raise ValueError(f"Directory source artifact is incomplete: {source_root}")
        actual = json.loads(manifest_path.read_text(encoding="utf-8"))
        if actual != expected_manifest:
            raise ValueError(
                f"Directory source artifact identity mismatch: {source_root}"
            )
        return data_path

    source_root.parent.mkdir(parents=True, exist_ok=True)
    staged = source_root.parent / f".tmp-{uuid.uuid4().hex}"
    staged.mkdir()
    try:
        write_source_snapshot(snapshot, staged / "data.lance")
        _write_synced_json(staged / "manifest.json", expected_manifest)
        staged.rename(source_root)
    finally:
        if staged.exists():
            shutil.rmtree(staged)
    return data_path


def _resolve_source(settings: WorkflowSettings) -> _ResolvedSource:
    import lance

    from .data.lance import fingerprint_external_lance_inputs

    if settings.dataset_path is not None:
        dataset_path = Path(settings.dataset_path).expanduser().resolve()
        if not dataset_path.is_dir():
            raise ValueError(f"Explicit Lance input does not exist: {dataset_path}")
        identity = fingerprint_external_lance_inputs(lance.dataset(dataset_path))
        return _ResolvedSource(
            input_kind="lance",
            source_identity=_source_identity(
                "lance", identity.image_digest, identity.caption_digest
            ),
            dataset_path=dataset_path,
            image_digest=identity.image_digest,
            caption_digest=identity.caption_digest,
            caption_counts=identity.caption_counts,
            source_root=None,
        )

    snapshot = discover_directory(Path(settings.train_data_dir))
    if not snapshot.records:
        raise ValueError(f"No supported images found in {settings.train_data_dir}")
    dataset_path = _ensure_directory_snapshot(snapshot, Path(settings.artifact_root))
    return _ResolvedSource(
        input_kind="directory",
        source_identity=_source_identity(
            "directory", snapshot.image_digest, snapshot.caption_digest
        ),
        dataset_path=dataset_path,
        image_digest=snapshot.image_digest,
        caption_digest=snapshot.caption_digest,
        caption_counts=snapshot.caption_counts,
        source_root=Path(settings.train_data_dir).expanduser().resolve(),
    )


def _validate_caption_coverage(source: _ResolvedSource, mode: str) -> None:
    if mode != "caption-guided":
        return
    total = sum(source.caption_counts.values())
    if source.caption_counts.get("valid", 0) == total:
        return
    details = ", ".join(
        f"{key}={source.caption_counts.get(key, 0)}"
        for key in ("valid", "missing", "empty", "unreadable")
    )
    raise ValueError(
        "Caption-guided mode requires one valid caption per image: " + details
    )


def _mode_input_digest(source: _ResolvedSource, mode: str) -> str:
    payload = {
        "schema_version": 2,
        "input_kind": source.input_kind,
        "image_digest": source.image_digest,
    }
    if mode == "caption-guided":
        payload["caption_digest"] = source.caption_digest
    return _digest(payload)


def _preview_base64(image) -> str:
    preview = image.convert("RGB").copy()
    preview.thumbnail((192, 192))
    buffer = io.BytesIO()
    preview.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _initial_coordinates(embeddings):
    import numpy as np

    values = np.asarray(embeddings, dtype=np.float32)
    if values.shape[1] >= 2:
        return values[:, :2]
    return np.pad(values, ((0, 0), (0, 2 - values.shape[1])))


def _embedding_data_digest(paths, styles, contents) -> str:
    import numpy as np

    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
    for values in (styles, contents):
        array = np.ascontiguousarray(values, dtype=np.float32)
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _generate_embedding_build(
    settings: WorkflowSettings,
    source: _ResolvedSource,
    backend,
    staged: Path,
) -> dict[str, object]:
    import lance
    import numpy as np

    from .data.lance import LanceImageDataset, build_embedding_table

    dataset = LanceImageDataset(source.dataset_path)
    styles = []
    contents = []
    paths = []
    previews = []
    for start in range(0, len(dataset), settings.batch_size):
        batch = [
            dataset[index]
            for index in range(start, min(len(dataset), start + settings.batch_size))
        ]
        batch_paths, images, captions = zip(*batch, strict=True)
        try:
            encoded = backend.encode(
                images,
                captions if settings.text_mode == "caption-guided" else None,
            )
            encoded.validate(expected_rows=len(images))
        except Exception as error:
            end = start + len(batch) - 1
            raise RuntimeError(
                f"Backend '{settings.backend}' failed for batch {start}:{end}; "
                f"first input: {batch_paths[0]}: {error}"
            ) from error
        if encoded.backend != backend.name or encoded.mode != settings.text_mode:
            raise ValueError("Backend output identity does not match the requested run")
        if encoded.model_fingerprint != backend.fingerprint:
            raise ValueError(
                "Backend output model fingerprint changed during inference"
            )
        styles.append(encoded.style_embeddings)
        contents.append(encoded.content_embeddings)
        paths.extend(batch_paths)
        previews.extend(_preview_base64(image) for image in images)

    if not styles:
        raise ValueError("The input dataset contains no readable images")
    style_embeddings = np.concatenate(styles).astype(np.float32, copy=False)
    content_embeddings = np.concatenate(contents).astype(np.float32, copy=False)
    table = build_embedding_table(
        paths,
        previews,
        style_embeddings,
        content_embeddings,
        _initial_coordinates(style_embeddings),
        _initial_coordinates(content_embeddings),
    )
    data_path = staged / "data.lance"
    lance.write_dataset(table, data_path)
    written = lance.dataset(data_path)
    if written.count_rows() != len(paths):
        raise ValueError("Written embedding artifact has the wrong row count")
    return {
        "data_digest": _embedding_data_digest(
            paths, style_embeddings, content_embeddings
        ),
        "input_kind": source.input_kind,
        "image_digest": source.image_digest,
        **(
            {"caption_digest": source.caption_digest}
            if settings.text_mode == "caption-guided"
            else {}
        ),
        "backend_name": backend.name,
        "resolved_text_mode": settings.text_mode,
        "embedding_dimensions": {
            "style": int(style_embeddings.shape[1]),
            "content": int(content_embeddings.shape[1]),
        },
        "row_count": len(paths),
        "created_at": datetime.now(UTC).isoformat(),
        "creation_command": settings.creation_command,
        "settings": {
            "batch_size": settings.batch_size,
            "precision": settings.precision,
        },
    }


def _build_export_parameters(settings: WorkflowSettings) -> dict[str, object]:
    if settings.clusterer == "hdbscan":
        return {
            "min_cluster_size": settings.min_cluster_size,
            "feature_set": "1",
        }
    if settings.clusterer == "finch":
        return {
            "k": settings.k_clusters,
            "partition_index": settings.finch_partition_index,
            "feature_set": "1",
        }
    return {"k": settings.k_clusters, "feature_set": "1"}


class WorkflowViewService:
    def __init__(
        self,
        base_dataframe,
        projection_manager,
        settings: WorkflowSettings,
        embedding_manifest_digest: str,
        source: _ResolvedSource,
    ):
        from .dashboard.app import (
            DEFAULT_FINCH_PARTITION_OPTIONS,
            build_default_view_specs,
        )

        self.base_dataframe = base_dataframe
        self.projection_manager = projection_manager
        self.settings = settings
        self.embedding_manifest_digest = embedding_manifest_digest
        self.source = source
        titles, parameter_sets = build_default_view_specs(
            settings.backend.upper(), settings.k_clusters
        )
        self.view_configs = [
            {"title": title, **parameters}
            for title, parameters in zip(titles, parameter_sets, strict=True)
        ]
        self.num_views = len(self.view_configs)
        self.reducer_options = projection_manager.get_reducer_options()
        self.default_reducer = (
            settings.reducer or projection_manager.get_default_reducer()
        )
        self.finch_partition_options = DEFAULT_FINCH_PARTITION_OPTIONS
        self.default_finch_partition_index = settings.finch_partition_index
        self._cache = {}

    def _cluster(self, view_config, partition_index):
        from .clustering.algorithms import (
            perform_finch,
            perform_hdbscan,
            perform_kmeans,
            resolve_finch_req_clust,
        )
        from .clustering.analysis import get_clustering_coords

        coordinates = get_clustering_coords(
            self.base_dataframe, view_config["feature_set"]
        )
        if view_config["clusterer"] == "hdbscan":
            return perform_hdbscan(
                coords=coordinates,
                min_cluster_size=self.settings.min_cluster_size,
            )
        if view_config["clusterer"] == "finch":
            return perform_finch(
                coords=coordinates,
                req_clust=resolve_finch_req_clust(view_config["k"]),
                partition_index=partition_index,
            )
        return perform_kmeans(coords=coordinates, k=view_config["k"])

    def get_view(self, reducer_name, view_index, finch_partition_index):
        active_partition = (
            finch_partition_index
            if self.view_configs[view_index]["clusterer"] == "finch"
            else None
        )
        cache_key = (reducer_name, view_index, active_partition)
        if cache_key in self._cache:
            return self._cache[cache_key]

        from .clustering.analysis import (
            get_clustering_coords,
            get_visual_coords,
            summarize_clusters_for_display,
        )
        from .clustering.export import ExportIdentity, export_clustered_images
        from .dashboard.app import build_view_output_dir
        from .dashboard.figures import create_cluster_figure
        from .data.lance import LanceImageDataset

        spec = self.projection_manager.build_spec(
            reducer_name, random_state=self.settings.random_state
        )
        projected = self.projection_manager.get_projected_dataframe(spec)
        view_config = self.view_configs[view_index]
        result = self._cluster(view_config, active_partition)
        representatives, centers = summarize_clusters_for_display(
            projected,
            result,
            feature_set=view_config["feature_set"],
            clustering_coords=get_clustering_coords(
                self.base_dataframe, view_config["feature_set"]
            ),
            visual_coords=get_visual_coords(projected, view_config["feature_set"]),
        )
        figure = create_cluster_figure(
            projected,
            result,
            representatives,
            centers,
            view_config["title"],
            view_config["feature_set"],
        )
        if self.settings.output_dir is not None:
            parameters = {
                key: value
                for key, value in view_config.items()
                if key not in {"title", "clusterer"}
            }
            if view_config["clusterer"] == "hdbscan":
                parameters["min_cluster_size"] = self.settings.min_cluster_size
            if active_partition is not None:
                parameters["partition_index"] = active_partition
            identity = ExportIdentity(
                self.embedding_manifest_digest,
                spec.digest(self.embedding_manifest_digest),
                result.algorithm_name,
                parameters,
                self.settings.random_state,
                implementation=result.implementation,
            )
            export_clustered_images(
                self.base_dataframe,
                result,
                build_view_output_dir(
                    self.settings.output_dir,
                    view_config,
                    finch_partition_index,
                    identity.digest(),
                ),
                identity,
                self.settings.symlink,
                source_reader=LanceImageDataset(self.source.dataset_path),
                source_root=self.source.source_root,
            )
        value = (figure, projected["image"].tolist())
        self._cache[cache_key] = value
        return value


def execute_workflow(
    settings: WorkflowSettings,
    *,
    backend=None,
    launch_dashboard: bool = False,
) -> WorkflowResult:
    configure_runtime_env()

    import lance

    from .artifacts import (
        ArtifactIdentity,
        ArtifactStore,
        artifact_manifest_digest,
        read_artifact_manifest,
    )
    from .clustering.export import ExportIdentity
    from .models import create_backend, get_supported_text_modes
    from .models.base import validate_backend_mode
    from .projection.manager import ProjectionManager

    supported_modes = (
        backend.supported_text_modes
        if backend is not None
        else get_supported_text_modes(settings.backend)
    )
    validate_backend_mode(settings.backend, supported_modes, settings.text_mode)
    source = _resolve_source(settings)
    _validate_caption_coverage(source, settings.text_mode)
    if backend is None:
        backend = create_backend(settings.backend, settings)
    if backend.name != settings.backend:
        raise ValueError(
            f"Requested backend '{settings.backend}' but constructed '{backend.name}'"
        )

    identity = ArtifactIdentity(
        input_digest=_mode_input_digest(source, settings.text_mode),
        backend=backend.name,
        mode=settings.text_mode,
        model_digest=backend.fingerprint,
        preprocessing_digest=getattr(
            backend, "preprocessing_fingerprint", backend.fingerprint
        ),
        schema_version=3,
    )
    store = ArtifactStore(Path(settings.artifact_root))
    embedding_path = None if settings.rebuild else store.resolve(identity)
    if embedding_path is None:
        with store.stage(identity) as staged:
            manifest = _generate_embedding_build(settings, source, backend, staged)
            embedding_path = store.publish(identity, staged, manifest)

    manifest = read_artifact_manifest(embedding_path)
    embedding_manifest_digest = artifact_manifest_digest(embedding_path)
    embedding_dataset = lance.dataset(embedding_path / "data.lance")
    if embedding_dataset.count_rows() != manifest["row_count"]:
        raise ValueError("Embedding artifact row count does not match its manifest")
    base_dataframe = embedding_dataset.to_table().to_pandas()

    projection_manager = ProjectionManager(
        base_dataframe,
        embedding_digest=embedding_manifest_digest,
        cache_root=Path(settings.artifact_root) / "projections" / "v2",
    )
    reducer_name = settings.reducer or projection_manager.get_default_reducer()
    projection_spec = projection_manager.build_spec(
        reducer_name, random_state=settings.random_state
    )
    projection_identity = projection_spec.digest(embedding_manifest_digest)
    export_identity = ExportIdentity(
        embedding_manifest_digest,
        projection_identity,
        settings.clusterer,
        _build_export_parameters(settings),
        settings.random_state,
    )
    view_service = WorkflowViewService(
        base_dataframe,
        projection_manager,
        settings,
        embedding_manifest_digest,
        source,
    )
    result = WorkflowResult(
        input_kind=source.input_kind,
        source_identity=source.source_identity,
        source_path=source.dataset_path,
        image_digest=source.image_digest,
        caption_digest=source.caption_digest,
        embedding_identity=identity,
        embedding_path=embedding_path,
        embedding_manifest_digest=embedding_manifest_digest,
        projection_spec=projection_spec,
        projection_identity=projection_identity,
        export_identity=export_identity,
        view_service=view_service,
    )
    if launch_dashboard:
        from .dashboard.app import run_dashboard

        run_dashboard(view_service)
    return result


def run(args) -> int:
    execute_workflow(
        WorkflowSettings.from_namespace(args),
        launch_dashboard=True,
    )
    return 0
