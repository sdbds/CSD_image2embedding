from dataclasses import replace

from csd_image2embedding.projection.manager import ProjectionSpec


def test_projection_digest_changes_for_seed_parameter_and_version():
    base = ProjectionSpec(
        "umap",
        {"metric": "cosine", "n_components": 2},
        42,
        "umap-1",
    )

    assert base.digest("emb") != replace(base, random_state=7).digest("emb")
    assert base.digest("emb") != replace(
        base,
        parameters={"metric": "euclidean", "n_components": 2},
    ).digest("emb")
    assert base.digest("emb") != replace(base, implementation_version="umap-2").digest(
        "emb"
    )


def test_projection_digest_is_canonical_but_depends_on_embedding_manifest():
    first = ProjectionSpec(
        "umap",
        {"metric": "cosine", "n_components": 2},
        42,
        "umap-1",
    )
    reordered = ProjectionSpec(
        "umap",
        {"n_components": 2, "metric": "cosine"},
        42,
        "umap-1",
    )

    assert first.digest("embedding-a") == reordered.digest("embedding-a")
    assert first.digest("embedding-a") != first.digest("embedding-b")
