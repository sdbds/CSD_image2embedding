from dash import dcc, html

from csd_image2embedding.dashboard.app import (
    build_default_view_specs,
    build_multi_view_layout,
    create_dashboard_app,
)


def _find_component_by_id(node, target_id):
    if getattr(node, "id", None) == target_id:
        return node
    children = getattr(node, "children", None)
    if children is None:
        return None
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        found = _find_component_by_id(child, target_id)
        if found is not None:
            return found
    return None


def test_layout_contains_global_projection_and_finch_controls():
    layout = build_multi_view_layout(
        num_views=4,
        reducer_options=[
            {"label": "PaCMAP", "value": "pacmap"},
            {"label": "UMAP", "value": "umap"},
        ],
        default_reducer="pacmap",
        finch_partition_options=[{"label": "0", "value": 0}],
        default_finch_partition_index=0,
    )

    assert isinstance(layout, html.Div)
    assert isinstance(_find_component_by_id(layout, "reducer-selector"), dcc.Dropdown)
    assert _find_component_by_id(layout, "reducer-selector").value == "pacmap"
    assert isinstance(
        _find_component_by_id(layout, "finch-partition-selector"), dcc.Dropdown
    )
    assert isinstance(_find_component_by_id(layout, "reducer-loading"), dcc.Loading)


def test_default_views_include_each_clusterer_for_style_and_content():
    titles, parameters = build_default_view_specs("CSD", 40)

    assert titles == [
        "[CSD] KMeans_style",
        "[CSD] HDBSCAN_style",
        "[CSD] FINCH_style",
        "[CSD] KMeans_content",
        "[CSD] HDBSCAN_content",
        "[CSD] FINCH_content",
    ]
    assert [item["clusterer"] for item in parameters] == [
        "kmeans",
        "hdbscan",
        "finch",
        "kmeans",
        "hdbscan",
        "finch",
    ]


def test_dashboard_app_reads_configuration_from_view_service():
    class FakeViewService:
        num_views = 2
        reducer_options = [{"label": "Legacy", "value": "legacy"}]
        default_reducer = "legacy"
        finch_partition_options = [{"label": "0", "value": 0}]
        default_finch_partition_index = 0

        def get_view(self, reducer_name, view_index, finch_partition_index):
            raise AssertionError((reducer_name, view_index, finch_partition_index))

    app = create_dashboard_app(FakeViewService())

    assert len(_find_component_by_id(app.layout, "tabs").children) == 2
