"""Dash layout, callbacks, view-service integration, and server startup."""

from __future__ import annotations

import base64
import io
import random
import socket
import webbrowser
from pathlib import Path

from dash import Dash, Input, Output, dcc, html, no_update
from PIL import Image

DEFAULT_FINCH_PARTITION_OPTIONS = [
    {"label": str(index), "value": index} for index in range(6)
]


def build_default_view_specs(model_label: str, k_clusters: int):
    titles = []
    parameters = []
    for suffix, feature_set in (("style", "1"), ("content", "2")):
        titles.extend(
            [
                f"[{model_label}] KMeans_{suffix}",
                f"[{model_label}] HDBSCAN_{suffix}",
                f"[{model_label}] FINCH_{suffix}",
            ]
        )
        parameters.extend(
            [
                {"clusterer": "kmeans", "k": k_clusters, "feature_set": feature_set},
                {"clusterer": "hdbscan", "feature_set": feature_set},
                {"clusterer": "finch", "k": k_clusters, "feature_set": feature_set},
            ]
        )
    return titles, parameters


def build_multi_view_layout(
    num_views: int,
    reducer_options,
    default_reducer: str,
    finch_partition_options=None,
    default_finch_partition_index: int = 1,
):
    if finch_partition_options is None:
        finch_partition_options = DEFAULT_FINCH_PARTITION_OPTIONS
    return html.Div(
        [
            html.H1("Multi-view Clustering Visualization"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        "Projection reducer",
                                        style={"fontWeight": "bold"},
                                    ),
                                    dcc.Dropdown(
                                        id="reducer-selector",
                                        options=reducer_options,
                                        value=default_reducer,
                                        clearable=False,
                                        searchable=False,
                                        style={"width": "320px"},
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        "FINCH partition",
                                        style={"fontWeight": "bold"},
                                    ),
                                    dcc.Dropdown(
                                        id="finch-partition-selector",
                                        options=finch_partition_options,
                                        value=default_finch_partition_index,
                                        clearable=False,
                                        searchable=False,
                                        style={"width": "180px"},
                                    ),
                                ]
                            ),
                        ],
                        style={
                            "display": "flex",
                            "gap": "16px",
                            "alignItems": "end",
                            "marginBottom": "16px",
                            "flexWrap": "wrap",
                        },
                    ),
                    html.Div(
                        id="reducer-message",
                        style={"marginBottom": "12px", "color": "#8b0000"},
                    ),
                    dcc.Loading(
                        id="reducer-loading",
                        type="default",
                        children=html.Div(
                            [
                                dcc.Tabs(
                                    id="tabs",
                                    value="tab-0",
                                    children=[
                                        dcc.Tab(
                                            label=f"View {index + 1}",
                                            value=f"tab-{index}",
                                        )
                                        for index in range(num_views)
                                    ],
                                ),
                                html.Div(id="tabs-content"),
                            ]
                        ),
                    ),
                ]
            ),
            dcc.Tooltip(id="graph-tooltip", direction="bottom"),
        ]
    )


def _tooltip_children(image_base64: str):
    image_data = base64.b64decode(image_base64)
    with Image.open(io.BytesIO(image_data)) as image:
        width, height = image.size
    return [
        html.Div(
            [
                html.Img(
                    src=f"data:image/jpeg;base64,{image_base64}",
                    style={
                        "width": f"{width}px",
                        "height": f"{height}px",
                        "display": "block",
                        "margin": "0 auto",
                    },
                )
            ]
        )
    ]


def create_dashboard_app(view_service) -> Dash:
    app = Dash(__name__, suppress_callback_exceptions=True)
    app.layout = build_multi_view_layout(
        num_views=view_service.num_views,
        reducer_options=view_service.reducer_options,
        default_reducer=view_service.default_reducer,
        finch_partition_options=getattr(
            view_service,
            "finch_partition_options",
            DEFAULT_FINCH_PARTITION_OPTIONS,
        ),
        default_finch_partition_index=getattr(
            view_service, "default_finch_partition_index", 1
        ),
    )

    @app.callback(
        Output("tabs-content", "children"),
        Output("reducer-message", "children"),
        Input("tabs", "value"),
        Input("reducer-selector", "value"),
        Input("finch-partition-selector", "value"),
    )
    def render_content(tab, reducer_name, finch_partition_index):
        view_index = int(tab.split("-")[1])
        try:
            figure, _ = view_service.get_view(
                reducer_name, view_index, finch_partition_index
            )
        except Exception as error:
            message = f"Failed to build {reducer_name} projection: {error}"
            return html.Div(message), message
        return html.Div([dcc.Graph(id="graph", figure=figure)]), ""

    @app.callback(
        Output("graph-tooltip", "show"),
        Output("graph-tooltip", "bbox"),
        Output("graph-tooltip", "children"),
        Input("graph", "hoverData"),
        Input("tabs", "value"),
        Input("reducer-selector", "value"),
        Input("finch-partition-selector", "value"),
    )
    def display_hover(hover_data, tab, reducer_name, finch_partition_index):
        if hover_data is None:
            return False, no_update, no_update
        try:
            _, images = view_service.get_view(
                reducer_name,
                int(tab.split("-")[1]),
                finch_partition_index,
            )
        except Exception:
            return False, no_update, no_update
        point = hover_data["points"][0]
        return (
            True,
            point["bbox"],
            _tooltip_children(images[point["pointNumber"]]),
        )

    return app


def resolve_active_finch_partition(view_config, finch_partition_index):
    if view_config["clusterer"] != "finch":
        return None
    return finch_partition_index


def build_view_output_dir(
    output_dir: Path,
    view_config,
    finch_partition_index: int,
    run_identity_digest: str,
) -> Path:
    suffix = (
        f"{view_config['title']}_p{finch_partition_index}"
        if view_config["clusterer"] == "finch"
        else view_config["title"]
    )
    return Path(output_dir) / suffix / "runs" / run_identity_digest


def find_free_port() -> int:
    while True:
        port = random.randint(49152, 65535)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
            try:
                connection.bind(("", port))
            except OSError:
                continue
            return port


def run_dashboard(view_service) -> Dash:
    app = create_dashboard_app(view_service)
    port = find_free_port()
    url = f"http://127.0.0.1:{port}/"
    print(f"Serving on {url}")
    webbrowser.open(url)
    app.run(port=port)
    return app
