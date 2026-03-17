from sklearn.cluster import KMeans
from hdbscan import HDBSCAN
import numpy as np
import random
import socket
import torch
from dash import dcc, html, Input, Output, no_update, Dash, callback_context
import plotly.graph_objects as go
from PIL import Image
import base64
import io
import os
from process_image import classify_images
import webbrowser
from clustering_utils import (
    get_clustering_coords,
    get_visual_coords,
    summarize_clusters_for_display,
)
from projection_cache import DEFAULT_REDUCER
from projection_manager import ProjectionManager
from cluster_result_utils import GenericClusteringResult


DEFAULT_FINCH_PARTITION_OPTIONS = [
    {"label": str(index), "value": index} for index in range(6)
]


def find_free_port():
    while True:
        port = random.randint(49152, 65535)  # Use dynamic/private port range
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                pass


def create_dash_app(fig, images):
    app = Dash(__name__)

    app.layout = html.Div(
        className="container",
        children=[
            dcc.Graph(id="graph", figure=fig, clear_on_unhover=True),
            dcc.Tooltip(id="graph-tooltip", direction="bottom"),
        ],
    )

    @app.callback(
        Output("graph-tooltip", "show"),
        Output("graph-tooltip", "bbox"),
        Output("graph-tooltip", "children"),
        Input("graph", "hoverData"),
    )
    def display_hover(hoverData):
        if hoverData is None:
            return False, no_update, no_update

        hover_data = hoverData["points"][0]
        bbox = hover_data["bbox"]
        num = hover_data["pointNumber"]

        image_base64 = images[num]
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        width, height = image.size
        children = [
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
                    ),
                ]
            )
        ]

        return True, bbox, children

    return app


def _to_numpy_array(value, dtype):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _perform_sklearn_kmeans(coords, k, algorithm_name):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(coords)
    kmeans.algorithm_name = algorithm_name
    kmeans.noise_label = None
    return kmeans


def _perform_flash_kmeans(coords, k):
    from flash_kmeans import FlashKMeans

    flash_kmeans = FlashKMeans(
        d=coords.shape[1],
        k=k,
        seed=42,
    )
    flash_kmeans.fit(torch.as_tensor(coords, dtype=torch.float32))

    labels = _to_numpy_array(flash_kmeans.cluster_ids_b, np.int32)
    centers = _to_numpy_array(flash_kmeans.centroids_b, np.float32)
    if labels.ndim > 1:
        labels = labels[0]
    if centers.ndim > 2:
        centers = centers[0]

    return GenericClusteringResult(
        labels_=labels,
        cluster_centers_=centers,
        algorithm_name="flash-kmeans",
        noise_label=None,
    )


def perform_kmeans(data=None, k=40, feature_set="1", coords=None):
    if coords is None:
        coords = get_clustering_coords(data, feature_set=feature_set)
    coords = np.asarray(coords, dtype=np.float32)

    try:
        return _perform_flash_kmeans(coords, k)
    except Exception:
        return _perform_sklearn_kmeans(
            coords,
            k,
            algorithm_name="kmeans-sklearn-fallback",
        )


def perform_hdbscan(data=None, min_cluster_size=5, feature_set="1", coords=None):
    if coords is None:
        coords = get_clustering_coords(data, feature_set=feature_set)

    # Perform HDBSCAN clustering
    hdbscan = HDBSCAN(
        min_cluster_size=min_cluster_size,
    )
    hdbscan.fit(coords)
    hdbscan.algorithm_name = "hdbscan"
    hdbscan.noise_label = -1

    return hdbscan


def resolve_finch_req_clust(k_clusters, default_k_clusters=40):
    if k_clusters == default_k_clusters:
        return None
    return k_clusters


def _select_finch_labels(labels, num_clust, requested_labels, req_clust, partition_index):
    if requested_labels is not None:
        return np.asarray(requested_labels)

    labels = np.asarray(labels)
    if labels.ndim == 1:
        return labels

    num_clust = list(num_clust or [])
    if req_clust is not None and num_clust:
        if req_clust in num_clust:
            return labels[:, num_clust.index(req_clust)]
        if req_clust > num_clust[0]:
            return labels[:, 0]

    if labels.shape[1] == 1:
        return labels[:, 0]
    if partition_index >= labels.shape[1]:
        return labels[:, -1]
    if partition_index < -labels.shape[1]:
        return labels[:, 0]

    return labels[:, partition_index]


def perform_finch(
    data=None,
    feature_set="1",
    coords=None,
    req_clust=None,
    partition_index=1,
):
    try:
        from finch import FINCH
    except ImportError as exc:
        raise ImportError(
            "FINCH clustering requires the 'finch-clust' package."
        ) from exc

    if coords is None:
        coords = get_clustering_coords(data, feature_set=feature_set)
    try:
        labels, num_clust, requested_labels = FINCH(
            coords,
            req_clust=req_clust,
            distance="cosine",
            verbose=False,
        )
    except UnboundLocalError as exc:
        if req_clust is None or "requested_c" not in str(exc):
            raise
        labels, num_clust, requested_labels = FINCH(
            coords,
            req_clust=None,
            distance="cosine",
            verbose=False,
        )

    labels = _select_finch_labels(
        labels,
        num_clust,
        requested_labels,
        req_clust,
        partition_index,
    )
    _, normalized_labels = np.unique(labels, return_inverse=True)
    return GenericClusteringResult(
        labels_=normalized_labels,
        algorithm_name="finch",
    )


def find_nearest_images(
    data,
    clustering_result,
    feature_set="1",
    clustering_coords=None,
    visual_coords=None,
):
    return summarize_clusters_for_display(
        data,
        clustering_result,
        feature_set=feature_set,
        clustering_coords=clustering_coords,
        visual_coords=visual_coords,
    )


def create_dash_fig(
    data, kmeans_result, nearest_images, cluster_centers, title, feature_set="1"
):
    # Extract x, y coordinates based on feature set
    if feature_set == "1":
        x = data["x1"].tolist()
        y = data["y1"].tolist()
    else:
        x = data["x2"].tolist()
        y = data["y2"].tolist()
    images = data["image"].tolist()

    # Determine the range for both axes
    max_range = max(max(x) - min(x), max(y) - min(y)) / 2
    center_x = (max(x) + min(x)) / 2
    center_y = (max(y) + min(y)) / 2

    # Create the scatter plot
    fig = go.Figure()

    # Add data points with enhanced color scheme
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(
                size=5,
                color=kmeans_result.labels_,
                colorscale="hsv",
                showscale=True,
                colorbar=dict(title="style"),
                opacity=0.8,
            ),
            name="Data Points",
        )
    )

    # Add cluster centers
    if len(cluster_centers) > 0:
        fig.add_trace(
            go.Scatter(
                x=cluster_centers[:, 0],
                y=cluster_centers[:, 1],
                mode="markers",
                marker=dict(
                    symbol="star",
                    size=15,
                    color="black",
                    line=dict(width=2, color="DarkSlateGrey"),
                ),
                name="Cluster Centers",
            )
        )

    # Add cluster centers and images

    fig.update_layout(
        title=title,
        width=1000,
        height=1000,
        xaxis=dict(
            range=[center_x - max_range, center_x + max_range],
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            range=[center_y - max_range, center_y + max_range],
        ),
        showlegend=False,
    )

    fig.update_traces(
        hoverinfo="none",
        hovertemplate=None,
    )
    # Add images
    if len(cluster_centers) == 0:
        return fig, images
    for i, (cx, cy) in enumerate(cluster_centers):
        fig.add_layout_image(
            dict(
                source=f"data:image/jpg;base64,{nearest_images[i]}",
                x=cx,
                y=cy,
                xref="x",
                yref="y",
                sizex=1,
                sizey=1,
                sizing="contain",
                opacity=1,
                layer="below",
            )
        )

    # Remove x and y axes ticks
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False))

    return fig, images


def create_unavailable_fig(title, message):
    fig = go.Figure()
    fig.update_layout(
        title=f"{title} (unavailable)",
        width=1000,
        height=1000,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[
            dict(
                text=message,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
            )
        ],
        showlegend=False,
    )
    return fig


def build_multi_view_layout(
    num_views,
    reducer_options,
    default_reducer,
    finch_partition_options=None,
    default_finch_partition_index=1,
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
                                        dcc.Tab(label=f"View {i+1}", value=f"tab-{i}")
                                        for i in range(num_views)
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


def create_multi_view_dash_app(
    get_view_data,
    reducer_options,
    default_reducer,
    num_views,
    finch_partition_options,
    default_finch_partition_index,
):
    app = Dash(__name__)

    app.layout = build_multi_view_layout(
        num_views=num_views,
        reducer_options=reducer_options,
        default_reducer=default_reducer,
        finch_partition_options=finch_partition_options,
        default_finch_partition_index=default_finch_partition_index,
    )

    @app.callback(
        Output("tabs-content", "children"),
        Output("reducer-message", "children"),
        Input("tabs", "value"),
        Input("reducer-selector", "value"),
        Input("finch-partition-selector", "value"),
    )
    def render_content(tab, reducer_name, finch_partition_index):
        index = int(tab.split("-")[1])
        try:
            fig, _ = get_view_data(reducer_name, index, finch_partition_index)
        except Exception as exc:
            error_text = f"Failed to build {reducer_name} projection: {exc}"
            return html.Div(error_text), error_text

        return (
            html.Div([dcc.Graph(id="graph", figure=fig, clear_on_unhover=True)]),
            "",
        )

    @app.callback(
        Output("graph-tooltip", "show"),
        Output("graph-tooltip", "bbox"),
        Output("graph-tooltip", "children"),
        Input("graph", "hoverData"),
        Input("tabs", "value"),
        Input("reducer-selector", "value"),
        Input("finch-partition-selector", "value"),
    )
    def display_hover(hoverData, tab, reducer_name, finch_partition_index):
        if hoverData is None:
            return False, no_update, no_update

        try:
            _, images = get_view_data(
                reducer_name,
                int(tab.split("-")[1]),
                finch_partition_index,
            )
        except Exception:
            return False, no_update, no_update

        hover_data = hoverData["points"][0]
        bbox = hover_data["bbox"]
        num = hover_data["pointNumber"]

        image_base64 = images[num]
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        width, height = image.size
        children = [
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
                    ),
                ]
            )
        ]

        return True, bbox, children

    return app


def resolve_active_finch_partition(view_config, finch_partition_index):
    if view_config["clusterer"] != "finch":
        return None
    return finch_partition_index


def build_view_output_dir(output_dir, view_config, finch_partition_index):
    if view_config["clusterer"] == "finch":
        return os.path.join(
            output_dir,
            f"{view_config['title']}_p{finch_partition_index}",
        )
    return os.path.join(output_dir, view_config["title"])


def make_multi_view_dash(
    datasets,
    titles,
    params_list,
    args,
    feature_set="1",
):
    datasets_df = datasets.to_table().to_pandas()
    projection_manager = ProjectionManager(datasets_df)
    reducer_options = projection_manager.get_reducer_options()
    default_reducer = projection_manager.get_default_reducer()
    view_configs = []
    feature_sets = sorted({params.get("feature_set", "1") for params in params_list})
    clustering_coords_cache = {
        feature_name: get_clustering_coords(datasets_df, feature_set=feature_name)
        for feature_name in feature_sets
    }

    for title, params in zip(titles, params_list):
        feature_set = params.get("feature_set", "1")
        clusterer = params.get("clusterer")
        if clusterer is None:
            clusterer = "hdbscan" if params.get("hdbscan", False) else "kmeans"

        view_configs.append(
            {
                "title": title,
                "feature_set": feature_set,
                "clusterer": clusterer,
                "k": params.get("k", 40),
            }
        )

    if default_reducer != DEFAULT_REDUCER:
        print(
            f"Default reducer '{DEFAULT_REDUCER}' unavailable. Falling back to '{default_reducer}'."
        )

    cluster_cache = {}
    rendered_view_cache = {}
    exported_views = set()

    def get_clustering_state(view_index, finch_partition_index):
        view_config = view_configs[view_index]
        active_finch_partition = resolve_active_finch_partition(
            view_config,
            finch_partition_index,
        )
        cache_key = (view_index, active_finch_partition)
        if cache_key in cluster_cache:
            return cluster_cache[cache_key]

        coords = clustering_coords_cache[view_config["feature_set"]]
        try:
            if view_config["clusterer"] == "hdbscan":
                clustering_result = perform_hdbscan(
                    min_cluster_size=args.min_cluster_size,
                    coords=coords,
                )
            elif view_config["clusterer"] == "finch":
                clustering_result = perform_finch(
                    coords=coords,
                    req_clust=resolve_finch_req_clust(view_config["k"]),
                    partition_index=active_finch_partition,
                )
            else:
                clustering_result = perform_kmeans(
                    k=view_config["k"],
                    coords=coords,
                )
            cluster_state = {"result": clustering_result, "error": None}
        except ImportError as exc:
            cluster_state = {"result": None, "error": str(exc)}

        cluster_cache[cache_key] = cluster_state
        return cluster_state

    def get_view_data(reducer_name, view_index, finch_partition_index):
        view_config = view_configs[view_index]
        active_finch_partition = resolve_active_finch_partition(
            view_config,
            finch_partition_index,
        )
        cache_key = (reducer_name, view_index, active_finch_partition)
        if cache_key in rendered_view_cache:
            return rendered_view_cache[cache_key]

        projected_df = projection_manager.get_projected_dataframe(reducer_name)
        cluster_state = get_clustering_state(view_index, finch_partition_index)

        if cluster_state["error"]:
            view_data = (
                create_unavailable_fig(
                    view_config["title"],
                    cluster_state["error"],
                ),
                projected_df["image"].tolist(),
            )
            rendered_view_cache[cache_key] = view_data
            return view_data

        clustering_result = cluster_state["result"]
        feature_set = view_config["feature_set"]
        nearest_images, cluster_centers = find_nearest_images(
            projected_df,
            clustering_result,
            feature_set=feature_set,
            clustering_coords=clustering_coords_cache[feature_set],
            visual_coords=get_visual_coords(projected_df, feature_set=feature_set),
        )
        fig, images = create_dash_fig(
            projected_df,
            clustering_result,
            nearest_images,
            cluster_centers,
            view_config["title"],
            feature_set=feature_set,
        )

        export_key = (view_index, active_finch_partition)
        if args.output_dir and export_key not in exported_views:
            classify_images(
                datasets_df,
                clustering_result,
                args,
                build_view_output_dir(
                    args.output_dir,
                    view_config,
                    finch_partition_index,
                ),
            )
            exported_views.add(export_key)

        view_data = (fig, images)
        rendered_view_cache[cache_key] = view_data
        return view_data

    app = create_multi_view_dash_app(
        get_view_data=get_view_data,
        reducer_options=reducer_options,
        default_reducer=default_reducer,
        num_views=len(view_configs),
        finch_partition_options=DEFAULT_FINCH_PARTITION_OPTIONS,
        default_finch_partition_index=getattr(args, "finch_partition_index", 1),
    )
    port = find_free_port()
    url = f"http://127.0.0.1:{port}/"
    print(f"Serving on {url}")
    print(f"To serve this over the Internet, run `ngrok http {port}`")
    webbrowser.open(url)
    app.run(port=port)
    return app


def make_dash_kmeans(datasets, title, k=50, hdbscan=False, output_dir="output"):
    datasets = datasets.to_table().to_pandas()
    kmeans_result = (
        perform_kmeans(datasets, k=k)
        if not hdbscan
        else perform_hdbscan(datasets, min_cluster_size=5)
    )
    if output_dir:
        classify_images(datasets, kmeans_result, output_dir)
    nearest_images, cluster_centers = find_nearest_images(datasets, kmeans_result)
    fig, images = create_dash_fig(
        datasets, kmeans_result, nearest_images, cluster_centers, title
    )
    app = create_dash_app(fig, images)
    port = find_free_port()
    url = f"http://127.0.0.1:{port}/"
    print(f"Serving on {url}")
    print(f"To serve this over the Internet, run `ngrok http {port}`")
    webbrowser.open(url)
    app.run(port=port)
    return app
