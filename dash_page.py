from sklearn.cluster import KMeans
from hdbscan import HDBSCAN
import numpy as np
import random
import socket
from dash import dcc, html, Input, Output, no_update, Dash, callback_context
import plotly.graph_objects as go
from PIL import Image
import base64
import io
import os
from scipy.spatial.distance import cdist
from process_image import classify_images


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


def perform_kmeans(data, k=40, feature_set="1"):
    # Extract x, y coordinates based on feature set
    if feature_set == "1":
        coords = data[["x1", "y1"]].to_numpy()
    else:
        coords = data[["x2", "y2"]].to_numpy()

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(coords)

    return kmeans


def perform_hdbscan(data, min_cluster_size=5, feature_set="1"):
    # Extract x, y coordinates based on feature set
    if feature_set == "1":
        coords = data[["x1", "y1"]].to_numpy()
    else:
        coords = data[["x2", "y2"]].to_numpy()

    # Perform HDBSCAN clustering
    hdbscan = HDBSCAN(
        min_cluster_size=min_cluster_size,
    )
    hdbscan.fit(coords)

    return hdbscan


def find_nearest_images(data, kmeans, feature_set="1"):
    if feature_set == "1":
        coords = data[["x1", "y1"]].to_numpy()
    else:
        coords = data[["x2", "y2"]].to_numpy()
    images = data["image"].tolist()

    if isinstance(kmeans, KMeans):

        # Calculate distances to cluster centers
        distances = cdist(coords, kmeans.cluster_centers_, metric="euclidean")

        # Find the index of the nearest point for each cluster
        nearest_indices = distances.argmin(axis=0)

        # Get the images nearest to each cluster center
        nearest_images = [images[i] for i in nearest_indices]

        return nearest_images, kmeans.cluster_centers_

    else:

        nearest_images = []
        cluster_centers = []

        # Calculate distances to cluster centers
        for label in np.unique(kmeans.labels_):
            if label == -1:  # Skip noise points
                continue
            cluster_indices = np.where(kmeans.labels_ == label)[0]
            cluster_coords = coords[cluster_indices]

            # Calculate the centroid of the cluster
            centroid = cluster_coords.mean(axis=0)
            cluster_centers.append(centroid)

            # Find the nearest point to the centroid
            distances = np.linalg.norm(cluster_coords - centroid, axis=1)
            nearest_index = cluster_indices[np.argmin(distances)]
            nearest_images.append(images[nearest_index])

        return nearest_images, np.array(cluster_centers)


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


def create_multi_view_dash_app(view_data):
    app = Dash(__name__)

    app.layout = html.Div(
        [
            html.H1("Multi-view Clustering Visualization"),
            html.Div(
                [
                    dcc.Tabs(
                        id="tabs",
                        value="tab-0",
                        children=[
                            dcc.Tab(label=f"View {i+1}", value=f"tab-{i}")
                            for i in range(len(view_data))
                        ],
                    ),
                    html.Div(id="tabs-content"),
                ]
            ),
            dcc.Tooltip(id="graph-tooltip", direction="bottom"),
        ]
    )

    @app.callback(Output("tabs-content", "children"), Input("tabs", "value"))
    def render_content(tab):
        index = int(tab.split("-")[1])
        fig, images = view_data[index]
        return html.Div([dcc.Graph(id="graph", figure=fig, clear_on_unhover=True)])

    @app.callback(
        Output("graph-tooltip", "show"),
        Output("graph-tooltip", "bbox"),
        Output("graph-tooltip", "children"),
        Input("graph", "hoverData"),
        Input("tabs", "value"),
    )
    def display_hover(hoverData, tab):
        if hoverData is None:
            return False, no_update, no_update

        index = int(tab.split("-")[1])
        _, images = view_data[index]

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


def make_multi_view_dash(
    datasets,
    titles,
    params_list,
    args,
    feature_set="1",
):
    view_data = []

    for title, params in zip(titles, params_list):
        datasets_df = datasets.to_table().to_pandas()

        feature_set = params.get("feature_set", "1")

        if params.get("hdbscan", False):
            clustering_result = perform_hdbscan(
                datasets_df,
                min_cluster_size=args.min_cluster_size,
                feature_set=feature_set,
            )
        else:
            clustering_result = perform_kmeans(
                datasets_df, k=params.get("k", 40), feature_set=feature_set
            )

        if args.output_dir:
            classify_images(
                datasets_df,
                clustering_result,
                args,
                os.path.join(args.output_dir, title),
            )

        nearest_images, cluster_centers = find_nearest_images(
            datasets_df, clustering_result, feature_set=feature_set
        )
        fig, images = create_dash_fig(
            datasets_df,
            clustering_result,
            nearest_images,
            cluster_centers,
            title,
            feature_set=feature_set,
        )
        view_data.append((fig, images))

    app = create_multi_view_dash_app(view_data)
    port = find_free_port()
    print(f"Serving on http://127.0.0.1:{port}/")
    print(f"To serve this over the Internet, run `ngrok http {port}`")
    app.run_server(port=port)
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
    print(f"Serving on http://127.0.0.1:{port}/")
    print(f"To serve this over the Internet, run `ngrok http {port}`")
    app.run_server(port=port)
    return app
