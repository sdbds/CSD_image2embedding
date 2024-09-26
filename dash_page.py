from sklearn.cluster import KMeans
import random
import socket
from dash import dcc, html, Input, Output, no_update, Dash
import plotly.graph_objects as go
from PIL import Image
import base64
import io
from scipy.spatial.distance import cdist

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


def perform_kmeans(data, k=20):
    # Extract x, y coordinates
    coords = data[['x', 'y']].to_numpy()

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(coords)

    return kmeans


def find_nearest_images(data, kmeans):
    coords = data[['x', 'y']].to_numpy()
    images = data["image"].tolist()

    # Calculate distances to cluster centers
    distances = cdist(coords, kmeans.cluster_centers_, metric="euclidean")

    # Find the index of the nearest point for each cluster
    nearest_indices = distances.argmin(axis=0)

    # Get the images nearest to each cluster center
    nearest_images = [images[i] for i in nearest_indices]

    return nearest_images, kmeans.cluster_centers_


def create_dash_fig(data, kmeans_result, nearest_images, cluster_centers, title):
    # Extract x, y coordinates
    x = data["x"].tolist()
    y = data["y"].tolist()
    images = data["image"].tolist()

    # Determine the range for both axes
    max_range = max(max(x) - min(x), max(y) - min(y)) / 2
    center_x = (max(x) + min(x)) / 2
    center_y = (max(y) + min(y)) / 2

    # Create the scatter plot
    fig = go.Figure()

    # Add data points
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(
                size=5,
                color=kmeans_result.labels_,
                colorscale="Viridis",
                showscale=False,
            ),
            name="Data Points",
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


def make_dash_kmeans(datasets, title, k=40):
    datasets = datasets.to_table().to_pandas()
    kmeans_result = perform_kmeans(datasets, k=k)
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