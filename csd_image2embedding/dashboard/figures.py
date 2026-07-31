"""Pure Plotly construction for cluster views."""

from __future__ import annotations

import plotly.graph_objects as go

from ..clustering.analysis import get_visual_column_names


def create_cluster_figure(
    data,
    clustering_result,
    representatives,
    centers,
    title: str,
    feature_set: str = "1",
) -> go.Figure:
    x_column, y_column = get_visual_column_names(feature_set)
    x_values = data[x_column].tolist()
    y_values = data[y_column].tolist()
    if not x_values:
        return create_unavailable_figure(title, "No rows are available")

    x_span = max(x_values) - min(x_values)
    y_span = max(y_values) - min(y_values)
    half_range = max(x_span, y_span, 1e-6) / 2
    center_x = (max(x_values) + min(x_values)) / 2
    center_y = (max(y_values) + min(y_values)) / 2

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="markers",
            marker={
                "size": 5,
                "color": clustering_result.labels_,
                "colorscale": "hsv",
                "showscale": True,
                "colorbar": {"title": "cluster"},
                "opacity": 0.8,
            },
            name="Data Points",
            hoverinfo="none",
            hovertemplate=None,
        )
    )
    if len(centers):
        figure.add_trace(
            go.Scatter(
                x=centers[:, 0],
                y=centers[:, 1],
                mode="markers",
                marker={
                    "symbol": "star",
                    "size": 15,
                    "color": "black",
                    "line": {"width": 2, "color": "DarkSlateGrey"},
                },
                name="Cluster Centers",
                hoverinfo="none",
                hovertemplate=None,
            )
        )
        for image, (center_horizontal, center_vertical) in zip(
            representatives, centers, strict=True
        ):
            figure.add_layout_image(
                {
                    "source": f"data:image/jpeg;base64,{image}",
                    "x": center_horizontal,
                    "y": center_vertical,
                    "xref": "x",
                    "yref": "y",
                    "sizex": max(half_range / 8, 0.1),
                    "sizey": max(half_range / 8, 0.1),
                    "sizing": "contain",
                    "opacity": 1,
                    "layer": "below",
                }
            )

    figure.update_layout(
        title=title,
        width=1000,
        height=1000,
        xaxis={
            "range": [center_x - half_range, center_x + half_range],
            "scaleanchor": "y",
            "scaleratio": 1,
            "visible": False,
        },
        yaxis={
            "range": [center_y - half_range, center_y + half_range],
            "visible": False,
        },
        showlegend=False,
    )
    return figure


def create_unavailable_figure(title: str, message: str) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(
        title=f"{title} (unavailable)",
        width=1000,
        height=1000,
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": message,
                "x": 0.5,
                "y": 0.5,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
            }
        ],
        showlegend=False,
    )
    return figure
