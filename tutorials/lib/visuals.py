"""Plotly charts used by the notebooks; each chart answers one learning question."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def rank_heatmap(values: np.ndarray, layers: Sequence[int], positions: Sequence[str], title: str) -> go.Figure:
    return px.imshow(
        values,
        x=list(positions),
        y=[f"layer {layer}" for layer in layers],
        aspect="auto",
        color_continuous_scale="Viridis_r",
        labels={"x": "token position", "y": "source layer", "color": "rank (lower is better)"},
        title=title,
    )


def rank_trajectories(jacobian: np.ndarray, logit: np.ndarray, layers: Sequence[int], concept: str) -> go.Figure:
    figure = go.Figure()
    figure.add_scatter(x=list(layers), y=jacobian, mode="lines+markers", name="Jacobian lens")
    figure.add_scatter(x=list(layers), y=logit, mode="lines+markers", name="Logit lens")
    figure.update_layout(title=f"How does {concept!r} change through depth?", xaxis_title="source layer", yaxis_title="rank")
    figure.update_yaxes(autorange="reversed")
    return figure


def residual_projection(points: np.ndarray, layers: Sequence[int], positions: Sequence[str]) -> go.Figure:
    figure = go.Figure()
    for index, layer in enumerate(layers):
        figure.add_scatter(
            x=points[index, :, 0], y=points[index, :, 1], mode="lines+markers", name=f"layer {layer}",
            text=list(positions), hovertemplate="position=%{text}<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
        )
    figure.update_layout(title="How does the selected residual trace move across layers?", xaxis_title="PC 1", yaxis_title="PC 2")
    return figure


def dictionary_geometry(active: int, coefficient: float) -> go.Figure:
    """A pedagogical 2D overcomplete dictionary, not a fitted J-space result."""
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    figure = go.Figure()
    for index, angle in enumerate(angles):
        figure.add_scatter(x=[0, np.cos(angle)], y=[0, np.sin(angle)], mode="lines+markers", name=f"dictionary {index}", showlegend=False)
    selected = angles[active % len(angles)]
    figure.add_scatter(x=[0, coefficient * np.cos(selected)], y=[0, coefficient * np.sin(selected)], mode="lines+markers", line={"width": 6}, name="active non-negative component")
    figure.update_layout(title="Toy sparse dictionary geometry — not a fitted model decomposition", xaxis={"scaleanchor": "y", "range": [-1.5, 1.5]}, yaxis={"range": [-1.5, 1.5]})
    return figure
