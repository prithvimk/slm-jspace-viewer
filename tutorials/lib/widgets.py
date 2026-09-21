"""ipywidgets explorers backed by compact tutorial artifacts."""
from __future__ import annotations

from typing import Any

import ipywidgets as widgets
from IPython.display import Markdown, display

from .artifacts import TutorialArtifact
from .visuals import dictionary_geometry, rank_heatmap, rank_trajectories, residual_projection


def display_lens_explorer(artifact: TutorialArtifact) -> None:
    concept = widgets.Dropdown(options=artifact.concepts, description="Concept:")
    position = widgets.IntSlider(min=0, max=len(artifact.manifest["positions"]) - 1, value=0, description="Position:")
    mode = widgets.ToggleButtons(options=["Jacobian rank", "Logit rank", "Rank advantage"], description="View:")
    output = widgets.Output()

    def redraw(_: Any = None) -> None:
        selected = artifact.concepts.index(concept.value)
        jacobian = artifact.metrics["jacobian_rank"][selected]
        logit = artifact.metrics["logit_rank"][selected]
        values = jacobian if mode.value == "Jacobian rank" else logit if mode.value == "Logit rank" else logit - jacobian
        with output:
            output.clear_output(wait=True)
            display(Markdown(f"**Question:** where is `{concept.value}` most available for verbal report?"))
            display(rank_heatmap(values, artifact.manifest["layers"], artifact.manifest["positions"], mode.value))
            display(rank_trajectories(jacobian[:, position.value], logit[:, position.value], artifact.manifest["layers"], concept.value))
            display(residual_projection(artifact.metrics["residual_projection"], artifact.manifest["layers"], artifact.manifest["positions"]))

    for control in (concept, position, mode):
        control.observe(redraw, names="value")
    display(widgets.VBox([widgets.HBox([concept, position]), mode, output]))
    redraw()


def display_dictionary_widget() -> None:
    active = widgets.IntSlider(min=0, max=7, value=0, description="Direction:")
    coefficient = widgets.FloatSlider(min=0, max=1.5, step=0.1, value=1.0, description="Coefficient:")
    output = widgets.Output()

    def redraw(_: Any = None) -> None:
        with output:
            output.clear_output(wait=True)
            display(dictionary_geometry(active.value, coefficient.value))

    active.observe(redraw, names="value")
    coefficient.observe(redraw, names="value")
    display(widgets.VBox([active, coefficient, output]))
    redraw()
