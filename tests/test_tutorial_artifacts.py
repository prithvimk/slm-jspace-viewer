from pathlib import Path

import numpy as np
import pytest

from tutorials.lib.artifacts import TutorialArtifact, TutorialArtifactError, synthetic_artifact
from tutorials.lib.visuals import dictionary_geometry, rank_heatmap


def test_synthetic_tutorial_artifact_round_trip(tmp_path: Path) -> None:
    artifact = synthetic_artifact(tmp_path / "tutorial")
    reopened = TutorialArtifact.open(tmp_path / "tutorial")
    assert artifact.concepts == ["Paris", "France", "capital", "answer"]
    assert reopened.metrics["jacobian_rank"].shape == (4, 4, 4)
    assert reopened.manifest["synthetic"] is True


def test_tutorial_visuals_are_plotly_figures() -> None:
    figure = rank_heatmap(np.ones((2, 2)), [0, 1], ["a", "b"], "test")
    assert len(figure.data) == 1
    assert len(dictionary_geometry(2, 0.7).data) == 9


def test_tutorial_artifact_requires_complete_payload(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text('{"tutorial_schema_version": 1}', encoding="utf-8")
    with pytest.raises(TutorialArtifactError, match="needs manifest"):
        TutorialArtifact.open(tmp_path)
