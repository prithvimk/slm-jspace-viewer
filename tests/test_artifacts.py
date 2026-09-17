import json
from pathlib import Path

import numpy as np
import pytest

from slm_jspace.artifacts import ArtifactError, ExperimentArtifact, write_artifact


def _manifest() -> dict:
    return {"artifact_id": "synthetic", "model": {"id": "test"}, "tokens": [{"id": 0, "text": "a", "position": 0}], "selected_layers": [0, 1], "selected_positions": [0, 1]}


def test_artifact_round_trip_and_lazy_cell_read(tmp_path: Path) -> None:
    residuals = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    jacobian = np.array([[[1, 3, 2, 0], [0, 1, 4, 2]], [[2, 0, 3, 1], [4, 0, 2, 1]]], dtype=np.float32)
    artifact = write_artifact(tmp_path / "artifact", _manifest(), residuals, jacobian, jacobian - 1)
    np.testing.assert_array_equal(artifact.scores("jacobian_lens", 0, 1), jacobian[0, 1])
    np.testing.assert_array_equal(artifact.rank("jacobian_lens", 2), np.array([[2, 1], [1, 2]]))
    reopened = ExperimentArtifact.open(tmp_path / "artifact")
    assert reopened.manifest["schema_version"] == 1


def test_artifact_refuses_overwrite(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    write_artifact(root, _manifest(), np.zeros((1, 1, 1)), np.zeros((1, 1, 2)), np.zeros((1, 1, 2)))
    with pytest.raises(ArtifactError, match="overwrite"):
        write_artifact(root, _manifest(), np.zeros((1, 1, 1)), np.zeros((1, 1, 2)), np.zeros((1, 1, 2)))


def test_missing_schema_is_reported(tmp_path: Path) -> None:
    root = tmp_path / "bad"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
    (root / "arrays.zarr").mkdir()
    with pytest.raises(ArtifactError, match="Unsupported"):
        ExperimentArtifact.open(root)
