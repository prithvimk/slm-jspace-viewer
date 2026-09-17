"""Versioned artifact storage, lazy reads, and rank calculations."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import SCHEMA_VERSION


class ArtifactError(ValueError):
    pass


@dataclass
class ExperimentArtifact:
    root: Path
    manifest: dict[str, Any]
    arrays: Any

    @classmethod
    def open(cls, root: str | Path) -> ExperimentArtifact:
        import zarr

        root = Path(root)
        manifest_path = root / "manifest.json"
        arrays_path = root / "arrays.zarr"
        if not manifest_path.is_file() or not arrays_path.is_dir():
            raise ArtifactError("Artifact must contain manifest.json and arrays.zarr")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ArtifactError(f"Unsupported artifact schema {manifest.get('schema_version')!r}")
        arrays = zarr.open_group(str(arrays_path), mode="r")
        for name in ("residuals", "scores/jacobian_lens", "scores/logit_lens"):
            if name not in arrays:
                raise ArtifactError(f"Artifact missing array {name}")
        return cls(root=root, manifest=manifest, arrays=arrays)

    def scores(self, method: str, layer_index: int, position_index: int) -> np.ndarray:
        if method not in ("jacobian_lens", "logit_lens"):
            raise ArtifactError(f"Unknown method {method}")
        return np.asarray(self.arrays[f"scores/{method}"][layer_index, position_index, :])

    def rank(self, method: str, token_id: int) -> np.ndarray:
        scores = self.arrays[f"scores/{method}"]
        pinned = np.asarray(scores[:, :, token_id])
        # Rank 1 is highest score; read one layer/position vector at a time.
        return np.asarray([[1 + np.count_nonzero(np.asarray(scores[layer, pos, :]) > pinned[layer, pos])
                            for pos in range(scores.shape[1])] for layer in range(scores.shape[0])])


def _create_array(group: Any, name: str, data: np.ndarray, chunks: tuple[int, ...]) -> None:
    if hasattr(group, "create_array"):
        group.create_array(name, data=data, chunks=chunks)
    else:  # Zarr v2 compatibility for durable artifacts.
        group.create_dataset(name, data=data, chunks=chunks, overwrite=True)


def write_artifact(root: str | Path, manifest: dict[str, Any], residuals: np.ndarray,
                   jacobian_scores: np.ndarray, logit_scores: np.ndarray) -> ExperimentArtifact:
    import zarr

    root = Path(root)
    if root.exists():
        raise ArtifactError(f"Refusing to overwrite existing artifact: {root}")
    root.mkdir(parents=True)
    complete_manifest = {"schema_version": SCHEMA_VERSION, **manifest}
    (root / "manifest.json").write_text(json.dumps(complete_manifest, indent=2, sort_keys=True), encoding="utf-8")
    group = zarr.open_group(str(root / "arrays.zarr"), mode="w")
    _create_array(group, "residuals", np.asarray(residuals, dtype=np.float32), (1, 1, residuals.shape[-1]))
    chunks = (1, 1, jacobian_scores.shape[-1])
    _create_array(group, "scores/jacobian_lens", np.asarray(jacobian_scores, dtype=np.float32), chunks)
    _create_array(group, "scores/logit_lens", np.asarray(logit_scores, dtype=np.float32), chunks)
    # Reserved for lossless sparse J-space arrays in the cross-model follow-on.
    group.require_group("jspace")
    return ExperimentArtifact.open(root)
