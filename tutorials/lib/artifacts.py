"""Compact, anonymous-download artifacts for the Colab tutorials."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

TUTORIAL_SCHEMA_VERSION = 1
DEFAULT_ARTIFACT_REPO = "krispri/slm-jspace-tutorial-artifacts"
DEFAULT_ARTIFACT_NAME = "qwen2.5-0.5b/v1"


class TutorialArtifactError(ValueError):
    """Raised when a downloaded tutorial artifact is incomplete or incompatible."""


@dataclass(frozen=True)
class TutorialArtifact:
    """A compact derivative of a full experiment artifact."""

    root: Path
    manifest: dict[str, Any]
    metrics: dict[str, np.ndarray]
    topk: dict[str, Any]

    @property
    def concepts(self) -> list[str]:
        return [str(item["label"]) for item in self.manifest["concepts"]]

    @classmethod
    def open(cls, root: str | Path) -> TutorialArtifact:
        root = Path(root)
        manifest_path = root / "manifest.json"
        metrics_path = root / "metrics.npz"
        topk_path = root / "topk.json"
        if not all(path.is_file() for path in (manifest_path, metrics_path, topk_path)):
            raise TutorialArtifactError("Tutorial artifact needs manifest.json, metrics.npz, and topk.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("tutorial_schema_version") != TUTORIAL_SCHEMA_VERSION:
            raise TutorialArtifactError("Unsupported tutorial artifact schema")
        with np.load(metrics_path, allow_pickle=False) as archive:
            metrics = {name: archive[name] for name in archive.files}
        required = {"jacobian_rank", "logit_rank", "jacobian_score", "logit_score", "residual_projection"}
        missing = required.difference(metrics)
        if missing:
            raise TutorialArtifactError(f"Tutorial artifact metrics missing: {sorted(missing)}")
        shape = metrics["jacobian_rank"].shape
        if len(shape) != 3 or any(metrics[name].shape != shape for name in required - {"residual_projection"}):
            raise TutorialArtifactError("Concept metrics must share [concept, layer, position] shape")
        if shape[0] != len(manifest.get("concepts", [])):
            raise TutorialArtifactError("Concept metadata does not match metric tensors")
        return cls(root=root, manifest=manifest, metrics=metrics, topk=json.loads(topk_path.read_text(encoding="utf-8")))


def download_tutorial_artifact(repo_id: str = DEFAULT_ARTIFACT_REPO, artifact_name: str = DEFAULT_ARTIFACT_NAME, revision: str = "main", cache_dir: str | Path | None = None) -> TutorialArtifact:
    """Download the public tutorial artifact without an HF token."""
    from huggingface_hub import hf_hub_download

    root = Path(cache_dir or ".jspace-tutorial-cache") / artifact_name
    root.mkdir(parents=True, exist_ok=True)
    for filename in ("manifest.json", "metrics.npz", "topk.json"):
        remote = f"{artifact_name}/{filename}"
        downloaded = hf_hub_download(repo_id, remote, repo_type="dataset", revision=revision)
        destination = root / filename
        if not destination.exists():
            destination.write_bytes(Path(downloaded).read_bytes())
    return TutorialArtifact.open(root)


def synthetic_artifact(root: str | Path, seed: int = 7) -> TutorialArtifact:
    """Create a clearly labelled local fallback for widget-development only."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    concepts = ["Paris", "France", "capital", "answer"]
    layers = [0, 7, 15, 22]
    positions = ["capital", "of", "France", "?"]
    shape = (len(concepts), len(layers), len(positions))
    baseline = rng.integers(10, 1000, size=shape)
    advantage = rng.integers(-100, 300, size=shape)
    metrics = {"jacobian_rank": np.maximum(1, baseline - advantage), "logit_rank": baseline, "jacobian_score": rng.normal(size=shape).astype(np.float32), "logit_score": rng.normal(size=shape).astype(np.float32), "residual_projection": rng.normal(size=(len(layers), len(positions), 2)).astype(np.float32)}
    manifest = {"tutorial_schema_version": TUTORIAL_SCHEMA_VERSION, "synthetic": True, "model": {"id": "synthetic-widget-demo"}, "concepts": [{"label": item, "token_id": index} for index, item in enumerate(concepts)], "layers": layers, "positions": positions, "warning": "Synthetic fallback only; do not interpret as a model result."}
    topk = {f"{layer}:{position}": {"jacobian_lens": [{"token": "Paris", "score": 2.1}], "logit_lens": [{"token": "France", "score": 1.6}]} for layer in range(len(layers)) for position in range(len(positions))}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    np.savez_compressed(root / "metrics.npz", **metrics)
    (root / "topk.json").write_text(json.dumps(topk, indent=2), encoding="utf-8")
    return TutorialArtifact.open(root)
