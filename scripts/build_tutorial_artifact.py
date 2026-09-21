"""Derive a compact tutorial artifact from a full local experiment artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from slm_jspace.artifacts import ExperimentArtifact
from tutorials.lib.artifacts import TUTORIAL_SCHEMA_VERSION


def _top_k(scores: np.ndarray, vocabulary: list[str], count: int) -> list[dict[str, object]]:
    indices = np.argsort(scores)[-count:][::-1]
    return [
        {"token_id": int(index), "token": vocabulary[int(index)], "score": float(scores[index])}
        for index in indices
    ]


def build(source: Path, destination: Path, concepts: list[int], top_k: int) -> None:
    artifact = ExperimentArtifact.open(source)
    destination.mkdir(parents=True, exist_ok=True)
    layers = artifact.manifest["selected_layers"]
    positions = artifact.manifest["selected_positions"]
    vocabulary = artifact.manifest.get("vocabulary")
    if not vocabulary:
        raise ValueError("Full artifact must include decoded vocabulary for tutorial export")
    shape = (len(concepts), len(layers), len(positions))
    metric_names = ("jacobian_rank", "logit_rank", "jacobian_score", "logit_score")
    metrics = {name: np.zeros(shape, dtype=np.float32) for name in metric_names}
    topk: dict[str, object] = {}
    for layer_index in range(len(layers)):
        for position_index in range(len(positions)):
            jacobian = artifact.scores("jacobian_lens", layer_index, position_index)
            logit = artifact.scores("logit_lens", layer_index, position_index)
            topk[f"{layer_index}:{position_index}"] = {
                "jacobian_lens": _top_k(jacobian, vocabulary, top_k),
                "logit_lens": _top_k(logit, vocabulary, top_k),
            }
            for concept_index, token_id in enumerate(concepts):
                metrics["jacobian_score"][concept_index, layer_index, position_index] = jacobian[token_id]
                metrics["logit_score"][concept_index, layer_index, position_index] = logit[token_id]
                metrics["jacobian_rank"][concept_index, layer_index, position_index] = 1 + np.count_nonzero(jacobian > jacobian[token_id])
                metrics["logit_rank"][concept_index, layer_index, position_index] = 1 + np.count_nonzero(logit > logit[token_id])
    residuals = np.asarray(artifact.arrays["residuals"])
    flattened = residuals.reshape(-1, residuals.shape[-1])
    centered = flattened - flattened.mean(axis=0, keepdims=True)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    metrics["residual_projection"] = (centered @ right[:2].T).reshape(*residuals.shape[:2], 2).astype(np.float32)
    manifest = {
        "tutorial_schema_version": TUTORIAL_SCHEMA_VERSION,
        "source_artifact_id": artifact.manifest.get("artifact_id"),
        "model": artifact.manifest["model"],
        "layers": layers,
        "positions": [artifact.manifest["tokens"][index]["text"] for index in positions],
        "concepts": [{"label": vocabulary[token_id], "token_id": token_id} for token_id in concepts],
        "top_k": top_k,
        "provenance": artifact.manifest,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    np.savez_compressed(destination / "metrics.npz", **metrics)
    (destination / "topk.json").write_text(json.dumps(topk, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--concept-token", type=int, action="append", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    arguments = parser.parse_args()
    build(arguments.source, arguments.destination, arguments.concept_token, arguments.top_k)
