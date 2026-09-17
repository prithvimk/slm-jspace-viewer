"""Common Jacobian- and logit-lens readout API."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class LensReadout(Protocol):
    name: str

    def read(self, residuals: Any, layers: list[int]) -> Any:
        """Return float32 scores shaped [layer, position, vocabulary]."""


def _final_norm_and_unembed(model: Any, vectors: Any) -> Any:
    """Use the exact final normalization and unembedding of the source model."""
    backbone = getattr(model, "model", model)
    normalizer = getattr(backbone, "norm", None)
    lm_head = getattr(model, "lm_head", None)
    if normalizer is None or lm_head is None:
        raise RuntimeError("Model must expose model.norm and lm_head for a logit lens")
    return lm_head(normalizer(vectors))


@dataclass
class LogitLens:
    model: Any
    name: str = "logit_lens"

    def read(self, residuals: Any, layers: list[int]) -> Any:
        return _final_norm_and_unembed(self.model, residuals).float()


@dataclass
class JacobianLens:
    model: Any
    jacobians: dict[int, Any]
    name: str = "jacobian_lens"

    @classmethod
    def load(cls, model: Any, path: str | Path) -> JacobianLens:
        import jlens

        fitted = jlens.JacobianLens.load(str(path))
        return cls(model=model, jacobians=fitted.jacobians)

    def read(self, residuals: Any, layers: list[int]) -> Any:
        import torch

        transported = []
        for index, layer in enumerate(layers):
            matrix = self.jacobians.get(layer)
            if matrix is None:
                raise KeyError(f"No fitted Jacobian for layer {layer}")
            transported.append(residuals[index] @ matrix.to(residuals.device, dtype=residuals.dtype).T)
        return _final_norm_and_unembed(self.model, torch.stack(transported)).float()


def top_k(scores: Any, tokenizer: Any, count: int = 20) -> list[dict[str, object]]:
    values, indices = scores.float().topk(count)
    return [
        {"token_id": int(token_id), "token": tokenizer.decode([int(token_id)]), "score": float(value)}
        for token_id, value in zip(indices.cpu().tolist(), values.cpu().tolist(), strict=True)
    ]


def token_id_for_concept(tokenizer: Any, concept: str) -> int:
    ids = tokenizer.encode(concept, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError("Pinned text must encode to exactly one token; use token ID for multi-token concepts")
    return int(ids[0])
