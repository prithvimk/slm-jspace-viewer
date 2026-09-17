"""Versioned, validated YAML configuration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from . import SCHEMA_VERSION


class ConfigError(ValueError):
    """Raised for an invalid or unsupported configuration."""


def _read(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a YAML object")
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ConfigError(f"{path} must use schema_version {SCHEMA_VERSION}")
    return raw


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    revision: str
    tokenizer_revision: str
    dtype: str
    device_map: str
    use_chat_template: bool
    system_prompt: str | None

    @classmethod
    def load(cls, path: str | Path) -> ModelConfig:
        raw = _read(path)
        required = ("model_id", "revision", "tokenizer_revision", "dtype", "device_map")
        missing = [key for key in required if not raw.get(key)]
        if missing:
            raise ConfigError(f"Model config missing: {', '.join(missing)}")
        return cls(
            model_id=str(raw["model_id"]), revision=str(raw["revision"]),
            tokenizer_revision=str(raw["tokenizer_revision"]), dtype=str(raw["dtype"]),
            device_map=str(raw["device_map"]), use_chat_template=bool(raw.get("use_chat_template", True)),
            system_prompt=raw.get("system_prompt"),
        )


@dataclass(frozen=True)
class LensFitConfig:
    model: Path
    output_dir: Path
    dataset: str
    dataset_config: str | None
    split: str
    seed: int
    prompt_count: int
    sequence_length: int
    layer_count: int
    dim_batch: int

    @classmethod
    def load(cls, path: str | Path) -> LensFitConfig:
        path = Path(path)
        raw = _read(path)
        config = cls(
            model=(path.parent / raw["model"]).resolve(), output_dir=(path.parent / raw["output_dir"]).resolve(),
            dataset=str(raw["dataset"]), dataset_config=raw.get("dataset_config"), split=str(raw.get("split", "train")),
            seed=int(raw["seed"]), prompt_count=int(raw["prompt_count"]),
            sequence_length=int(raw["sequence_length"]), layer_count=int(raw.get("layer_count", 8)),
            dim_batch=int(raw.get("dim_batch", 1)),
        )
        if config.prompt_count <= 0 or config.sequence_length <= 1 or config.layer_count <= 0 or config.dim_batch <= 0:
            raise ConfigError("prompt_count, sequence_length, layer_count, and dim_batch must be positive")
        return config


@dataclass(frozen=True)
class ExperimentConfig:
    model: Path
    lens_path: Path
    output_dir: Path
    name: str
    messages: list[dict[str, str]]
    prompt_positions: list[int]
    layer_count: int
    max_new_tokens: int

    @classmethod
    def load(cls, path: str | Path) -> ExperimentConfig:
        path = Path(path)
        raw = _read(path)
        messages = raw.get("messages")
        if not isinstance(messages, list) or not messages or not all(isinstance(item, dict) and item.get("role") and item.get("content") for item in messages):
            raise ConfigError("messages must be a non-empty list of role/content objects")
        positions = [int(position) for position in raw.get("prompt_positions", [])]
        if not positions:
            raise ConfigError("prompt_positions must explicitly select at least one prompt position")
        config = cls(
            model=(path.parent / raw["model"]).resolve(), lens_path=(path.parent / raw["lens_path"]).resolve(),
            output_dir=(path.parent / raw["output_dir"]).resolve(), name=str(raw["name"]), messages=messages,
            prompt_positions=positions, layer_count=int(raw.get("layer_count", 8)),
            max_new_tokens=int(raw.get("max_new_tokens", 64)),
        )
        if config.layer_count <= 0 or config.max_new_tokens <= 0:
            raise ConfigError("layer_count and max_new_tokens must be positive")
        return config
