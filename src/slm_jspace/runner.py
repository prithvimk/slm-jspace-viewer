"""Lens fitting and generate-first experiment extraction."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .artifacts import write_artifact
from .config import ExperimentConfig, LensFitConfig, ModelConfig
from .lenses import JacobianLens, LogitLens
from .modeling import capture_residuals, greedy_generate, jlens_adapter, load_model
from .selection import evenly_spaced_layers, resolve_positions


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _versions() -> dict[str, str]:
    packages = ("torch", "transformers", "datasets", "zarr", "jlens", "slm-jspace-viewer")
    return {package: importlib.metadata.version(package) for package in packages if _has_package(package)}


def _has_package(package: str) -> bool:
    try:
        importlib.metadata.version(package)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def smoke_test(model_path: str | Path) -> dict[str, Any]:
    config = ModelConfig.load(model_path)
    loaded = load_model(config)
    rendered = loaded.render_messages([{"role": "user", "content": "Reply with the word: ready"}])
    ids = loaded.tokenizer(rendered, return_tensors="pt").input_ids
    layers = evenly_spaced_layers(loaded.layer_total)
    residuals, logits = capture_residuals(loaded, ids, layers)
    adapter = jlens_adapter(loaded)
    return {
        "model_id": config.model_id, "layer_total": loaded.layer_total, "selected_layers": layers,
        "input_tokens": int(ids.shape[-1]), "residual_shape": list(residuals.shape),
        "logits_shape": list(logits.shape), "adapter_type": type(adapter).__name__,
    }


def _fineweb_prompts(config: LensFitConfig, tokenizer: Any) -> tuple[list[str], str]:
    from datasets import load_dataset

    dataset = load_dataset(config.dataset, config.dataset_config, split=config.split, streaming=True)
    shuffled = dataset.shuffle(seed=config.seed, buffer_size=max(1_000, config.prompt_count * 10))
    prompts: list[str] = []
    for record in shuffled:
        text = str(record.get("text", "")).strip()
        if not text:
            continue
        ids = tokenizer(text, truncation=True, max_length=config.sequence_length).input_ids
        if len(ids) >= 2:
            prompts.append(tokenizer.decode(ids, skip_special_tokens=False))
        if len(prompts) == config.prompt_count:
            break
    if len(prompts) != config.prompt_count:
        raise RuntimeError(f"FineWeb stream produced {len(prompts)} usable prompts, expected {config.prompt_count}")
    corpus_hash = hashlib.sha256("\n".join(prompts).encode("utf-8")).hexdigest()
    return prompts, corpus_hash


def fit_lens(config_path: str | Path) -> Path:
    """Fit with Anthropic's estimator and persist its native immutable lens file."""
    import jlens

    config = LensFitConfig.load(config_path)
    model_config = ModelConfig.load(config.model)
    loaded = load_model(model_config)
    prompts, corpus_hash = _fineweb_prompts(config, loaded.tokenizer)
    layers = evenly_spaced_layers(loaded.layer_total, config.layer_count)
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    lens_path = output / "lens.pt"
    # `jlens.fit` is deliberately delegated to the reference implementation.
    fitted = jlens.fit(
        jlens_adapter(loaded), prompts=prompts, source_layers=layers,
        dim_batch=config.dim_batch, max_seq_len=config.sequence_length,
    )
    fitted.save(str(lens_path))
    metadata = {
        "schema_version": 1, "lens_id": hashlib.sha256(lens_path.read_bytes()).hexdigest(),
        "model": {"id": model_config.model_id, "revision": model_config.revision, "dtype": model_config.dtype},
        "layers": layers, "fit": {"dataset": config.dataset, "dataset_config": config.dataset_config,
        "split": config.split, "seed": config.seed, "prompt_count": config.prompt_count,
        "sequence_length": config.sequence_length, "dim_batch": config.dim_batch, "corpus_sha256": corpus_hash},
        "versions": _versions(), "git_commit": _git_revision(), "created_at": datetime.now(UTC).isoformat(),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return lens_path


def run_experiment(config_path: str | Path) -> Path:
    config = ExperimentConfig.load(config_path)
    model_config = ModelConfig.load(config.model)
    loaded = load_model(model_config)
    rendered = loaded.render_messages(config.messages)
    prompt_ids = loaded.tokenizer(rendered, return_tensors="pt").input_ids
    sequence = greedy_generate(loaded, prompt_ids, config.max_new_tokens)
    layers = evenly_spaced_layers(loaded.layer_total, config.layer_count)
    residuals, _ = capture_residuals(loaded, sequence.unsqueeze(0), layers)
    positions = resolve_positions(prompt_ids.shape[-1], sequence.shape[-1], config.prompt_positions)
    selected = residuals[:, positions, :]
    jacobian = JacobianLens.load(loaded.model, config.lens_path).read(selected, layers)
    logit = LogitLens(loaded.model).read(selected, layers)
    lens_metadata_path = config.lens_path.parent / "metadata.json"
    lens_metadata = json.loads(lens_metadata_path.read_text(encoding="utf-8")) if lens_metadata_path.is_file() else {}
    run_id = f"{config.name}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    destination = config.output_dir / run_id
    tokens = [int(token) for token in sequence.tolist()]
    manifest = {
        "artifact_id": run_id, "created_at": datetime.now(UTC).isoformat(),
        "model": {"id": model_config.model_id, "revision": model_config.revision,
                  "tokenizer_revision": model_config.tokenizer_revision, "dtype": model_config.dtype,
                  "device_map": model_config.device_map, "quantized": False},
        "prompt": {"messages": config.messages, "rendered": rendered, "prompt_length": int(prompt_ids.shape[-1])},
        "generation": {"strategy": "greedy", "max_new_tokens": config.max_new_tokens,
                       "generated_token_count": int(sequence.shape[-1] - prompt_ids.shape[-1])},
        "tokens": [{"id": token, "text": loaded.tokenizer.decode([token]), "position": index,
                    "kind": "prompt" if index < prompt_ids.shape[-1] else "generated"}
                   for index, token in enumerate(tokens)],
        # This small decoded lookup keeps concept pinning offline and avoids loading a tokenizer in the viewer.
        "vocabulary": [loaded.tokenizer.decode([index]) for index in range(int(jacobian.shape[-1]))],
        "selected_layers": layers,
        "normalized_depths": [layer / max(1, loaded.layer_total - 1) for layer in layers],
        "selected_positions": positions,
        "lens": {"path": str(config.lens_path), **lens_metadata}, "versions": _versions(), "git_commit": _git_revision(),
    }
    write_artifact(destination, manifest, selected.cpu().numpy(), jacobian.cpu().numpy(), logit.cpu().numpy())
    return destination
