"""Lazy Hugging Face model integration and residual capture."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import ModelConfig


def _torch_dtype(name: str) -> Any:
    import torch

    try:
        return getattr(torch, name)
    except AttributeError as error:
        raise ValueError(f"Unsupported torch dtype: {name}") from error


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    config: ModelConfig

    @property
    def layer_total(self) -> int:
        value = getattr(self.model.config, "num_hidden_layers", None)
        if not isinstance(value, int):
            raise TypeError("Model config does not expose num_hidden_layers")
        return value

    def render_messages(self, messages: list[dict[str, str]]) -> str:
        if self.config.system_prompt:
            messages = [{"role": "system", "content": self.config.system_prompt}, *messages]
        if self.config.use_chat_template:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return "\n".join(item["content"] for item in messages)


def load_model(config: ModelConfig) -> LoadedModel:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.tokenizer_revision)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id, revision=config.revision, dtype=_torch_dtype(config.dtype), device_map=config.device_map,
    ).eval()
    return LoadedModel(model=model, tokenizer=tokenizer, config=config)


def jlens_adapter(loaded: LoadedModel) -> Any:
    import jlens

    return jlens.from_hf(loaded.model, loaded.tokenizer)


def capture_residuals(loaded: LoadedModel, token_ids: Any, layers: list[int]) -> tuple[Any, Any]:
    """Return residuals in source-layer order and the standard model logits."""
    import torch

    device = next(loaded.model.parameters()).device
    inputs = {"input_ids": token_ids.to(device), "attention_mask": torch.ones_like(token_ids, device=device)}
    with torch.inference_mode():
        outputs = loaded.model(**inputs, output_hidden_states=True, use_cache=False)
    # Transformers exposes embedding state at index zero, then the output of each block.
    residuals = torch.stack([outputs.hidden_states[layer + 1][0] for layer in layers]).float()
    return residuals, outputs.logits[0].float()


def greedy_generate(loaded: LoadedModel, prompt_ids: Any, max_new_tokens: int) -> Any:
    import torch

    device = next(loaded.model.parameters()).device
    with torch.inference_mode():
        result = loaded.model.generate(
            input_ids=prompt_ids.to(device), do_sample=False, max_new_tokens=max_new_tokens,
            pad_token_id=loaded.tokenizer.pad_token_id or loaded.tokenizer.eos_token_id,
        )
    return result[0].detach().cpu()
