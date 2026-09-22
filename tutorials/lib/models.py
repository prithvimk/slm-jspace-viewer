"""Small, explicit model helpers for interactive Colab cells."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass
class TutorialModel:
    model: Any
    tokenizer: Any
    model_id: str


def load_tutorial_model(model_id: str = DEFAULT_MODEL_ID) -> TutorialModel:
    """Load the anonymous Qwen default using the active Colab-friendly dtype."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    has_cuda = torch.cuda.is_available()
    # T4 is a Turing GPU (compute capability 7.5) and accelerates FP16, not
    # BF16. Ampere and later GPUs natively accelerate BF16 Tensor Core work.
    dtype = (
        torch.bfloat16
        if has_cuda and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
        if has_cuda
        else torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="auto" if has_cuda else None,
    ).eval()
    return TutorialModel(model=model, tokenizer=tokenizer, model_id=model_id)


def render_and_generate(loaded: TutorialModel, prompt: str, max_new_tokens: int = 48, seed: int = 42) -> tuple[str, list[str]]:
    """Return a visible rationale/output and its decoded tokens."""
    import torch

    torch.manual_seed(seed)
    messages = [{"role": "user", "content": prompt}]
    rendered = loaded.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = loaded.tokenizer(rendered, return_tensors="pt").input_ids
    device = next(loaded.model.parameters()).device
    with torch.inference_mode():
        generated = loaded.model.generate(encoded.to(device), do_sample=False, max_new_tokens=max_new_tokens)[0].cpu()
    decoded = loaded.tokenizer.decode(generated, skip_special_tokens=True)
    return decoded, [loaded.tokenizer.decode([int(token)]) for token in generated]


def residual_trace(loaded: TutorialModel, prompt: str) -> tuple[list[int], Any]:
    """Capture all block residuals for one prompt; cap prompt length in notebooks."""
    import torch

    encoded = loaded.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).input_ids
    device = next(loaded.model.parameters()).device
    with torch.inference_mode():
        outputs = loaded.model(input_ids=encoded.to(device), output_hidden_states=True, use_cache=False)
    residuals = torch.stack([state[0].float().cpu() for state in outputs.hidden_states[1:]])
    return encoded[0].tolist(), residuals
