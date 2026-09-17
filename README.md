# Gemma 3 J-Space Viewer

An offline-first experiment pipeline and Streamlit viewer for comparing a
Jacobian lens with a logit-lens baseline on precomputed Gemma 3 traces.

## Setup

```powershell
uv sync --group dev
uv run jspace smoke-test configs/model.gemma-270m.yaml
```

Gemma checkpoints require accepting Google's license on Hugging Face before the
model commands can download weights. The viewer itself needs only an artifact:

```powershell
uv run jspace view artifacts/<experiment-id>
```

## Workflow

1. Run `smoke-test` to verify the model and the `jlens` Hugging Face adapter.
2. Run `fit-lens` with `configs/lens.dev.yaml` (five prompts) or
   `configs/lens.research.yaml` (100 prompts).
3. Run `run-experiment` with a fitted lens and `configs/experiment.yaml`.
4. Inspect the saved artifact with `view`.

Artifacts store dense scores for every configured layer and position so they
can be explored without CUDA or model weights.

The Jacobian lens always reserves the final transformer block as its target;
configured source layers are chosen evenly from the preceding blocks.
