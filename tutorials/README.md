# J-Space Colab mini-course

This public mini-course teaches visible reasoning, residual-stream probes, the
logit lens, the Jacobian lens, and J-space fundamentals using small language
models. It is designed for researchers and technically curious readers who
know linear algebra and want a concrete interpretability workflow.

| Notebook | Focus | Login required |
| --- | --- | --- |
| `01_prerequisites_and_visible_reasoning.ipynb` | Chain-of-thought limits, tokenization, generation | No |
| `02_residual_streams_and_logit_lens.ipynb` | Residual streams and logit-lens readouts | No |
| `03_jacobian_lens_and_jspace_explorer.ipynb` | Compact public artifact and J-space geometry | No |
| `04_optional_t4_microfit_and_gemma_comparison.ipynb` | Tiny Qwen fit and optional gated Gemma path | Gemma only |

The default model is `Qwen/Qwen2.5-0.5B-Instruct`. It is downloaded from
Hugging Face and fits the free-T4 target. The tutorial artifact is downloaded
anonymously from the public dataset `krispri/slm-jspace-tutorial-artifacts`.

## Open in Colab

- [01 — visible reasoning](https://colab.research.google.com/github/prithvimk/slm-jspace-viewer/blob/feature/colab-jspace-tutorial/notebooks/01_prerequisites_and_visible_reasoning.ipynb)
- [02 — residual streams and logit lens](https://colab.research.google.com/github/prithvimk/slm-jspace-viewer/blob/feature/colab-jspace-tutorial/notebooks/02_residual_streams_and_logit_lens.ipynb)
- [03 — Jacobian lens and J-space](https://colab.research.google.com/github/prithvimk/slm-jspace-viewer/blob/feature/colab-jspace-tutorial/notebooks/03_jacobian_lens_and_jspace_explorer.ipynb)
- [04 — optional T4 micro-fit](https://colab.research.google.com/github/prithvimk/slm-jspace-viewer/blob/feature/colab-jspace-tutorial/notebooks/04_optional_t4_microfit_and_gemma_comparison.ipynb)

## Interpretation boundary

Visible chain-of-thought is generated text. A Jacobian-lens readout is a probe
of a representation's average future verbal effect. Neither is a verified
transcript of hidden private reasoning. Treat the notebooks as tools for
forming and testing hypotheses, not as a mind-reading interface.

## Publishing artifacts

Generate a compact derivative from a complete experiment artifact, then publish
it with an authenticated maintainer token:

```bash
uv run python scripts/build_tutorial_artifact.py artifacts/<run> tutorial-out --concept-token 123
uv run python scripts/publish_tutorial_artifact.py tutorial-out
```

Learners never need a Hugging Face token for the default three notebooks.
