"""Streamlit presentation for precomputed artifacts only."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.express as px
import streamlit as st

from .artifacts import ArtifactError, ExperimentArtifact


def _top_k(scores: np.ndarray, vocabulary: list[str], count: int) -> list[dict[str, object]]:
    ids = np.argpartition(scores, -count)[-count:]
    ids = ids[np.argsort(scores[ids])[::-1]]
    return [{"rank": rank + 1, "token_id": int(token), "token": vocabulary[int(token)], "score": float(scores[token])}
            for rank, token in enumerate(ids)]


def run(artifact_path: str | Path) -> None:
    st.set_page_config(page_title="J-Space Viewer", layout="wide")
    st.title("Gemma J-Space Viewer")
    try:
        artifact = ExperimentArtifact.open(artifact_path)
    except ArtifactError as error:
        st.error(str(error))
        return
    manifest = artifact.manifest
    tokens = manifest["tokens"]
    vocabulary_size = artifact.arrays["scores/jacobian_lens"].shape[-1]
    st.caption(f"{manifest['model']['id']} · artifact {manifest['artifact_id']} · offline artifact mode")
    input_mode = st.radio("Pin concept by", ["Token ID", "Token text", "Observed token"], horizontal=True)
    if input_mode == "Token ID":
        token_id = int(st.number_input("Vocabulary token ID", min_value=0, max_value=vocabulary_size - 1, value=0))
    elif input_mode == "Token text":
        concept = st.text_input("Exactly-one-token text")
        matches = [index for index, text in enumerate(manifest.get("vocabulary", [])) if text == concept]
        if not concept:
            st.info("Enter the decoded text of exactly one vocabulary token, or choose Token ID.")
            return
        if len(matches) != 1:
            st.error("That text is not uniquely one token. Pin it by token ID instead.")
            return
        token_id = matches[0]
    else:
        options = {f"{item['position']}: {item['text']!r} ({item['id']})": int(item["id"]) for item in tokens}
        token_id = options[st.selectbox("Observed token", list(options))]
    metric = st.selectbox("Heatmap", ["Jacobian rank", "Logit rank", "Jacobian rank advantage"])
    jacobian_rank = artifact.rank("jacobian_lens", token_id)
    logit_rank = artifact.rank("logit_lens", token_id)
    values = jacobian_rank if metric == "Jacobian rank" else logit_rank if metric == "Logit rank" else logit_rank - jacobian_rank
    labels = [f"{tokens[position]['position']}: {tokens[position]['text']!r}" for position in manifest["selected_positions"]]
    figure = px.imshow(values, x=labels, y=[str(layer) for layer in manifest["selected_layers"]],
                      color_continuous_scale="Viridis", aspect="auto", labels={"x": "token position", "y": "source layer", "color": metric})
    st.plotly_chart(figure, use_container_width=True)
    left, right = st.columns(2)
    with left:
        layer_index = st.selectbox("Layer", range(len(manifest["selected_layers"])), format_func=lambda index: str(manifest["selected_layers"][index]))
    with right:
        position_index = st.selectbox("Position", range(len(manifest["selected_positions"])), format_func=lambda index: labels[index])
    columns = st.columns(2)
    for column, method, title in zip(columns, ("jacobian_lens", "logit_lens"), ("Jacobian lens", "Logit lens"), strict=True):
        with column:
            st.subheader(title)
            scores = artifact.scores(method, layer_index, position_index)
            st.dataframe(_top_k(scores, manifest.get("vocabulary", []), min(20, scores.size)), use_container_width=True)
    with st.expander("Full provenance"):
        st.json(manifest)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: streamlit run viewer.py -- <artifact-directory>")
    run(sys.argv[1])
