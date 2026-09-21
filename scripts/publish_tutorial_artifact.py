"""Publish a prepared compact artifact to the public Hugging Face dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Requires a maintainer HF token; never used by learners.")
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--repo-id", default="krispri/slm-jspace-tutorial-artifacts")
    parser.add_argument("--path-in-repo", default="qwen2.5-0.5b/v1")
    arguments = parser.parse_args()
    api = HfApi()
    api.create_repo(arguments.repo_id, repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(
        repo_id=arguments.repo_id,
        repo_type="dataset",
        folder_path=str(arguments.artifact),
        path_in_repo=arguments.path_in_repo,
        commit_message="Publish tutorial artifact",
    )
