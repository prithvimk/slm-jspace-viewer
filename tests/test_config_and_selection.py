from pathlib import Path

import pytest

from slm_jspace.config import ConfigError, ExperimentConfig
from slm_jspace.selection import evenly_spaced_layers, resolve_positions


def test_evenly_spaced_layers_includes_both_ends() -> None:
    assert evenly_spaced_layers(18, 8) == [0, 2, 5, 7, 10, 12, 15, 17]


def test_positions_include_selected_prompt_and_every_generated() -> None:
    assert resolve_positions(5, 8, [-1, 1]) == [4, 1, 5, 6, 7]


def test_positions_reject_out_of_range_prompt() -> None:
    with pytest.raises(ValueError, match="outside prompt"):
        resolve_positions(5, 8, [5])


def test_experiment_requires_explicit_prompt_positions(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text("schema_version: 1\nmodel: m.yaml\nlens_path: l.pt\noutput_dir: out\nname: x\nmessages:\n  - role: user\n    content: hello\nprompt_positions: []\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="prompt_positions"):
        ExperimentConfig.load(path)


def test_experiment_paths_are_resolved_from_config_location(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text("schema_version: 1\nmodel: model.yaml\nlens_path: lenses/lens.pt\noutput_dir: artifacts\nname: x\nmessages:\n  - role: user\n    content: hello\nprompt_positions: [-1]\n", encoding="utf-8")
    config = ExperimentConfig.load(path)
    assert config.output_dir == (tmp_path / "artifacts").resolve()
    assert config.lens_path == (tmp_path / "lenses/lens.pt").resolve()
