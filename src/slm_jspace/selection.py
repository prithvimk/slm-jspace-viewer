"""Layer and token-position selection helpers."""
from __future__ import annotations


def evenly_spaced_layers(layer_total: int, requested: int = 8) -> list[int]:
    if layer_total <= 0 or requested <= 0:
        raise ValueError("layer_total and requested must be positive")
    if requested >= layer_total:
        return list(range(layer_total))
    # Rounding yields deterministic coverage of both earliest and latest blocks.
    return sorted({round(index * (layer_total - 1) / (requested - 1)) for index in range(requested)}) if requested > 1 else [0]


def jacobian_source_layers(layer_total: int, requested: int = 8) -> list[int]:
    """Choose source blocks strictly before the final target block.

    ``jlens.fit`` transports every source residual into the basis of its final
    block, so its source layer indices must satisfy ``source < target``.
    """
    if layer_total < 2:
        raise ValueError("A Jacobian lens requires at least two transformer blocks")
    return evenly_spaced_layers(layer_total - 1, requested)


def resolve_positions(prompt_length: int, total_length: int, configured: list[int]) -> list[int]:
    positions: list[int] = []
    for position in configured:
        absolute = prompt_length + position if position < 0 else position
        if not 0 <= absolute < prompt_length:
            raise ValueError(f"Prompt position {position} is outside prompt length {prompt_length}")
        positions.append(absolute)
    positions.extend(range(prompt_length, total_length))
    return list(dict.fromkeys(positions))
