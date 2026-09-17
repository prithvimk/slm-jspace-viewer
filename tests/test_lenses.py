import pytest

from slm_jspace.lenses import token_id_for_concept


class Tokenizer:
    def encode(self, concept: str, add_special_tokens: bool = False) -> list[int]:
        return [5] if concept == "one" else [1, 2]


def test_concept_requires_one_token() -> None:
    assert token_id_for_concept(Tokenizer(), "one") == 5
    with pytest.raises(ValueError, match="exactly one token"):
        token_id_for_concept(Tokenizer(), "many")
