from __future__ import annotations


def tokenize(text: str) -> list[str]:
    return [token for token in text.lower().split() if token]
