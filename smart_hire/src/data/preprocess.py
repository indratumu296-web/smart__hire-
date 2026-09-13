from __future__ import annotations

import re


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()
