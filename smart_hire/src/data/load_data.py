from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_jobs(path: str | Path) -> pd.DataFrame:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)
