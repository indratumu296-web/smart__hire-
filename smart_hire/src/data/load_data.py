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
    return pd.DataFrame(rows).fillna("")


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path).fillna("")


def load_resumes(path: str | Path) -> pd.DataFrame:
    frame = load_csv(path)
    required = {"Resume_str", "Category"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Resume dataset is missing columns: {', '.join(sorted(missing))}")
    return frame.loc[:, ["Resume_str", "Category"]].rename(
        columns={"Resume_str": "text", "Category": "label"}
    )
