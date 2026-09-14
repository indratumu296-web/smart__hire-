from __future__ import annotations

import re

from sklearn.feature_extraction.text import TfidfVectorizer


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z0-9+#.-]*", normalize_text(text))


def make_tfidf_vectorizer(*, max_features: int = 10000, min_df: int = 1) -> TfidfVectorizer:
    return TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=min_df,
    )
