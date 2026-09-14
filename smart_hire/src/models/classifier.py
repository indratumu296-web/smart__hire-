from __future__ import annotations

from collections.abc import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_classifier(
    texts: Iterable[str],
    labels: Iterable[str],
    *,
    max_features: int = 5000,
    min_df: int = 2,
) -> Pipeline:
    """Train a TF-IDF resume role classifier."""
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=max_features, min_df=min_df)),
        ("classifier", LogisticRegression(max_iter=500)),
    ])
    model.fit(list(texts), list(labels))
    return model


def predict_roles(model: Pipeline, texts: Iterable[str]) -> list[str]:
    """Predict role labels for one or more resume texts."""
    return model.predict(list(texts)).tolist()


def predict_role(resume_text: str) -> str:
    from src.models.recommender import _role
    return _role(resume_text)
