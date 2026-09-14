from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_classifier(
    texts: Iterable[str],
    labels: Iterable[str],
    *,
    max_features: int = 5000,
    min_df: int = 1,
) -> Pipeline:
    """Train a TF-IDF resume role classifier."""
    text_values = list(texts)
    label_values = list(labels)
    if len(text_values) != len(label_values):
        raise ValueError("texts and labels must contain the same number of records.")
    if len(set(label_values)) < 2:
        raise ValueError("At least two role categories are required to train the classifier.")
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=max_features, min_df=min_df)),
        ("classifier", LogisticRegression(max_iter=500)),
    ])
    model.fit(text_values, label_values)
    return model


def save_classifier(model: Pipeline, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_classifier(path: str | Path) -> Pipeline:
    return joblib.load(path)


def predict_roles(model: Pipeline, texts: Iterable[str]) -> list[str]:
    """Predict role labels for one or more resume texts."""
    return model.predict(list(texts)).tolist()


def predict_role(resume_text: str, model: Pipeline | None = None) -> str:
    if model is not None:
        return str(model.predict([resume_text])[0])
    from src.models.recommender import infer_role

    return infer_role(resume_text)
