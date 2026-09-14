from __future__ import annotations

from collections.abc import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def train_fit_predictor(
	texts: Iterable[str],
	labels: Iterable[str],
	*,
	max_features: int = 5000,
	min_df: int = 2,
) -> Pipeline:
	"""Train the notebook's TF-IDF role-fit predictor."""
	predictor = Pipeline([
		("tfidf", TfidfVectorizer(stop_words="english", max_features=max_features, min_df=min_df)),
		("classifier", LogisticRegression(max_iter=500)),
	])
	predictor.fit(list(texts), list(labels))
	return predictor


def predict_fit(model: Pipeline, resume_text: str) -> str:
	"""Predict the best-fitting role for one resume."""
	return str(model.predict([resume_text])[0])
