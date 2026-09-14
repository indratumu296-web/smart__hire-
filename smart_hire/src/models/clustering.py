from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def cluster_texts(
	texts: Iterable[str],
	*,
	n_clusters: int = 6,
	max_features: int = 2500,
	min_df: int = 2,
	random_state: int = 42,
) -> tuple[list[int], TfidfVectorizer, KMeans]:
	"""Cluster text records and return labels plus fitted artifacts."""
	values = [str(text) for text in texts]
	if not values:
		raise ValueError("At least one text record is required.")
	if n_clusters < 1 or n_clusters > len(values):
		raise ValueError("n_clusters must be between 1 and the number of texts.")
	vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features, min_df=min_df)
	matrix = vectorizer.fit_transform(values)
	model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
	return model.fit_predict(matrix).tolist(), vectorizer, model


def cluster_jobs(
	jobs: pd.DataFrame,
	*,
	text_columns: tuple[str, ...] = ("job_title", "job_description"),
	n_clusters: int = 6,
) -> tuple[pd.DataFrame, TfidfVectorizer, KMeans]:
	"""Add a cluster label to a job frame using its text columns."""
	missing = [column for column in text_columns if column not in jobs.columns]
	if missing:
		raise ValueError(f"Missing job text columns: {', '.join(missing)}")
	texts = jobs[list(text_columns)].fillna("").astype(str).agg(" ".join, axis=1)
	labels, vectorizer, model = cluster_texts(texts, n_clusters=n_clusters)
	result = jobs.copy()
	result["cluster"] = labels
	return result, vectorizer, model


def top_cluster_terms(
	vectorizer: TfidfVectorizer,
	model: KMeans,
	*,
	limit: int = 10,
) -> dict[int, list[str]]:
	"""Return the highest-weight terms describing each cluster."""
	terms = vectorizer.get_feature_names_out()
	return {
		cluster_id: terms[center.argsort()[-limit:][::-1]].tolist()
		for cluster_id, center in enumerate(model.cluster_centers_)
	}
