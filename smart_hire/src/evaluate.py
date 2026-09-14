from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score


def classification_metrics(
	expected: Iterable[str],
	predicted: Iterable[str],
) -> dict[str, float]:
	"""Return stable summary metrics for role predictions."""
	expected_values = list(expected)
	predicted_values = list(predicted)
	if len(expected_values) != len(predicted_values):
		raise ValueError("expected and predicted must contain the same number of labels.")
	if not expected_values:
		raise ValueError("At least one label is required for evaluation.")
	return {
		"accuracy": float(accuracy_score(expected_values, predicted_values)),
		"f1_weighted": float(
			f1_score(expected_values, predicted_values, average="weighted", zero_division=0)
		),
	}


def classification_report_text(
	expected: Iterable[str],
	predicted: Iterable[str],
) -> str:
	"""Return scikit-learn's readable per-class classification report."""
	expected_values = list(expected)
	predicted_values = list(predicted)
	if len(expected_values) != len(predicted_values):
		raise ValueError("expected and predicted must contain the same number of labels.")
	return classification_report(expected_values, predicted_values, zero_division=0)


def ranking_metrics(
	ranked_jobs: pd.DataFrame,
	relevant_jobs: Sequence[int] | set[int],
	*,
	k: int = 5,
) -> dict[str, float]:
	"""Measure precision and recall for the first ``k`` ranked job rows.

	``relevant_jobs`` contains the original dataframe indexes of relevant jobs.
	"""
	if k < 1:
		raise ValueError("k must be at least 1.")
	relevant = set(relevant_jobs)
	if not relevant:
		raise ValueError("At least one relevant job index is required.")
	top_indexes = set(ranked_jobs.head(k).index)
	hits = len(top_indexes & relevant)
	return {
		"precision_at_k": hits / min(k, len(ranked_jobs)),
		"recall_at_k": hits / len(relevant),
	}
