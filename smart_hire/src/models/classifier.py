from __future__ import annotations


def predict_role(resume_text: str) -> str:
    from src.models.recommender import _role
    return _role(resume_text)
