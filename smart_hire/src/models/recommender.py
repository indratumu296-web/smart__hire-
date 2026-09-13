from __future__ import annotations

import re

import pandas as pd

ROLE_TERMS = {
    "Data Analyst": {"python", "sql", "tableau", "power bi", "excel", "analytics", "pandas", "statistics"},
    "Software Engineer": {"python", "java", "javascript", "react", "api", "git", "software", "programming"},
    "Human Resources": {"recruitment", "talent", "hr", "employee", "payroll", "onboarding", "human resources"},
    "Accountant": {"accounting", "audit", "tax", "ledger", "finance", "tally", "reconciliation"},
    "Sales": {"sales", "client", "crm", "lead", "business development", "negotiation", "revenue"},
    "Marketing": {"marketing", "seo", "content", "social media", "campaign", "brand", "digital marketing"},
}


def _terms(text: str) -> set[str]:
    normalized = re.sub(r"[^a-z0-9+#. ]", " ", text.lower())
    return {term for term in ROLE_TERMS.values() for term in term if term in normalized}


def _role(text: str) -> str:
    normalized = text.lower()
    scores = {role: sum(term in normalized for term in terms) for role, terms in ROLE_TERMS.items()}
    return max(scores, key=scores.get) if max(scores.values()) else "Generalist"


def match_jobs(resume_text: str, jobs: pd.DataFrame) -> tuple[pd.DataFrame, str, list[str]]:
    detected = sorted(_terms(resume_text))
    predicted = _role(resume_text)
    if jobs.empty:
        return pd.DataFrame(), predicted, detected

    result = jobs.copy()
    searchable = result.fillna("").astype(str).agg(" ".join, axis=1).str.lower()
    normalized = resume_text.lower()
    result["match_score"] = searchable.map(lambda row: sum(term in row for term in detected) / max(len(detected), 1))
    if predicted != "Generalist":
        result["match_score"] += searchable.map(lambda row: 0.15 if any(term in row for term in ROLE_TERMS[predicted]) else 0)
    return result.sort_values("match_score", ascending=False).reset_index(drop=True), predicted, detected
