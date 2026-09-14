from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.features.match_features import extract_skills, skill_gap
from src.features.text_features import make_tfidf_vectorizer

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


def infer_role(text: str) -> str:
    normalized = text.lower()
    scores = {role: sum(term in normalized for term in terms) for role, terms in ROLE_TERMS.items()}
    return max(scores, key=scores.get) if max(scores.values()) else "Generalist"


_role = infer_role


@dataclass
class JobRecommender:
    jobs: pd.DataFrame
    vectorizer: object
    matrix: object

    @classmethod
    def fit(cls, jobs: pd.DataFrame) -> "JobRecommender":
        if jobs.empty:
            raise ValueError("At least one job is required to fit the recommender.")
        prepared = jobs.copy().fillna("")
        if "text" not in prepared:
            title = prepared.get("job_title", pd.Series("", index=prepared.index)).astype(str)
            description = prepared.get("job_description", pd.Series("", index=prepared.index)).astype(str)
            prepared["text"] = title + " " + description
        vectorizer = make_tfidf_vectorizer()
        matrix = vectorizer.fit_transform(prepared["text"].astype(str))
        return cls(prepared, vectorizer, matrix)

    def recommend(self, resume_text: str, top_n: int = 10) -> pd.DataFrame:
        if top_n < 1:
            raise ValueError("top_n must be at least 1.")
        query = self.vectorizer.transform([resume_text])
        scores = cosine_similarity(query, self.matrix).ravel()
        result = self.jobs.copy()
        result["match_score"] = scores
        return result.sort_values("match_score", ascending=False).head(top_n).reset_index(drop=True)


def recommend_jobs(resume_text: str, jobs: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Rank jobs by TF-IDF cosine similarity to the resume."""
    if jobs.empty:
        return pd.DataFrame()
    return JobRecommender.fit(jobs).recommend(resume_text, top_n=top_n)


def match_jobs(resume_text: str, jobs: pd.DataFrame, top_n: int = 10) -> tuple[pd.DataFrame, str, list[str]]:
    detected = sorted(_terms(resume_text))
    predicted = infer_role(resume_text)
    return recommend_jobs(resume_text, jobs, top_n=top_n), predicted, detected


def skill_gap_report(resume_text: str, job: pd.Series) -> dict[str, list[str]]:
    job_text = str(job.get("text", ""))
    if not job_text:
        job_text = f"{job.get('job_title', '')} {job.get('job_description', '')}"
    resume_skills = extract_skills(resume_text)
    required_skills = extract_skills(job_text)
    return {
        "resume_skills": resume_skills,
        "required_skills": required_skills,
        "missing_skills": skill_gap(resume_text, job_text),
    }
