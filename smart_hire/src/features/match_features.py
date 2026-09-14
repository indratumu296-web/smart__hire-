from __future__ import annotations

import re


DEFAULT_SKILLS = {
    "python", "sql", "java", "javascript", "typescript", "react", "excel", "tableau",
    "power bi", "pandas", "numpy", "scikit-learn", "machine learning", "statistics",
    "data analysis", "communication", "leadership", "recruitment", "payroll", "accounting",
    "audit", "finance", "marketing", "seo", "sales", "crm", "git", "aws", "azure",
    "docker", "rest api", "agile", "project management",
}


def extract_skills(text: str, skills: set[str] | None = None) -> list[str]:
    normalized = re.sub(r"\s+", " ", str(text).lower())
    vocabulary = skills or DEFAULT_SKILLS
    return sorted(skill for skill in vocabulary if re.search(rf"(?<!\w){re.escape(skill)}(?!\w)", normalized))

def overlap_score(resume_text: str, job_text: str, skills: list[str]) -> float:
    if not skills:
        return 0.0
    job_text = job_text.lower()
    return sum(skill in job_text for skill in skills) / len(skills)


def skill_gap(resume_text: str, job_text: str, skills: set[str] | None = None) -> list[str]:
    resume_skills = set(extract_skills(resume_text, skills))
    job_skills = set(extract_skills(job_text, skills))
    return sorted(job_skills - resume_skills)
