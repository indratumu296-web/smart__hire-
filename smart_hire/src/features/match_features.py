from __future__ import annotations


def overlap_score(resume_text: str, job_text: str, skills: list[str]) -> float:
    if not skills:
        return 0.0
    job_text = job_text.lower()
    return sum(skill in job_text for skill in skills) / len(skills)
