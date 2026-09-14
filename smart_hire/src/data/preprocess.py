from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from src.data.load_data import load_jobs, load_resumes


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def clean_resumes(resumes: pd.DataFrame) -> pd.DataFrame:
    result = resumes.copy()
    result["text"] = result["text"].map(clean_text)
    result["label"] = result["label"].map(clean_text)
    return result[(result["text"].str.len() > 0) & (result["label"].str.len() > 0)].reset_index(drop=True)


def build_job_corpus(jobs: pd.DataFrame) -> pd.DataFrame:
    result = jobs.copy().fillna("")
    title = result.get("job_title", pd.Series("", index=result.index)).map(clean_text)
    description = result.get("job_description", pd.Series("", index=result.index)).map(clean_text)
    result["text"] = (title + " " + description).map(clean_text)
    return result[result["text"].str.len() > 0].reset_index(drop=True)


def preprocess_datasets(resume_path: str | Path, jobs_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    resumes = clean_resumes(load_resumes(resume_path))
    jobs = build_job_corpus(load_jobs(jobs_path))
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    jobs.to_csv(INTERIM_DATA_DIR / "job_corpus.csv", index=False)
    jobs.to_csv(PROCESSED_DATA_DIR / "jobs_clean.csv", index=False)
    resumes.to_csv(PROCESSED_DATA_DIR / "resumes_clean.csv", index=False)
    return resumes, jobs


if __name__ == "__main__":
    from src.config import find_job_dataset, find_resume_dataset

    resume_path = find_resume_dataset()
    jobs_path = find_job_dataset()
    if resume_path is None or jobs_path is None:
        raise FileNotFoundError("Could not find both the resume CSV and job JSONL dataset.")
    resume_frame, job_frame = preprocess_datasets(resume_path, jobs_path)
    print(f"Wrote {len(resume_frame):,} resumes and {len(job_frame):,} jobs.")
