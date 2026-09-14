from src.features.text_features import tokenize
from src.features.match_features import extract_skills, skill_gap
from src.models.recommender import recommend_jobs
import pandas as pd


def test_tokenize_returns_lowercase_tokens():
    assert tokenize("Python SQL") == ["python", "sql"]


def test_skill_gap_returns_job_skills_missing_from_resume():
    assert skill_gap("Python", "Python SQL") == ["sql"]


def test_recommender_ranks_similar_jobs_first():
    jobs = pd.DataFrame(
        {
            "job_title": ["Data analyst", "Chef"],
            "job_description": ["Python SQL analytics", "Kitchen food preparation"],
        }
    )
    result = recommend_jobs("Python SQL data analysis", jobs, top_n=2)
    assert result.iloc[0]["job_title"] == "Data analyst"
    assert result.iloc[0]["match_score"] > result.iloc[1]["match_score"]
