from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
RESUME_DATASET = "Resume.csv"
JOB_DATASET_DIRECTORY = "marketing_sample_for_naukri_com-naukri_com_job_data__20201001_20201231__5k_data.csv"


def find_resume_dataset() -> Path | None:
	candidates = [
		RAW_DATA_DIR / RESUME_DATASET,
		REPOSITORY_ROOT / "resume.csv" / "Resume" / RESUME_DATASET,
		REPOSITORY_ROOT / "resume.csv" / "data" / RESUME_DATASET,
	]
	return next((path for path in candidates if path.exists()), None)


def find_job_dataset() -> Path | None:
	candidates = [
		RAW_DATA_DIR / JOB_DATASET_DIRECTORY,
		REPOSITORY_ROOT / JOB_DATASET_DIRECTORY,
		PROJECT_ROOT / JOB_DATASET_DIRECTORY,
	]
	for directory in candidates:
		files = list(directory.glob("*.ldjson")) if directory.is_dir() else []
		if files:
			return files[0]
	return None
