# SmartHire — Resume-to-Job Matching & Career Guidance Engine

Classical-ML project (no LLMs). Upload a resume → get matching jobs, a predicted
role category, and a skill-gap report.

## Core scope
1. Resume category classifier — supervised (TF-IDF → Logistic Regression)
2. Job recommender — unsupervised (TF-IDF + cosine similarity, top-N)
3. Skill-gap report — job skills minus resume skills

Optional: fit predictor, clustering + topics, salary regression.

## Requirements
- **Python 3.10+** (developed on 3.12)
- **git** (to clone)
- A free **Kaggle account** (to download the datasets)

## Getting started

### 1. Clone the repo
```
git clone <> SmartHire
cd SmartHire
```
> Replace `<REPO_URL>` with the repository's clone URL (e.g.
> `https://github.com/<user>/SmartHire.git`). If you already have the folder
> locally, just `cd` into it.

### 2. Create a virtual environment and install dependencies
```
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Windows (cmd)
.venv\Scripts\activate.bat
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Download the datasets
The datasets are **not** committed (see `.gitignore`) — download them into
`data/raw/` once:
```
python download_data.py
```
The first run asks for your Kaggle credentials (username + API key from
Kaggle → Settings → **API** → **Create New Token**). This fetches the Resume
and Naukri datasets; LinkedIn is optional.

Filenames are already wired up in `src/config.py`, so you only edit that file if
your downloaded filenames differ. **Full download details + manual alternatives:**
see [`data/DATASETS.md`](data/DATASETS.md).

### 4. Build the cleaned data / job corpus
Merge and clean the raw files into model-ready CSVs (`data/processed/`):
```
python -m src.data.preprocess
```
This writes `data/interim/job_corpus.csv`, `data/processed/jobs_clean.csv`, and
`data/processed/resumes_clean.csv`. Run all commands from the project root so the
`src` package imports resolve.

### 5. Explore & train (notebooks in VS Code)
The notebooks run in **VS Code** (install the **Python** and **Jupyter**
extensions if prompted) — no standalone Jupyter server needed:

1. Open the `SmartHire` folder in VS Code.
2. Open a notebook, e.g. `notebooks/01_eda.ipynb`.
3. Top-right, click **Select Kernel → Python Environments** and choose the venv
   at `.venv\Scripts\python.exe` (shown as `.venv (Python 3.12)`).
4. Run cells with **Shift+Enter**.

Run the notebooks in order **01 → 05**. Each notebook is one module of the
project; move reusable code from a notebook into the matching `src/` file, and
save trained models to `models/` (as `.pkl` via joblib) so the app can load them
without retraining.

> Prefer the classic browser UI instead? Add `jupyter` to `requirements.txt`
> (or `pip install jupyter`) and run `jupyter notebook`. The pinned dependency
> is `ipykernel`, which is all VS Code needs.

### 6. Launch the web app
```
streamlit run app/streamlit_app.py
```
Opens the SmartHire portal in your browser (default <http://localhost:8501>).

> **Current status:** the data pipeline (steps 3–4) is fully working. The model
> code in `src/models/` (`classifier.py`, `recommender.py`) and the Streamlit UI
> in `app/streamlit_app.py` are still stubs — build them via the notebooks first
> (step 5). Until then the app page will be empty.

## Project structure
```
smarthire/
├── README.md                          
├── requirements.txt                   
├── .gitignore                       
│
├── data/                             
│   ├── raw/                           
│   ├── interim/                      
│   └── processed/                    
│
├── notebooks/                        
│   ├── 01_eda.ipynb                  
│   ├── 02_resume_classifier.ipynb     
│   ├── 03_recommender.ipynb          
│   ├── 04_clustering_topics.ipynb    
│   └── 05_fit_predictor.ipynb       
│
├── src/                              
│   ├── config.py                    
│   ├── data/
│   │   ├── load_data.py            
│   │   └── preprocess.py             
│   ├── features/
│   │   ├── text_features.py         
│   │   └── match_features.py        
│   ├── models/
│   │   ├── classifier.py            
│   │   ├── recommender.py           
│   │   ├── clustering.py            
│   │   └── fit_predictor.py         
│   ├── parsing/
│   │   └── resume_parser.py         
│   └── evaluate.py                 
│
├── models/                          
│
├── app/
│   └── streamlit_app.py             
├── reports/
│   └── figures/                    
│
└── tests/
    └── test_features.py             
```

### What each part is for
- **`data/`** — Keep raw downloads in `raw/` untouched; write cleaned versions to
  `interim/` then `processed/`. The whole folder is git-ignored so datasets never
  get committed.
- **`notebooks/`** — Where you experiment and see results. Run 01 → 05 in order;
  each notebook is one module of the project.
- **`src/`** — Once code works in a notebook, move the reusable function here so the
  notebooks and the app can both import it. `config.py` holds every path so nothing
  is hard-coded.
- **`models/`** — Trained models saved as `.pkl` (via joblib) so the app can load
  them without retraining.
- **`app/`** — The Streamlit portal. It only wires together pieces that already work
  in `src/`, so build it last.
- **`reports/`** — Figures for the write-up and the final report.
- **`tests/`** — Optional sanity checks for feature functions.
