# SmartHire: Resume-to-Job Matching

SmartHire is a classical-ML career guidance engine. Upload a resume to get a
predicted role category, cosine-similarity job recommendations, and a skill-gap
report. It does not use LLMs.

## Run

From this folder:

```powershell
python -m pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Upload a PDF, DOCX, or TXT resume. The app extracts the text locally, predicts a
resume category with TF-IDF plus Logistic Regression, ranks jobs with TF-IDF
cosine similarity, and compares the resume skills with the top job's skills.

To build cleaned datasets from the supplied raw files:

```powershell
python -m src.data.preprocess
```

## Structure

- `app/streamlit_app.py` - Streamlit web portal
- `src/parsing/resume_parser.py` - PDF, DOCX, and TXT extraction
- `src/models/classifier.py` - TF-IDF plus Logistic Regression role classifier
- `src/models/recommender.py` - TF-IDF plus cosine-similarity top-N recommender
- `src/features/match_features.py` - skill extraction and skill-gap analysis
- `src/data/preprocess.py` - cleaned resume and job corpus generation
- `requirements.txt` - Python dependencies

The optional `clustering.py`, `fit_predictor.py`, and `evaluate.py` modules
provide extensions for clustering, shortlisting, and model evaluation.
