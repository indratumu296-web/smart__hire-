# SmartHire

A local Streamlit resume intelligence prototype based on the supplied resume and Naukri job datasets.

## Run

From this folder:

```powershell
python -m pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Upload a PDF, DOCX, or TXT resume. The app extracts the text locally, detects role vocabulary, and ranks jobs from the parent workspace's Naukri JSONL dataset.

## Structure

- `app/streamlit_app.py` - Streamlit web portal
- `src/parsing/resume_parser.py` - PDF, DOCX, and TXT extraction
- `src/models/recommender.py` - transparent role and job matching baseline
- `requirements.txt` - Python dependencies
