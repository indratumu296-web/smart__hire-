from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# The app lives in <project>/app locally and on Streamlit Cloud.
PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.parsing.resume_parser import extract_resume_text
from src.models.recommender import match_jobs

st.set_page_config(
    page_title="SmartHire | Resume intelligence",
    page_icon="6",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    :root { --ink:#17212b; --muted:#66727d; --line:#dce3e8; --blue:#1769e0; --mint:#d9f4e9; --paper:#f7f9f7; }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stSidebar"] { background: #11202b; }
    [data-testid="stSidebar"] * { color: #edf6f2 !important; }
    h1, h2, h3, p, label, .stMarkdown { font-family: 'Space Grotesk', sans-serif; }
    .mono { font-family: 'DM Mono', monospace; letter-spacing: .02em; }
    .eyebrow { color: var(--blue); font-family:'DM Mono',monospace; font-size:.72rem; text-transform:uppercase; letter-spacing:.13em; font-weight:500; }
    .hero { padding: 2rem 0 1.6rem; border-bottom: 1px solid var(--line); }
    .hero h1 { font-size: clamp(2.4rem, 5vw, 4.9rem); line-height:.98; letter-spacing:-.06em; margin:.4rem 0 1rem; max-width: 760px; }
    .hero p { color:var(--muted); max-width: 600px; font-size:1.05rem; }
    .metric { border-top: 3px solid var(--blue); padding: .8rem 0; }
    .metric-value { font-size:1.8rem; font-weight:700; }
    .metric-label { color:var(--muted); font-size:.8rem; text-transform:uppercase; letter-spacing:.08em; }
    .job { background:#fff; border:1px solid var(--line); border-radius:8px; padding:1rem 1.1rem; margin:.55rem 0; }
    .job-title { font-weight:700; font-size:1.08rem; }
    .job-meta { color:var(--muted); font-size:.86rem; margin-top:.25rem; }
    .score { color:var(--blue); font-family:'DM Mono',monospace; font-size:.82rem; float:right; }
    .stButton > button { border-radius:5px; font-family:'Space Grotesk',sans-serif; font-weight:600; }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_jobs() -> pd.DataFrame:
    relative_path = "marketing_sample_for_naukri_com-naukri_com_job_data__20201001_20201231__5k_data.csv"
    search_roots = (PROJECT, ROOT)
    files = [
        file
        for search_root in search_roots
        for file in search_root.glob(f"{relative_path}/*.ldjson")
    ]
    if not files:
        return pd.DataFrame()
    rows = []
    with files[0].open(encoding="utf-8") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def get_jobs() -> pd.DataFrame:
    return load_jobs()


jobs = get_jobs()

with st.sidebar:
    st.markdown("<div class='eyebrow'>SmartHire / 01</div>", unsafe_allow_html=True)
    st.markdown("## Resume desk")
    st.caption("Upload a resume to identify the strongest role signals and surface relevant jobs from the included Naukri corpus.")
    st.divider()
    st.markdown("**Pipeline**")
    st.markdown("`01` extract  →  `02` classify  →  `03` match")
    st.divider()
    st.caption("Local data mode · no resume leaves this machine")

st.markdown(
    "<section class='hero'><div class='eyebrow'>Candidate intelligence workspace</div><h1>Make the next role feel less random.</h1><p>Drop in a resume. SmartHire reads the signal, predicts a target role, and ranks the closest opportunities in your local job corpus.</p></section>",
    unsafe_allow_html=True,
)

uploaded = st.file_uploader("Resume upload", type=["pdf", "docx", "txt"], label_visibility="visible")

if not uploaded:
    left, right = st.columns([1.4, 1])
    with left:
        st.subheader("Start with a resume")
        st.write("Supported formats: PDF, DOCX, or plain text. The first pass is intentionally transparent: extracted text, role evidence, and match scores are all visible.")
    with right:
        st.markdown("<div class='metric'><div class='metric-value'>%s</div><div class='metric-label'>jobs indexed</div></div>" % f"{len(jobs):,}", unsafe_allow_html=True)
        st.markdown("<div class='metric'><div class='metric-value'>local</div><div class='metric-label'>processing mode</div></div>", unsafe_allow_html=True)
    st.stop()

try:
    resume_text = extract_resume_text(uploaded)
except ValueError as error:
    st.error(str(error))
    st.stop()

if not resume_text.strip():
    st.warning("The file did not contain readable text. Try exporting the resume as PDF, DOCX, or TXT.")
    st.stop()

matches, predicted_role, skills = match_jobs(resume_text, jobs)

st.markdown(f"<div class='eyebrow'>Analysis complete / {uploaded.name}</div>", unsafe_allow_html=True)
metric_cols = st.columns(4)
for column, value, label in zip(metric_cols, [predicted_role, len(skills), len(matches), len(resume_text.split())], ["predicted role", "skills detected", "jobs ranked", "words analyzed"]):
    with column:
        st.markdown(f"<div class='metric'><div class='metric-value'>{value}</div><div class='metric-label'>{label}</div></div>", unsafe_allow_html=True)

left, right = st.columns([1.35, 1])
with left:
    st.subheader("Recommended opportunities")
    if matches.empty:
        st.info("No matching jobs were found in the local corpus.")
    else:
        for _, job in matches.head(8).iterrows():
            title = str(job.get("job_title", "Untitled role"))
            company = str(job.get("company_name", "Company not listed"))
            location = str(job.get("city", "Location not listed"))
            score = float(job.get("match_score", 0))
            st.markdown(f"<div class='job'><span class='score'>{score:.0%} match</span><div class='job-title'>{title}</div><div class='job-meta'>{company} · {location}</div></div>", unsafe_allow_html=True)
with right:
    st.subheader("Role signal")
    st.markdown(f"### {predicted_role}")
    st.caption("Inferred from the role vocabulary and skill evidence found in the uploaded resume.")
    if skills:
        st.write("Detected skills")
        st.write(" · ".join(skills[:18]))
    with st.expander("View extracted text"):
        st.text_area("Resume text", resume_text, height=300, label_visibility="collapsed")

st.divider()
st.caption("SmartHire is a decision-support prototype. Review the original job listing before applying.")
