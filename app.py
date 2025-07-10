import streamlit as st
import fitz
import docx2txt
import os
import re
import nltk
import pandas as pd
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')

# ---------- TEXT PROCESSING ----------

def extract_text(file):
    ext = os.path.splitext(file.name)[1].lower()
    if ext == ".pdf":
        text = ""
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
        return text
    elif ext == ".docx":
        return docx2txt.process(file)
    else:
        return ""

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in nltk.corpus.stopwords.words('english')]
    return ' '.join(tokens)

def extract_sections(text):
    sections = {
        "skills": "",
        "experience": "",
        "education": "",
        "achievements": ""
    }

    text_lower = text.lower()
    idxs = {
        "skills": text_lower.find("skill"),
        "experience": text_lower.find("experience"),
        "education": text_lower.find("education"),
        "achievements": text_lower.find("achievement")
    }

    sorted_idxs = sorted([(i, s) for s, i in idxs.items() if i != -1])
    for i in range(len(sorted_idxs)):
        start_idx, sec = sorted_idxs[i]
        end_idx = sorted_idxs[i + 1][0] if i + 1 < len(sorted_idxs) else len(text)
        sections[sec] = preprocess_text(text[start_idx:end_idx])

    return sections

# ---------- SCORING ----------

def scale_score(raw_score):
    # Scale from [0, 1] to [70, 90]
    return 70 + (raw_score * 20)

def match_sections(jd_sections, resume_sections, weights):
    total_score = 0.0
    section_scores = {}

    for section in weights:
        jd_text = jd_sections.get(section, "")
        resume_text = resume_sections.get(section, "")

        if jd_text.strip() == "" or resume_text.strip() == "":
            score = 0
        else:
            vectorizer = TfidfVectorizer()
            tfidf = vectorizer.fit_transform([jd_text, resume_text])
            score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

        section_scores[section] = scale_score(score)
        total_score += weights[section] * scale_score(score)

    return round(total_score, 2), section_scores

def save_to_excel(results):
    df = pd.DataFrame(results)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Match Results')
    return output.getvalue()

# ---------- STREAMLIT APP ----------

st.set_page_config(page_title="Resume Matcher", layout="centered")
st.title("📄 Automated Resume Matcher with Dynamic Filters")

st.sidebar.header("Upload Files")
jd_file = st.sidebar.file_uploader("Upload Job Description (PDF/DOCX)", type=['pdf', 'docx'])
resume_files = st.sidebar.file_uploader("Upload Resumes (PDF/DOCX)", type=['pdf', 'docx'], accept_multiple_files=True)

# Section selection
aspects = st.sidebar.multiselect(
    "🧠 Choose Resume Sections to Match",
    ["Skills", "Experience", "Education", "Achievements"]
)

aspect_map = {
    "Skills": "skills",
    "Experience": "experience",
    "Education": "education",
    "Achievements": "achievements"
}

# Load section choices
selected_sections = [aspect_map[a] for a in aspects]
if not selected_sections:
    selected_sections = ["skills", "experience", "education", "achievements"]

# Build weights
weights = {sec: 1 / len(selected_sections) for sec in selected_sections}

# Session state storage
if 'jd_sections' not in st.session_state:
    st.session_state.jd_sections = None
if 'resume_data' not in st.session_state:
    st.session_state.resume_data = []

# Handle Upload & Extraction
if jd_file and resume_files:
    jd_text = extract_text(jd_file)
    jd_sections = extract_sections(jd_text)
    st.session_state.jd_sections = jd_sections

    resumes_data = []
    for resume in resume_files:
        text = extract_text(resume)
        sections = extract_sections(text)
        resumes_data.append({"name": resume.name, "sections": sections})

    st.session_state.resume_data = resumes_data
    st.success("✅ Files uploaded and processed!")

# Match Results if JD & Resumes already uploaded
if st.session_state.jd_sections and st.session_state.resume_data:
    results = []
    for resume in st.session_state.resume_data:
        total, scores = match_sections(st.session_state.jd_sections, resume["sections"], weights)
        result = {
            "Resume Name": resume["name"],
            "Total Match Score": total
        }
        for sec in selected_sections:
            result[f"{sec.capitalize()} Score"] = round(scores[sec], 2)
        results.append(result)

    sorted_results = sorted(results, key=lambda x: x["Total Match Score"], reverse=True)

    st.subheader("📊 Matching Results")
    for res in sorted_results:
        st.markdown(f"**{res['Resume Name']}** — Match Score: `{res['Total Match Score']}%`")

    excel_data = save_to_excel(sorted_results)
    st.download_button(
        label="📥 Download Excel Report",
        data=excel_data,
        file_name='resume_match_results.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
else:
    st.info("⬅️ Please upload a job description and some resumes to begin.")
