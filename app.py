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

# ---------- TEXT PROCESSING HELPERS ----------

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
        "education": ""
    }

    text_lower = text.lower()
    skill_idx = text_lower.find("skill")
    exp_idx = text_lower.find("experience")
    edu_idx = text_lower.find("education")

    indexes = sorted([(skill_idx, 'skills'), (exp_idx, 'experience'), (edu_idx, 'education')])
    indexes = [(i, sec) for i, sec in indexes if i != -1]

    if not indexes:
        return sections  # All sections empty

    for i in range(len(indexes)):
        start_idx = indexes[i][0]
        sec = indexes[i][1]
        end_idx = indexes[i+1][0] if i+1 < len(indexes) else len(text)
        section_text = text[start_idx:end_idx]
        sections[sec] = preprocess_text(section_text)

    return sections

# ---------- MATCHING FUNCTION ----------

def match_sections(jd_sections, resume_sections, weights={'skills': 0.4, 'experience': 0.4, 'education': 0.2}):
    total_score = 0.0
    section_scores = {}

    for section in jd_sections.keys():
        jd_text = jd_sections[section]
        resume_text = resume_sections[section]

        if jd_text.strip() == "" or resume_text.strip() == "":
            score = 0
        else:
            vectorizer = TfidfVectorizer()
            tfidf = vectorizer.fit_transform([jd_text, resume_text])
            score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

        section_scores[section] = score
        total_score += weights[section] * score

    return total_score, section_scores

# ---------- EXCEL OUTPUT ----------

def save_to_excel(results):
    df = pd.DataFrame(results)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Match Results')
    return output.getvalue()

# ---------- STREAMLIT APP ----------

st.set_page_config(page_title="Resume Matcher", layout="centered")
st.title("📄 Automated Resume Matcher with Section-wise Analysis")

st.sidebar.header("Upload Files")
jd_file = st.sidebar.file_uploader("Upload Job Description (PDF/DOCX)", type=['pdf', 'docx'])
resume_files = st.sidebar.file_uploader("Upload Resumes (PDF/DOCX)", type=['pdf', 'docx'], accept_multiple_files=True)

if st.sidebar.button("🔍 Match Resumes"):
    if not jd_file or not resume_files:
        st.warning("Please upload both job description and at least one resume.")
    else:
        jd_raw = extract_text(jd_file)
        jd_sections = extract_sections(jd_raw)

        results = []

        for resume in resume_files:
            resume_raw = extract_text(resume)
            resume_sections = extract_sections(resume_raw)
            total_score, section_scores = match_sections(jd_sections, resume_sections)

            results.append({
                "Resume Name": resume.name,
                "Total Match Score": round(total_score, 2),
                "Skills Score": round(section_scores['skills'], 2),
                "Experience Score": round(section_scores['experience'], 2),
                "Education Score": round(section_scores['education'], 2)
            })

        sorted_results = sorted(results, key=lambda x: x["Total Match Score"], reverse=True)

        st.subheader("🏆 Top Matching Resumes")
        for res in sorted_results:
            st.markdown(f"**{res['Resume Name']}** — Match Score: `{res['Total Match Score']}`")

        excel_data = save_to_excel(sorted_results)
        st.download_button(
            label="📥 Download Excel Report",
            data=excel_data,
            file_name='sectionwise_resume_match_results.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
