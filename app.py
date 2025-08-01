import streamlit as st
import fitz  # PyMuPDF
import json
import spacy
import pandas as pd
from io import StringIO
from sentence_transformers import SentenceTransformer, util

# Load BERT model for sentence similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Proficiency Estimation
def estimate_proficiency(text, skill):
    text = text.lower()
    skill = skill.lower()
    if f"advanced {skill}" in text or f"expert in {skill}" in text:
        return "Advanced"
    elif f"intermediate {skill}" in text or f"good in {skill}" in text:
        return "Intermediate"
    elif f"basic {skill}" in text or f"familiar with {skill}" in text:
        return "Beginner"
    else:
        return "Uncertain"

# Load job and course data
with open("data/job_data.json", "r") as f:
    job_data = json.load(f)

with open("data/courses.json", "r") as f:
    course_data = json.load(f)

# Keywords for simple skill matching
SKILL_KEYWORDS = [
    "python", "java", "c++", "sql", "excel", "machine learning", "html", "css",
    "react", "django", "tensorflow", "keras", "git", "quickbooks", "powerpoint",
    "problem solving", "data visualization", "statistics", "accounting"
]

# PDF Text Extraction
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Skill extraction
def extract_skills(text):
    text_lower = text.lower()
    return list(set([skill for skill in SKILL_KEYWORDS if skill in text_lower]))

# Job match logic
def calculate_match(resume_skills, job_skills):
    matched = [s for s in job_skills if s in resume_skills]
    missing = [s for s in job_skills if s not in resume_skills]
    score = (len(matched) / len(job_skills)) * 100
    return round(score, 2), matched, missing

# Semantic similarity
def semantic_match(resume_text, job_text):
    embeddings = model.encode([resume_text, job_text], convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return round(float(similarity[0][0]) * 100, 2)

# Streamlit UI
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>AI-Powered Resume Analyzer</h1>", unsafe_allow_html=True)
st.write("Upload your resume to get matched job suggestions and learning resources.")

uploaded_file = st.file_uploader("📤 Upload Resume (PDF)", type="pdf")

if uploaded_file is not None:
    resume_text = extract_text_from_pdf(uploaded_file)
    st.success("Resume uploaded and processed!")

    resume_skills = extract_skills(resume_text)

    st.subheader("Extracted Skills with Proficiency")
    if resume_skills:
        for skill in resume_skills:
            level = estimate_proficiency(resume_text, skill)
            st.markdown(f"- **{skill.title()}** — _{level}_")
    else:
        st.info("No skills found in your resume.")
    st.markdown("---")

    st.subheader("Job Match Suggestions")

    output = StringIO()
    output.write("Resume Analysis Report\n")
    output.write("=" * 40 + "\n\n")

    for job in job_data:
        score, matched, missing = calculate_match(resume_skills, job["required_skills"])
        job_text = job["job_title"] + " requires skills like " + ", ".join(job["required_skills"])
        semantic_score = semantic_match(resume_text, job_text)

        st.markdown(f"**Job Title:** {job['job_title']}")
        st.markdown(f"**Match Score:** {score}%")
        st.markdown(f"**Semantic Similarity Score:** {semantic_score}%")
        st.markdown(f"**Matched Skills:** {', '.join(matched) if matched else 'None'}")
        st.markdown(f"**Missing Skills:** {', '.join(missing) if missing else 'None'}")

        if missing:
            st.markdown("**Suggested Courses:**")
            shown_courses = set()
            for skill in missing:
                if skill in course_data:
                    for course in course_data[skill]:
                        course_key = course["name"]
                        if course_key not in shown_courses:
                            st.markdown(f"- [{course['name']}]({course['url']})")
                            shown_courses.add(course_key)

        st.markdown("---")

        # Add to report
        output.write(f"Job Title: {job['job_title']}\n")
        output.write(f"Match Score: {score}%\n")
        output.write(f"Semantic Similarity Score: {semantic_score}%\n")
        output.write(f"Matched Skills: {', '.join(matched) if matched else 'None'}\n")
        output.write(f"Missing Skills: {', '.join(missing) if missing else 'None'}\n")

        if score < 70:
            output.write("Suggestion: Consider improving your skills in the missing areas.\n")
        elif score < 90:
            output.write("You're close! A few more skills would boost your chances.\n")
        else:
            output.write("Excellent fit for this role!\n")

        output.write("Recommended Courses:\n")
        for skill in missing:
            if skill in course_data:
                for course in course_data[skill]:
                    output.write(f"- {course['name']}: {course['url']}\n")

        output.write("\n" + "-" * 40 + "\n\n")

    st.markdown("### Download Your Report")
    st.download_button(
        label="📄 Download Report as TXT",
        data=output.getvalue().encode("utf-8"),
        file_name="resume_analysis.txt",
        mime="text/plain"
    )
