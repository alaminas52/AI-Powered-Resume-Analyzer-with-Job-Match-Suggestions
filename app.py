import streamlit as st
import fitz  # PyMuPDF
import json
import spacy

# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Load job and course data
with open("data/job_data.json", "r") as f:
    job_data = json.load(f)

with open("data/courses.json", "r") as f:
    course_data = json.load(f)

# Sample skill keywords (for simple extraction)
SKILL_KEYWORDS = [
    "python", "java", "c++", "sql", "excel", "machine learning", "html", "css",
    "react", "django", "tensorflow", "keras", "git", "quickbooks", "powerpoint",
    "problem solving", "data visualization", "statistics", "accounting"
]

# Extract text from PDF
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Extract skills
def extract_skills(text):
    skills = []
    text_lower = text.lower()
    for skill in SKILL_KEYWORDS:
        if skill in text_lower:
            skills.append(skill)
    return list(set(skills))

# Match logic
def calculate_match(resume_skills, job_skills):
    matched = [s for s in job_skills if s in resume_skills]
    missing = [s for s in job_skills if s not in resume_skills]
    score = (len(matched) / len(job_skills)) * 100
    return round(score, 2), matched, missing

# Streamlit UI
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🤖 AI-Powered Resume Analyzer</h1>", unsafe_allow_html=True)
st.write("Upload your resume to get matched job suggestions and learning resources.")

uploaded_file = st.file_uploader("📤 Upload Resume (PDF)", type="pdf")

if uploaded_file is not None:
    resume_text = extract_text_from_pdf(uploaded_file)
    st.success("✅ Resume uploaded and processed!")

    doc = nlp(resume_text)
    resume_skills = extract_skills(resume_text)

    st.subheader("🔍 Extracted Skills")
    st.write(", ".join(resume_skills) if resume_skills else "No skills found.")
    st.subheader("💼 Job Match Suggestions")

    for job in job_data:
        score, matched, missing = calculate_match(resume_skills, job["required_skills"])
        st.markdown(f"**🧑‍💻 Job Title:** {job['job_title']}")
        st.markdown(f"**✅ Match Score:** {score}%")
        st.markdown(f"**🎯 Matched Skills:** {', '.join(matched) if matched else 'None'}")
        st.markdown(f"**❌ Missing Skills:** {', '.join(missing) if missing else 'None'}")

        if missing:
            st.markdown("**📘 Suggested Courses:**")
            shown_courses = set()
            for skill in missing:
                if skill in course_data:
                    for course in course_data[skill]:
                        if course not in shown_courses:
                            st.markdown(f"- {course}")
                            shown_courses.add(course)

        st.markdown("---")

        
        import pandas as pd
    from io import StringIO

    output = StringIO()

    for job in job_data:
        score, matched, missing = calculate_match(resume_skills, job["required_skills"])
        output.write(f"Job Title: {job['job_title']}\n")
        output.write(f"Match Score: {score}%\n")
        output.write(f"Matched Skills: {', '.join(matched) if matched else 'None'}\n")
        output.write(f"Missing Skills: {', '.join(missing) if missing else 'None'}\n")
        output.write("Suggested Courses:\n")
        for skill in missing:
            if skill in course_data:
                for course in course_data[skill]:
                    output.write(f" - {course}\n")
        output.write("-" * 50 + "\n")

    st.markdown("### 📥 Download Your Report")
    st.download_button(
        label="Download Report as TXT",
        data=output.getvalue(),
        file_name="resume_analysis.txt",
        mime="text/plain",
        key="download_report"
    )
       