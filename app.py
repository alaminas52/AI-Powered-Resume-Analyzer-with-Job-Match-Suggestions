import streamlit as st
import fitz  # PyMuPDF
import json
import spacy
import pandas as pd
from io import StringIO
from sentence_transformers import SentenceTransformer, util
from collections import Counter
import base64
import matplotlib.pyplot as plt

# ===================== Skill Tips for Improvement =====================
skill_tips = {
    "sql": "Practice SQL queries on free platforms like LeetCode or W3Schools.",
    "python": "Contribute to open-source or solve problems on HackerRank.",
    "excel": "Explore advanced Excel topics like pivot tables and formulas.",
    "statistics": "Learn basic probability, distributions, and hypothesis testing.",
    "machine learning": "Complete beginner-friendly ML courses on Coursera or Kaggle.",
    "html": "Build small web pages using HTML/CSS from tutorials.",
    "git": "Learn version control basics from GitHub learning lab.",
}

# ===================== Load BERT Model for Sentence Similarity =====================
model = SentenceTransformer('all-MiniLM-L6-v2')

# ===================== Load spaCy Model =====================
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# ===================== Proficiency Estimation =====================
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

# ===================== Load Data =====================
with open("data/job_data.json", "r") as f:
    job_data = json.load(f)

with open("data/courses.json", "r") as f:
    course_data = json.load(f)

# ===================== Keywords for Simple Skill Matching =====================
SKILL_KEYWORDS = [
    "python", "java", "c++", "sql", "excel", "machine learning", "html", "css",
    "react", "django", "tensorflow", "keras", "git", "quickbooks", "powerpoint",
    "problem solving", "data visualization", "statistics", "accounting"
]

# ===================== PDF Text Extraction =====================
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ===================== Skill Extraction =====================
def extract_skills(text):
    text_lower = text.lower()
    return list(set([skill for skill in SKILL_KEYWORDS if skill in text_lower]))

# ===================== Job Match Logic =====================
def calculate_match(resume_skills, job_skills):
    matched = [s for s in job_skills if s in resume_skills]
    missing = [s for s in job_skills if s not in resume_skills]
    score = (len(matched) / len(job_skills)) * 100
    return round(score, 2), matched, missing

# ===================== Semantic Similarity =====================
def semantic_match(resume_text, job_text):
    embeddings = model.encode([resume_text, job_text], convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return round(float(similarity[0][0]) * 100, 2)

# ===================== Image to Base64 =====================
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

banner_base64 = get_base64_image("banner.png")  # Banner image
logo_base64 = get_base64_image("logo.png")      # AI logo image

# ===================== Banner + Title =====================
st.markdown(
    f"""
    <div style="
        background-image: url('data:image/png;base64,{banner_base64}');
        background-size: cover;
        background-position: center;
        padding: 40px;
        border-radius: 10px;
        text-align: center;">
        
        <img src="data:image/png;base64,{logo_base64}" 
             alt="AI Logo" style="width:60px; vertical-align: middle; margin-right: 10px;">
        <span style="font-size: 40px; font-weight: bold; color: #4CAF50; vertical-align: middle;">
            AI-Powered Resume Analyzer
        </span>
        
        <p style="color: white; font-size: 18px; margin-top: 10px;">
            Upload your resume to get matched job suggestions and learning resources.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ===================== File Uploader =====================
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
    skill_gap_counter = Counter()
    all_missing_skills = set()

    output.write("Resume Analysis Report\n")
    output.write("=" * 40 + "\n\n")

    for job in job_data:
        score, matched, missing = calculate_match(resume_skills, job["required_skills"])
        for skill in missing:
            skill_gap_counter[skill] += 1
        all_missing_skills.update(missing)

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

    if skill_gap_counter:
        st.markdown("### 🔍 Summary Insights")
        st.subheader("📉 Most Common Skill Gaps")
        for skill, count in skill_gap_counter.most_common():
            st.markdown(f"- **{skill.title()}** missing in {count} job(s)")

        fig, ax = plt.subplots()
        ax.bar([s.title() for s, _ in skill_gap_counter.most_common()],
               [count for _, count in skill_gap_counter.most_common()])
        ax.set_ylabel("Count")
        ax.set_title("Top Missing Skills Across Jobs")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    if all_missing_skills:
        st.subheader("📚 Suggested Courses for All Missing Skills")
        shown_courses = set()
        for skill in all_missing_skills:
            if skill in course_data:
                for course in course_data[skill]:
                    course_key = course["name"]
                    if course_key not in shown_courses:
                        st.markdown(f"- [{course['name']}]({course['url']}) — _({skill.title()})_")
                        shown_courses.add(course_key)

        matched_count = len(resume_skills)
        missing_count = len(all_missing_skills)
        fig2, ax2 = plt.subplots()
        ax2.pie([matched_count, missing_count],
                labels=["Matched Skills", "Missing Skills"],
                autopct="%1.1f%%", colors=["#4CAF50", "#FF5733"])
        st.pyplot(fig2)

    st.markdown("### Download Your Report")
    output.write("📉 Skill Gap Summary:\n")
    for skill, count in skill_gap_counter.most_common():
        output.write(f"- {skill.title()}: Missing in {count} job(s)\n")
    output.write("\n")

    output.write("📚 Course Suggestions (Grouped):\n")
    for skill in all_missing_skills:
        if skill in course_data:
            for course in course_data[skill]:
                output.write(f"- {course['name']}: {course['url']} ({skill.title()})\n")

    st.subheader("📝 Resume Summary")
    st.markdown(f"**Total Skills Found:** {len(resume_skills)}")
    st.markdown(f"**Top Missing Skills Across Jobs:** {', '.join([s.title() for s, _ in skill_gap_counter.most_common(3)])}")
    st.subheader("💡 Skill Improvement Tips")
    for skill in all_missing_skills:
        if skill in skill_tips:
            st.markdown(f"- **{skill.title()}**: {skill_tips[skill]}")

    st.download_button(
        label="📄 Download Report as TXT",
        data=output.getvalue().encode("utf-8"),
        file_name="resume_analysis.txt",
        mime="text/plain"
    )