import streamlit as st
import fitz  # PyMuPDF
import json
import spacy
import pandas as pd
from io import StringIO, BytesIO
from sentence_transformers import SentenceTransformer, util
from collections import Counter
from PIL import Image
import base64
import matplotlib.pyplot as plt
from datetime import datetime

try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    st.warning("PDF report generation is not available. Please install fpdf with: pip install fpdf")

# Skill tips for improvement
skill_tips = {
    "sql": "Practice SQL queries on free platforms like LeetCode or W3Schools.",
    "python": "Contribute to open-source or solve problems on HackerRank.",
    "excel": "Explore advanced Excel topics like pivot tables and formulas.",
    "statistics": "Learn basic probability, distributions, and hypothesis testing.",
    "machine learning": "Complete beginner-friendly ML courses on Coursera or Kaggle.",
    "html": "Build small web pages using HTML/CSS from tutorials.",
    "git": "Learn version control basics from GitHub learning lab.",
}

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

# Load job and course data with UTF-8 encoding
with open("data/job_data.json", "r", encoding="utf-8") as f:
    job_data = json.load(f)

with open("data/courses.json", "r", encoding="utf-8") as f:
    course_data = json.load(f)

# Keywords for simple skill matching
SKILL_KEYWORDS = [
    "python", "java", "c++", "sql", "excel", "machine learning", "html", "css",
    "react", "django", "tensorflow", "keras", "git", "quickbooks", "powerpoint",
    "problem solving", "data visualization", "statistics", "accounting", "javascript",
    "financial reporting"
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

# Create PDF report
def create_pdf_report(content, skill_gap_counter, all_missing_skills):
    if not PDF_AVAILABLE:
        return None
        
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_fill_color(44, 62, 80)
    pdf.rect(0, 0, 210, 30, 'F')
    pdf.set_font("Arial", 'B', 20)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(200, 10, txt="RESUME ANALYSIS REPORT", ln=1, align='C')
    
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(200, 10, txt=f"Generated on: {datetime.now().strftime('%B %d, %Y')}", ln=1, align='C')
    pdf.ln(15)
    
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(41, 128, 185)
    pdf.cell(200, 10, txt="SKILL GAP SUMMARY", ln=1)
    pdf.ln(5)
    
    pdf.set_font("Arial", size=12)
    pdf.set_text_color(0, 0, 0)
    for skill, count in skill_gap_counter.most_common():
        pdf.cell(200, 8, txt=f"- {skill.title()}: Missing in {count} job(s)", ln=1)
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(41, 128, 185)
    pdf.cell(200, 10, txt="SUGGESTED COURSES", ln=1)
    pdf.ln(5)
    
    pdf.set_font("Arial", size=12)
    pdf.set_text_color(0, 0, 0)
    shown_courses = set()
    for skill in all_missing_skills:
        if skill in course_data:
            for course in course_data[skill]:
                course_key = course["name"]
                if course_key not in shown_courses:
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(200, 8, txt=f"- {course['name']} ({skill.title()}):", ln=1)
                    pdf.set_font("Arial", size=12)
                    pdf.set_text_color(0, 0, 255)
                    pdf.cell(200, 8, txt=f"  {course['url']}", ln=1)
                    pdf.set_text_color(0, 0, 0)
                    shown_courses.add(course_key)
        else:
            pdf.cell(200, 8, txt=f"- No course suggestions found for {skill.title()}", ln=1)
    pdf.ln(15)
    
    pdf.set_draw_color(41, 128, 185)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(15)
    
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(41, 128, 185)
    pdf.cell(200, 10, txt="DETAILED ANALYSIS", ln=1)
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    for line in content.split('\n'):
        if line.startswith("Job Title:"):
            pdf.set_font("Arial", 'B', 14)
            pdf.set_text_color(39, 174, 96)
            pdf.cell(200, 10, txt=line, ln=1)
            pdf.set_font("Arial", size=12)
            pdf.set_text_color(0, 0, 0)
        elif line.startswith("Match Score:") or line.startswith("Semantic Similarity Score:"):
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 8, txt=line, ln=1)
            pdf.set_font("Arial", size=12)
        else:
            pdf.cell(200, 8, txt=line, ln=1)
    
    pdf.set_y(-15)
    pdf.set_font('Arial', 'I', 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, 'Created by Al-Amin | GitHub: https://github.com/alaminas52/AI-Powered-Resume-Analyzer-with-Job-Match-Suggestions', 0, 0, 'C')
    
    return pdf.output(dest='S').encode('latin1')

# --- Banner + Logo Setup ---
def get_base64_of_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

banner_base64 = get_base64_of_image("banner.png")
logo_base64 = get_base64_of_image("logo.png")

st.markdown(
    f"""
    <div style="
        background-image: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)), url('data:image/png;base64,{banner_base64}');
        background-size: cover;
        background-position: center;
        padding: 40px;
        border-radius: 10px;
        text-align: center;">
        <img src="data:image/png;base64,{logo_base64}" alt="AI Logo" width="60" style="vertical-align: middle; margin-right: 15px;">
        <span style="font-size: 40px; font-weight: bold; color: #4CAF50; vertical-align: middle;">
            AI-Powered Resume Analyzer
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("Upload your resume to get matched job suggestions and learning resources.")

uploaded_file = st.file_uploader("📤 Upload Resume (PDF)", type="pdf")

if uploaded_file is not None:
    resume_text = extract_text_from_pdf(uploaded_file)
    st.success("Resume uploaded and processed!")

    resume_skills = extract_skills(resume_text)

    st.subheader("🔍 Extracted Skills with Proficiency")
    if resume_skills:
        for skill in resume_skills:
            level = estimate_proficiency(resume_text, skill)
            st.markdown(f"- **{skill.title()}** — _{level}_")
    else:
        st.info("No skills found in your resume.")
    st.markdown("---")

    st.subheader("💼 Job Match Suggestions")

    output = StringIO()
    skill_gap_counter = Counter()
    all_missing_skills = set()

    output.write("Resume Analysis Report\n")
    output.write("=" * 50 + "\n\n")

    for job in job_data:
        score, matched, missing = calculate_match(resume_skills, job["required_skills"])
        for skill in missing:
            skill_gap_counter[skill] += 1
        all_missing_skills.update(missing)

        job_text = job["job_title"] + " requires skills like " + ", ".join(job["required_skills"])
        semantic_score = semantic_match(resume_text, job_text)

        st.markdown(f"**🏷 Job Title:** {job['job_title']}")
        st.markdown(f"**📊 Match Score:** {score}%")
        st.markdown(f"**🤝 Semantic Similarity Score:** {semantic_score}%")
        st.markdown(f"**✅ Matched Skills:** {', '.join(matched) if matched else 'None'}")
        st.markdown(f"**❌ Missing Skills:** {', '.join(missing) if missing else 'None'}")

        if missing:
            st.markdown("**📚 Suggested Courses:**")
            shown_courses = set()
            for skill in missing:
                if skill in course_data:
                    for course in course_data[skill]:
                        course_key = course["name"]
                        if course_key not in shown_courses:
                            st.markdown(f"- [{course['name']}]({course['url']}) — _({skill.title()})_")
                            shown_courses.add(course_key)
                else:
                    st.markdown(f"- No course suggestions found for: {skill}")

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

        if missing:
            output.write("Recommended Courses:\n")
            for skill in missing:
                if skill in course_data:
                    for course in course_data[skill]:
                        output.write(f"- {course['name']}: {course['url']}\n")
                else:
                    output.write(f"- No course suggestions found for: {skill}\n")

        output.write("\n" + "-" * 50 + "\n\n")

    # Summary
    if skill_gap_counter:
        st.markdown("### 📉 Summary Insights")
        st.subheader("📌 Most Common Skill Gaps")
        for skill, count in skill_gap_counter.most_common():
            st.markdown(f"- **{skill.title()}**: Missing in {count} job(s)")

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
            else:
                st.markdown(f"- No course suggestions found for: {skill}")

    # Skill Match Visualization
    st.markdown("### 📊 Skill Match Visualization")
    col1, col2 = st.columns(2)
    
    with col1:
        if all_missing_skills or resume_skills:
            labels = ["Matched Skills", "Missing Skills"]
            sizes = [len(resume_skills), len(all_missing_skills)]
            fig1, ax1 = plt.subplots()
            ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax1.axis('equal')
            ax1.set_title('Skill Match Percentage')
            st.pyplot(fig1)
    
    with col2:
        if skill_gap_counter:
            fig2, ax2 = plt.subplots()
            skills, counts = zip(*skill_gap_counter.most_common())
            bars = ax2.bar(skills, counts)
            ax2.set_ylabel('Count')
            ax2.set_title('Top Missing Skills Across Jobs')
            plt.xticks(rotation=45)
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height}',
                        ha='center', va='bottom')
            st.pyplot(fig2)

    # Report Download
    st.markdown("### 📥 Download Your Report")
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="📄 Download Report as TXT",
            data=output.getvalue().encode("utf-8"),
            file_name="resume_analysis.txt",
            mime="text/plain"
        )
    
    with col2:
        if PDF_AVAILABLE:
            pdf_report = create_pdf_report(output.getvalue(), skill_gap_counter, all_missing_skills)
            st.download_button(
                label="📑 Download Report as PDF",
                data=pdf_report,
                file_name="resume_analysis.pdf",
                mime="application/pdf"
            )
        else:
            st.button("📑 PDF Download (Unavailable)", disabled=True)
            st.info("Install fpdf with: pip install fpdf")

# Footer
st.markdown(
    """
    <div style="text-align: center; margin-top: 50px; padding: 10px; border-top: 1px solid #eee;">
        <p>Created by Al-Amin | <a href="https://github.com/alaminas52/AI-Powered-Resume-Analyzer-with-Job-Match-Suggestions" target="_blank">GitHub Repository</a></p>
    </div>
    """,
    unsafe_allow_html=True
)
