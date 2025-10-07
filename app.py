# Full updated app.py — AI-Powered Resume Analyzer with O*NET integration
# Changes requested by user:
# - Keep SBERT similarity % as main Match Score
# - Keep Detected Occupation inside O*NET section
# - Move O*NET Skills Analysis to the bottom of HR view (after charts / tables)
# - Keep Matched / Missing skills + course recommendations INSIDE O*NET section
# - Remove debug/list-button and "composite score" label
# - Keep Candidate and HR views behavior (candidate = single upload)
# - Keep other code and logic untouched as much as possible
# Note: user asked to "just send the code" — no extra text is included.

import streamlit as st
import fitz  # PyMuPDF
import io
import re
import pandas as pd
from collections import Counter
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from PIL import Image
import tempfile
import base64
import requests
from typing import List, Tuple, Dict

# Convert logo.png to base64 so Streamlit can show it inline
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# =============================
# Config & Styling
# =============================
st.set_page_config(page_title="AI-Powered-Resume-Analyzer-with-Job-Match-Suggestions", layout='wide')

CUSTOM_CSS = """
<style>
    .main .block-container {
        max-width: 1050px; 
        padding-top: 1.0rem; 
        padding-bottom: 1.5rem;
    }
    .title-text {
        color: #2ECC71 !important;
        font-weight: 700;
        text-align: center;
        font-size: 2rem;
        margin-left: 10px;
    }
    .hero-section {
        background-image: url("https://raw.githubusercontent.com/alaminas52/AI-Powered-Resume-Analyzer-with-Job-Match-Suggestions/main/banner.png");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        height: 220px;
        width: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 2rem;
    }
    .hero-overlay {
        background: rgba(0, 0, 0, 0.4);
        width: 100%;
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .hero-content {
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .hero-logo {
        height: 50px;
        width: auto;
        margin-right: 12px;
    }
    .footer {
        text-align: center !important;
        margin-top: 2rem;
        color: #666;
        font-size: 0.9rem;
    }
    .footer a {
        color: #2ECC71;
        text-decoration: none;
        font-weight: 600;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    .pill {
        display:inline-block;
        padding:4px 8px;
        margin:2px;
        border-radius:12px;
        background:#e8f5e9;
        color:#1b5e20;
        font-size:0.9rem;
    }
    .pill.bad {
        background:#ffebee;
        color:#b71c1c;
    }
    .category-header {
        background-color: #f0f8ff;
        padding: 8px;
        border-radius: 5px;
        margin: 10px 0;
        font-weight: bold;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Hero section with inline base64 logo
st.markdown(
    """
    <div class="hero-section">
        <div class="hero-overlay">
            <div class="hero-content">
                <img src="https://raw.githubusercontent.com/alaminas52/AI-Powered-Resume-Analyzer-with-Job-Match-Suggestions/main/logo.png" class="hero-logo" alt="Logo"/>
                <h1 class="title-text">AI-Powered Resume Analyzer with Job Match Suggestions</h1>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# =============================
# O*NET Skills Database Integration
# =============================

class ONETSkillsManager:
    def __init__(self):
        self.skills_db = {}
        self.load_onet_data()
    
    def load_onet_data(self):
        """Load O*NET skills data"""
        self.skills_db = {
            # Content Skills (Basic)
            "reading comprehension": {"category": "basic", "level": "content"},
            "active listening": {"category": "basic", "level": "content"},
            "writing": {"category": "basic", "level": "content"},
            "speaking": {"category": "basic", "level": "content"},
            "mathematics": {"category": "basic", "level": "content"},
            
            # Process Skills (Basic)
            "critical thinking": {"category": "basic", "level": "process"},
            "active learning": {"category": "basic", "level": "process"},
            "learning strategies": {"category": "basic", "level": "process"},
            "monitoring": {"category": "basic", "level": "process"},
            
            # Social Skills (Basic)
            "social perceptiveness": {"category": "basic", "level": "social"},
            "coordination": {"category": "basic", "level": "social"},
            "persuasion": {"category": "basic", "level": "social"},
            "negotiation": {"category": "basic", "level": "social"},
            "instructing": {"category": "basic", "level": "social"},
            "service orientation": {"category": "basic", "level": "social"},
            
            # Complex Problem Solving Skills
            "complex problem solving": {"category": "cross-functional", "level": "advanced"},
            
            # Technical Skills
            "operations analysis": {"category": "technical", "level": "technical"},
            "technology design": {"category": "technical", "level": "technical"},
            "equipment selection": {"category": "technical", "level": "technical"},
            "installation": {"category": "technical", "level": "technical"},
            "programming": {"category": "technical", "level": "technical"},
            "quality control analysis": {"category": "technical", "level": "technical"},
            "operations monitoring": {"category": "technical", "level": "technical"},
            "systems analysis": {"category": "technical", "level": "technical"},
            "troubleshooting": {"category": "technical", "level": "technical"},
            
            # Software Development
            "software development": {"category": "technical", "level": "technical"},
            "web development": {"category": "technical", "level": "technical"},
            "database management": {"category": "technical", "level": "technical"},
            "software testing": {"category": "technical", "level": "technical"},
            
            # Business & Management
            "project management": {"category": "business", "level": "advanced"},
            "leadership": {"category": "business", "level": "advanced"},
            "strategic planning": {"category": "business", "level": "advanced"},
            "budget management": {"category": "business", "level": "advanced"},
            "team management": {"category": "business", "level": "advanced"},
            
            # Data & Analytics
            "data analysis": {"category": "analytical", "level": "technical"},
            "data visualization": {"category": "analytical", "level": "technical"},
            "statistical analysis": {"category": "analytical", "level": "technical"},
            "machine learning": {"category": "analytical", "level": "advanced"},
            "research": {"category": "analytical", "level": "technical"},
            
            # Marketing & Sales
            "digital marketing": {"category": "marketing", "level": "technical"},
            "content marketing": {"category": "marketing", "level": "technical"},
            "social media marketing": {"category": "marketing", "level": "technical"},
            "sales": {"category": "marketing", "level": "technical"},
            "customer service": {"category": "marketing", "level": "basic"},
            
            # Healthcare
            "patient care": {"category": "healthcare", "level": "technical"},
            "medical knowledge": {"category": "healthcare", "level": "technical"},
            "healthcare management": {"category": "healthcare", "level": "advanced"},
            
            # Education
            "teaching": {"category": "education", "level": "technical"},
            "curriculum development": {"category": "education", "level": "advanced"},
            "student assessment": {"category": "education", "level": "technical"},
            
            # Creative
            "graphic design": {"category": "creative", "level": "technical"},
            "video editing": {"category": "creative", "level": "technical"},
            "photography": {"category": "creative", "level": "technical"},
        }
    
    def detect_occupation_from_jd(self, job_description: str) -> str:
        """Detect the closest occupation from job description"""
        occupation_keywords = {
            "software development": ["software", "developer", "programming", "code", "algorithm", "backend", "frontend"],
            "data science": ["data science", "machine learning", "data analysis", "statistics", "python", "r programming"],
            "project management": ["project management", "pmp", "agile", "scrum", "budget", "stakeholder"],
            "business management": ["manager", "management", "strategy", "business", "leadership", "team"],
            "marketing": ["marketing", "digital marketing", "seo", "social media", "brand", "campaign"],
            "healthcare": ["patient", "medical", "healthcare", "hospital", "clinical", "nursing"],
            "education": ["teaching", "education", "curriculum", "student", "classroom", "lesson"],
            "finance": ["finance", "financial", "accounting", "investment", "banking", "audit"],
            "design": ["design", "graphic", "creative", "ui/ux", "adobe", "photoshop"]
        }
        
        jd_lower = job_description.lower()
        best_match = "general"
        max_matches = 0
        
        for occupation, keywords in occupation_keywords.items():
            matches = sum(1 for keyword in keywords if re.search(rf'\b{re.escape(keyword)}\b', jd_lower))
            if matches > max_matches:
                max_matches = matches
                best_match = occupation
        
        return best_match
    
    def extract_skills_with_onet(self, text: str) -> List[str]:
        """Extract skills using O*NET taxonomy"""
        found_skills = set()
        text_lower = text.lower()
        
        # Check for O*NET skills with better matching
        for skill, info in self.skills_db.items():
            if self._skill_in_text(skill, text_lower):
                found_skills.add(skill)
        
        # Expanded common variations
        skill_variations = {
            "programming": ["coding", "software development", "developing software", "programmer", "coder"],
            "data analysis": ["analyzing data", "data analytics", "statistical analysis", "data mining", "data processing"],
            "project management": ["managing projects", "project planning", "project coordinator", "pm"],
            "communication": ["communicating", "verbal communication", "written communication", "presentation", "public speaking"],
            "leadership": ["leading teams", "team leadership", "managing people", "team lead", "supervisor"],
            "machine learning": ["ml", "ai", "artificial intelligence", "deep learning", "neural networks"],
            "python": ["python programming", "python developer", "python script", "django", "flask"],
            "java": ["java programming", "java developer", "spring boot", "j2ee"],
            "javascript": ["js", "react", "angular", "vue", "node", "typescript"],
            "sql": ["database", "mysql", "postgresql", "oracle", "sql server"],
            "aws": ["amazon web services", "cloud computing", "ec2", "s3"],
            "docker": ["containerization", "docker container", "dockerfile"],
            "kubernetes": ["k8s", "container orchestration"],
        }
        
        for canonical, variations in skill_variations.items():
            if any(self._skill_in_text(var, text_lower) for var in variations + [canonical]):
                found_skills.add(canonical)
        
        return sorted(list(found_skills))
    
    def _skill_in_text(self, skill: str, text: str) -> bool:
        """Check if a skill is mentioned in text"""
        return re.search(rf'\b{re.escape(skill)}\b', text) is not None
    
    def get_skill_category(self, skill: str) -> str:
        """Get the category of a skill"""
        return self.skills_db.get(skill, {}).get('category', 'other')
    
    def recommend_courses_based_on_skill_gap(self, missing_skills: List[str], occupation: str) -> List[Tuple[str, str]]:
        """Recommend courses based on skill gaps and occupation"""
        course_recommendations = []
        
        # Map skills to common course platforms
        skill_course_map = {
            "programming": ("Python Programming Bootcamp", "https://www.coursera.org/specializations/python"),
            "software development": ("Software Development Fundamentals", "https://www.coursera.org/specializations/software-development"),
            "data analysis": ("Data Analysis with Python", "https://www.coursera.org/learn/data-analysis-with-python"),
            "machine learning": ("Machine Learning by Andrew Ng", "https://www.coursera.org/learn/machine-learning"),
            "project management": ("Project Management Principles", "https://www.coursera.org/learn/project-management"),
            "leadership": ("Strategic Leadership", "https://www.coursera.org/learn/leadership"),
            "digital marketing": ("Digital Marketing Specialization", "https://www.coursera.org/specializations/digital-marketing"),
            "communication": ("Business Communication", "https://www.coursera.org/learn/communication-skills"),
            "data visualization": ("Data Visualization with Tableau", "https://www.coursera.org/learn/data-visualization-tableau"),
            "web development": ("Web Development Bootcamp", "https://www.udemy.com/course/the-complete-web-development-bootcamp/"),
            "graphic design": ("Graphic Design Specialization", "https://www.coursera.org/specializations/graphic-design"),
            "teaching": ("Teaching English as a Foreign Language", "https://www.coursera.org/learn/teach-english-now"),
            "patient care": ("Patient Care Technician Training", "https://www.coursera.org/learn/patient-care"),
            "financial analysis": ("Financial Analysis Specialization", "https://www.coursera.org/specializations/financial-analysis"),
        }
        
        # Prioritize occupation-specific skills
        occupation_priority = {
            "software development": ["programming", "software development", "web development"],
            "data science": ["data analysis", "machine learning", "programming"],
            "project management": ["project management", "leadership", "communication"],
            "business management": ["leadership", "strategic planning", "budget management"],
            "marketing": ["digital marketing", "content marketing", "social media marketing"],
            "healthcare": ["patient care", "medical knowledge", "healthcare management"],
            "education": ["teaching", "curriculum development", "student assessment"],
            "finance": ["financial analysis", "accounting", "risk management"],
            "design": ["graphic design", "video editing", "photography"]
        }
        
        priority_skills = occupation_priority.get(occupation, [])
        
        # Add priority skills first
        for skill in priority_skills:
            if skill in missing_skills and skill in skill_course_map:
                course_recommendations.append(skill_course_map[skill])
        
        # Then add other missing skills
        for skill in missing_skills:
            if skill in skill_course_map and skill_course_map[skill] not in course_recommendations:
                course_recommendations.append(skill_course_map[skill])
        
        return course_recommendations[:6]  # Limit to 6 courses

# Initialize O*NET manager
onet_manager = ONETSkillsManager()

# =============================
# Your Original Data & Helpers (KEEP THIS)
# =============================

# Curated skill taxonomy (compact). Expand as needed.
SKILL_DB = {
    "python": ["python", "python programming"],
    "java": ["java"],
    "c++": ["c++", "cpp"],
    "javascript": ["javascript", "js"],
    "sql": ["sql", "mysql", "postgresql", "postgres", "mssql"],
    "r programming": ["r programming", "r language"],
    "excel": ["microsoft excel", "excel"],
    "power bi": ["power bi"],
    "tableau": ["tableau"],
    "nlp": ["nlp", "natural language processing"],
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning", "neural networks"],
    "data analysis": ["data analysis", "data analytics", "analytics"],
    "statistics": ["statistics", "statistical analysis"],
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "aws": ["aws", "amazon web services"],
    "azure": ["azure", "microsoft azure"],
    "gcp": ["gcp", "google cloud"],
    "project management": ["project management", "pmp"],
    "communication": ["communication", "communications", "presentation"],
    "leadership": ["leadership"],
    "customer service": ["customer service", "customer support"],
    "sales": ["sales", "sales operations"],
    "teaching": ["teaching", "lecturing", "instructor"],
}

# Course links (clickable) - KEEP YOUR ORIGINAL
SKILL_COURSES = {
    "python": ("Complete Python Bootcamp (Udemy)", "https://www.udemy.com/course/complete-python-bootcamp/"),
    "machine learning": ("Machine Learning by Andrew Ng (Coursera)", "https://www.coursera.org/learn/machine-learning"),
    "deep learning": ("Deep Learning Specialization (Coursera)", "https://www.coursera.org/specializations/deep-learning"),
    "sql": ("SQL for Data Analysis (Udemy)", "https://www.udemy.com/topic/sql/"),
    "excel": ("Excel Skills for Business (Coursera)", "https://www.coursera.org/specializations/excel"),
    "power bi": ("Power BI Desktop for Business Intelligence (Udemy)", "https://www.udemy.com/topic/power-bi/"),
    "tableau": ("Data Visualization with Tableau (Coursera)", "https://www.coursera.org/learn/data-visualization-tableau"),
    "project management": ("PMP Exam Prep", "https://www.pmi.org/certifications/project-management-pmp"),
    "communication": ("Successful Negotiation: Essential Strategies (Coursera)", "https://www.coursera.org/learn/negotiation"),
    "docker": ("Docker Mastery (Udemy)", "https://www.udemy.com/topic/docker/"),
    "kubernetes": ("Kubernetes for the Absolute Beginners (Udemy)", "https://www.udemy.com/topic/kubernetes/"),
    "aws": ("AWS Certified Cloud Practitioner (Coursera)", "https://www.coursera.org/learn/aws-cloud-technical-essentials"),
}

STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before being below between both but by can did do does doing down during each few for from further had has have having he her here hers herself him himself his how i if in into is it its itself just me more most my myself no nor not of off on once only or other our ours ourselves out over own same she should so some such than that the their theirs them themselves then there these they this those through to too under until up very was we were what when where which while who whom why with you your yours yourself yourselves
""".split())

LOCATION_WORDS = set(["united", "kingdom", "york", "london", "dhaka", "bangladesh", "usa", "canada", "india", "australia"])

# Build alias index
@st.cache_data
def build_skill_alias_index():
    index = {}
    for canon, aliases in SKILL_DB.items():
        for a in aliases + [canon]:
            index[a.lower()] = canon
    return index

SKILL_ALIAS_INDEX = build_skill_alias_index()

# Token regex
_token_re = re.compile(r"[A-Za-z][A-Za-z\-\+\#]{1,}")

def clean_tokens(text: str):
    text = text.lower()
    tokens = _token_re.findall(text)
    tokens = [t for t in tokens if t not in STOPWORDS and t not in LOCATION_WORDS and len(t) >= 3]
    return tokens

# Extract skills using alias matching (deterministic) - KEEP YOUR ORIGINAL
def extract_skills(text: str):
    found = set()
    lt = (text or "").lower()
    for alias, canon in SKILL_ALIAS_INDEX.items():
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", lt):
            found.add(canon)
    noisy = {"business", "with", "that", "customer", "kingdom", "york"}
    found = {f for f in found if f not in noisy}
    return sorted(found)

# Compute similarity
@st.cache_resource
def load_embedding_model(name: str = "all-MiniLM-L6-v2"):
    return SentenceTransformer(name)

def compute_similarity(model, jd_text, resumes_texts):
    jd_emb = model.encode(jd_text, convert_to_tensor=True)
    res_embs = model.encode(resumes_texts, convert_to_tensor=True)
    sims = util.cos_sim(jd_emb, res_embs)[0].cpu().tolist()
    return sims

# Contact & experience extraction - KEEP YOUR ORIGINAL
def extract_contact_info(text: str):
    name = text.split('\n')[0].strip()[:80] if text else 'Unknown'
    email_match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    phone_match = re.search(r'\+?\d[\d\s\-]{7,}\d', text)
    email = email_match.group(0) if email_match else 'N/A'
    phone = phone_match.group(0) if phone_match else 'N/A'
    exp_years = None
    m = re.search(r'(\d{1,2})\+?\s*(?:years?|yrs?)', text.lower())
    if m:
        try:
            exp_years = int(m.group(1))
        except:
            exp_years = None
    return name, email, phone, exp_years

# Robust PDF text extraction with per-page OCR fallback - KEEP YOUR ORIGINAL
def extract_text_from_pdf(file_bytes):
    """Return (text, used_ocr, errors_list). Tries text extraction first; if a page has no text or extraction fails,
    it renders that page to an image and runs OCR (pytesseract) on the image.
    """
    used_ocr = False
    errors = []
    text_pages = []

    # try to open with PyMuPDF
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        # try writing to temp file and re-open
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            tmp.write(file_bytes)
            tmp.flush()
            tmp.close()
            doc = fitz.open(tmp.name)
        except Exception as e2:
            errors.append(f'Failed to open PDF: {e2}')
            return "", used_ocr, errors

    # iterate pages and extract text; fallback to OCR per page if needed
    for pno in range(len(doc)):
        page_text = ""
        try:
            page = doc.load_page(pno)
            page_text = page.get_text("text")
        except Exception as e:
            errors.append(f"Page {pno} extraction error: {e}")
            page_text = ""

        if not page_text or len(page_text.strip()) < 30:
            # attempt OCR for this page
            try:
                used_ocr = True
                pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
                mode = "RGB" if pix.n < 4 else "RGBA"
                img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                try:
                    import pytesseract
                except Exception:
                    errors.append("pytesseract not installed; install pytesseract + tesseract-ocr to enable OCR.")
                    # append whatever text we have (may be empty)
                    text_pages.append(page_text)
                    continue
                ocr_text = pytesseract.image_to_string(img)
                if ocr_text and ocr_text.strip():
                    text_pages.append(ocr_text)
                else:
                    text_pages.append(page_text)
            except Exception as e:
                errors.append(f"OCR failed on page {pno}: {e}")
                text_pages.append(page_text)
        else:
            text_pages.append(page_text)

    combined = "\n".join(text_pages).strip()
    return combined, used_ocr, errors

# Simple PDF/text reader - KEEP YOUR ORIGINAL
def extract_text_from_txt(file_bytes):
    try:
        return file_bytes.decode('utf-8', errors='ignore')
    except:
        return file_bytes.decode('latin1', errors='ignore')

# Generate PDF report - KEEP YOUR ORIGINAL
def generate_pdf_report(df, job_description):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Job Description:", styles['Heading2']))
    elements.append(Paragraph(job_description.replace('\n', '<br/>'), styles['Normal']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Batch Analysis Results", styles['Heading2']))

    for _, row in df.iterrows():
        elements.append(Paragraph(f"Resume: {row['filename']}", styles['Heading3']))
        elements.append(Paragraph(f"Similarity: {row['similarity_pct']}%", styles['Normal']))
        elements.append(Paragraph(f"Skills: {row['skills']}", styles['Normal']))
        if 'email' in row and 'phone' in row:
            elements.append(Paragraph(f"Email: {row['email']} | Phone: {row['phone']}", styles['Normal']))
        elements.append(Spacer(1, 8))

    doc.build(elements)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf

# =============================
# Enhanced O*NET Analysis Function (UPDATED)
# =============================

def enhanced_onet_analysis(resume_text, job_description):
    """Enhanced analysis using O*NET skills database"""
    
    # Detect occupation from job description
    occupation = onet_manager.detect_occupation_from_jd(job_description)
    
    # Use YOUR ORIGINAL skill extraction for both resume and JD
    resume_skills_original = extract_skills(resume_text)
    jd_skills_original = extract_skills(job_description)
    
    # Also get O*NET skills for additional insights
    resume_skills_onet = onet_manager.extract_skills_with_onet(resume_text)
    jd_skills_onet = onet_manager.extract_skills_with_onet(job_description)
    
    # COMBINE both skill sets, but prioritize your original method
    all_resume_skills = list(dict.fromkeys(resume_skills_original + resume_skills_onet))
    all_jd_skills = list(dict.fromkeys(jd_skills_original + jd_skills_onet))
    
    # Analyze matches and gaps using COMBINED skills
    matched_skills = [skill for skill in all_jd_skills if skill in all_resume_skills]
    missing_skills = [skill for skill in all_jd_skills if skill not in all_resume_skills]
    
    # Calculate match percentage based on JD skills detected
    if all_jd_skills:
        match_percentage = len(matched_skills) / len(all_jd_skills) * 100
    else:
        match_percentage = 0.0
    
    # Categorize skills for better display
    skill_categories = {}
    for skill in all_jd_skills:
        category = onet_manager.get_skill_category(skill)
        if category not in skill_categories:
            skill_categories[category] = {"matched": [], "missing": []}
        if skill in matched_skills:
            skill_categories[category]["matched"].append(skill)
        else:
            skill_categories[category]["missing"].append(skill)
    
    # Get course recommendations
    courses = onet_manager.recommend_courses_based_on_skill_gap(missing_skills, occupation)
    
    return {
        "occupation": occupation,
        "resume_skills": all_resume_skills,
        "jd_skills": all_jd_skills,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "skill_categories": skill_categories,
        "courses": courses,
        "match_percentage": match_percentage
    }

def display_onet_analysis(analysis_result, filename):
    """Display O*NET enhanced analysis results (keeps matched/missing & courses inside O*NET section)"""
    
    st.markdown(f"### 🎯 O*NET Skills Analysis for {filename}")
    
    # Occupation and match score
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Detected Occupation", analysis_result["occupation"].replace("_", " ").title())
    with col2:
        st.metric("Skills Match Score (O*NET)", f"{analysis_result['match_percentage']:.1f}%")
    
    # Skills by category (all inside O*NET section)
    st.markdown("#### 📊 Skills Analysis by Category")
    
    for category, skills_data in analysis_result["skill_categories"].items():
        matched_skills = skills_data["matched"]
        missing_skills = skills_data["missing"]
        
        if matched_skills or missing_skills:
            with st.expander(f"{category.title()} Skills (✅ {len(matched_skills)} | ❌ {len(missing_skills)})"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**✅ Matched Skills**")
                    if matched_skills:
                        for skill in matched_skills:
                            st.markdown(f"<span class='pill'>{skill}</span>", unsafe_allow_html=True)
                    else:
                        st.write("—")
                
                with col2:
                    st.markdown("**❌ Missing Skills**")
                    if missing_skills:
                        for skill in missing_skills:
                            st.markdown(f"<span class='pill bad'>{skill}</span>", unsafe_allow_html=True)
                    else:
                        st.write("—")
    
    # Course recommendations (inside O*NET section)
    st.markdown("#### 🎓 Personalized Learning Recommendations")
    
    if analysis_result["courses"]:
        for title, url in analysis_result["courses"]:
            st.markdown(f"- **[{title}]({url})**")
    else:
        st.info("No specific course recommendations available. Focus on developing the missing skills listed above.")
    
    st.markdown("---")

# =============================
# Sidebar (KEEP YOUR ORIGINAL)
# =============================
with st.sidebar:
    # Add logo in sidebar
    try:
        logo = Image.open("logo.png")
        st.image(logo, use_container_width=True)
    except:
        # If local file not found, try to load from GitHub
        try:
            logo_url = "https://raw.githubusercontent.com/alaminas52/AI-Powered-Resume-Analyzer-with-Job-Match-Suggestions/main/logo.png"
            logo = Image.open(requests.get(logo_url, stream=True).raw)
            st.image(logo, use_container_width=True)
        except:
            st.write("Logo image not found")
        
    st.header("Settings")
    # add unique keys to avoid duplicate IDs
    EMBEDDING_MODEL = st.selectbox("Embedding model", ["all-MiniLM-L6-v2"], index=0, key="embed_model")
    MAX_FILES = st.slider("Max resumes per batch", 1, 10, 5, key="max_files")
    VIEW = st.radio("Mode", ["HR View", "Candidate View"], index=0, key="view_mode")
    
    # Add O*NET toggle
    USE_ONET = st.checkbox("Use O*NET Skills Database", value=True, 
                          help="Enhanced skills analysis using official O*NET taxonomy", key="use_onet")

st.markdown("---")

st.subheader("Step 1 — Paste/Upload Job Description")
job_title = st.text_input("Job Title (optional)", key="job_title")
job_description = st.text_area("Job Description", height=180, key="job_description")

# =============================
# Step 2 uploader (different behavior for HR vs Candidate) - KEEP YOUR ORIGINAL
# =============================
if VIEW == "HR View":
    st.subheader(f"Step 2 — Upload candidate resumes (up to {MAX_FILES} files)")
    uploaded_files = st.file_uploader("Upload resumes", type=['pdf','txt','docx'], accept_multiple_files=True, key="uploader_hr")
    if uploaded_files and len(uploaded_files) > MAX_FILES:
        st.warning(f"You uploaded {len(uploaded_files)} files but the current limit is {MAX_FILES}. Only the first {MAX_FILES} will be processed.")
        uploaded_files = uploaded_files[:MAX_FILES]
else:
    st.subheader("Step 2 — Upload your resume (single file)")
    uploaded_file = st.file_uploader("Upload your resume", type=['pdf','txt','docx'], accept_multiple_files=False, key="uploader_candidate")
    # normalize to list so downstream code can reuse same logic
    uploaded_files = [uploaded_file] if uploaded_file else []

process_button = st.button("Analyze batch", key="analyze_btn")

if process_button:
    if not job_description.strip():
        st.error("Please paste the Job Description in Step 1 before analyzing.")
    else:
        with st.spinner("Processing resumes..."):
            model = load_embedding_model(EMBEDDING_MODEL)

            resume_texts, filenames, skill_lists, contacts = [], [], [], []
            ocr_used_files = []
            ocr_errors = []

            for f in uploaded_files:
                # if file is None skip
                if f is None:
                    continue
                fname = f.name
                data = f.read()
                text = ""
                used_ocr = False
                errors = []

                try:
                    if fname.lower().endswith('.pdf'):
                        text, used_ocr, errors = extract_text_from_pdf(data)
                    elif fname.lower().endswith('.txt'):
                        text = extract_text_from_txt(data)
                    elif fname.lower().endswith('.docx'):
                        try:
                            import docx
                            doc = docx.Document(io.BytesIO(data))
                            text = "\n".join([p.text for p in doc.paragraphs])
                        except Exception as e:
                            errors.append(f"docx read error: {e}")
                    else:
                        text = ""
                except Exception as e:
                    errors.append(str(e))

                if used_ocr:
                    ocr_used_files.append(fname)
                if errors:
                    ocr_errors.extend([f"{fname}: {m}" for m in errors])

                if not text:
                    st.warning(f"No text extracted from {fname} — the file may be a scanned PDF or encrypted. OCR attempted if available.")

                resumeskills = extract_skills(text)
                contact = extract_contact_info(text)

                resume_texts.append(text)
                filenames.append(fname)
                skill_lists.append(resumeskills)
                contacts.append(contact)

            sims = compute_similarity(model, job_description, resume_texts) if resume_texts else []

            df = pd.DataFrame({
                'filename': filenames,
                'similarity': sims,
                'skills': [', '.join(s) for s in skill_lists],
                'text_snippet': [(t[:400] + '...') if len(t) > 400 else t for t in resume_texts],
                'name': [c[0] for c in contacts],
                'email': [c[1] for c in contacts],
                'phone': [c[2] for c in contacts],
                'experience_years': [c[3] if c[3] is not None else '' for c in contacts]
            })
            if not df.empty:
                df['similarity_pct'] = (df['similarity'] * 100).round(2)
                df = df.sort_values('similarity', ascending=False).reset_index(drop=True)
            else:
                df['similarity_pct'] = []

        # Inform about OCR usage / errors
        if ocr_used_files:
            st.info(f"OCR was used for: {', '.join(ocr_used_files)}")
            st.markdown("""
        If OCR results look poor, install Tesseract and the `pytesseract` Python package.

        **Windows:** install Tesseract from [GitHub](https://github.com/tesseract-ocr/tesseract) and add it to PATH.  
        **Ubuntu/Debian:** `sudo apt install tesseract-ocr`  
        Then run: `pip install pytesseract pillow`.
        """)

        if ocr_errors:
            st.write("\n")
            st.warning("Some extraction warnings/errors occurred (see below).")
            for e in ocr_errors:
                st.text(e)

        # Branch behaviour based on selected view
        if VIEW == "HR View":
            # ===== HR view: show batch ranking, charts, downloads =====
            st.subheader(f"Batch Results — Top {min(5, len(df))} Resumes")
            top_df = df.head(min(5, len(df))) if not df.empty else pd.DataFrame()

            header_cols = st.columns([2, 1, 2, 3])
            header_cols[0].markdown("**Resume**")
            header_cols[1].markdown("**Score (%)**")
            header_cols[2].markdown("**Skills**")
            header_cols[3].markdown("**Candidate Info**")

            for i, row in top_df.iterrows():
                c0, c1, c2, c3 = st.columns([2, 1, 2, 3])
                c0.write(row['filename'])
                c1.write(f"{row['similarity_pct']}")
                c2.write(row['skills'])
                exp_text = f"{row['experience_years']} years" if row['experience_years'] != '' else "Not specified"
                info = f"Name: {row['name']}\nEmail: {row['email']}\nPhone: {row['phone']}\nExperience: {exp_text}"
                c3.write(info)

            st.markdown("---")

            # ===== Your Original Skill Gap Analysis (as fallback) =====
            if not USE_ONET:
                st.subheader("Skill Gap Analysis & Course Suggestions (Top Candidates)")
                jd_skills = extract_skills(job_description)
                st.markdown(f"**JD Skills Detected:** {' '.join([f'`{s}`' for s in jd_skills]) if jd_skills else 'None detected'}")

                for i, row in top_df.iterrows():
                    st.markdown(f"### {row['filename']}")
                    resume_skills = [s.strip() for s in row['skills'].split(',') if s.strip()]
                    jd_skill_list = jd_skills
                    missing = [s for s in jd_skill_list if s not in resume_skills]

                    colL, colR = st.columns(2)
                    with colL:
                        st.markdown("**✅ Matched Skills**")
                        matched = [s for s in jd_skill_list if s in resume_skills]
                        if matched:
                            st.markdown(" ".join([f"<span class='pill'>{m}</span>" for m in matched]), unsafe_allow_html=True)
                        else:
                            st.write("—")
                    with colR:
                        st.markdown("**❌ Missing Skills**")
                        if missing:
                            st.markdown(" ".join([f"<span class='pill bad'>{m}</span>" for m in missing]), unsafe_allow_html=True)
                        else:
                            st.write("—")

                    # Courses with links
                    courses = []
                    for s in missing:
                        if s in SKILL_COURSES:
                            courses.append(SKILL_COURSES[s])
                    if courses:
                        st.markdown("**🎓 Recommended Courses**")
                        for title, url in courses:
                            st.markdown(f"- [{title}]({url})")
                    st.markdown("<hr>", unsafe_allow_html=True)

            # ===== Detailed Table + Downloads =====
            st.subheader("Detailed Results")
            if not df.empty:
                st.dataframe(df[['filename', 'similarity_pct', 'skills', 'email', 'phone', 'experience_years']])
                csv_bytes = df[['filename', 'similarity_pct', 'skills', 'email', 'phone', 'experience_years']].to_csv(index=False).encode('utf-8')
                st.download_button("Download results as CSV", data=csv_bytes, file_name="batch_match_results.csv", mime='text/csv', key="dl_csv_hr")
                pdf_report = generate_pdf_report(df[['filename','similarity_pct','skills','email','phone','experience_years']], job_description)
                st.download_button("Download results as PDF", data=pdf_report, file_name="batch_match_results.pdf", mime='application/pdf', key="dl_pdf_hr")
                st.success("Batch analysis complete.")
            else:
                st.info("No resumes were processed.")

            # ===== Charts (smaller) =====
            if not df.empty:
                st.subheader("Top Match Score Chart")
                fig, ax = plt.subplots(figsize=(6, 3))
                bars = ax.bar(top_df['filename'], top_df['similarity_pct'])
                ax.set_ylabel('Match Score (%)')
                ax.set_xlabel('Resume')
                ax.set_title('Top Resume Matches')
                plt.xticks(rotation=30, ha='right')
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
                try:
                    best_match_idx = top_df['similarity_pct'].idxmax()
                    bars[best_match_idx].set_color('#1E88E5')
                    best_score = top_df['similarity_pct'].max()
                    best_file = top_df.loc[best_match_idx, 'filename']
                    ax.set_title(f"Best Match: {best_file} ({best_score:.1f}%)", fontsize=11, fontweight='bold')
                except Exception:
                    pass
                centered = st.columns([1, 6, 1])
                with centered[1]:
                    st.pyplot(fig, use_container_width=False)

            # ===== O*NET Enhanced Analysis Section (moved to bottom of HR view) =====
            if USE_ONET and not df.empty:
                st.subheader("🎯 O*NET Enhanced Skills Analysis")
                st.info("Using O*NET Skills Database for comprehensive skills matching across professions")

                for i, row in top_df.iterrows():
                    # Find the corresponding resume text
                    idx = df[df['filename'] == row['filename']].index
                    if len(idx) == 0:
                        continue
                    resume_text = resume_texts[idx[0]]
                    analysis_result = enhanced_onet_analysis(resume_text, job_description)
                    # Display O*NET analysis (matched/missing skills + courses inside section)
                    display_onet_analysis(analysis_result, row['filename'])

        else:
            # ===== Candidate View: personalized single-resume analysis =====
            st.header("Candidate View — Personalized Analysis")
            if not uploaded_files:
                st.info("Upload your resume (PDF/TXT/DOCX) to get a personal analysis.")
            else:
                # Use first (and only) uploaded file for candidate-focused display
                f = uploaded_files[0]
                if f is None:
                    st.info("Upload your resume (single file) to get analysis.")
                else:
                    fname = f.name
                    # find corresponding row in df
                    try:
                        row = df[df['filename'] == fname].iloc[0]
                        resume_text = resume_texts[df[df['filename'] == fname].index[0]]
                    except Exception:
                        row = None
                        resume_text = ""

                    st.subheader(f"Analysis for: {fname}")
                    if row is not None:
                        score = row['similarity_pct']
                        st.metric("Match Score (%)", f"{score}")
                        st.markdown("**Candidate Info**")
                        st.write(f"Name: {row['name']}")
                        st.write(f"Email: {row['email']}")
                        exp_text = f"{row['experience_years']} years" if row['experience_years'] != '' else "Not specified"
                        st.write(f"Experience: {exp_text}")

                        # ===== O*NET Enhanced Analysis for Candidate View (kept inside candidate view) =====
                        if USE_ONET:
                            analysis_result = enhanced_onet_analysis(resume_text, job_description)
                            display_onet_analysis(analysis_result, fname)
                        else:
                            # ===== Your Original Candidate View Analysis =====
                            resume_skills = [s.strip() for s in row['skills'].split(',') if s.strip()]
                            jd_skill_list = extract_skills(job_description)
                            matched = [s for s in jd_skill_list if s in resume_skills]
                            missing = [s for s in jd_skill_list if s not in resume_skills]

                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**✅ Matched Skills**")
                                if matched:
                                    st.markdown(" ".join([f"<span class='pill'>{m}</span>" for m in matched]), unsafe_allow_html=True)
                                else:
                                    st.write("—")
                            with col2:
                                st.markdown("**❌ Missing Skills**")
                                if missing:
                                    st.markdown(" ".join([f"<span class='pill bad'>{m}</span>" for m in missing]), unsafe_allow_html=True)
                                else:
                                    st.write("—")

                            # Recommended courses
                            courses = []
                            for s in missing:
                                if s in SKILL_COURSES:
                                    courses.append(SKILL_COURSES[s])
                            if courses:
                                st.markdown("**🎓 Recommended Courses**")
                                for title, url in courses:
                                    st.markdown(f"- [{title}]({url})")
                            else:
                                st.markdown("**🎓 Recommended Courses**")
                                st.write("No direct course matches in the curated dataset. Consider learning resources on the missing topics.")

                            # Simple improvement suggestions (heuristic)
                            st.markdown("**🛠 Improvement Suggestions**")
                            tips = []
                            if score < 40:
                                tips.append("Consider tailoring your resume more closely to the job description — add relevant keywords and concrete project outcomes.")
                            if 'python' in missing and 'python' in SKILL_DB:
                                tips.append("Add details on Python projects (repo links, frameworks used).")
                            if 'sql' in missing:
                                tips.append("Include SQL examples: queries, datasets, or reporting work.")
                            if 'aws' in missing:
                                tips.append("Mention cloud experience (services used, deployments).")
                            if not tips:
                                tips.append("Good alignment. Consider highlighting measurable achievements (numbers/impact) to boost score further.")
                            for t in tips:
                                st.write(f"- {t}")

                    else:
                        st.warning("Could not find analysis row for the uploaded resume. Re-run the analysis or upload a single resume and try again.")

# Add footer with developer credit and GitHub link
st.markdown("---")
st.markdown(
    """
    <div class="footer">
        Developed by <a href="https://github.com/alaminas52" target="_blank">AI-Amin</a> | 
        <a href="https://github.com/alaminas52/AI-Powered-Resume-Analyzer-with-Job-Match-Suggestions" target="_blank">GitHub Repository</a>
    </div>
    """,
    unsafe_allow_html=True
)
