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
import requests  # Add this import

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
# Data & Helpers
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

# Course links (clickable)
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

# Extract skills using alias matching (deterministic)
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

# Contact & experience extraction
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


# Robust PDF text extraction with per-page OCR fallback

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

# Simple PDF/text reader

def extract_text_from_txt(file_bytes):
    try:
        return file_bytes.decode('utf-8', errors='ignore')
    except:
        return file_bytes.decode('latin1', errors='ignore')

# Generate PDF report

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
# UI
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
    EMBEDDING_MODEL = st.selectbox("Embedding model", ["all-MiniLM-L6-v2"], index=0)
    MAX_FILES = st.slider("Max resumes per batch", 1, 10, 5)
    VIEW = st.radio("Mode", ["HR View", "Candidate View"], index=0)

st.markdown("---")

st.subheader("Step 1 — Paste/Upload Job Description")
job_title = st.text_input("Job Title (optional)")
job_description = st.text_area("Job Description", height=180)

st.subheader(f"Step 2 — Upload candidate resumes (up to {MAX_FILES} files)")
uploaded_files = st.file_uploader("Upload resumes", type=['pdf','txt','docx'], accept_multiple_files=True)

if uploaded_files and len(uploaded_files) > MAX_FILES:
    st.warning(f"You uploaded {len(uploaded_files)} files but the current limit is {MAX_FILES}. Only the first {MAX_FILES} will be processed.")
    uploaded_files = uploaded_files[:MAX_FILES]

process_button = st.button("Analyze batch")

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

            sims = compute_similarity(model, job_description, resume_texts)

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
            df['similarity_pct'] = (df['similarity'] * 100).round(2)
            df = df.sort_values('similarity', ascending=False).reset_index(drop=True)

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

        # ===== Summary Top N =====
        st.subheader(f"Batch Results — Top {min(5, len(df))} Resumes")
        top_df = df.head(min(5, len(df)))

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

        # ===== Detailed Table + Downloads =====
        st.subheader("Detailed Results")
        st.dataframe(df[['filename', 'similarity_pct', 'skills', 'email', 'phone', 'experience_years']])

        csv_bytes = df[['filename', 'similarity_pct', 'skills', 'email', 'phone', 'experience_years']].to_csv(index=False).encode('utf-8')
        st.download_button("Download results as CSV", data=csv_bytes, file_name="batch_match_results.csv", mime='text/csv')

        pdf_report = generate_pdf_report(df[['filename','similarity_pct','skills','email','phone','experience_years']], job_description)
        st.download_button("Download results as PDF", data=pdf_report, file_name="batch_match_results.pdf", mime='application/pdf')

        st.success("Batch analysis complete.")

      # ===== Charts (smaller, centered, no overlaps) =====
        st.subheader("Top Match Score Chart")

        # Create smaller figure
        fig, ax = plt.subplots(figsize=(6, 3))  

        bars = ax.bar(top_df['filename'], top_df['similarity_pct'])
        ax.set_ylabel('Match Score (%)')
        ax.set_xlabel('Resume')
        ax.set_title('Top Resume Matches')

        # Rotate x-labels for readability
        plt.xticks(rotation=30, ha='right')

        # Add value labels above bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

        # Highlight the best match
        best_match_idx = top_df['similarity_pct'].idxmax()
        bars[best_match_idx].set_color('#1E88E5')
        best_score = top_df['similarity_pct'].max()
        best_file = top_df.loc[best_match_idx, 'filename']

        # Put annotation above chart instead of overlapping bars
        ax.set_title(f"Best Match: {best_file} ({best_score:.1f}%)", fontsize=11, fontweight='bold')

        # Center the chart in Streamlit
        centered = st.columns([1, 6, 1])   # left, middle, right columns
        with centered[1]:
            st.pyplot(fig, use_container_width=False)
                



        # ===== Skill Gap & Course Suggestions (icon UI) =====
        st.subheader("Skill Gap Analysis & Course Suggestions")
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

        # ===== Candidate View (self-improvement) =====
        if VIEW == "Candidate View":
            st.subheader("Your Development Plan (Based on JD)")
            st.markdown("Use the gaps above to pick 2–3 skills to improve first. Track progress weekly and update your resume accordingly.")

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
