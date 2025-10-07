
import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Sample resume text (copy it from your extracted result or load from extract_text.py)
with open("D:/Al-AMin/resume-analyzer/resume_samples/resume_text.txt", "r", encoding="utf-8") as f:
    resume_text = f.read()

# Process text with spaCy
doc = nlp(resume_text)

# Function to extract NAME (looks for proper nouns in the top part)
def extract_name(doc):
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return "Not Found"

# Sample list of common tech skills
SKILL_KEYWORDS = [
    "python", "java", "c++", "sql", "excel", "machine learning", "deep learning",
    "html", "css", "react", "django", "tensorflow", "keras", "powerpoint", "quickbooks"
]

def extract_skills(text):
    skills = []
    text_lower = text.lower()
    for skill in SKILL_KEYWORDS:
        if skill.lower() in text_lower:
            skills.append(skill)
    return list(set(skills))

# Extract degree keywords
DEGREE_KEYWORDS = ["bachelor", "master", "bsc", "msc", "phd", "diploma"]

def extract_education(text):
    text_lower = text.lower()
    edu = []
    for word in DEGREE_KEYWORDS:
        if word in text_lower:
            edu.append(word)
    return list(set(edu))

# Show results
print("🔹 Name:", extract_name(doc))
print("🔹 Skills:", extract_skills(resume_text))
print("🔹 Education:", extract_education(resume_text))
