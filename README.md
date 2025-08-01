
🤖 AI-Powered Resume Analyzer with Job Match Suggestions
This project uses NLP and transformer models to analyze a user's resume, extract relevant skills, and recommend best-matched job roles along with personalized course suggestions for missing skills. It also provides an overall match score and semantic similarity between the resume and job descriptions.

🔍 Features
✅ Resume skill extraction

📊 Proficiency estimation (Beginner, Intermediate, Advanced)

🧠 Semantic job matching using BERT

📘 Learning resource recommendations for missing skills

📥 Downloadable analysis report

🌐 Tailored for the Bangladeshi job market

🚀 Demo
Coming soon — Live Streamlit App Deployment

🛠️ How It Works
Upload Resume (PDF)

Resume text is extracted and analyzed

Skills are matched with predefined job profiles

Semantic similarity between resume and job roles is calculated

Missing skills are identified

Recommended courses are displayed

Report can be downloaded in .txt format

📁 Project Structure
bash
Copy
Edit
📦 AI-Powered-Resume-Analyzer
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── data/
│   ├── job_data.json       # Job roles and required skills
│   └── courses.json        # Courses for missing skill suggestions
└── README.md               # You're reading this!
⚙️ Installation & Run Locally
🧩 Prerequisites
Python 3.8+

pip or pipenv

Git (to clone the repo)

🔧 Setup
bash
Copy
Edit
# Clone the repository
git clone https://github.com/alaminas52/AI-Powered-Resume-Analyzer-with-Job-Match-Suggestions.git

# Navigate to the project directory
cd AI-Powered-Resume-Analyzer-with-Job-Match-Suggestions

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
▶️ Run the App
bash
Copy
Edit
streamlit run app.py
Then open the app in your browser (usually http://localhost:8501).

🧠 Technologies Used
Python

Streamlit

spaCy

PyMuPDF

Sentence Transformers (BERT)

Pandas

📄 Sample Resume Format
Make sure your resume includes:

Clearly mentioned skills (e.g., Python, SQL, Excel)

Role-specific experiences or projects

Keywords like "advanced", "intermediate", "familiar with", etc., for better proficiency estimation

📦 Sample Data
🔧 job_data.json
json
Copy
Edit
[
  {
    "job_title": "Data Analyst",
    "required_skills": ["excel", "sql", "data visualization", "statistics"]
  },
  {
    "job_title": "Web Developer",
    "required_skills": ["html", "css", "javascript", "react"]
  }
]
📚 courses.json
json
Copy
Edit
{
  "python": [
    { "name": "Learn Python", "url": "https://example.com/python" }
  ]
}
🧑‍🎓 Developed By
Al-Amin

MSc Thesis Project — AI Application in Healthcare
Supervised by [Your Supervisor's Name]

✅ To Do / Future Improvements
Add multilingual resume support

Deploy to Streamlit Cloud / Hugging Face Spaces

More job profiles for Bangladeshi market

Save user history & analytics

📃 License
This project is licensed under the MIT License.

