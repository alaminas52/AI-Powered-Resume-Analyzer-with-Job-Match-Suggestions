
import json

# Resume skills from NLP output
resume_skills = ['c++', 'quickbooks', 'sql', 'powerpoint', 'excel']

# Load job data
with open("data/job_data.json", "r") as f:
    job_data = json.load(f)

# Load course data
with open("data/courses.json", "r") as f:
    course_data = json.load(f)

# Match logic
def calculate_match_score(resume_skills, job_skills):
    matched = [skill for skill in job_skills if skill in resume_skills]
    missing = [skill for skill in job_skills if skill not in resume_skills]
    score = (len(matched) / len(job_skills)) * 100
    return round(score, 2), matched, missing

# Print results
print("🔍 Job Match Suggestions:\n")
for job in job_data:
    score, matched_skills, missing_skills = calculate_match_score(resume_skills, job['required_skills'])

    print(f"🧑‍💻 Job Title: {job['job_title']}")
    print(f"✅ Match Score: {score}%")
    print(f"🎯 Matched Skills: {matched_skills}")
    print(f"❌ Missing Skills: {missing_skills}")

    # Recommend courses
    if missing_skills:
        print("📘 Suggested Courses:")
        for skill in missing_skills:
            if skill in course_data:
                for course in course_data[skill]:
                    print(f"   - {course}")
    print("-" * 50)
