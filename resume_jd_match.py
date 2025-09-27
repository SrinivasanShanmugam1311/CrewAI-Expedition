"""
Resume vs Job Description Matching using CrewAI.

Supports: .txt, .pdf, .doc/.docx resumes
Usage:
    export OPENAI_API_KEY="sk-..."
    python resume_jd_match_crewai.py resume.pdf job_description.txt
"""

import re
import sys
from pathlib import Path
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import docx
from crewai import Agent, Task, Crew
from langchain_openai import OpenAI

# Load .env variables
load_dotenv()

# -------------------------
# Step 1: Read resume file
# -------------------------
def read_resume(file_path):
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"{file_path} not found")
    
    if p.suffix.lower() == ".txt":
        return p.read_text()
    
    elif p.suffix.lower() == ".pdf":
        text = ""
        reader = PdfReader(str(p))
        for page in reader.pages:
            text += page.extract_text() + " "
        return text.strip()
    
    elif p.suffix.lower() in [".doc", ".docx"]:
        doc = docx.Document(str(p))
        text = " ".join([para.text for para in doc.paragraphs])
        return text.strip()
    
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")

# -------------------------
# Step 2: Preprocess text
# -------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    words = text.split()
    return " ".join(words)

# -------------------------
# Step 3: CrewAI Setup
# -------------------------
def run_crewai_match(resume_text, jd_text):
    # Define LLM
    llm = OpenAI(model="gpt-4o-mini", temperature=0)

    # Define agent
    matcher_agent = Agent(
        role="Resume Matcher",
        goal="Compare resumes with job descriptions using embeddings and reasoning.",
        backstory="You are an expert HR recruiter skilled in semantic matching of resumes with job descriptions.",
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    # Define task
    task = Task(
        description=f"""
        Analyze the given resume and job description.

        Resume:
        {resume_text}

        Job Description:
        {jd_text}

        Your job:
        1. Compute a semantic similarity score between the resume and JD (0-1).
        2. Identify overlapping keywords.
        3. List keywords that appear only in the resume.
        4. List keywords that appear only in the job description.

        Return the result strictly in JSON format with keys:
        similarity_score, matching_keywords, resume_only_keywords, jd_only_keywords
        """,
        agent=matcher_agent,
        expected_output="JSON with semantic similarity score and keyword analysis."
    )

    # Crew
    crew = Crew(
        agents=[matcher_agent],
        tasks=[task],
        verbose=True
    )

    result = crew.kickoff()
    return result

# -------------------------
# Step 4: Main
# -------------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python resume_jd_match_crewai.py resume_file jd_file")
        sys.exit(1)

    resume_file = sys.argv[1]
    jd_file = sys.argv[2]

    resume_text = read_resume(resume_file)
    jd_text = Path(jd_file).read_text()

    # Preprocess
    resume_text_clean = preprocess_text(resume_text)
    jd_text_clean = preprocess_text(jd_text)

    # Run CrewAI
    result = run_crewai_match(resume_text_clean, jd_text_clean)
    print("\nFinal Result:")
    print(result)
