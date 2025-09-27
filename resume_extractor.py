import os
import json
import traceback
import argparse
import PyPDF2
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI  # ✅ LLM from LangChain
from crewai import Agent, Task, Crew     # ✅ Agent, Task, Crew from CrewAI


# --------------------------
# PDF Extraction Helper
# --------------------------
def extract_pdf_text(pdf_path: str) -> str:
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


# --------------------------
# Resume to JSON with CrewAI
# --------------------------
def resume_to_json(resume_text: str, llm):
    # Define Agent
    resume_agent = Agent(
        role="Resume Parser",
        goal="Convert resumes into structured JSON format",
        backstory="Expert at extracting structured information from unstructured resumes.",
        llm=llm,
        verbose=True
    )

    # Task description
    prompt = f"""
    Extract the following information from this resume text and return it strictly as valid JSON:
    {{
      "Name": "",
      "Email": "",
      "Phone": "",
      "Education": "",
      "Experience": [],
      "Skills": []
    }}

    Resume Text:
    {resume_text}

    IMPORTANT:
    - Return ONLY valid JSON
    - Do not include ```json or any extra text
    """

    resume_task = Task(
        description=prompt,
        expected_output="Valid JSON with resume details",
        agent=resume_agent
    )

    # Crew to run task
    crew = Crew(agents=[resume_agent], tasks=[resume_task], verbose=True)
    result = crew.kickoff()

    return result.raw  # raw response (should be JSON)


# --------------------------
# CLI Entry Point
# --------------------------
def main():
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in .env file")

    # Define LLM from LangChain
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

    parser = argparse.ArgumentParser(description="Extract resume info to JSON using CrewAI + LangChain")
    parser.add_argument("--pdf", type=str, help="Path to the PDF resume")
    parser.add_argument("--text", type=str, help="Direct resume text")
    parser.add_argument("--output", type=str, default="resume_output.json", help="Output JSON file name")
    args = parser.parse_args()

    try:
        # Get resume text
        if args.pdf:
            resume_text = extract_pdf_text(args.pdf)
        elif args.text:
            resume_text = args.text
        else:
            raise ValueError("Please provide either --pdf or --text")

        # Run CrewAI pipeline
        result_str = resume_to_json(resume_text, llm)

        # Try parsing JSON safely
        try:
            data = json.loads(result_str)
        except json.JSONDecodeError:
            print("⚠️ Model returned invalid JSON. Cleaning response...")
            cleaned = result_str.strip().replace("```json", "").replace("```", "")
            data = json.loads(cleaned)

        # Pretty print JSON
        pretty_json = json.dumps(data, indent=4)
        print(pretty_json)

        # Save JSON
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(pretty_json)
        print(f"\n✅ Saved JSON to {args.output}")

    except Exception as e:
        print("Error:", e)
        traceback.print_exc()


if __name__ == "__main__":
    main()
