"""
Code Explainer using CrewAI (agents, tasks, crew) with LangChain LLM.
Takes a code snippet (file or pasted) and explains it in plain English.

Usage:
    export OPENAI_API_KEY="sk-..."
    python code_explainer_crewai.py path/to/code.py
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# ✅ LangChain LLM
from langchain_openai import ChatOpenAI

# ✅ CrewAI orchestration
from crewai import Agent, Task, Crew

# Load environment variables
load_dotenv()


# ---------------------------
# CrewAI Code Explainer
# ---------------------------
def explain_code_with_crewai(code: str) -> str:
    """Run CrewAI agent + task to explain code in plain English."""
    llm = ChatOpenAI(temperature=0)

    # Define the agent
    explainer_agent = Agent(
        role="Senior Developer and Teacher",
        goal="Explain code in simple, beginner-friendly language.",
        backstory="You are an expert software engineer and mentor who can simplify complex code.",
        llm=llm,
        verbose=False,
    )

    # Define the task
    task_description = f"""
    Explain the following code in clear, simple terms for a beginner.
    Highlight:
    - what the code does
    - main functions/classes and responsibilities
    - potential gotchas
    - a one-sentence summary

    Code:
    {code}
    """

    explanation_task = Task(
        description=task_description,
        expected_output="A plain-English explanation of the code, formatted for clarity.",
        agent=explainer_agent,
    )

    # Create a crew and run it
    crew = Crew(agents=[explainer_agent], tasks=[explanation_task], verbose=False)
    result = crew.kickoff()

    # ✅ Extract clean output (instead of full CrewResult object)
    if hasattr(result, "raw"):
        return result.raw
    elif hasattr(result, "output"):
        return result.output
    return str(result)


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python code_explainer_crewai.py path/to/code.py")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print("❌ File not found:", path)
        sys.exit(1)

    code_text = path.read_text()
    explanation = explain_code_with_crewai(code_text)

    print("\n--- Explanation ---\n")
    print(explanation)
