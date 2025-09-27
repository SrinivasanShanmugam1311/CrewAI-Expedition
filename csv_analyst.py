"""
CSV Analyst using CrewAI (agents, tasks, crew) and LangChain LLM.
Answers natural language questions about a CSV file.
"""

import sys
import pandas as pd
from dotenv import load_dotenv

# ✅ LLM from LangChain
from langchain_openai import ChatOpenAI

# ✅ CrewAI orchestration
from crewai import Agent, Task, Crew

# Load environment variables
load_dotenv()


# ---------------------------
# CrewAI CSV Analyst
# ---------------------------
def run_agent(csv_path: str):
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"📊 CSV loaded successfully with shape {df.shape} and columns {list(df.columns)}")

    # LangChain LLM
    llm = ChatOpenAI(temperature=0)

    # CrewAI Agent
    csv_agent = Agent(
        role="CSV Analyst",
        goal="Answer questions about the CSV dataset accurately using Python and pandas.",
        backstory="A data analyst skilled in interpreting CSV files and extracting insights using pandas.",
        llm=llm,
        verbose=True
    )

    print("🤖 Ask questions about the data (type 'exit' to quit).")

    while True:
        q = input("\nQuestion: ").strip()
        if q.lower() in ("exit", "quit"):
            break

        # Pass DataFrame as context
        qa_task = Task(
            description=f"""You are given a CSV dataset with shape {df.shape} and columns {list(df.columns)}.
Answer the following question using pandas operations only:

Question: {q}""",
            expected_output="A concise, accurate answer in natural language (with calculations if needed).",
            agent=csv_agent
        )

        crew = Crew(agents=[csv_agent], tasks=[qa_task], verbose=True)
        result = crew.kickoff()

        print("\n✅ Answer:\n", result)


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python csv_analyst_crewai.py data.csv")
        sys.exit(1)

    run_agent(sys.argv[1])
