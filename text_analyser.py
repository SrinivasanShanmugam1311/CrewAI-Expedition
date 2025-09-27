from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI   # ✅ use LangChain LLM
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Step 1: Define the LLM (LangChain)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Step 2: Define the Agent and attach the LLM
analyzer_agent = Agent(
    role="Text Analyzer",
    goal="Analyze text and return counts in JSON format.",
    backstory="You are an expert text analyzer.",
    llm=llm,   # ✅ Attach LangChain LLM here
    #verbose=True,
    allow_delegation=False
)

# Step 3: Define the Task for the Agent
txt_file = Path("sample.txt")
text = txt_file.read_text()

analysis_task = Task(
    description=f"""
    Analyze the text below and return:
    1) Number of characters
    2) Number of words
    3) Number of paragraphs
    4) Number of sentences

    Text:
    {text}

    Output JSON with keys:
    characters, words, paragraphs, sentences
    """,
    agent=analyzer_agent,
    expected_output="JSON object with keys: characters, words, paragraphs, sentences"
)

# Step 4: Create a Crew with that Agent
crew = Crew(
    agents=[analyzer_agent],  # ✅ Crew knows which Agent (with LangChain LLM) to use
    tasks=[analysis_task],
    #verbose=True,
    tracing=False
)

# Step 5: Run the Crew
result = crew.kickoff()
print(result)
