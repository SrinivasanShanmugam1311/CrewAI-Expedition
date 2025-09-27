"""
Given a list of article URLs, fetch (via newspaper3k), summarize each article,
and produce a short digest.

Usage:
    export OPENAI_API_KEY="sk-..."
    python news_summarizer.py https://example.com/article1 https://example.com/article2
"""

import sys
from dotenv import load_dotenv
from newspaper import Article
from langchain_openai import ChatOpenAI   # ✅ LLM from LangChain
from crewai import Agent, Task, Crew      # ✅ Agent/Task/Crew from CrewAI

# Load env vars
load_dotenv()

SUMMARY_PROMPT = """You are a concise news summarizer.
                    Given the article text, produce:
                    1) Headline (single line)
                    2) 2-3 sentence summary
                    3) 1-sentence "why it matters"

                    Article:
                    {article}
                    """

# --------------------------
# Article Fetcher
# --------------------------
def fetch_article(url: str) -> str:
    art = Article(url)
    art.download()
    art.parse()
    return art.title + "\n\n" + art.text


# --------------------------
# CrewAI Summarization
# --------------------------
def summarize_with_crewai(article_text: str, llm) -> str:
    summarizer_agent = Agent(
        role="News Summarizer",
        goal="Summarize news articles clearly and concisely.",
        backstory="Expert journalist who creates short, impactful digests.",
        llm=llm,
        verbose=True
    )

    task = Task(
        description=SUMMARY_PROMPT.format(article=article_text),
        expected_output="Headline + 2-3 sentence summary + why it matters",
        agent=summarizer_agent
    )

    crew = Crew(agents=[summarizer_agent], tasks=[task], verbose=True)
    result = crew.kickoff()

    return result.raw  # raw response string


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python news_summarizer.py <article_url> [article_url...]")
        sys.exit(1)

    # ✅ Define LLM from LangChain
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    urls = sys.argv[1:]
    for u in urls:
        print(f"\nFetching: {u}")
        try:
            text = fetch_article(u)
            print("Summarizing with CrewAI...")
            out = summarize_with_crewai(text, llm)
            print("\n--- Summary ---\n", out)
        except Exception as e:
            print("❌ Failed for", u, e)
