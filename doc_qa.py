"""
Document Q&A using LangChain (LLM + embeddings) and CrewAI (agents, tasks, crew).
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# ✅ LangChain PDF + embeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# ✅ CrewAI orchestration
from crewai import Agent, Task, Crew

# Load environment variables
load_dotenv()


# ---------------------------
# Build VectorStore
# ---------------------------
def build_vectorstore(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    emb = OpenAIEmbeddings()
    db = FAISS.from_documents(chunks, emb)
    return db


# ---------------------------
# CrewAI Q&A
# ---------------------------
def run_doc_qa(db, llm):
    retriever = db.as_retriever(search_kwargs={"k": 4})

    qa_agent = Agent(
        role="Document Q&A Assistant",
        goal="Answer user questions based only on the provided PDF document.",
        backstory="An expert assistant that retrieves and summarizes information from uploaded PDFs.",
        llm=llm,
        verbose=True
    )

    print("📖 Document Q&A. Type 'exit' to quit.")
    while True:
        q = input("\nQuestion: ").strip()
        if q.lower() in ("exit", "quit"):
            break

        # Get document chunks from retriever
        context_docs = retriever.get_relevant_documents(q)
        context_text = "\n\n".join([doc.page_content for doc in context_docs])

        # CrewAI task
        qa_task = Task(
            description=f"""Answer the question using only the following document context:

{context_text}

Question: {q}""",
            expected_output="A concise, accurate answer in natural language.",
            agent=qa_agent
        )

        crew = Crew(agents=[qa_agent], tasks=[qa_task], verbose=True)
        result = crew.kickoff()

        print("\n✅ Answer:\n", result)


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python doc_qa_crewai.py /path/to/doc.pdf")
        sys.exit(1)

    pdf = sys.argv[1]
    if not Path(pdf).exists():
        print("❌ PDF not found:", pdf)
        sys.exit(1)

    print("📚 Building vectorstore (this may take a minute)...")
    db = build_vectorstore(pdf)
    print("✅ Vectorstore ready.")

    # ✅ LangChain LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Run interactive Q&A via CrewAI
    run_doc_qa(db, llm)
