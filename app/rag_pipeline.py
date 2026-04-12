import os
import google.generativeai as genai
from groq import Groq
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from typing import List

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class GeminiEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model="models/gemini-embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result["embedding"])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=text,
            task_type="retrieval_query"
        )
        return result["embedding"]


def build_vectorstore(chunks):
    print("Building vector store...")
    embeddings = GeminiEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("Vector store ready!")
    return vectorstore


def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print("RAG chain ready!")
    return retriever


def ask_question(retriever, question: str):
    source_docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in source_docs)

    prompt = f"""You are a helpful assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't have enough information to answer this."

Context:
{context}

Question:
{question}

Answer:"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    sources = []
    for i, doc in enumerate(source_docs):
        sources.append({
            "chunk": i + 1,
            "page": doc.metadata.get("page", "unknown"),
            "preview": doc.page_content[:150]
        })

    return {
        "answer": response.choices[0].message.content,
        "sources": sources
    }