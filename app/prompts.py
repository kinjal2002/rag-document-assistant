from langchain.prompts import PromptTemplate

# This is the instruction we give to the LLM
# {context} = the relevant chunks retrieved from the document
# {question} = the user's question

RAG_PROMPT_TEMPLATE = """
You are a helpful assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't have enough information to answer this."

Context:
{context}

Question:
{question}

Answer:
"""

# Wrap it in LangChain's PromptTemplate object
rag_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=RAG_PROMPT_TEMPLATE
)