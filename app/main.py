import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.utils import load_and_split_document, get_upload_path
from app.rag_pipeline import build_vectorstore, build_rag_chain, ask_question, submit_feedback

app = FastAPI(
    title="RAG Document Assistant",
    description="Upload a document and ask questions about it!",
    version="1.0.0"
)

retriever = None


@app.get("/")
def root():
    return {"message": "RAG Document Assistant is running!"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global retriever

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    save_path = get_upload_path(file.filename)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"File saved: {save_path}")

    chunks = load_and_split_document(save_path)
    vectorstore = build_vectorstore(chunks)
    retriever = build_rag_chain(vectorstore)

    return {
        "message": f"Document '{file.filename}' processed successfully!",
        "chunks_created": len(chunks)
    }


class QuestionRequest(BaseModel):
    question: str


class FeedbackRequest(BaseModel):
    run_id: str
    score: int   # 1 = helpful, 0 = not helpful
    comment: str = ""


@app.post("/feedback")
def feedback(request: FeedbackRequest):
    if request.score not in (0, 1):
        raise HTTPException(status_code=400, detail="score must be 0 or 1.")
    submit_feedback(request.run_id, request.score, request.comment)
    return {"message": "Feedback submitted to LangSmith."}


@app.post("/ask")
def ask(request: QuestionRequest):
    global retriever

    if retriever is None:
        raise HTTPException(
            status_code=400,
            detail="No document uploaded yet. Please upload a PDF first."
        )

    result = ask_question(retriever, request.question)
    return result


@app.get("/health")
def health():
    return {
        "status": "ok",
        "document_loaded": retriever is not None
    }