import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split_document(file_path: str):
    
    
    # Step 1: Load the PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # Step 2: Split into chunks
    # chunk_size = max characters per chunk
    # chunk_overlap = how many characters overlap between chunks
    # (overlap prevents losing meaning at chunk boundaries)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    
    chunks = splitter.split_documents(documents)
    
    print(f"Document loaded: {len(documents)} page(s)")
    print(f"Total chunks created: {len(chunks)}")
    
    return chunks


def get_upload_path(filename: str) -> str:
    """
    Returns the full path where an uploaded file should be saved.
    """
    uploads_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    return os.path.join(uploads_dir, filename)