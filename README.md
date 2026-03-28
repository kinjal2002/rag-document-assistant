# \# RAG Document Assistant 🤖

# 

# A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and query them using natural language.

# 

# \## 🚀 Features

# \- Upload any PDF document

# \- Ask natural language questions about the document

# \- Get accurate, context-aware answers powered by Google Gemini

# \- See which document chunks were used to generate the answer

# \- REST API built with FastAPI

# 

# \## 🛠️ Tech Stack

# \- \*\*Python\*\* - Core language

# \- \*\*LangChain\*\* - AI orchestration framework

# \- \*\*Google Gemini API\*\* - LLM and embeddings

# \- \*\*FAISS\*\* - Vector similarity search

# \- \*\*FastAPI\*\* - REST API backend

# \- \*\*Uvicorn\*\* - ASGI server

# 

# \## ⚙️ Setup

# 

# 1\. Clone the repository

# 2\. Create virtual environment: `python -m venv venv`

# 3\. Activate: `venv\\Scripts\\activate`

# 4\. Install dependencies: `pip install -r requirements.txt`

# 5\. Add your Gemini API key to `.env`:

# ```

# GOOGLE\_API\_KEY=your-key-here

# ```

# 6\. Run the server: `uvicorn app.main:app --reload`

# 7\. Open `http://127.0.0.1:8000/docs`

# 

# \## 📌 API Endpoints

# | Method | Endpoint | Description |

# |--------|----------|-------------|

# | GET | `/` | Health check |

# | POST | `/upload` | Upload a PDF document |

# | POST | `/ask` | Ask a question about the document |

# | GET | `/health` | Check if document is loaded |

# 

# \## 🏗️ Architecture

# ```

# PDF Upload → Text Extraction → Chunking → Vector Embeddings → FAISS Index

# &#x20;                                                                     ↓

# User Question → Embed Question → Similarity Search → Top 3 Chunks → Gemini LLM → Answer

# ```

