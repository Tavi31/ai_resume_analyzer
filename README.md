# 🧠 AI Resume Analyzer (Core Backend)

This project implements the **core backend system** described in the *Project Proposal – AI Resume Analyzer*.

It provides:
- ✅ **Batch resume parsing & chunking** (PDF, TXT)
- ✅ **Embedding generation & vector storage** (Gemini + pgvector)
- ✅ **Vector similarity search** for job queries
- ✅ **Retrieval-Augmented Generation (RAG)** for candidate–role analysis
- ✅ **FastAPI REST endpoints** for programmatic access


---

## 📚 Tech Stack

| Component | Technology |
|------------|-------------|
| **Language** | Python 3.10+ |
| **LLM & Embeddings** | Google Gemini (`gemini-2.0-flash-lite` + `models/gemini-embedding-004`) |
| **Database** | PostgreSQL + `pgvector` extension (tested on [Neon.tech](https://neon.tech)) |
| **Server Framework** | FastAPI + Uvicorn |
| **ORM / Query Layer** | SQLAlchemy 2.x |
| **Text Parsing** | LangChain’s `RecursiveCharacterTextSplitter`, PyPDF |
| **Environment Management** | python-dotenv |
| **Vector Math** | NumPy |

---

## 🧩 Features Overview

| Feature | Description |
|----------|--------------|
| **1️⃣ Batch Parsing** | Reads all `.txt` / `.pdf` files in the `resumes/` directory, cleans & chunks them. |
| **2️⃣ Embedding Generation** | Uses Gemini embeddings (`gemini-embedding-004`, 768-dim) for each chunk. |
| **3️⃣ Storage** | Persists text + vectors in PostgreSQL using `pgvector`. |
| **4️⃣ Retrieval** | Performs top-K cosine similarity search between job description & resume embeddings. |
| **5️⃣ RAG Response** | Generates concise, cited answers with Gemini (`gemini-2.0-flash-lite`). |
| **6️⃣ FastAPI Endpoints** | `/ingest` for batch ingestion and `/chat` for interactive Q&A. |

---

## ⚙️ Setup Guide

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-user>/ai_resume_analyzer.git
cd ai_resume_analyzer

2️⃣ Create & Activate Virtual Environment
python -m venv .venv_pg
.\.venv_pg\Scripts\Activate.ps1
pip install --upgrade pip
(On Linux/macOS: source .venv_pg/bin/activate)

3️⃣ Install Dependencies
pip install -r requirements_pg.txt
Example requirements_pg.txt:
google-generativeai==0.8.3
fastapi==0.115.0
uvicorn==0.30.6
pydantic==2.9.2
python-dotenv==1.0.1
pypdf==5.0.1
numpy==1.26.4
psycopg[binary]
SQLAlchemy==2.0.36
pgvector==0.3.3
langchain==0.3.3
langchain-community==0.3.2

🔑 Environment Variables
Create a file named .env in your project root:
GEMINI_API_KEY=your_google_api_key_here
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:PORT/DBNAME
(For Neon, click “Connection Details” → Copy the psycopg3 connection URL and prefix with postgresql+psycopg://.)

🧱 PostgreSQL Setup (Neon / Local)
In your database SQL editor, run:
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS resume_chunks (
  id UUID PRIMARY KEY,
  source_file TEXT NOT NULL,
  source_path TEXT NOT NULL,
  chunk_index INT NOT NULL,
  text TEXT NOT NULL,
  embedding VECTOR(768) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_resume_chunks_embedding
  ON resume_chunks
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);

🧾 Directory Structure
ai_resume_analyzer/
├─ main.py
├─ vector_store_pg.py
├─ requirements_pg.txt
├─ .env
├─ resumes/
│  ├─ alice.txt
│  ├─ bob.txt
│  ├─ carol.txt
│  └─ (optional PDFs)
├─ job.txt
└─ README.md

🚀 Usage
1️⃣ Ingest Resumes
python main.py ingest --resumes ./resumes
Example output:
{
  "added_files": 4,
  "added_chunks": 11
}

2️⃣ Ask a Question (CLI)
python main.py ask --job ./job.txt --question "Which candidate is the best fit and why?"
Example output:
=== ANSWER ===

Alice Johnson is the best fit.

- Alice Johnson: Strong Python + ML pipeline background. [source: alice.txt#0]

=== CITATIONS ===
- alice.txt#0
- bob.txt#0
- carol.txt#0

3️⃣ API Mode (FastAPI)
Run the API server:
uvicorn main:app --host 0.0.0.0 --port 8000
Interactive docs:
👉 http://localhost:8000/docs

/ingest – POST
{
  "resumes_dir": "./resumes"
}
/chat – POST
{
  "job_description": "We are hiring a machine learning engineer...",
  "question": "Which candidate fits best and why?",
  "top_k": 6
}
💡 Example Prompt Flow
Job description is read from job.txt.

Embedding vector of job text + question is generated.

Most relevant resume chunks are retrieved via pgvector.

Gemini (Flash Lite) generates a concise answer referencing those snippets.

Output includes structured text and [source: filename#chunk] citations.

⚖️ Free-Tier Notes
Service	Notes
Google Gemini API	Free-tier allows limited embedding + generation calls per minute. Use the gemini-embedding-001 model for efficiency.
Neon PostgreSQL	Free serverless tier works well; latency is fine for small datasets. Ensure vector(768) matches the embedding size.

🧠 Troubleshooting
Issue	Fix
psycopg.errors.SyntaxError near :	Use CAST(:embedding AS vector) in SQL insert.
No resumes are indexed yet	Verify DB row count: SELECT COUNT(*) FROM resume_chunks;
json_invalid error in API	Ensure Content-Type: application/json and escape newlines.
Embeddings mismatch	Ensure embed_texts_safe() returns same length as all_meta.
Gemini quota exceeded	Retry later or reduce top_k.

🧩 Deliverables Summary (as per Proposal)
Requirement	Implemented
Backend System	✅
Batch resume parsing & chunking	✅
Embedding generation & storage	✅
Vector similarity search	✅
RAG-based response generation	✅
Public GitHub Repository	✅
README with setup, usage, free-tier notes	✅
Clean, modular, documented code	✅

🧑‍💻 Development Notes
All embeddings and queries are L2-normalized for consistent cosine similarity.

Chunks are created around ~500 characters with 50-character overlap.

Postgres ivfflat index improves search performance significantly.

No local .npy store is used in this PG version — all data resides in Neon.

🪪 License
MIT License © 2025

🏁 Example End-to-End Flow
bash
Copy code
# 1️⃣ Activate environment
.\.venv_pg\Scripts\Activate.ps1

# 2️⃣ Ingest resumes
python main.py ingest --resumes .\resumes

# 3️⃣ Ask a question
python main.py ask --job .\job.txt --question "Which candidate is the best fit and why?"

# 4️⃣ Start API
uvicorn main:app --host 0.0.0.0 --port 8000
You now have a fully functional AI Resume Analyzer backend
