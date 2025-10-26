#!/usr/bin/env python3
"""
AI Resume Analyzer — Core (pgvector + FastAPI)
----------------------------------------------
Features:
- Ingest multiple resumes (PDF/TXT), extract & chunk text
- Generate embeddings with Google Gemini (gemini-embedding-001, 768-d)
- Store chunks + vectors in PostgreSQL (pgvector)
- Vector similarity search for job queries
- Retrieval-Augmented Generation with Gemini (gemini-2.0-flash-lite)
- Minimal FastAPI endpoints + CLI commands

Setup:
1) Python 3.10+
2) pip install -r requirements_pg.txt
3) .env in project root:
   GEMINI_API_KEY=your_key_here
   DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:PORT/DBNAME

Run:
- CLI:
  python main.py ingest --resumes ./resumes
  python main.py ask --job "./job.txt" --question "Who are the top 3 matches and why?"
  python main.py chat --job "./job.txt"

- API:
  uvicorn main:app --host 0.0.0.0 --port 8000
  POST /ingest  { "resumes_dir": "./resumes" }
  POST /chat    { "job_description": "...", "question": "...", "top_k": 6 }
"""

from __future__ import annotations

import os
import re
import json
import uuid
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import google.generativeai as genai

# pgvector helpers
from vector_store_pg import (
    get_engine,
    insert_chunks_with_embeddings,
    topk as pg_topk,
)

# -----------------------------
# Configuration & Constants
# -----------------------------
EMBED_MODEL = "models/gemini-embedding-004"    # 768-dim stable embeddings
GEN_MODEL   = "models/gemini-2.0-flash-lite"   # you verified this works
CHUNK_TARGET_CHARS = 500
CHUNK_OVERLAP_CHARS = 50
TOP_K_DEFAULT = 6
MAX_CONTEXT_TOKENS_APPROX = 4000  # rough character cap for prompt safety


# -----------------------------
# Utils
# -----------------------------
def load_env_and_init_gemini():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY missing. Put it in a .env file.")
    genai.configure(api_key=api_key)


def read_file_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        pages = []
        for p in reader.pages:
            try:
                pages.append(p.extract_text() or "")
            except Exception:
                pages.append("")
        return "\n".join(pages)
    elif path.suffix.lower() in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    else:
        return ""  # unsupported types ignored


def clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


def chunk_text(text: str, target: int = CHUNK_TARGET_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=target,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [c for c in splitter.split_text(clean_text(text)) if c.strip()]


def embed_texts_safe(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """Return one embedding per input text (same length), with a safe fallback."""
    out: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            resp = genai.embed_content(
                model=EMBED_MODEL,
                content=batch,
            )
            if "embeddings" in resp:
                embs = [
                    (e.get("values") if isinstance(e, dict) and "values" in e else e)
                    for e in resp["embeddings"]
                ]
            elif "embedding" in resp:
                embs = [resp["embedding"]]
            else:
                embs = []
        except Exception:
            embs = []

        # If batch size mismatches, fall back to per-item calls for this batch
        if len(embs) != len(batch):
            for t in batch:
                try:
                    r = genai.embed_content(model=EMBED_MODEL, content=t)
                    if "embedding" in r:
                        v = r["embedding"]
                    elif "embeddings" in r and r["embeddings"]:
                        v = (
                            r["embeddings"][0].get("values")
                            if isinstance(r["embeddings"][0], dict) and "values" in r["embeddings"][0]
                            else r["embeddings"][0]
                        )
                    else:
                        v = None
                    if v is None:
                        v = [0.0] * 768
                    out.append(v)
                except Exception:
                    out.append([0.0] * 768)
            continue

        out.extend(embs)

    # L2-normalize
    import numpy as np
    arr = np.array(out, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    arr = (arr / norms).astype(np.float32)
    return arr.tolist()


def trim_for_prompt(text: str, limit_chars: int) -> str:
    if len(text) <= limit_chars:
        return text
    head = int(limit_chars * 0.6)
    tail = limit_chars - head
    return text[:head] + "\n...\n" + text[-tail:]


# -----------------------------
# Ingestion
# -----------------------------
def ingest_resumes(resumes_dir: str) -> Dict[str, Any]:
    load_env_and_init_gemini()
    p = Path(resumes_dir)
    if not p.exists() or not p.is_dir():
        raise RuntimeError(f"Directory not found: {resumes_dir}")

    files = sorted([f for f in p.glob("**/*") if f.suffix.lower() in {".pdf", ".txt", ".md"}])
    if not files:
        return {"added_files": 0, "added_chunks": 0}

    all_chunks: List[str] = []
    all_meta: List[Dict[str, Any]] = []
    for f in files:
        txt = read_file_text(f)
        if not txt.strip():
            continue
        for i, c in enumerate(chunk_text(txt)):
            all_chunks.append(c)
            all_meta.append({
                "id": str(uuid.uuid4()),
                "source_file": str(f.name),
                "source_path": str(f),
                "chunk_index": i,
                "text_preview": c[:200],
                "text": c,  # store full text
            })

    if not all_chunks:
        return {"added_files": len(files), "added_chunks": 0}

    # Get embeddings with strict one-to-one alignment
    embs = embed_texts_safe(all_chunks)  # -> List[List[float]] same length as all_meta

    # Sanity guard
    if len(embs) != len(all_meta):
        min_len = min(len(embs), len(all_meta))
        all_meta = all_meta[:min_len]
        embs = embs[:min_len]

    engine = get_engine()
    insert_chunks_with_embeddings(engine, all_meta, embs)
    return {"added_files": len(files), "added_chunks": len(all_meta)}


# -----------------------------
# RAG Chat
# -----------------------------
SYSTEM_INSTRUCTIONS = (
    "You are an AI Resume Analyzer. Given a job description and retrieved resume snippets, "
    "answer questions about candidate-role fit. Be concise, structured, and cite sources using "
    'the format [source: FILENAME#CHUNK]. If making ranked recommendations, provide a short bullet list '
    "with a one-line rationale for each. Do not invent facts beyond the snippets."
)

def build_prompt(job_description: str,
                 question: str,
                 contexts: List[Dict[str, Any]]) -> str:
    ctx_lines = []
    for c in contexts:
        tag = f"{c['source_file']}#{c['chunk_index']}"
        snippet = trim_for_prompt(c["text"], 1200)
        ctx_lines.append(f"--- {tag} ---\n{snippet}\n")
    ctx_block = "\n".join(ctx_lines)
    jd_block = trim_for_prompt(job_description, 2000)
    q_block = question.strip()

    prompt = (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"JOB DESCRIPTION:\n{jd_block}\n\n"
        f"RETRIEVED RESUME SNIPPETS:\n{ctx_block}\n"
        f"QUESTION:\n{q_block}\n\n"
        f"Answer now. Always cite like [source: FILENAME#CHUNK]."
    )
    return trim_for_prompt(prompt, MAX_CONTEXT_TOKENS_APPROX * 4)


def retrieve(job_description: str, question: str, top_k: int = TOP_K_DEFAULT):
    load_env_and_init_gemini()
    engine = get_engine()

    # Build query text and embed
    composite = f"Job requirements:\n{job_description}\n\nQuestion:\n{question}"
    q = embed_texts_safe([composite])[0]
    if q and isinstance(q[0], (list, tuple)):  # defensive flatten
        q = q[0]

    # Vector search
    rows = pg_topk(engine, q, k=top_k)

    # Fallback: if vector search returns nothing but table has data, return first few chunks
    if not rows:
        from sqlalchemy import text
        with engine.begin() as conn:
            cnt = conn.execute(text("SELECT COUNT(*) FROM resume_chunks")).scalar()
            if cnt and cnt > 0:
                rows = conn.execute(text("""
                    SELECT id, source_file, source_path, chunk_index, text
                    FROM resume_chunks
                    ORDER BY source_file, chunk_index
                    LIMIT :k
                """), {"k": top_k}).mappings().all()

    contexts = [{
        "id": r["id"],
        "source_file": r["source_file"],
        "source_path": r["source_path"],
        "chunk_index": r["chunk_index"],
        "text_preview": r["text"][:200],
        "text": r["text"],
    } for r in rows]

    return contexts, list(range(len(contexts)))


def generate_answer(job_description: str, question: str, top_k: int = TOP_K_DEFAULT) -> Dict[str, Any]:
    contexts, _ = retrieve(job_description, question, top_k=top_k)
    if not contexts:
        return {
            "answer": "No matching snippets found in the vector DB (pgvector). Try re-ingesting or adjusting your query.",
            "citations": [],
            "used_chunks": []
        }

    prompt = build_prompt(job_description, question, contexts)
    model = genai.GenerativeModel(GEN_MODEL)
    resp = model.generate_content(prompt)
    text = resp.text.strip() if hasattr(resp, "text") and resp.text else "(No response)"
    citations = [f"{c['source_file']}#{c['chunk_index']}" for c in contexts]
    return {"answer": text, "citations": citations, "used_chunks": contexts}


# -----------------------------
# FastAPI (minimal)
# -----------------------------
try:
    from fastapi import FastAPI
    from pydantic import BaseModel, Field

    app = FastAPI(title="AI Resume Analyzer (Core)")

    class IngestReq(BaseModel):
        resumes_dir: str = Field(..., description="Directory containing PDF/TXT resumes")

    class ChatReq(BaseModel):
        job_description: str
        question: str
        top_k: int = TOP_K_DEFAULT

    @app.post("/ingest")
    def api_ingest(req: IngestReq):
        res = ingest_resumes(req.resumes_dir)
        return {"status": "ok", **res}

    @app.post("/chat")
    def api_chat(req: ChatReq):
        res = generate_answer(req.job_description, req.question, top_k=req.top_k)
        return res

except Exception:
    app = None  # If FastAPI isn't installed, CLI still works


# -----------------------------
# CLI
# -----------------------------
def cli():
    parser = argparse.ArgumentParser(description="AI Resume Analyzer (Core)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="Ingest resumes from a directory")
    p_ing.add_argument("--resumes", required=True, help="Path to resumes directory")

    p_ask = sub.add_parser("ask", help="One-shot question (job + question)")
    p_ask.add_argument("--job", required=True, help="Path to a job description text file")
    p_ask.add_argument("--question", required=True, help="Your question")
    p_ask.add_argument("--top_k", type=int, default=TOP_K_DEFAULT, help="Top-K chunks to retrieve")

    p_chat = sub.add_parser("chat", help="Interactive chat loop")
    p_chat.add_argument("--job", required=True, help="Path to a job description text file")
    p_chat.add_argument("--top_k", type=int, default=TOP_K_DEFAULT, help="Top-K chunks to retrieve")

    args = parser.parse_args()

    if args.cmd == "ingest":
        out = ingest_resumes(args.resumes)
        print(json.dumps(out, indent=2))
        return

    if args.cmd == "ask":
        job_desc = Path(args.job).read_text(encoding="utf-8", errors="ignore")
        out = generate_answer(job_desc, args.question, top_k=args.top_k)
        print("\n=== ANSWER ===\n")
        print(out["answer"])
        print("\n=== CITATIONS ===")
        for c in out["citations"]:
            print("-", c)
        return

    if args.cmd == "chat":
        job_desc = Path(args.job).read_text(encoding="utf-8", errors="ignore")
        print("Interactive chat. Type 'exit' to quit.")
        while True:
            q = input("\nYou: ").strip()
            if q.lower() in {"exit", "quit"}:
                break
            out = generate_answer(job_desc, q, top_k=args.top_k)
            print("\nAI:\n" + out["answer"])
            if out.get("citations"):
                print("\nCitations:", ", ".join(out["citations"]))


if __name__ == "__main__":
    cli()
