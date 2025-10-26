# vector_store_pg.py
from __future__ import annotations
import os, uuid
from typing import List, Dict, Any

from sqlalchemy import create_engine, text as sqltext
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool
from sqlalchemy import text as sqltext, bindparam
from typing import Sequence


def get_engine() -> Engine:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL missing in .env")
    return create_engine(url, poolclass=NullPool, future=True)


def insert_chunks_with_embeddings(
    engine, metas, embeddings
) -> None:
    if not metas or not embeddings:
        return
    if len(metas) != len(embeddings):
        raise ValueError("metas and embeddings length mismatch")

    # Force SQLAlchemy to recognize the :embedding bind
    sql = sqltext("""
        INSERT INTO resume_chunks (id, source_file, source_path, chunk_index, text, embedding)
        VALUES (:id, :source_file, :source_path, :chunk_index, :text, CAST(:embedding AS vector))
        ON CONFLICT (id) DO NOTHING
    """)

    rows = []
    for meta, emb in zip(metas, embeddings):
        rows.append({
            "id": meta.get("id"),
            "source_file": meta["source_file"],
            "source_path": meta["source_path"],
            "chunk_index": meta["chunk_index"],
            "text": meta["text"],
            # pass as a string like "[0.1,0.2,...]"
            "embedding": "[" + ",".join(str(float(x)) for x in emb) + "]",
        })

    with engine.begin() as conn:
        # executemany style
        conn.execute(sql, rows)

def _vector_to_str(vec: Sequence) -> str:
    """Serialize a vector to '[v1,v2,...]'. Flattens if nested like [[...]]."""
    # If it's nested (e.g., [[...]]), unwrap once
    if len(vec) > 0 and isinstance(vec[0], (list, tuple)):
        vec = vec[0]  # flatten one level
    # Ensure all items are floats
    return "[" + ",".join(str(float(x)) for x in vec) + "]"

def topk(engine: Engine, query_embedding, k: int = 6) -> List[Dict[str, Any]]:
    sql = sqltext("""
        SELECT id, source_file, source_path, chunk_index, text,
               1 - (embedding <=> CAST(:q AS vector)) AS cosine_sim
        FROM resume_chunks
        ORDER BY embedding <=> CAST(:q AS vector)
        LIMIT :k
    """)
    q = _vector_to_str(query_embedding)
    with engine.begin() as conn:
        rows = conn.execute(sql, {"q": q, "k": k}).mappings().all()
    return [dict(r) for r in rows]
