import os
from contextlib import asynccontextmanager

import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "pdf_chunks"
DB_PATH = os.getenv("VECTOR_DB_PATH", "chroma_db")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        app.state.collection = client.get_collection(COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' not found in '{DB_PATH}'. "
            "Run ingest.py first."
        ) from exc
    yield


app = FastAPI(
    title="PDF Vector Retrieval API",
    description="Returns semantically similar PDF chunks; no LLM or conversational layer.",
    version="1.0.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, description="Plain-text search query")
    top_k: int = Field(default=5, ge=1, le=50)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
def query_chunks(request: QueryRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="query must not be blank")

    vector = app.state.model.encode(
        [query], normalize_embeddings=True
    ).tolist()[0]
    result = app.state.collection.query(
        query_embeddings=[vector],
        n_results=min(request.top_k, app.state.collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    rows = []
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    for text, metadata, distance in zip(documents, metadatas, distances):
        # Chroma cosine distance = 1 - cosine similarity.
        rows.append({
            "chunk_text": text,
            "page_number": metadata["page_number"],
            "score": round(1.0 - float(distance), 6),
        })
    return rows
