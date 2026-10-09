# PDF Vectorisation Pipeline

A minimal PDF ingestion and vector-retrieval API built with Python, PyMuPDF, Sentence Transformers, ChromaDB, and FastAPI. It uses direct library calls only—no LangChain, LlamaIndex, agent framework, LLM call, keyword search, or full-text search.

## Requirements

- Python 3.10+
- Internet access on first run to download the embedding model
- A text-based PDF (OCR for scanned/image-only PDFs is not included)

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 1. Ingest a PDF

```bash
python ingest.py --file document.pdf
```

The script extracts text page-by-page, records one-based page numbers, creates overlapping chunks, generates embeddings, and persists them in the local `chroma_db/` directory. You can customize chunking:

```bash
python ingest.py --file document.pdf --chunk-size 800 --overlap 120
```

Re-running ingestion replaces the existing `pdf_chunks` collection to prevent stale or duplicate records.

## 2. Start the retrieval API

```bash
uvicorn server:app --host 127.0.0.1 --port 8000
```

Interactive API docs are available at `http://127.0.0.1:8000/docs`.

## 3. Query the API

```bash
curl -X POST http://127.0.0.1:8000/query ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"What are the ingestion steps?\",\"top_k\":3}"
```

On macOS/Linux, use the same curl command with `\` line continuations or send the JSON body in one line. Example request:

```json
{
  "query": "What are the ingestion steps?",
  "top_k": 3
}
```

The `/query` response is an array of raw chunk results:

```json
[
  {
    "chunk_text": "Example retrieved text from the source PDF",
    "page_number": 1,
    "score": 0.812345
  }
]
```

The score is cosine similarity (higher is more similar). Actual values and chunk text depend on the ingested PDF and model version. The endpoint returns only retrieved chunks, not a generated answer.

## 4. Generate results.json for three queries

With the server running and your PDF already ingested:

```bash
python query_examples.py
```

This calls `/query` for three assessment-oriented queries and writes the actual responses to `results.json`. Run this against the assessment platform's sample PDF before submitting.

## Design choices

- **Embedding model — `sentence-transformers/all-MiniLM-L6-v2`:** a compact sentence embedding model that runs locally after its first download, avoiding API keys and per-request embedding charges. The same model and normalization are used during ingestion and querying.
- **Vector database — ChromaDB:** persistent local storage with vector similarity search and metadata for source page numbers. The collection uses cosine distance.
- **Chunk size — 800 characters:** keeps passages reasonably focused while retaining enough context for retrieval. Character-based splitting is simple and predictable; a sentence boundary is preferred when one occurs near the end of a chunk.
- **Overlap — 120 characters:** retains some context across adjacent chunks so that content near a chunk boundary is less likely to be lost. The overlap is 15% of the nominal chunk size.
- **PDF parser — PyMuPDF:** extracts text from multiple pages and preserves each page's one-based number in metadata. Image-only/scanned PDFs require OCR, which is outside this implementation's scope.
- **API — FastAPI:** validates request input and exposes one `POST /query` retrieval endpoint plus a simple health check.

## Project structure

```text
.
├── ingest.py
├── server.py
├── query_examples.py
├── requirements.txt
├── results.json
├── README.md
└── .gitignore
```

## Notes for submission

1. Ingest the sample PDF supplied by the assessment platform.
2. Start the API and run `python query_examples.py` so `results.json` contains real results from that PDF.
3. If deploying, set `VECTOR_DB_PATH` to a persistent writable directory and keep the embedding model available in the deployment environment.
4. Create a **private** GitHub repository as required by the assessment. Add the requested evaluators and include the live deployment URL in this README after deploying. A deployment is not created automatically by this code package.
