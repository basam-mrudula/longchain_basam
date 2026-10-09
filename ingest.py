import argparse
from pathlib import Path

import chromadb
import fitz
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "pdf_chunks"
DB_PATH = "chroma_db"


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120):
    """Split text into character-based chunks with a small overlap."""
    text = " ".join(text.split())
    if not text:
        return []
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Prefer a natural boundary near the end of the target chunk.
        if end < len(text):
            boundary = max(text.rfind(". ", start, end),
                           text.rfind("? ", start, end),
                           text.rfind("! ", start, end))
            if boundary > start + int(chunk_size * 0.6):
                end = boundary + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def main():
    parser = argparse.ArgumentParser(description="Ingest a PDF into ChromaDB.")
    parser.add_argument("--file", required=True, help="Path to the source PDF")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=120)
    parser.add_argument("--db-path", default=DB_PATH)
    args = parser.parse_args()

    pdf_path = Path(args.file)
    if not pdf_path.is_file():
        parser.error(f"PDF file not found: {pdf_path}")

    client = chromadb.PersistentClient(path=args.db_path)
    # Re-ingestion replaces the collection to avoid duplicate/stale chunks.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    model = SentenceTransformer(MODEL_NAME)
    ids, documents, metadatas = [], [], []

    with fitz.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf, start=1):
            page_text = page.get_text("text").strip()
            for chunk_index, chunk in enumerate(
                chunk_text(page_text, args.chunk_size, args.overlap)
            ):
                ids.append(f"page-{page_index}-chunk-{chunk_index}")
                documents.append(chunk)
                metadatas.append({
                    "source": pdf_path.name,
                    "page_number": page_index,
                    "chunk_index": chunk_index,
                })

    if not documents:
        raise SystemExit("No text found in the PDF. It may be scanned; OCR is not included.")

    # Encode in batches to avoid excessive memory use on larger PDFs.
    embeddings = model.encode(
        documents, batch_size=32, normalize_embeddings=True,
        show_progress_bar=True
    ).tolist()

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    print(f"Ingested {len(documents)} chunks from {pdf_path.name} into {args.db_path}")


if __name__ == "__main__":
    main()
