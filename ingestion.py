from pathlib import Path
import hashlib
import re
import chromadb
from sentence_transformers import SentenceTransformer
from .config import settings
from .document_loader import extract_pdf, infer_doc_metadata
from .chunking import structure_aware_recursive_split

def _stable_id(doc_id: str, page: int, chunk_index: int, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{doc_id}-p{page}-c{chunk_index}-{digest}"

def ingest_policies(reset: bool = False) -> dict:
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    if reset:
        try:
            client.delete_collection(settings.collection_name)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    model = SentenceTransformer(settings.embedding_model)
    ids, docs, metas = [], [], []
    pdf_paths = sorted(settings.policy_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {settings.policy_dir}")
    for pdf_path in pdf_paths:
        pages = extract_pdf(pdf_path)
        first_text = "\n".join(p["text"] for p in pages[:2])
        doc_meta = infer_doc_metadata(pdf_path, first_text)
        for page in pages:
            chunks = structure_aware_recursive_split(
                page["text"], settings.chunk_size_tokens, settings.chunk_overlap_tokens
            )
            for idx, chunk in enumerate(chunks):
                ids.append(_stable_id(doc_meta["document_id"], page["page"], idx, chunk))
                docs.append(chunk)
                metas.append({**doc_meta, "page": page["page"], "chunk_index": idx})
    if not docs:
        raise ValueError("PDFs were found, but no extractable text was found.")
    embeddings = model.encode(docs, normalize_embeddings=True, show_progress_bar=True).tolist()
    # Upsert makes re-running ingestion safe and avoids duplicate records.
    batch_size = 128
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.upsert(
            ids=ids[start:end], documents=docs[start:end],
            metadatas=metas[start:end], embeddings=embeddings[start:end]
        )
    return {"documents": len(pdf_paths), "chunks": len(ids),
            "collection": settings.collection_name, "persist_directory": str(settings.chroma_dir)}

if __name__ == "__main__":
    print(ingest_policies())
