from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]

def _path(value: str, default: str) -> Path:
    p = Path(os.getenv(value, default))
    return p if p.is_absolute() else ROOT / p

@dataclass(frozen=True)
class Settings:
    policy_dir: Path = _path("POLICY_DIR", "data/policies")
    chroma_dir: Path = _path("CHROMA_DIR", "storage/chroma")
    collection_name: str = os.getenv("CHROMA_COLLECTION", "nexora_policy_chunks")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    chunk_size_tokens: int = int(os.getenv("CHUNK_SIZE_TOKENS", "500"))
    chunk_overlap_tokens: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "75"))
    dense_candidates: int = int(os.getenv("DENSE_CANDIDATES", "12"))
    bm25_candidates: int = int(os.getenv("BM25_CANDIDATES", "12"))
    rerank_candidates: int = int(os.getenv("RERANK_CANDIDATES", "10"))
    top_k: int = int(os.getenv("TOP_K", "4"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))
    demo_mode: bool = os.getenv("DEMO_MODE", "true").lower() == "true"

settings = Settings()
