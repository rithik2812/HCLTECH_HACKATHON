from pathlib import Path
import pickle
import re
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from .config import settings
from .schemas import RetrievedChunk

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())

class HybridRetriever:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name, metadata={"hnsw:space": "cosine"}
        )
        self.embedder = SentenceTransformer(settings.embedding_model)
        self.reranker = CrossEncoder(settings.reranker_model)
        self._load_bm25_corpus()

    def _load_bm25_corpus(self):
        result = self.collection.get(include=["documents", "metadatas"])
        self.ids = result.get("ids", [])
        self.docs = result.get("documents", []) or []
        self.metas = result.get("metadatas", []) or []
        self.bm25 = BM25Okapi([tokenize(d) for d in self.docs]) if self.docs else None
        self.index = {doc_id: i for i, doc_id in enumerate(self.ids)}

    def _dense(self, query: str, n: int) -> list[dict]:
        if not self.ids:
            return []
        qvec = self.embedder.encode([query], normalize_embeddings=True).tolist()
        res = self.collection.query(query_embeddings=qvec, n_results=min(n, len(self.ids)))
        hits = []
        for rank, (doc_id, doc, meta, dist) in enumerate(zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ), start=1):
            hits.append({"id": doc_id, "text": doc, "metadata": meta,
                         "rank": rank, "distance": float(dist)})
        return hits

    def _sparse(self, query: str, n: int) -> list[dict]:
        if not self.bm25 or not self.docs:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [{"id": self.ids[i], "text": self.docs[i], "metadata": self.metas[i],
                 "rank": rank, "bm25_score": float(scores[i])}
                for rank, i in enumerate(ranked, start=1) if scores[i] > 0]

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or settings.top_k
        dense = self._dense(query, settings.dense_candidates)
        sparse = self._sparse(query, settings.bm25_candidates)
        # Reciprocal Rank Fusion: score = sum(1 / (k + rank)).
        fused = {}
        for source, hits in (("dense", dense), ("bm25", sparse)):
            for h in hits:
                item = fused.setdefault(h["id"], {**h, "rrf_score": 0.0,
                                                  "dense_rank": None, "bm25_rank": None})
                item["rrf_score"] += 1.0 / (settings.rrf_k + h["rank"])
                item[f"{source}_rank"] = h["rank"]
        candidates = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)
        candidates = candidates[:settings.rerank_candidates]
        if not candidates:
            return []
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs)
        for c, score in zip(candidates, scores):
            c["reranker_score"] = float(score)
        candidates.sort(key=lambda x: x["reranker_score"], reverse=True)
        return [RetrievedChunk(
            chunk_id=c["id"], text=c["text"], metadata=c["metadata"],
            dense_rank=c["dense_rank"], bm25_rank=c["bm25_rank"],
            rrf_score=c["rrf_score"], reranker_score=c["reranker_score"]
        ) for c in candidates[:top_k]]
