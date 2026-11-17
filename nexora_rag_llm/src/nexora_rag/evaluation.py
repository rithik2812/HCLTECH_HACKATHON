"""
Small retrieval evaluation helper.
Expected JSONL format: {"query": "...", "relevant_document_ids": ["NB-COM-POL-001"]}
Use manually labelled queries; document-level relevance is a starting point.
"""
import json, math
from pathlib import Path
from .retrieval import HybridRetriever

def evaluate(jsonl_path: str, ks=(3, 5)):
    retriever = HybridRetriever()
    rows = [json.loads(line) for line in Path(jsonl_path).read_text().splitlines() if line.strip()]
    totals = {f"recall@{k}": 0.0 for k in ks}
    totals.update({f"mrr@{k}": 0.0 for k in ks})
    for row in rows:
        relevant = set(row["relevant_document_ids"])
        hits = retriever.retrieve(row["query"], top_k=max(ks))
        ranked = [h.metadata.get("document_id") for h in hits]
        for k in ks:
            top = ranked[:k]
            totals[f"recall@{k}"] += len(set(top) & relevant) / max(1, len(relevant))
            rr = next((1.0 / (i + 1) for i, d in enumerate(top) if d in relevant), 0.0)
            totals[f"mrr@{k}"] += rr
    return {key: value / max(1, len(rows)) for key, value in totals.items()}
