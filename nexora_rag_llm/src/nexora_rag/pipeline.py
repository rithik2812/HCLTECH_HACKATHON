from .retrieval import HybridRetriever
from .context_builder import build_context
from .llm import LLMRouter
from .schemas import DraftResponse

class NexoraAssistant:
    def __init__(self):
        self.retriever = HybridRetriever()
        self.llm = LLMRouter()

    def answer(self, question: str, category: str = "moderate",
               customer_facts: dict | None = None,
               deterministic_checks: dict | None = None,
               top_k: int | None = None) -> dict:
        customer_facts = customer_facts or {}
        deterministic_checks = deterministic_checks or {}
        chunks = self.retriever.retrieve(question, top_k=top_k)
        context = build_context(chunks)
        draft: DraftResponse = self.llm.generate(
            user_request=question,
            category=category,
            customer_facts=customer_facts,
            deterministic_checks=deterministic_checks,
            policy_context=context,
        )
        return {
            "draft": draft.model_dump(),
            "retrieved_sources": [
                {"chunk_id": c.chunk_id, "document_id": c.metadata.get("document_id"),
                 "document_name": c.metadata.get("document_name"),
                 "version": c.metadata.get("version"), "page": c.metadata.get("page"),
                 "rrf_score": c.rrf_score, "reranker_score": c.reranker_score}
                for c in chunks
            ],
            "disclaimer": "Synthetic hackathon demo only. Draft requires human review; no message is sent.",
        }
