from pydantic import BaseModel, Field
from typing import List, Optional

class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: dict
    dense_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    rrf_score: Optional[float] = None
    reranker_score: Optional[float] = None

class DraftResponse(BaseModel):
    subject: str = ""
    body: str
    call_to_action: str = ""
    policy_references: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    status: str = "DRAFT_FOR_HUMAN_REVIEW"
