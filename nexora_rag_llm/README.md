# Nexora Bank — RAG + LLM module

A modular Python implementation of the policy retrieval and grounded message-drafting layer for the Nexora Bank synthetic hackathon demo.

## Included policy PDFs

This ZIP includes 5 uploaded PDFs:
- `data/policies/Nexora_Bank_Fee_Waiver_Policy_v1.0.pdf`
- `data/policies/Nexora_Bank_Offer_Eligibility_and_Validation_Policy_v1.0.pdf`
- `data/policies/Nexora_Bank_Bonus_Points_Policy_v1.0.pdf`
- `data/policies/Nexora_Bank_Customer_Communication_Policy_v1.0.pdf`
- `data/policies/Nexora_Bank_Cashback_Rewards_Terms_Conditions_v1.0.pdf`

**Sixth document:** add `customer_engagement_strategy_policy.pdf` to `data/policies/` when you create it. The ingestion pipeline automatically discovers every PDF in that folder. It is not included because it was not among the uploaded files.

All Nexora documents are explicitly synthetic/demo-only and not production-approved. Keep that status in UI and audit logs.

## Architecture

1. PDF loading and page-aware extraction (PyMuPDF)
2. Text cleaning and structure-aware recursive chunking (section/paragraph first, oversized text split with overlap)
3. Sentence Transformers embeddings (`BAAI/bge-small-en-v1.5`, normalized)
4. Persistent ChromaDB collection using cosine distance
5. Dense semantic retrieval from ChromaDB
6. BM25 sparse keyword retrieval
7. Reciprocal Rank Fusion (RRF) to combine ranked lists
8. CrossEncoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
9. Top-K context selection with source metadata
10. Grounded LLM drafting via Gemini, Groq, or OpenRouter
11. JSON/Pydantic output validation and source traceability

## Setup

Python 3.10 or 3.11 recommended.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
cp .env.example .env
```

Add at least one provider key to `.env`.

### Provider configuration

**Gemini**
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=...
LLM_MODEL=gemini-2.5-flash
```

**Groq**
```env
LLM_PROVIDER=groq
GROQ_API_KEY=...
GROQ_MODEL=llama-3.3-70b-versatile
```

**OpenRouter**
```env
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=openai/gpt-4o-mini
```

Provider model names and availability can change. Set the model identifier supported by your own account/provider. The code does not assume that a free tier or a particular model is always available.

Fallback routing is OFF by default. To enable it:
```env
ENABLE_LLM_FALLBACK=true
LLM_FALLBACK_ORDER=groq,openrouter,gemini
```
The primary provider is tried first; fallbacks are attempted only after a provider error. All providers need their own configured key.

## Run

From the project root:

```bash
# First-time ingestion (also re-upserts on subsequent runs)
python -m src.nexora_rag.cli ingest

# Rebuild the collection from scratch
python -m src.nexora_rag.cli ingest --reset

# Ask a policy question
python -m src.nexora_rag.cli ask --question "What are the conditions for the cashback offer?"

# Example with structured, verified facts and deterministic checks
python -m src.nexora_rag.cli ask \
  --question "Draft a message explaining the offer status." \
  --category moderate \
  --customer-facts '{"customer_ref":"CUST-DEMO-001","preferred_channel":"email","marketing_consent":true}' \
  --checks '{"offer_id":"NX-CB-5PCT-300","eligibility_status":"ELIGIBLE","offer_active":true,"ledger_status":"not_posted"}'
```

Do not put real customer data in this demo. The JSON fields above are illustrative only.

## Integration into your main application

```python
from src.nexora_rag.pipeline import NexoraAssistant

assistant = NexoraAssistant()
result = assistant.answer(
    question="Draft a message about the customer's eligible offer.",
    category="moderate",
    customer_facts={"customer_ref": "CUST-DEMO-001", "preferred_channel": "email"},
    deterministic_checks={"eligibility_status": "ELIGIBLE", "consent_status": "PERMITTED"},
    top_k=4,
)
print(result["draft"])
print(result["retrieved_sources"])
```

## Important separation of responsibilities

- **RAG:** retrieves and explains policy terms. It does not determine customer-specific eligibility.
- **PostgreSQL / authoritative application data:** customer profile, consent, campaign enrolment, transactions, billing, points/cashback/fee ledgers and communication history.
- **Python policy engine:** deterministic eligibility, consent, channel, frequency, duplicate-suppression and billing checks. Do not delegate these to the LLM.
- **XGBoost or another ML model:** customer engagement segment classification. RAG does not classify customers.
- **LLM:** drafts wording using verified structured facts, deterministic outcomes and retrieved policy text.
- **Human review:** required before any communication is sent.

This matches the supplied Nexora fee-waiver policy (RAG explains terms, deterministic checks decide; pages 2–4), offer policy (deterministic eligibility; pages 2–4), and communication policy (consent, frequency, channel and human-review controls; pages 2–5).

## Important implementation notes

- The chunker is structure-aware and page-aware. It tries to preserve section and paragraph boundaries; very long paragraphs are split using a word-count approximation to tokens. If exact token budgeting becomes important, swap in the tokenizer for the embedding/LLM model.
- BM25 is built in memory from the Chroma collection when the retriever starts. For a small policy corpus this is simple and practical. For a large corpus, persist a versioned sparse index and rebuild it during ingestion.
- Re-ingestion uses stable IDs and Chroma upsert. Use `--reset` when you want a clean rebuild after changing chunking or embedding settings.
- The CrossEncoder is applied after RRF to a bounded candidate pool; it is not run over every chunk.
- Retrieved references include document ID, version and page. Keep these in your audit record.
- Do not index customer statements or private customer records into this shared policy collection.

## Evaluation

Create `tests/retrieval_eval.jsonl` with one query per line, for example:
```json
{"query":"What is the bonus points earning threshold?","relevant_document_ids":["NB-BP-POL-001"]}
{"query":"What are the promotional SMS frequency limits?","relevant_document_ids":["NB-COM-POL-001"]}
```
Then call `evaluate("tests/retrieval_eval.jsonl")` from Python. Start with 30–50 manually labelled queries. Compare dense-only, BM25-only, hybrid RRF and hybrid+reranker; track Recall@K, MRR@K, latency and failure cases.

## Security / demo boundaries

- Never commit `.env` or API keys.
- Do not send real customer messages from this module. There is no delivery integration.
- Never claim a fee waiver, cashback or points award has been applied unless the corresponding authoritative ledger confirms it.
- Unknown, stale or conflicting evidence should lead to a safe no-claim / human-review result.
- These synthetic policies are not legal, regulatory or production bank policy.
