from pathlib import Path
import re
import fitz

def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove repeated page footer/header fragments common in policy PDFs.
    text = re.sub(r"(?im)^\s*NEXORA BANK\s*$", "", text)
    text = re.sub(r"(?im)^\s*NEXORA BANK\s*\|.*$", "", text)
    return text.strip()

def extract_pdf(path: Path) -> list[dict]:
    """Extract text page-wise and retain page numbers for traceability."""
    doc = fitz.open(path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = clean_text(page.get_text("text"))
        if text:
            pages.append({"page": page_num, "text": text})
    return pages

def infer_doc_metadata(path: Path, page_text: str = "") -> dict:
    name = path.stem
    text = page_text[:5000]
    doc_id_match = re.search(r"\bNB-[A-Z0-9-]{5,}\b", text)
    version_match = re.search(r"\bVersion\s+([0-9.]+)", text, re.I)
    date_match = re.search(r"\bEffective date\s*[:|]?\s*([0-9A-Za-z ,/-]+)", text, re.I)
    doc_id = doc_id_match.group(0) if doc_id_match else name.upper().replace("_", "-")[:80]
    if "Fee_Waiver" in name:
        category = "fee_waiver"
    elif "Offer_Eligibility" in name:
        category = "offer_eligibility"
    elif "Bonus_Points" in name:
        category = "bonus_points"
    elif "Communication" in name:
        category = "customer_communication"
    elif "Cashback" in name:
        category = "cashback_rewards"
    elif "engagement_strategy" in name.lower():
        category = "engagement_strategy"
    else:
        category = "general_policy"
    return {
        "document_id": doc_id,
        "document_name": path.name,
        "category": category,
        "version": version_match.group(1) if version_match else "unknown",
        "effective_date": date_match.group(1).strip() if date_match else "unknown",
        "approval_status": "synthetic_demo_not_production_approved",
        "source_status": "synthetic_hackathon_demo",
    }
