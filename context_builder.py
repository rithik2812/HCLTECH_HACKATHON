from .schemas import RetrievedChunk

def build_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No policy passages were retrieved. Do not invent policy terms."
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        m = chunk.metadata
        source = (
            f"[Source {i} | Document ID: {m.get('document_id')} | "
            f"Document: {m.get('document_name')} | Version: {m.get('version')} | "
            f"Page: {m.get('page')} | Status: {m.get('approval_status')}]"
        )
        blocks.append(f"{source}\n{chunk.text}")
    return "\n\n---\n\n".join(blocks)
