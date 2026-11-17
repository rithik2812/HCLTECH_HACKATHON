from src.nexora_rag.chunking import structure_aware_recursive_split

def test_chunker_returns_content():
    text = "1. Purpose\n\nThis policy explains the purpose.\n\n2. Scope\n\nThis policy applies to demo customers."
    chunks = structure_aware_recursive_split(text, chunk_size=20, chunk_overlap=3)
    assert chunks
    assert all(c.strip() for c in chunks)
