import re
from typing import List, Dict

HEADING = re.compile(
    r"(?m)^(?:\s*(?:\d+(?:\.\d+)*\.?\s+.+|[A-Z][A-Z /&()\-]{5,}|"
    r"Appendix\s+[A-Z].*))$"
)

def _token_words(text: str) -> list[str]:
    # A practical word-token approximation. Replace with model tokenizer if exact
    # token accounting is required for a production corpus.
    return re.findall(r"\S+", text)

def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

def _split_oversized(text: str, size: int, overlap: int) -> list[str]:
    words = _token_words(text)
    if len(words) <= size:
        return [text]
    chunks, start = [], 0
    step = max(1, size - overlap)
    while start < len(words):
        chunks.append(" ".join(words[start:start + size]))
        start += step
    return chunks

def structure_aware_recursive_split(text: str, chunk_size: int = 500,
                                    chunk_overlap: int = 75) -> list[str]:
    """
    First separate policy sections/headings, then paragraphs, then recursively
    split oversized paragraphs. Keep clauses and bullets together where possible.
    """
    lines = text.splitlines()
    sections, current = [], []
    for line in lines:
        if HEADING.match(line.strip()) and current:
            sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current).strip())

    output = []
    for section in sections:
        paragraphs = _split_paragraphs(section)
        buffer = ""
        for para in paragraphs:
            candidate = (buffer + "\n\n" + para).strip() if buffer else para
            if len(_token_words(candidate)) <= chunk_size:
                buffer = candidate
            else:
                if buffer:
                    output.extend(_split_oversized(buffer, chunk_size, chunk_overlap))
                if len(_token_words(para)) > chunk_size:
                    output.extend(_split_oversized(para, chunk_size, chunk_overlap))
                    buffer = ""
                else:
                    buffer = para
        if buffer:
            output.extend(_split_oversized(buffer, chunk_size, chunk_overlap))
    return [x for x in output if x.strip()]
