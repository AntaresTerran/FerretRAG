from __future__ import annotations


def chunk_text(text: str, chunk_words: int = 300, overlap: int = 50) -> list[str]:
    words = text.split()
    if not words:
        return []
    if chunk_words <= 0:
        raise ValueError("chunk_words must be greater than zero")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_words:
        raise ValueError("overlap must be smaller than chunk_words")

    chunks: list[str] = []
    step = chunk_words - overlap
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_words])
        if chunk:
            chunks.append(chunk)
        if start + chunk_words >= len(words):
            break
    return chunks
