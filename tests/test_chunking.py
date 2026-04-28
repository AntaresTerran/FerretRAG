from __future__ import annotations

import pytest

from ferret_rag.index.chunking import chunk_text


def test_chunk_text_uses_overlap() -> None:
    text = " ".join(str(number) for number in range(10))

    chunks = chunk_text(text, chunk_words=5, overlap=2)

    assert chunks == ["0 1 2 3 4", "3 4 5 6 7", "6 7 8 9"]


def test_chunk_text_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="overlap must be smaller"):
        chunk_text("hello world", chunk_words=2, overlap=2)
