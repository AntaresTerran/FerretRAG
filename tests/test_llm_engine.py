from __future__ import annotations

from ferret_rag.index.store import SearchResult, SourceChunk
from ferret_rag.llm.engine import _estimate_tokens, _fit_context_to_budget


def test_fit_context_to_budget_trims_long_context() -> None:
    context = [
        SearchResult(
            chunk=SourceChunk(
                id=f"id-{index}",
                file_path=f"C:/docs/file-{index}.md",
                file_name=f"file-{index}.md",
                file_type="md",
                chunk_id=index,
                text=" ".join(["ferret"] * 900),
                file_hash="hash",
                modified_time=1,
            ),
            score=1,
        )
        for index in range(8)
    ]

    fitted = _fit_context_to_budget(
        question="Where is the answer?",
        context=context,
        n_ctx=1024,
        max_tokens=256,
    )

    assert _estimate_tokens(fitted) <= 1024 - 256
    assert "file-0.md" in fitted
    assert "file-7.md" not in fitted
