from __future__ import annotations

import sys
import types
from pathlib import Path

from ferret_rag.index.store import SearchResult, SourceChunk
from ferret_rag.llm.engine import LocalChatEngine, _estimate_tokens, _fit_context_to_budget


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


def test_llama_context_window_error_retries_with_less_context(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[int] = []

    class FakeLlama:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def create_chat_completion(self, messages, max_tokens):
            content = str(messages[1]["content"])
            calls.append(len(content))
            if len(calls) < 3:
                raise ValueError("Requested tokens (4289) exceed context window of 4096")
            return {"choices": [{"message": {"content": "Found it."}}]}

    fake_module = types.SimpleNamespace(__version__="0.3.21", Llama=FakeLlama)
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"not a real model")
    engine = LocalChatEngine(model_path=model_path, n_ctx=1024, max_tokens=128)
    context = [
        SearchResult(
            chunk=SourceChunk(
                id=f"id-{index}",
                file_path=f"C:/docs/file-{index}.md",
                file_name=f"file-{index}.md",
                file_type="md",
                chunk_id=index,
                text=" ".join(["needle"] * 500),
                file_hash="hash",
                modified_time=1,
            ),
            score=1,
        )
        for index in range(4)
    ]

    assert engine._try_llama("Where is the answer?", context) == "Found it."
    assert len(calls) == 3
    assert calls[0] > calls[1] > calls[2]
