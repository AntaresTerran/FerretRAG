from __future__ import annotations

import sys
import types
from pathlib import Path

from ferret_rag.index.store import SearchResult, SourceChunk
from ferret_rag.llm.engine import LocalChatEngine, _estimate_tokens, _fit_context_to_budget
from ferret_rag.llm.gpu import GpuDetection


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


def test_llama_passes_configured_gpu_layers(monkeypatch, tmp_path: Path) -> None:
    received_layers: list[int] = []

    class FakeLlama:
        def __init__(self, *args, **kwargs) -> None:
            received_layers.append(kwargs["n_gpu_layers"])

        def create_chat_completion(self, messages, max_tokens):
            return {"choices": [{"message": {"content": "Using configured layers."}}]}

    fake_module = types.SimpleNamespace(__version__="0.3.21", Llama=FakeLlama)
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"not a real model")
    engine = LocalChatEngine(model_path=model_path, n_ctx=1024, max_tokens=128, gpu_layers=17)

    assert engine._try_llama("Question?", [_result()]) == "Using configured layers."
    assert received_layers == [17]


def test_set_gpu_layers_resets_loaded_runtime(monkeypatch, tmp_path: Path) -> None:
    received_layers: list[int] = []

    class FakeLlama:
        def __init__(self, *args, **kwargs) -> None:
            received_layers.append(kwargs["n_gpu_layers"])

        def create_chat_completion(self, messages, max_tokens):
            return {"choices": [{"message": {"content": "Runtime switched."}}]}

    fake_module = types.SimpleNamespace(__version__="0.3.21", Llama=FakeLlama)
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"not a real model")
    engine = LocalChatEngine(model_path=model_path, n_ctx=1024, max_tokens=128, gpu_layers=0)

    assert engine._try_llama("Question?", [_result()]) == "Runtime switched."
    engine.set_gpu_layers(12)
    assert engine._try_llama("Question?", [_result()]) == "Runtime switched."

    assert received_layers == [0, 12]


def test_llama_auto_gpu_layers_use_detected_backend(monkeypatch, tmp_path: Path) -> None:
    received_layers: list[int] = []

    class FakeLlama:
        def __init__(self, *args, **kwargs) -> None:
            received_layers.append(kwargs["n_gpu_layers"])

        def create_chat_completion(self, messages, max_tokens):
            return {"choices": [{"message": {"content": "Using auto GPU."}}]}

    fake_module = types.SimpleNamespace(__version__="0.3.21", Llama=FakeLlama)
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)
    monkeypatch.setattr(
        "ferret_rag.llm.engine.detect_gpu_backend",
        lambda: GpuDetection(
            vendor="NVIDIA",
            backend="cuda",
            gpu_available=True,
            install_backend="cuda",
            wheel_index="https://example.invalid/cu124",
            build_from_source=False,
            message="CUDA detected.",
        ),
    )
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"not a real model")
    engine = LocalChatEngine(model_path=model_path, n_ctx=1024, max_tokens=128)

    assert engine._try_llama("Question?", [_result()]) == "Using auto GPU."
    assert received_layers == [-1]
    assert engine.runtime_status()["gpu_mode"] == "auto-gpu"


def test_failed_gpu_and_cpu_load_returns_snippet_fallback(monkeypatch, tmp_path: Path) -> None:
    received_layers: list[int] = []

    class FakeLlama:
        def __init__(self, *args, **kwargs) -> None:
            received_layers.append(kwargs["n_gpu_layers"])
            raise RuntimeError(f"load failed for {kwargs['n_gpu_layers']}")

    fake_module = types.SimpleNamespace(__version__="0.3.21", Llama=FakeLlama)
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)
    monkeypatch.setattr(
        "ferret_rag.llm.engine.detect_gpu_backend",
        lambda: GpuDetection(
            vendor="NVIDIA",
            backend="cuda",
            gpu_available=True,
            install_backend="cuda",
            wheel_index="https://example.invalid/cu124",
            build_from_source=False,
            message="CUDA detected.",
        ),
    )
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"not a real model")
    engine = LocalChatEngine(model_path=model_path, n_ctx=1024, max_tokens=128)

    answer = engine.answer("Question?", [_result()])

    assert received_layers == [-1, 0]
    assert engine._llm is None
    assert "LLM runtime note: GPU model load failed" in answer
    assert "CPU retry failed" in answer


def _result() -> SearchResult:
    return SearchResult(
        chunk=SourceChunk(
            id="id-1",
            file_path="C:/docs/file.md",
            file_name="file.md",
            file_type="md",
            chunk_id=1,
            text="The answer is in this local snippet.",
            file_hash="hash",
            modified_time=1,
        ),
        score=1,
    )
