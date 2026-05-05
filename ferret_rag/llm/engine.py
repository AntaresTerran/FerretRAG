from __future__ import annotations

from pathlib import Path

from ferret_rag.index.store import SearchResult
from ferret_rag.llm.gguf import read_gguf_architecture
from ferret_rag.llm.gpu import GpuDetection, detect_gpu_backend, resolve_gpu_layers


class LocalChatEngine:
    def __init__(
        self,
        model_path: Path,
        n_ctx: int = 4096,
        max_tokens: int = 512,
        gpu_layers: str | int = "auto",
    ) -> None:
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.max_tokens = max_tokens
        self.gpu_layers = gpu_layers
        self._llm = None
        self._loaded_gpu_layers: int | None = None
        self.last_error: str | None = None
        self._gpu_detection: GpuDetection | None = None
        self._gpu_mode = "unknown"
        self._gpu_message = "GPU runtime has not been checked yet."
        self._gpu_fallback_message: str | None = None

    def set_model_path(self, model_path: Path) -> None:
        if model_path.resolve() == self.model_path.resolve():
            return
        self.model_path = model_path
        self._llm = None
        self._loaded_gpu_layers = None
        self.last_error = None
        self._gpu_fallback_message = None

    def answer(self, question: str, context: list[SearchResult]) -> str:
        if not context:
            return (
                "I could not find matching indexed snippets yet. "
                "Index a folder or ask about content that exists in the current index."
            )

        if self.model_path.exists():
            model_answer = self._try_llama(question, context)
            if model_answer:
                return model_answer

        runtime_note = ""
        if self.last_error:
            runtime_note = f"\n\nLLM runtime note: {self.last_error}"

        snippets = "\n\n".join(
            f"- {result.chunk.text[:500]} ({result.chunk.file_path})" for result in context
        )
        return (
            "Local model execution is not available in this environment yet, "
            "so here are the most relevant indexed snippets:\n\n"
            f"{snippets}"
            f"{runtime_note}"
        )

    def _try_llama(self, question: str, context: list[SearchResult]) -> str | None:
        try:
            import llama_cpp
            from llama_cpp import Llama
        except ImportError:
            self.last_error = "llama-cpp-python is not installed."
            return None

        architecture = read_gguf_architecture(self.model_path)
        version = getattr(llama_cpp, "__version__", "0")
        if architecture == "gemma4" and _version_tuple(version) <= (0, 3, 19):
            self.last_error = (
                f"model architecture '{architecture}' needs a newer llama-cpp-python "
                f"than installed version {version}."
            )
            return None

        try:
            if self._llm is None:
                requested_gpu_layers, _, _ = self._resolve_gpu_layers()
                try:
                    self._load_llama(Llama, requested_gpu_layers)
                except Exception as exc:
                    self._llm = None
                    self._loaded_gpu_layers = None
                    if requested_gpu_layers == 0:
                        raise
                    self._gpu_fallback_message = (
                        f"GPU model load failed ({exc}); retried on CPU."
                    )
                    self.last_error = self._gpu_fallback_message
                    try:
                        self._load_llama(Llama, 0)
                    except Exception as cpu_exc:
                        self.last_error = (
                            f"{self._gpu_fallback_message} CPU retry failed ({cpu_exc})."
                        )
                        return None
                    self._gpu_mode = "cpu-fallback"
                    self._gpu_message = self._gpu_fallback_message

            response = None
            for context_scale in (1.0, 0.55, 0.3):
                context_text = _fit_context_to_budget(
                    question=question,
                    context=context,
                    n_ctx=self.n_ctx,
                    max_tokens=self.max_tokens,
                    context_scale=context_scale,
                )
                try:
                    response = self._invoke_llama(question, context_text)
                    break
                except Exception as exc:
                    if not _is_context_window_error(exc) or context_scale == 0.3:
                        raise
                    self.last_error = f"{exc}; retrying with less retrieved context."
        except Exception as exc:
            self.last_error = str(exc)
            return None

        if response is None:
            self.last_error = "llama-cpp-python returned no response."
            return None

        choices = response.get("choices", [])
        if not choices:
            self.last_error = "llama-cpp-python returned no choices."
            return None

        self.last_error = None
        choice = choices[0]
        if "message" in choice:
            return str(choice["message"].get("content", "")).strip() or None
        return str(choice.get("text", "")).strip() or None

    def _invoke_llama(self, question: str, context_text: str) -> dict:
        if hasattr(self._llm, "create_chat_completion"):
            return self._llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Answer using only the provided local context. "
                            "If the answer is not in the context, say so."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context_text}\n\nQuestion: {question}",
                    },
                ],
                max_tokens=self.max_tokens,
            )

        prompt = (
            "Use only the local context below to answer the question. "
            "If the answer is not in the context, say so.\n\n"
            f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"
        )
        return self._llm(
            prompt,
            max_tokens=self.max_tokens,
            stop=["Question:"],
            echo=False,
        )

    def _load_llama(self, llama_class, gpu_layers: int) -> None:
        self._llm = llama_class(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_gpu_layers=gpu_layers,
            verbose=False,
        )
        self._loaded_gpu_layers = gpu_layers

    def runtime_status(self) -> dict[str, object]:
        architecture = read_gguf_architecture(self.model_path) if self.model_path.exists() else None
        llama_version: str | None = None
        llama_available = False
        compatible = False
        message = "Model file is missing."

        try:
            import llama_cpp
        except Exception as exc:
            message = f"llama-cpp-python is unavailable: {exc}"
        else:
            llama_available = True
            llama_version = getattr(llama_cpp, "__version__", "unknown")
            compatible = self._is_compatible(architecture, llama_version)
            message = "ready" if compatible else _compatibility_message(architecture, llama_version)

        if self.last_error:
            message = self.last_error
        requested_gpu_layers, gpu_mode, gpu_message = self._resolve_gpu_layers()
        actual_gpu_layers = (
            self._loaded_gpu_layers
            if self._loaded_gpu_layers is not None
            else requested_gpu_layers
        )
        if self._loaded_gpu_layers == 0 and requested_gpu_layers != 0:
            gpu_mode = "cpu-fallback"
            gpu_message = self._gpu_fallback_message or "GPU load failed; running on CPU."

        return {
            "model_path": str(self.model_path),
            "model_exists": self.model_path.exists(),
            "architecture": architecture,
            "llama_cpp_available": llama_available,
            "llama_cpp_version": llama_version,
            "is_compatible": compatible,
            "gpu_layers": actual_gpu_layers,
            "gpu_mode": gpu_mode,
            "gpu_message": gpu_message,
            "last_error": self.last_error,
            "message": message,
        }

    def _is_compatible(self, architecture: str | None, version: str | None) -> bool:
        if not self.model_path.exists() or architecture is None or version is None:
            return False
        return not (architecture == "gemma4" and _version_tuple(version) <= (0, 3, 19))

    def _resolve_gpu_layers(self) -> tuple[int, str, str]:
        needs_detection = str(self.gpu_layers).strip().lower() == "auto"
        if needs_detection and self._gpu_detection is None:
            self._gpu_detection = detect_gpu_backend()
        gpu_layers, gpu_mode, gpu_message = resolve_gpu_layers(
            self.gpu_layers,
            detection=self._gpu_detection,
        )
        self._gpu_mode = gpu_mode
        self._gpu_message = gpu_message
        return gpu_layers, gpu_mode, gpu_message


def _version_tuple(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for part in version.split("."):
        if not part.isdigit():
            break
        parts.append(int(part))
    return tuple(parts)


def _compatibility_message(architecture: str | None, version: str | None) -> str:
    if architecture is None:
        return "Could not read model architecture."
    if architecture == "gemma4" and version and _version_tuple(version) <= (0, 3, 19):
        return f"Model architecture 'gemma4' needs a newer llama-cpp-python than {version}."
    return "Model is not compatible with the current runtime."


def _fit_context_to_budget(
    question: str,
    context: list[SearchResult],
    n_ctx: int,
    max_tokens: int,
    context_scale: float = 1.0,
) -> str:
    overhead_tokens = 700
    question_tokens = _estimate_tokens(question)
    raw_budget = n_ctx - max_tokens - overhead_tokens - question_tokens
    budget = max(int(raw_budget * 0.72 * context_scale), 64)
    parts: list[str] = []
    used = 0

    for result in context:
        header = _source_header(result)
        text = result.chunk.text.strip()
        candidate = f"{header}\n{text}"
        candidate_tokens = _estimate_tokens(candidate)
        remaining = budget - used
        if remaining <= 0:
            break
        if candidate_tokens > remaining:
            token_budget = max(remaining - _estimate_tokens(header), 32)
            trimmed_text = _trim_to_token_budget(text, token_budget)
            if trimmed_text:
                parts.append(f"{header}\n{trimmed_text}")
            break
        parts.append(candidate)
        used += candidate_tokens

    return "\n\n".join(parts)


def _source_header(result: SearchResult) -> str:
    chunk = result.chunk
    location = f"page {chunk.page_num}" if chunk.page_num else chunk.label or chunk.file_type
    return f"[{chunk.file_name} | {location} | chunk {chunk.chunk_id}]"


def _trim_to_token_budget(text: str, token_budget: int) -> str:
    if token_budget <= 0:
        return ""
    words = text.split()
    approx_words = max(int(token_budget / 1.35), 1)
    return " ".join(words[:approx_words])


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(int(len(text.split()) * 1.65), int(len(text) / 3.2))


def _is_context_window_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "exceed" in message and ("context" in message or "n_ctx" in message)
