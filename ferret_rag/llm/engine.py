from __future__ import annotations

from pathlib import Path

from ferret_rag.index.store import SearchResult
from ferret_rag.llm.gguf import read_gguf_architecture


class LocalChatEngine:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self._llm = None
        self.last_error: str | None = None

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
                self._llm = Llama(model_path=str(self.model_path), n_ctx=4096, verbose=False)

            context_text = "\n\n".join(result.chunk.text for result in context)
            prompt = (
                "Use only the local context below to answer the question. "
                "If the answer is not in the context, say so.\n\n"
                f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"
            )
            response = self._llm(prompt, max_tokens=512, stop=["Question:"], echo=False)
        except Exception as exc:
            self.last_error = str(exc)
            return None

        choices = response.get("choices", [])
        if not choices:
            self.last_error = "llama-cpp-python returned no choices."
            return None

        self.last_error = None
        return str(choices[0].get("text", "")).strip() or None


def _version_tuple(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for part in version.split("."):
        if not part.isdigit():
            break
        parts.append(int(part))
    return tuple(parts)
