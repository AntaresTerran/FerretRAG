from __future__ import annotations

from pathlib import Path

from ferret_rag.index.store import SearchResult
from ferret_rag.llm.gguf import read_gguf_architecture


class LocalChatEngine:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self._llm = None
        self.last_error: str | None = None

    def set_model_path(self, model_path: Path) -> None:
        if model_path.resolve() == self.model_path.resolve():
            return
        self.model_path = model_path
        self._llm = None
        self.last_error = None

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
            if hasattr(self._llm, "create_chat_completion"):
                response = self._llm.create_chat_completion(
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
                    max_tokens=512,
                )
            else:
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
        choice = choices[0]
        if "message" in choice:
            return str(choice["message"].get("content", "")).strip() or None
        return str(choice.get("text", "")).strip() or None

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

        return {
            "model_path": str(self.model_path),
            "model_exists": self.model_path.exists(),
            "architecture": architecture,
            "llama_cpp_available": llama_available,
            "llama_cpp_version": llama_version,
            "is_compatible": compatible,
            "last_error": self.last_error,
            "message": message,
        }

    def _is_compatible(self, architecture: str | None, version: str | None) -> bool:
        if not self.model_path.exists() or architecture is None or version is None:
            return False
        return not (architecture == "gemma4" and _version_tuple(version) <= (0, 3, 19))


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
