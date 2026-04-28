from __future__ import annotations

from pathlib import Path

from ferret_rag.index.store import SearchResult


class LocalChatEngine:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self._llm = None

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

        snippets = "\n\n".join(
            f"- {result.chunk.text[:500]} ({result.chunk.file_path})" for result in context
        )
        return (
            "Local model execution is not available in this environment yet, "
            "so here are the most relevant indexed snippets:\n\n"
            f"{snippets}"
        )

    def _try_llama(self, question: str, context: list[SearchResult]) -> str | None:
        try:
            from llama_cpp import Llama
        except ImportError:
            return None

        if self._llm is None:
            self._llm = Llama(model_path=str(self.model_path), n_ctx=4096, verbose=False)

        context_text = "\n\n".join(result.chunk.text for result in context)
        prompt = (
            "Use only the local context below to answer the question. "
            "If the answer is not in the context, say so.\n\n"
            f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"
        )
        response = self._llm(prompt, max_tokens=512, stop=["Question:"], echo=False)
        choices = response.get("choices", [])
        if not choices:
            return None
        return str(choices[0].get("text", "")).strip() or None
