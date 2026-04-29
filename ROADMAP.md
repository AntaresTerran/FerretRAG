# FerretRAG Roadmap To Release Candidate

## Summary

Bring FerretRAG from the current working MVP to a Windows-first release candidate: local folder indexing, reliable retrieval, local GGUF chat, model/settings UI, polished source display, packaging, tests, and release documentation.

## RC Feature Set

- Local-only FastAPI app with static web UI.
- Folder indexing with incremental refresh.
- Supported files: TXT, MD, HTML, CSV, PDF, DOCX, XLSX.
- Persistent local index under `data/`.
- Local GGUF chat through `llama-cpp-python`.
- Model selection between compatible local GGUF files.
- Clear source snippets with file path, chunk id, and preview.
- Windows executable or packaged launch script.
- No cloud APIs in core app.
- Tests, linting, README, troubleshooting, and release notes.

## Phases

1. **Stabilize MVP:** clean encoding issues, add structured API responses, add config/model APIs, and improve error handling.
2. **Indexing and retrieval quality:** richer chunk metadata, page/sheet-aware loaders, per-file failure reporting, and source-card improvements.
3. **Local LLM runtime:** model registry, compatible/incompatible model status, chat-style prompting, and clear fallback behavior.
4. **UI completion:** better folder workflow, model/status panel, source cards, loading states, copy answer, and clear chat.
5. **Packaging and release:** Windows packaging, launch behavior, model setup docs, and release notes.
6. **RC hardening:** GitHub Actions, manual QA checklist, privacy checklist, and clean release candidate build.

## RC Gates

- App starts locally and opens in the browser.
- Qwen model loads and answers locally.
- Incompatible models fail gracefully.
- Mixed document folders index with useful skipped/indexed/failed reporting.
- Sources include file path, type, chunk id, and page/sheet context where available.
- Ruff, pytest, and manual QA pass.
