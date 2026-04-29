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

1. **Stabilize MVP:** complete.
2. **Indexing and retrieval quality:** complete.
3. **Local LLM runtime:** complete for CPU-compatible models.
4. **UI completion:** complete for RC baseline.
5. **Packaging and release:** script and docs added; clean-machine validation remains.
6. **RC hardening:** CI and checklists added; manual QA remains.

## RC Gates

- App starts locally and opens in the browser.
- Qwen model loads and answers locally.
- Incompatible models fail gracefully.
- Mixed document folders index with useful skipped/indexed/failed reporting.
- Sources include file path, type, chunk id, and page/sheet context where available.
- Ruff, pytest, and manual QA pass.
