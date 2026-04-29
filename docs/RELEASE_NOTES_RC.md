# FerretRAG Release Candidate Notes

## Included

- Local FastAPI app with static web UI.
- Local document indexing for TXT, MD, HTML, CSV, PDF, DOCX, and XLSX.
- Incremental indexing with SHA-256 file hashes.
- Chroma-backed retrieval with deterministic local embeddings.
- Local GGUF chat through `llama-cpp-python` when a compatible runtime and model are installed.
- Model registry and model switching UI.
- Source cards with file metadata and snippets.
- Windows-first development and packaging scripts.

## Known Limitations

- The pinned Python 3.13 CPU wheel for `llama-cpp-python` supports Qwen and Llama test models but not the current Gemma 4 GGUF.
- Packaging is a script-based smoke path until PyInstaller details are validated on a clean Windows machine.
- GPU acceleration is not part of the RC gate.
- Streaming responses and agentic tools are post-RC.

## Required Checks

- `ruff check .`
- `pytest`
- Manual QA checklist in `docs/QA_CHECKLIST.md`
- Privacy checklist in `docs/PRIVACY_CHECKLIST.md`
