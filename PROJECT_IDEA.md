# FerretRAG - "Slinking through your files"

## Tagline

A single-executable, privacy-first Python tool that lets you chat with the contents of folders on your machine. Inference, indexing, and storage should happen locally by default.

## Why FerretRAG?

- **Privacy is mandatory:** local files stay on disk and are not sent to cloud APIs.
- **Low setup:** the long-term goal is a Windows executable that opens a browser to a local app.
- **Model freedom:** ship with a compatible GGUF model later, while allowing users to choose another local GGUF.
- **Cross-platform mindset:** Windows is the primary target, but the codebase should stay portable.

## First MVP

The first milestone is a working local RAG loop:

1. Start a FastAPI server.
2. Serve a static local web UI.
3. Let the user enter a folder path.
4. Index supported documents from that folder.
5. Store chunks and metadata locally.
6. Retrieve relevant snippets for a question.
7. Answer using a local GGUF model when `llama-cpp-python` is available.

If local model execution is not available during early development, the app should still return relevant snippets so the indexing and retrieval path can be tested.

## Success Metrics

| Goal | Success Metric |
| --- | --- |
| Local-only behavior | No external API calls in core indexing, retrieval, or chat. |
| Default model path | `models/gemma-4-E2B-it-UD-IQ3_XXS.gguf` is detected locally. |
| Incremental indexing | Unchanged files are skipped using SHA-256 hashes. |
| File coverage | TXT, MD, HTML, CSV, PDF, DOCX, and XLSX are supported where parser dependencies are installed. |
| Usable UI | Folder indexing, chat, status, and source snippets are available on the first screen. |
| Git hygiene | GGUF models, runtime data, virtualenvs, and build outputs are excluded from git. |

## Architecture

```text
Browser UI
   |
   | HTTP JSON
   v
FastAPI app
   |
   +-- Local document loaders
   +-- Chunking and file hashing
   +-- Local vector/search store
   +-- llama-cpp-python GGUF chat engine
```

## Key Components

### Backend

- FastAPI serves the API and static UI.
- `/api/health` reports server, model, and chunk status.
- `/api/index` indexes a folder path.
- `/api/sources` lists indexed chunks.
- `/api/chat` retrieves snippets and generates an answer.

### Document Ingestion

| File type | Parser |
| --- | --- |
| TXT, MD | Python text reading |
| HTML | Python HTML parser |
| CSV | Python `csv` module |
| PDF | PyMuPDF |
| DOCX | python-docx |
| XLSX | openpyxl |

### Indexing

- Use SHA-256 file hashes to skip unchanged files.
- Chunk text into roughly 300-word windows with overlap.
- Store metadata including file path, chunk id, text, and file hash.
- Use Chroma for the intended vector store once embedding behavior is finalized.
- Keep a simple local JSON keyword fallback during early development.

### Local Model

- Development default path: `models/Qwen3.5-9B-UD-IQ3_XXS.gguf`.
- Original bundled-model candidate: `models/gemma-4-E2B-it-UD-IQ3_XXS.gguf`.
- Runtime: `llama-cpp-python`.
- The model file is ignored by git.
- Redistribution and bundling terms must be checked before release.
- Install CPU inference first; GPU acceleration is a later hardware-specific milestone.
- Use Qwen or Llama as development defaults until a newer llama.cpp build supports the Gemma 4 GGUF.

### UI Palette

Inspired by the included sleek silver/albino ferret design reference:

| Palette element | Hex | Usage |
| --- | --- | --- |
| Light Gray | `#F5F5F5` | Page background |
| Soft Silver | `#C0C0C0` | Secondary panels |
| Charcoal | `#333333` | Primary text |
| Deep Purple | `#43285D` | Accent and source markers |
| Bright Lime Green | `#84FA63` | Primary indexing action |
| Electric Blue | `#007BFF` | Chat action and active states |

## Repository Layout

```text
FerretRAG/
|-- ferret_rag/
|   |-- api/
|   |-- core/
|   |-- index/
|   |-- llm/
|   `-- ui/
|-- models/
|-- data/
|-- tests/
|-- scripts/
|-- AGENTS.md
|-- README.md
|-- PROJECT_IDEA.md
|-- config.yaml.example
`-- pyproject.toml
```

## Later Milestones

- Replace or augment the JSON search fallback with Chroma embeddings.
- Add server-sent events for streaming model responses.
- Add GPU detection and model settings in the UI.
- Add file watcher support for optional automatic refresh.
- Add packaging with PyInstaller and a Windows-first release workflow.
- Add GitHub Actions for linting, testing, and release builds.
- Add optional agentic tools such as `open_file`, `refresh_index`, and `summarize_chunk`.

## Open Questions

- Confirm the license and redistribution terms for the default GGUF model.
- Decide the embedding strategy for Chroma: model-native embeddings, a bundled embedding model, or a lightweight local fallback.
- Decide whether v1 should include only manual refresh or also a file watcher.
- Decide how much GPU configuration should be automatic versus user-controlled.
