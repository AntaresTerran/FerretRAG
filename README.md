# FerretRAG

FerretRAG is a privacy-first local RAG app for chatting with folders on your machine. The first milestone is a small working MVP: index local documents, retrieve relevant chunks, and chat through a local model.

## Local-Only Promise

FerretRAG is designed so inference, indexing, and storage happen on your computer. The core app should not call external APIs unless a future feature explicitly asks for that and the user enables it.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"
```

This installs the app, document parsers, Chroma, and development tools. It does not install
`llama-cpp-python` by default because that package may require a local C++ build toolchain on
Windows.

## Optional Local LLM Runtime

The retrieval MVP works without `llama-cpp-python`; chat responses will show the most relevant
local snippets if the runtime is missing. To enable GGUF generation later, install the LLM extra:

```powershell
.venv\Scripts\pip install -e ".[llm]"
```

If pip tries to build `llama-cpp-python` from source and fails with `nmake` or compiler errors,
install Microsoft Visual Studio Build Tools, or use a prebuilt wheel that matches your Python,
Windows, and acceleration target.

## Run

```powershell
.venv\Scripts\python -m ferret_rag
```

The app starts a local FastAPI server and opens the UI at `http://127.0.0.1:8765`.

## Test and Lint

```powershell
.venv\Scripts\pytest
.venv\Scripts\ruff check .
```

## Model File

The default local model is expected at:

```text
models/gemma-4-E2B-it-UD-IQ3_XXS.gguf
```

GGUF files are ignored by git because they are large. Before shipping a release with a bundled model, verify the model license and redistribution terms.

## MVP Scope

- FastAPI backend with health, indexing, sources, and chat endpoints.
- Static HTML/CSS/JS UI using the silver/albino ferret palette.
- Incremental file indexing based on SHA-256 hashes.
- TXT, MD, HTML, CSV, PDF, DOCX, and XLSX parsing.
- Chroma-backed vector search when available, with a local JSON keyword fallback for early development.
