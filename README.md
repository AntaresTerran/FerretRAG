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
local snippets if the runtime is missing.

For a first Windows test, prefer the official prebuilt CPU wheel index. The requirements file pins
the wheel version because the newest PyPI release may not have a matching Windows/Python wheel yet:

```powershell
.venv\Scripts\pip install -r requirements\llm-cpu.txt
.venv\Scripts\python scripts\check_llm.py
```

At the moment this CPU wheel can install on Python 3.13, but it may be too old for the included
Gemma 4 GGUF. If `scripts\check_llm.py` reports unknown architecture `gemma4`, the runtime install
worked but the model needs a newer llama.cpp build.

The project still exposes `.[llm]`, but that uses pip's normal package resolution. On Windows it
may try to build from source and fail with `nmake`, `CMAKE_C_COMPILER`, or compiler errors. That is
expected unless Visual Studio Build Tools or MinGW are installed and configured.

GPU acceleration should be handled after the CPU path works. Official CUDA prebuilt wheels require
a matching Python, Windows, and CUDA combination; if no matching wheel exists, the package must be
built locally.

Practical paths from here:

- Use a GGUF model architecture supported by the pinned CPU wheel.
- Install a newer `llama-cpp-python` from source with Visual Studio Build Tools or MinGW.
- Use a Python/CUDA combination that has a matching newer prebuilt wheel.

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

## Recommended Next Milestones

1. Make the retrieval MVP comfortable: better source display, index progress, and clearer errors.
2. Get `llama-cpp-python` working with the CPU wheel path.
3. Switch the chat engine to llama-cpp chat completion once the runtime import succeeds.
4. Choose a GPU strategy only after CPU inference works.

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
