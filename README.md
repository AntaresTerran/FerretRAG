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

You can test a specific model file:

```powershell
.venv\Scripts\python scripts\check_llm.py models\Llama-3.2-1B-Instruct-UD-Q8_K_XL.gguf
.venv\Scripts\python scripts\check_llm.py models\Qwen3.5-9B-UD-IQ3_XXS.gguf
```

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

Known local model status with the pinned CPU wheel:

| Model | Architecture | Status |
| --- | --- | --- |
| `gemma-4-E2B-it-UD-IQ3_XXS.gguf` | `gemma4` | Does not load with `llama-cpp-python 0.3.19`. |
| `Llama-3.2-1B-Instruct-UD-Q8_K_XL.gguf` | `llama` | Loads successfully. |
| `Qwen3.5-9B-UD-IQ3_XXS.gguf` | `qwen35` | Loads successfully. |

To run the app with a compatible model, copy `config.yaml.example` to `config.yaml` and set:

```yaml
model:
  path: models/Qwen3.5-9B-UD-IQ3_XXS.gguf
```

## Run

```powershell
.venv\Scripts\python -m ferret_rag
```

The app starts a local FastAPI server and opens the UI at `http://127.0.0.1:8765`.

You can also use the development launcher:

```powershell
scripts\run_dev.cmd --port 8765
```

## Test and Lint

```powershell
.venv\Scripts\pytest
.venv\Scripts\ruff check .
```

## Packaging

Windows packaging is script-based for the release-candidate path:

```powershell
scripts\build_windows.cmd
```

The script expects PyInstaller to be installed in the active virtual environment. The PowerShell
variants are also available, but Windows execution policy may block `.ps1` files. GGUF model files
are not bundled by default; users should place compatible models under `models/`.

Install PyInstaller into this project venv before packaging:

```powershell
.venv\Scripts\pip install pyinstaller
```

## Release Candidate Docs

- `ROADMAP.md`
- `docs/QA_CHECKLIST.md`
- `docs/PRIVACY_CHECKLIST.md`
- `docs/RELEASE_NOTES_RC.md`

## Recommended Next Milestones

1. Validate the Windows packaging script on a clean machine.
2. Polish model selection persistence.
3. Decide whether a bundled compatible GGUF is allowed for release.
4. Choose a GPU strategy after the CPU release candidate is stable.

## Model File

The current development default local model is expected at:

```text
models/Qwen3.5-9B-UD-IQ3_XXS.gguf
```

GGUF files are ignored by git because they are large. Before shipping a release with a bundled model,
verify the model license and redistribution terms.

## MVP Scope

- FastAPI backend with health, indexing, sources, and chat endpoints.
- Static HTML/CSS/JS UI using the silver/albino ferret palette.
- Incremental file indexing based on SHA-256 hashes.
- TXT, MD, HTML, CSV, PDF, DOCX, and XLSX parsing.
- Chroma-backed vector search when available, with a local JSON keyword fallback for early development.
