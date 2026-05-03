# FerretRAG Manual QA Checklist

Run this before tagging a release candidate.

## Environment

- Fresh virtual environment created.
- `pip install -e ".[dev]"` succeeds.
- `pip install -r requirements/llm-cpu.txt` succeeds or the missing runtime is documented.
- `ruff check .` passes.
- `pytest` passes.

## App Startup

- `scripts/run_dev.cmd --port 8765 --no-browser` starts the server.
- Browser loads `http://127.0.0.1:8765`.
- `/favicon.ico` returns without a 404.
- `/api/health`, `/api/config`, `/api/models`, and `/api/runtime` return 200.

## Models

- Qwen model is listed as selected when present.
- Llama 3.2 model is listed as compatible when present.
- Gemma 4 model is listed as unsupported with the pinned CPU runtime.
- Switching to a compatible model updates the UI status.
- Missing or incompatible models do not crash chat.

## Indexing

- Empty or invalid folder path shows a clear error.
- Mixed folder with TXT, MD, HTML, CSV, PDF, DOCX, XLSX indexes.
- Broken files are reported in `failures` and do not stop the whole run.
- Re-indexing unchanged files reports skipped files.
- Indexed folders and files appear in the Indexed panel.
- Removing an indexed file removes its chunks from sources and search.
- Removing an indexed folder removes all files under that folder.
- Source cards show file name, path, type/page/sheet, chunk id, and snippet.

## Chat

- Chat shows a loading state while generating.
- Compatible model returns an answer.
- Missing runtime or incompatible model falls back to snippets with a clear runtime note.
- Copy answer copies the latest assistant response.
- Clear chat resets messages and sources.

## Packaging Smoke

- `scripts/build_windows.cmd` either builds successfully or gives a clear PyInstaller setup message.
- PyInstaller must be installed inside this project's `.venv`, not only globally.
- Packaged app starts on a machine with the documented model setup.
