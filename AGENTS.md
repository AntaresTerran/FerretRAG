# AGENTS.md

## Project Overview
- **Project:** FerretRAG - a privacy-first local RAG app for chatting with files in folders on the user's machine.
- **Target user:** Privacy-conscious users who want local document search and chat without cloud APIs.
- **My skill level:** Intermediate.
- **Stack:** Python, FastAPI, static HTML/CSS/JS, optional llama-cpp-python, Chroma, pytest, ruff.

## Commands
- **Install:** `python -m venv .venv` then `.venv\Scripts\pip install -e ".[dev]"`
- **Optional LLM runtime:** `.venv\Scripts\pip install -e ".[llm]"`
- **Preferred CPU LLM install:** `.venv\Scripts\pip install -r requirements\llm-cpu.txt`
- **Dev:** `.venv\Scripts\python -m ferret_rag`
- **Build:** Packaging is a later milestone. Use PyInstaller once packaging scripts exist.
- **Test:** `.venv\Scripts\pytest`
- **Lint:** `.venv\Scripts\ruff check .`

## Do
- Read existing code before modifying anything.
- Match existing patterns, naming, and style.
- Keep all inference, indexing, and storage local by default.
- Keep changes small and scoped to what was asked.
- Handle errors clearly. Do not hide failures.
- Run tests after changes when dependencies are available.
- Add focused tests for new features.
- Keep `models/*.gguf` out of git.

## Don't
- Install new dependencies without asking.
- Delete or overwrite user files without confirming.
- Hardcode secrets, API keys, credentials, or private paths.
- Add cloud API calls to the core app without explicit approval.
- Commit model files, runtime data, virtualenvs, or build artifacts.
- Push, deploy, publish releases, or force-push without permission.
- Rewrite working code unless explicitly asked.

## When Stuck
- If a task is large, break it into steps and confirm the plan first.
- If an error cannot be fixed in two focused attempts, stop and explain the issue.

## Testing
- Run existing tests after any change when the environment is ready.
- Add at least one test for new features.
- Never skip or delete tests just to make a run pass.

## Git
- Use small, focused commits with descriptive messages.
- Use Conventional Commit style when practical, such as `feat:`, `fix:`, or `docs:`.
- Never force-push.

## Response Style
- Use clear and concise messages.
- Use plain English when explaining work to the user.
- Avoid long sentences and long paragraphs.
