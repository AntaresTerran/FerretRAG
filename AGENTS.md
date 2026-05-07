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
- **GPU/runtime installer:** `.venv\Scripts\python scripts\install_llm_runtime.py`
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
- Treat user-visible runtime goals as acceptance criteria, not just implementation tasks.
- If a requested feature depends on a local runtime, driver, build tool, or package install, verify the real runtime behavior when possible.
- If reaching the goal requires installing dependencies or build tools, explain what is needed and ask before installing.

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
- If the next step is risky, slow, or changes the user's environment, ask the user to choose a path instead of ending with a note.
- Do not present a workaround, warning, or documentation note as completion when the requested behavior still does not work.

## LLM and GPU Runtime
- The goal "run the LLM on GPU" is only complete when the app actually loads the model with GPU offload on the target machine.
- For NVIDIA, Apple Silicon, AMD, and Intel, check the detected backend and the installed `llama-cpp-python` runtime.
- For AMD and Intel on Windows, expect Vulkan to require a locally built or Vulkan-enabled `llama-cpp-python` runtime.
- After GPU changes, verify more than API fields: run a real model load or `scripts\check_llm.py` with GPU enabled when a model and runtime are available.
- If verification shows CPU fallback, treat that as an unfinished requirement. Ask before installing or rebuilding the runtime needed to finish it.
- Surface the active CPU/GPU mode clearly in the UI, but UI status alone is not proof that GPU execution works.

## Testing
- Run existing tests after any change when the environment is ready.
- Add at least one test for new features.
- Never skip or delete tests just to make a run pass.
- For runtime-sensitive work, combine automated tests with a real smoke test of the relevant runtime path.

## Git
- Use small, focused commits with descriptive messages.
- Use Conventional Commit style when practical, such as `feat:`, `fix:`, or `docs:`.
- Never force-push.

## Response Style
- Use clear and concise messages.
- Use plain English when explaining work to the user.
- Avoid long sentences and long paragraphs.
