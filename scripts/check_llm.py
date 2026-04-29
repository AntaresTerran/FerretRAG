from __future__ import annotations

import importlib.util
import platform
import sys
from pathlib import Path

from ferret_rag.llm.gguf import read_gguf_architecture

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "Qwen3.5-9B-UD-IQ3_XXS.gguf"


def main() -> None:
    model_path = _model_path_from_args()
    print(f"Python: {platform.python_version()} ({platform.architecture()[0]})")
    print(f"Platform: {platform.platform()}")
    print(f"Model: {model_path}")
    print(f"Model exists: {model_path.exists()}")
    architecture = read_gguf_architecture(model_path) if model_path.exists() else None
    print(f"Model architecture: {architecture or 'unknown'}")

    spec = importlib.util.find_spec("llama_cpp")
    if spec is None:
        print("llama_cpp import: missing")
        print("Install CPU runtime with: .venv\\Scripts\\pip install -r requirements\\llm-cpu.txt")
        return

    import llama_cpp

    version = getattr(llama_cpp, "__version__", "unknown")
    print(f"llama_cpp import: ok ({version})")

    if not model_path.exists():
        print("Skipping model load because the GGUF file is missing.")
        return

    if architecture == "gemma4" and _version_tuple(version) <= (0, 3, 19):
        print(
            f"Skipping model load: architecture '{architecture}' needs a newer "
            f"llama-cpp-python than installed version {version}."
        )
        return

    from llama_cpp import Llama

    print("Loading model with a tiny context to verify the runtime...")
    try:
        llm = Llama(model_path=str(model_path), n_ctx=512, n_gpu_layers=0, verbose=False)
    except Exception as exc:
        print(f"Model load: failed ({exc})")
        print(
            "If the error mentions unknown architecture 'gemma4', this llama-cpp-python "
            "build is too old for the bundled model."
        )
        return

    output = llm(
        "Q: Say hello in one short sentence.\nA:",
        max_tokens=24,
        stop=["Q:"],
        echo=False,
    )
    text = output["choices"][0]["text"].strip()
    print(f"Model response: {text}")


def _model_path_from_args() -> Path:
    if len(sys.argv) <= 1:
        return DEFAULT_MODEL

    path = Path(sys.argv[1])
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _version_tuple(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for part in version.split("."):
        if not part.isdigit():
            break
        parts.append(int(part))
    return tuple(parts)


if __name__ == "__main__":
    main()
