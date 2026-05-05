from __future__ import annotations

import importlib.util
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ferret_rag.llm.gguf import read_gguf_architecture  # noqa: E402
from ferret_rag.llm.gpu import detect_gpu_backend, resolve_gpu_layers  # noqa: E402

DEFAULT_MODEL = PROJECT_ROOT / "models" / "Qwen3.5-9B-UD-IQ3_XXS.gguf"


def main() -> None:
    model_path, configured_gpu_layers = _args()
    print(f"Python: {platform.python_version()} ({platform.architecture()[0]})")
    print(f"Platform: {platform.platform()}")
    print(f"Model: {model_path}")
    print(f"Model exists: {model_path.exists()}")
    architecture = read_gguf_architecture(model_path) if model_path.exists() else None
    print(f"Model architecture: {architecture or 'unknown'}")
    detection = detect_gpu_backend()
    gpu_layers, gpu_mode, gpu_message = resolve_gpu_layers(
        configured_gpu_layers,
        detection=detection,
    )
    print(f"Detected GPU backend: {detection.backend}")
    print(f"GPU layers: {gpu_layers} ({gpu_mode})")
    print(f"GPU note: {gpu_message}")
    if detection.build_from_source:
        print("GPU install note: this backend may require the installer to build from source.")

    spec = importlib.util.find_spec("llama_cpp")
    if spec is None:
        print("llama_cpp import: missing")
        print("Install runtime with: .venv\\Scripts\\python scripts\\install_llm_runtime.py")
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
        llm = Llama(
            model_path=str(model_path),
            n_ctx=512,
            n_gpu_layers=gpu_layers,
            verbose=False,
        )
    except Exception as exc:
        print(f"Model load: failed ({exc})")
        if gpu_layers != 0:
            print("Retrying with CPU because GPU load failed...")
            try:
                llm = Llama(
                    model_path=str(model_path),
                    n_ctx=512,
                    n_gpu_layers=0,
                    verbose=False,
                )
            except Exception as cpu_exc:
                print(f"CPU retry: failed ({cpu_exc})")
                return
            print("CPU retry: ok")
        else:
            print(
                "If the error mentions unknown architecture 'gemma4', this llama-cpp-python "
                "build is too old for the bundled model."
            )
            return
    else:
        print("Model load: ok")
        if gpu_layers == 0:
            print("GPU offload: disabled")
        else:
            print("GPU offload: requested")

    if architecture == "gemma4" and _version_tuple(version) <= (0, 3, 19):
        print(
            "If the error mentions unknown architecture 'gemma4', this llama-cpp-python "
            "build is too old for the bundled model."
        )

    output = llm(
        "Q: Say hello in one short sentence.\nA:",
        max_tokens=24,
        stop=["Q:"],
        echo=False,
    )
    text = output["choices"][0]["text"].strip()
    print(f"Model response: {text}")


def _args() -> tuple[Path, str | int]:
    gpu_layers: str | int = "auto"
    args = sys.argv[1:]
    if "--gpu-layers" in args:
        index = args.index("--gpu-layers")
        try:
            gpu_layers = args[index + 1]
        except IndexError:
            raise SystemExit("--gpu-layers requires a value") from None
        del args[index : index + 2]

    if not args:
        return DEFAULT_MODEL, gpu_layers

    path = Path(args[0])
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path, gpu_layers


def _version_tuple(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for part in version.split("."):
        if not part.isdigit():
            break
        parts.append(int(part))
    return tuple(parts)


if __name__ == "__main__":
    main()
