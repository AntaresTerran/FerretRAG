from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ferret_rag.llm.gpu import (  # noqa: E402
    detect_gpu_backend,
    install_command,
    install_environment,
)


def main() -> None:
    args = _parse_args()
    detection = detect_gpu_backend()
    command = install_command(detection)
    install_env = install_environment(detection)

    print("FerretRAG local LLM runtime installer")
    print(f"Detected backend: {detection.backend}")
    print(f"Detected vendor: {detection.vendor or 'none'}")
    print(detection.message)
    print()
    print("Install command:")
    for name, value in install_env.items():
        print(f"{name}={value}")
    print(" ".join(command))

    if args.dry_run:
        print()
        print("Dry run only; no packages were installed.")
        return

    print()
    if detection.build_from_source:
        print(
            "Building llama-cpp-python from source for this GPU backend. "
            "This can take several minutes and requires local build tools."
        )
    else:
        print("Installing prebuilt llama-cpp-python runtime.")
    env = os.environ.copy()
    env.update(install_env)
    subprocess.run(command, check=True, env=env)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Install the best llama-cpp-python runtime for this machine. "
            "This prefers prebuilt wheels and builds Vulkan locally when needed."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the selected backend and pip command without installing anything.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
