from __future__ import annotations

import platform
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass

CUDA_WHEEL_VERSIONS = {
    "12.1": "cu121",
    "12.2": "cu122",
    "12.3": "cu123",
    "12.4": "cu124",
    "12.5": "cu125",
}
CPU_WHEEL_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/cpu"
LLAMA_CPP_PACKAGE = "llama-cpp-python"

RunCommand = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class GpuDetection:
    vendor: str | None
    backend: str
    gpu_available: bool
    install_backend: str
    wheel_index: str
    build_from_source: bool
    message: str


def detect_gpu_backend(
    *,
    run: RunCommand = subprocess.run,
    system: str | None = None,
    machine: str | None = None,
    version_info: tuple[int, int] | None = None,
) -> GpuDetection:
    system_name = system or platform.system()
    machine_name = (machine or platform.machine()).lower()
    python_version = version_info or (sys.version_info.major, sys.version_info.minor)

    if system_name == "Darwin" and machine_name in {"arm64", "aarch64"}:
        if _python_supports_prebuilt_gpu_wheel(python_version):
            return GpuDetection(
                vendor="Apple",
                backend="metal",
                gpu_available=True,
                install_backend="metal",
                wheel_index="https://abetlen.github.io/llama-cpp-python/whl/metal",
                build_from_source=False,
                message=(
                    "Apple Silicon detected; Metal GPU runtime is available "
                    "as a prebuilt wheel."
                ),
            )
        return _cpu_detection(
            vendor="Apple",
            message=(
                "Apple Silicon detected, but the Metal prebuilt wheel supports Python "
                "3.10-3.12. Using the CPU wheel for this Python version."
            ),
        )

    nvidia = _detect_nvidia(run)
    if nvidia:
        cuda_wheel = CUDA_WHEEL_VERSIONS.get(nvidia)
        if cuda_wheel and _python_supports_prebuilt_gpu_wheel(python_version):
            return GpuDetection(
                vendor="NVIDIA",
                backend="cuda",
                gpu_available=True,
                install_backend="cuda",
                wheel_index=f"https://abetlen.github.io/llama-cpp-python/whl/{cuda_wheel}",
                build_from_source=False,
                message=f"NVIDIA GPU with CUDA {nvidia} detected; CUDA runtime can be installed.",
            )
        reason = (
            f"CUDA {nvidia} is not available as a safe prebuilt wheel"
            if cuda_wheel is None
            else "CUDA prebuilt wheels support Python 3.10-3.12"
        )
        return _vulkan_detection(
            vendor="NVIDIA",
            message=(
                f"NVIDIA GPU detected, but {reason}. FerretRAG will build the "
                "Vulkan runtime so this GPU can still be used."
            ),
        )

    vendor = _detect_display_vendor(run, system_name)
    if vendor in {"AMD", "Intel", "NVIDIA"}:
        return _vulkan_detection(
            vendor=vendor,
            message=(
                f"{vendor} GPU detected. FerretRAG will build a Vulkan-enabled "
                "llama-cpp-python runtime for cross-vendor GPU offload."
            ),
        )

    return _cpu_detection(vendor=None, message="No supported GPU runtime was detected. Using CPU.")


def resolve_gpu_layers(
    gpu_layers: str | int,
    detection: GpuDetection | None = None,
) -> tuple[int, str, str]:
    if isinstance(gpu_layers, int):
        if gpu_layers <= 0:
            return 0, "cpu", "GPU offload is disabled by configuration."
        return gpu_layers, "gpu", f"Configured to offload {gpu_layers} layer(s) to the GPU."

    value = str(gpu_layers).strip().lower()
    if value == "auto":
        active_detection = detection or detect_gpu_backend()
        if active_detection.gpu_available and active_detection.install_backend != "cpu":
            return -1, "auto-gpu", active_detection.message
        return 0, "auto-cpu", active_detection.message

    try:
        parsed = int(value)
    except ValueError:
        return 0, "cpu", f"Invalid gpu_layers value {gpu_layers!r}; using CPU."
    return resolve_gpu_layers(parsed, detection=detection)


def install_command(detection: GpuDetection) -> list[str]:
    if detection.build_from_source:
        return [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            "--no-binary=llama-cpp-python",
            LLAMA_CPP_PACKAGE,
        ]

    return [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--force-reinstall",
        "--only-binary=:all:",
        "--extra-index-url",
        detection.wheel_index,
        LLAMA_CPP_PACKAGE,
    ]


def install_environment(detection: GpuDetection) -> dict[str, str]:
    if detection.build_from_source and detection.install_backend == "vulkan":
        return {"CMAKE_ARGS": "-DGGML_VULKAN=ON", "FORCE_CMAKE": "1"}
    return {}


def _cpu_detection(vendor: str | None, message: str) -> GpuDetection:
    return GpuDetection(
        vendor=vendor,
        backend="cpu",
        gpu_available=False,
        install_backend="cpu",
        wheel_index=CPU_WHEEL_INDEX,
        build_from_source=False,
        message=message,
    )


def _vulkan_detection(vendor: str, message: str) -> GpuDetection:
    return GpuDetection(
        vendor=vendor,
        backend="vulkan",
        gpu_available=True,
        install_backend="vulkan",
        wheel_index="",
        build_from_source=True,
        message=message,
    )


def _python_supports_prebuilt_gpu_wheel(version_info: tuple[int, int]) -> bool:
    return version_info in {(3, 10), (3, 11), (3, 12)}


def _detect_nvidia(run: RunCommand) -> str | None:
    try:
        result = run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if result.returncode != 0:
        return None
    output = f"{result.stdout}\n{result.stderr}"
    match = re.search(r"CUDA Version:\s*(\d+\.\d+)", output)
    return match.group(1) if match else "unknown"


def _detect_display_vendor(run: RunCommand, system_name: str) -> str | None:
    commands = {
        "Windows": [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
        ],
        "Linux": ["sh", "-c", "lspci 2>/dev/null | grep -Ei 'vga|3d|display'"],
    }
    command = commands.get(system_name)
    if command is None:
        return None

    try:
        result = run(command, capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None

    output = f"{result.stdout}\n{result.stderr}".lower()
    if "nvidia" in output or "geforce" in output or " rtx " in output:
        return "NVIDIA"
    if "amd" in output or "radeon" in output:
        return "AMD"
    if "intel" in output or "arc" in output:
        return "Intel"
    return None
