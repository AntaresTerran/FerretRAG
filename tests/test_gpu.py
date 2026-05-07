from __future__ import annotations

import subprocess

from ferret_rag.llm.gpu import detect_gpu_backend, install_command


def test_detects_nvidia_cuda_prebuilt_wheel() -> None:
    detection = detect_gpu_backend(
        run=_runner({"nvidia-smi": "CUDA Version: 12.4"}),
        system="Windows",
        version_info=(3, 12),
    )

    assert detection.vendor == "NVIDIA"
    assert detection.backend == "cuda"
    assert detection.gpu_available is True
    assert detection.wheel_index.endswith("/cu124")


def test_nvidia_unsupported_python_selects_vulkan_build() -> None:
    detection = detect_gpu_backend(
        run=_runner({"nvidia-smi": "CUDA Version: 12.4"}),
        system="Windows",
        version_info=(3, 13),
    )

    assert detection.vendor == "NVIDIA"
    assert detection.backend == "vulkan"
    assert detection.gpu_available is True
    assert detection.build_from_source is True


def test_detects_apple_metal_prebuilt_wheel() -> None:
    detection = detect_gpu_backend(
        run=_runner({}),
        system="Darwin",
        machine="arm64",
        version_info=(3, 12),
    )

    assert detection.vendor == "Apple"
    assert detection.backend == "metal"
    assert detection.gpu_available is True


def test_detects_amd_and_selects_vulkan_build() -> None:
    detection = detect_gpu_backend(
        run=_runner({"powershell": "AMD Radeon RX 7800 XT"}),
        system="Windows",
        version_info=(3, 12),
    )

    assert detection.vendor == "AMD"
    assert detection.backend == "vulkan"
    assert detection.gpu_available is True
    assert detection.build_from_source is True


def test_detects_nvidia_display_without_smi_and_selects_vulkan_build() -> None:
    detection = detect_gpu_backend(
        run=_runner({"powershell": "NVIDIA GeForce RTX 4070"}),
        system="Windows",
        version_info=(3, 12),
    )

    assert detection.vendor == "NVIDIA"
    assert detection.backend == "vulkan"
    assert detection.gpu_available is True
    assert detection.build_from_source is True


def test_detects_intel_and_selects_vulkan_build() -> None:
    detection = detect_gpu_backend(
        run=_runner({"powershell": "Intel Arc A770"}),
        system="Windows",
        version_info=(3, 12),
    )

    assert detection.vendor == "Intel"
    assert detection.backend == "vulkan"
    assert detection.gpu_available is True
    assert detection.build_from_source is True


def test_no_gpu_selects_cpu_install() -> None:
    detection = detect_gpu_backend(
        run=_runner({}),
        system="Windows",
        version_info=(3, 12),
    )

    assert detection.vendor is None
    assert detection.backend == "cpu"
    assert install_command(detection)[-1] == "llama-cpp-python"


def _runner(outputs: dict[str, str]):
    def run(command, *args, **kwargs):
        executable = command[0] if isinstance(command, list) else command
        output = outputs.get(executable)
        if output is None:
            return subprocess.CompletedProcess(command, 1, "", "")
        return subprocess.CompletedProcess(command, 0, output, "")

    return run
