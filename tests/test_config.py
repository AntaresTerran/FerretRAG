from __future__ import annotations

from pathlib import Path

from ferret_rag.core.config import AppConfig
from ferret_rag.core.paths import app_root, resource_root


def test_config_loads_relative_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
server:
  host: 127.0.0.1
  port: 9999
  open_browser: false
model:
  path: models/example.gguf
index:
  data_dir: data
  chunk_words: 123
  chunk_overlap: 12
  top_k: 7
""",
        encoding="utf-8",
    )

    config = AppConfig.load(config_path)

    assert config.server.port == 9999
    assert config.model.path.name == "example.gguf"
    assert config.model.gpu_layers == "auto"
    assert config.index.chunk_words == 123
    assert config.index.top_k == 7


def test_config_loads_gpu_layers_integer(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  path: models/example.gguf
  gpu_layers: 24
""",
        encoding="utf-8",
    )

    config = AppConfig.load(config_path)

    assert config.model.gpu_layers == 24


def test_config_loads_gpu_layers_zero(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  path: models/example.gguf
  gpu_layers: 0
""",
        encoding="utf-8",
    )

    config = AppConfig.load(config_path)

    assert config.model.gpu_layers == 0


def test_paths_resolve_in_development() -> None:
    assert (app_root() / "pyproject.toml").exists()
    assert resource_root() == app_root()
