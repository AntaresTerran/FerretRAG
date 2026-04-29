from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ferret_rag.core.paths import app_root

PROJECT_ROOT = app_root()


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    open_browser: bool = True


@dataclass(frozen=True)
class ModelConfig:
    path: Path = PROJECT_ROOT / "models" / "Qwen3.5-9B-UD-IQ3_XXS.gguf"
    n_ctx: int = 4096
    gpu_layers: str | int = "auto"


@dataclass(frozen=True)
class IndexConfig:
    data_dir: Path = PROJECT_ROOT / "data"
    chunk_words: int = 300
    chunk_overlap: int = 50
    top_k: int = 5


@dataclass(frozen=True)
class AppConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    index: IndexConfig = field(default_factory=IndexConfig)

    @classmethod
    def load(cls, path: Path | None = None) -> AppConfig:
        config_path = path or PROJECT_ROOT / "config.yaml"
        if not config_path.exists():
            return cls()

        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        return cls(
            server=_server_config(raw.get("server", {})),
            model=_model_config(raw.get("model", {})),
            index=_index_config(raw.get("index", {})),
        )


def _server_config(raw: dict[str, Any]) -> ServerConfig:
    return ServerConfig(
        host=str(raw.get("host", "127.0.0.1")),
        port=int(raw.get("port", 8765)),
        open_browser=bool(raw.get("open_browser", True)),
    )


def _model_config(raw: dict[str, Any]) -> ModelConfig:
    model_path = Path(str(raw.get("path", ModelConfig.path)))
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    return ModelConfig(
        path=model_path,
        n_ctx=int(raw.get("n_ctx", 4096)),
        gpu_layers=raw.get("gpu_layers", "auto"),
    )


def _index_config(raw: dict[str, Any]) -> IndexConfig:
    data_dir = Path(str(raw.get("data_dir", "data")))
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir

    return IndexConfig(
        data_dir=data_dir,
        chunk_words=int(raw.get("chunk_words", 300)),
        chunk_overlap=int(raw.get("chunk_overlap", 50)),
        top_k=int(raw.get("top_k", 5)),
    )
