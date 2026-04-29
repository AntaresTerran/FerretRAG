from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from ferret_rag.api.app import create_app
from ferret_rag.core.config import AppConfig, IndexConfig, ModelConfig, ServerConfig


def test_health_endpoint(tmp_path: Path) -> None:
    config = AppConfig(
        server=ServerConfig(open_browser=False),
        model=ModelConfig(path=tmp_path / "missing.gguf"),
        index=IndexConfig(data_dir=tmp_path / "data"),
    )
    client = TestClient(create_app(config))

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "model_path" in response.json()


def test_config_endpoint(tmp_path: Path) -> None:
    config = AppConfig(
        server=ServerConfig(open_browser=False),
        model=ModelConfig(path=tmp_path / "missing.gguf"),
        index=IndexConfig(data_dir=tmp_path / "data"),
    )
    client = TestClient(create_app(config))

    response = client.get("/api/config")

    assert response.status_code == 200
    assert response.json()["model"]["path"].endswith("missing.gguf")


def test_models_endpoint(tmp_path: Path) -> None:
    config = AppConfig(
        server=ServerConfig(open_browser=False),
        model=ModelConfig(path=tmp_path / "missing.gguf"),
        index=IndexConfig(data_dir=tmp_path / "data"),
    )
    client = TestClient(create_app(config))

    response = client.get("/api/models")

    assert response.status_code == 200
    assert "models" in response.json()
    assert "selected_model" in response.json()


def test_runtime_endpoint(tmp_path: Path) -> None:
    config = AppConfig(
        server=ServerConfig(open_browser=False),
        model=ModelConfig(path=tmp_path / "missing.gguf"),
        index=IndexConfig(data_dir=tmp_path / "data"),
    )
    client = TestClient(create_app(config))

    response = client.get("/api/runtime")

    assert response.status_code == 200
    assert response.json()["model_exists"] is False


def test_select_model_endpoint(tmp_path: Path) -> None:
    model = tmp_path / "test.gguf"
    model.write_bytes(b"not a real gguf")
    config = AppConfig(
        server=ServerConfig(open_browser=False),
        model=ModelConfig(path=tmp_path / "missing.gguf"),
        index=IndexConfig(data_dir=tmp_path / "data"),
    )
    client = TestClient(create_app(config))

    response = client.post("/api/model", json={"path": str(model)})

    assert response.status_code == 200
    assert response.json()["runtime"]["model_path"] == str(model)


def test_favicon_endpoint(tmp_path: Path) -> None:
    config = AppConfig(
        server=ServerConfig(open_browser=False),
        model=ModelConfig(path=tmp_path / "missing.gguf"),
        index=IndexConfig(data_dir=tmp_path / "data"),
    )
    client = TestClient(create_app(config))

    response = client.get("/favicon.ico")

    assert response.status_code in {200, 204}


def test_index_and_chat_endpoint(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notes.md").write_text("FerretRAG keeps private files local.", encoding="utf-8")
    config = AppConfig(
        server=ServerConfig(open_browser=False),
        model=ModelConfig(path=tmp_path / "missing.gguf"),
        index=IndexConfig(data_dir=tmp_path / "data", chunk_words=20, chunk_overlap=2),
    )
    client = TestClient(create_app(config))

    index_response = client.post("/api/index", json={"path": str(docs)})
    chat_response = client.post("/api/chat", json={"message": "private files"})

    assert index_response.status_code == 200
    assert chat_response.status_code == 200
    assert index_response.json()["failures"] == []
    assert chat_response.json()["sources"]
    assert chat_response.json()["sources"][0]["file_name"] == "notes.md"
