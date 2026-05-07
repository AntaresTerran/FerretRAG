from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from ferret_rag.api import app as api_app
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
    assert "gpu_mode" in response.json()


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


def test_model_catalog_endpoint(tmp_path: Path) -> None:
    config = AppConfig(
        server=ServerConfig(open_browser=False),
        model=ModelConfig(path=tmp_path / "missing.gguf"),
        index=IndexConfig(data_dir=tmp_path / "data"),
    )
    client = TestClient(create_app(config))

    response = client.get("/api/model-catalog")
    body = response.json()

    assert response.status_code == 200
    assert body["models"]
    assert {"id", "name", "repo", "filename", "size_label", "notes"} <= body["models"][0].keys()


def test_model_filesystem_endpoint_lists_gguf_files(tmp_path: Path) -> None:
    (tmp_path / "chosen.gguf").write_bytes(b"not a real model")
    (tmp_path / "notes.txt").write_text("skip me", encoding="utf-8")
    config = AppConfig(
        server=ServerConfig(open_browser=False),
        model=ModelConfig(path=tmp_path / "missing.gguf"),
        index=IndexConfig(data_dir=tmp_path / "data"),
    )
    client = TestClient(create_app(config))

    response = client.get("/api/model-filesystem", params={"path": str(tmp_path)})
    body = response.json()

    assert response.status_code == 200
    assert [file["name"] for file in body["files"]] == ["chosen.gguf"]
    assert body["unsupported_files_count"] == 1


def test_download_model_endpoint_downloads_and_selects(tmp_path: Path, monkeypatch) -> None:
    catalog_item = api_app.ModelCatalogItem(
        id="tiny-test",
        name="Tiny Test Model",
        repo="example/tiny",
        filename="tiny.gguf",
        size_label="tiny",
        notes="test fixture",
    )

    def fake_download(_url: str, destination: Path) -> None:
        destination.write_bytes(b"not a real model")

    monkeypatch.setattr(api_app, "MODEL_CATALOG", [catalog_item])
    monkeypatch.setattr(api_app, "app_root", lambda: tmp_path)
    monkeypatch.setattr(api_app, "_download_file", fake_download)
    config = AppConfig(
        server=ServerConfig(open_browser=False),
        model=ModelConfig(path=tmp_path / "missing.gguf"),
        index=IndexConfig(data_dir=tmp_path / "data"),
    )
    client = TestClient(create_app(config))

    response = client.post("/api/model/download", json={"id": "tiny-test"})
    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "downloaded"
    assert body["model"]["path"] == str(tmp_path / "models" / "tiny.gguf")
    assert body["runtime"]["model_path"] == str(tmp_path / "models" / "tiny.gguf")


def test_shutdown_endpoint_schedules_shutdown(tmp_path: Path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(api_app, "_schedule_shutdown", lambda: calls.append("scheduled"))
    config = AppConfig(
        server=ServerConfig(open_browser=False),
        model=ModelConfig(path=tmp_path / "missing.gguf"),
        index=IndexConfig(data_dir=tmp_path / "data"),
    )
    client = TestClient(create_app(config))

    response = client.post("/api/shutdown")

    assert response.status_code == 200
    assert response.json()["status"] == "shutting_down"
    assert calls == ["scheduled"]


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
    assert "gpu_layers" in response.json()


def test_runtime_update_endpoint_switches_gpu_layers(tmp_path: Path) -> None:
    config = AppConfig(
        server=ServerConfig(open_browser=False),
        model=ModelConfig(path=tmp_path / "missing.gguf"),
        index=IndexConfig(data_dir=tmp_path / "data"),
    )
    client = TestClient(create_app(config))

    response = client.post("/api/runtime", json={"gpu_layers": 0})
    config_response = client.get("/api/config")

    assert response.status_code == 200
    assert response.json()["gpu_layers"] == 0
    assert response.json()["gpu_mode"] == "cpu"
    assert config_response.json()["model"]["gpu_layers"] == 0


def test_runtime_update_endpoint_rejects_invalid_gpu_layers(tmp_path: Path) -> None:
    config = AppConfig(
        server=ServerConfig(open_browser=False),
        model=ModelConfig(path=tmp_path / "missing.gguf"),
        index=IndexConfig(data_dir=tmp_path / "data"),
    )
    client = TestClient(create_app(config))

    response = client.post("/api/runtime", json={"gpu_layers": "banana"})

    assert response.status_code == 400
    assert "gpu_layers" in response.json()["detail"]


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


def test_index_single_file_endpoint(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.md"
    file_path.write_text("FerretRAG indexes single files too.", encoding="utf-8")
    config = AppConfig(
        server=ServerConfig(open_browser=False),
        model=ModelConfig(path=tmp_path / "missing.gguf"),
        index=IndexConfig(data_dir=tmp_path / "data", chunk_words=20, chunk_overlap=2),
    )
    client = TestClient(create_app(config))

    response = client.post("/api/index", json={"path": str(file_path)})
    state_response = client.get("/api/index")

    assert response.status_code == 200
    assert response.json()["files_indexed"] == 1
    assert state_response.json()["files"][0]["file_name"] == "notes.md"


def test_index_rejects_unsupported_single_file(tmp_path: Path) -> None:
    file_path = tmp_path / "archive.zip"
    file_path.write_text("not indexable", encoding="utf-8")
    config = AppConfig(
        server=ServerConfig(open_browser=False),
        model=ModelConfig(path=tmp_path / "missing.gguf"),
        index=IndexConfig(data_dir=tmp_path / "data"),
    )
    client = TestClient(create_app(config))

    response = client.post("/api/index", json={"path": str(file_path)})

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_filesystem_endpoint_lists_folders_and_supported_files(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (tmp_path / "notes.md").write_text("index me", encoding="utf-8")
    (tmp_path / "image.png").write_text("skip me", encoding="utf-8")
    config = AppConfig(
        server=ServerConfig(open_browser=False),
        model=ModelConfig(path=tmp_path / "missing.gguf"),
        index=IndexConfig(data_dir=tmp_path / "data"),
    )
    client = TestClient(create_app(config))

    response = client.get("/api/filesystem", params={"path": str(tmp_path)})
    body = response.json()

    assert response.status_code == 200
    assert body["current_path"] == str(tmp_path.resolve())
    assert "docs" in [folder["name"] for folder in body["folders"]]
    assert [file["name"] for file in body["files"]] == ["notes.md"]
    assert body["unsupported_files_count"] == 1


def test_filesystem_endpoint_rejects_missing_path(tmp_path: Path) -> None:
    config = AppConfig(
        server=ServerConfig(open_browser=False),
        model=ModelConfig(path=tmp_path / "missing.gguf"),
        index=IndexConfig(data_dir=tmp_path / "data"),
    )
    client = TestClient(create_app(config))

    response = client.get("/api/filesystem", params={"path": str(tmp_path / "missing")})

    assert response.status_code == 400
    assert "Path does not exist" in response.json()["detail"]


def test_index_state_and_remove_endpoint(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    file_path = docs / "notes.md"
    file_path.write_text("FerretRAG keeps private files local.", encoding="utf-8")
    config = AppConfig(
        server=ServerConfig(open_browser=False),
        model=ModelConfig(path=tmp_path / "missing.gguf"),
        index=IndexConfig(data_dir=tmp_path / "data", chunk_words=20, chunk_overlap=2),
    )
    client = TestClient(create_app(config))

    client.post("/api/index", json={"path": str(docs)})
    state_response = client.get("/api/index")
    remove_response = client.request("DELETE", "/api/index", json={"path": str(file_path)})
    final_state = client.get("/api/index")

    assert state_response.status_code == 200
    assert state_response.json()["roots"][0]["active_file_count"] == 1
    assert remove_response.status_code == 200
    assert remove_response.json()["files_removed"] == 1
    assert final_state.json()["chunks_total"] == 0
