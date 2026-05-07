from __future__ import annotations

import os
import string
import threading
import urllib.request
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ferret_rag.core.config import AppConfig
from ferret_rag.core.paths import app_root, resource_root
from ferret_rag.index.loaders import SUPPORTED_EXTENSIONS
from ferret_rag.index.store import LocalIndex
from ferret_rag.llm.engine import LocalChatEngine


class IndexRequest(BaseModel):
    path: str


class ChatRequest(BaseModel):
    message: str
    top_k: int | None = None


class ModelSelectRequest(BaseModel):
    path: str


class ModelDownloadRequest(BaseModel):
    id: str


class RuntimeUpdateRequest(BaseModel):
    gpu_layers: str | int


class RemoveIndexRequest(BaseModel):
    path: str


class FailureResponse(BaseModel):
    file_path: str
    error: str


class IndexResponse(BaseModel):
    status: str
    path: str
    files_found: int
    files_indexed: int
    files_skipped: int
    files_failed: int
    chunks_total: int
    failures: list[FailureResponse]


class RemoveIndexResponse(BaseModel):
    status: str
    path: str
    files_removed: int
    chunks_removed: int
    roots_removed: int
    chunks_total: int


class IndexedRootResponse(BaseModel):
    path: str
    file_count: int
    active_file_count: int
    chunk_count: int
    last_result: dict[str, int]


class IndexedFileResponse(BaseModel):
    file_path: str
    file_name: str
    file_type: str
    chunk_count: int
    modified_time: float
    root_path: str | None = None


class IndexStateResponse(BaseModel):
    roots: list[IndexedRootResponse]
    files: list[IndexedFileResponse]
    chunks_total: int


class SourceResponse(BaseModel):
    id: str | None = None
    file_path: str
    file_name: str
    file_type: str
    chunk_id: int
    text: str
    file_hash: str | None = None
    modified_time: float
    page_num: int | None = None
    label: str | None = None
    score: float | None = None


class SourcesResponse(BaseModel):
    sources: list[SourceResponse]


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]


class HealthResponse(BaseModel):
    status: str
    model_exists: bool
    model_path: str
    model_architecture: str | None
    model_compatible: bool
    gpu_layers: int
    gpu_mode: str
    gpu_message: str
    runtime_message: str
    chunks: int


class ServerConfigResponse(BaseModel):
    host: str
    port: int
    open_browser: bool


class ModelConfigResponse(BaseModel):
    path: str
    n_ctx: int
    gpu_layers: str | int


class IndexConfigResponse(BaseModel):
    data_dir: str
    chunk_words: int
    chunk_overlap: int
    top_k: int


class ConfigResponse(BaseModel):
    server: ServerConfigResponse
    model: ModelConfigResponse
    index: IndexConfigResponse


class ModelInfo(BaseModel):
    name: str
    path: str
    size_bytes: int
    architecture: str | None
    is_selected: bool
    is_compatible: bool
    status: str


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
    selected_model: str


class ModelCatalogItem(BaseModel):
    id: str
    name: str
    repo: str
    filename: str
    size_label: str
    notes: str


class ModelCatalogResponse(BaseModel):
    models: list[ModelCatalogItem]


class ModelDownloadResponse(BaseModel):
    status: str
    model: ModelInfo
    runtime: RuntimeResponse


class RuntimeResponse(BaseModel):
    model_path: str
    model_exists: bool
    architecture: str | None
    llama_cpp_available: bool
    llama_cpp_version: str | None
    is_compatible: bool
    gpu_layers: int
    gpu_mode: str
    gpu_message: str
    last_error: str | None
    message: str


class ModelSelectResponse(BaseModel):
    status: str
    runtime: RuntimeResponse


class FilesystemEntryResponse(BaseModel):
    name: str
    path: str
    kind: str
    modified_time: float | None = None


class FilesystemResponse(BaseModel):
    current_path: str | None
    parent_path: str | None
    folders: list[FilesystemEntryResponse]
    files: list[FilesystemEntryResponse]
    unsupported_files_count: int


class ShutdownResponse(BaseModel):
    status: str


MODEL_CATALOG = [
    ModelCatalogItem(
        id="llama32-1b-q4",
        name="Llama 3.2 1B Instruct Q4",
        repo="bartowski/Llama-3.2-1B-Instruct-GGUF",
        filename="Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        size_label="about 0.8 GB",
        notes="Fast starter model for smoke tests and lightweight machines.",
    ),
    ModelCatalogItem(
        id="gemma3-4b-q4",
        name="Gemma 3 4B IT Q4",
        repo="unsloth/gemma-3-4b-it-GGUF",
        filename="gemma-3-4b-it-Q4_K_M.gguf",
        size_label="about 2.6 GB",
        notes="Balanced local chat model; works with the newer llama.cpp runtime.",
    ),
    ModelCatalogItem(
        id="qwen35-9b-iq3",
        name="Qwen3.5 9B UD IQ3 XXS",
        repo="unsloth/Qwen3.5-9B-GGUF",
        filename="Qwen3.5-9B-UD-IQ3_XXS.gguf",
        size_label="about 4.0 GB",
        notes="Stronger answers if you have enough memory and GPU headroom.",
    ),
]


def create_app(config: AppConfig | None = None) -> FastAPI:
    app_config = config or AppConfig.load()
    selected_model_path = app_config.model.path
    index = LocalIndex(
        data_dir=app_config.index.data_dir,
        chunk_words=app_config.index.chunk_words,
        overlap=app_config.index.chunk_overlap,
    )
    chat = LocalChatEngine(
        selected_model_path,
        n_ctx=app_config.model.n_ctx,
        gpu_layers=app_config.model.gpu_layers,
    )
    ui_dir = resource_root() / "ferret_rag" / "ui"
    icons_dir = resource_root() / "icons"

    app = FastAPI(title="FerretRAG", version="0.1.0")
    app.mount("/static", StaticFiles(directory=ui_dir), name="static")
    if icons_dir.exists():
        app.mount("/icons", StaticFiles(directory=icons_dir), name="icons")

    @app.get("/")
    def home() -> FileResponse:
        return FileResponse(ui_dir / "index.html")

    @app.get("/favicon.ico", response_model=None)
    def favicon():
        icon_path = icons_dir / "icon_round.png"
        if not icon_path.exists():
            return Response(status_code=204)
        return FileResponse(icon_path)

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        runtime = RuntimeResponse(**chat.runtime_status())
        return HealthResponse(
            status="ok",
            model_exists=runtime.model_exists,
            model_path=runtime.model_path,
            model_architecture=runtime.architecture,
            model_compatible=runtime.is_compatible,
            gpu_layers=runtime.gpu_layers,
            gpu_mode=runtime.gpu_mode,
            gpu_message=runtime.gpu_message,
            runtime_message=runtime.message,
            chunks=len(index.sources()),
        )

    @app.get("/api/config", response_model=ConfigResponse)
    def get_config() -> ConfigResponse:
        return ConfigResponse(
            server=ServerConfigResponse(
                host=app_config.server.host,
                port=app_config.server.port,
                open_browser=app_config.server.open_browser,
            ),
            model=ModelConfigResponse(
                path=str(chat.model_path),
                n_ctx=app_config.model.n_ctx,
                gpu_layers=chat.gpu_layers,
            ),
            index=IndexConfigResponse(
                data_dir=str(app_config.index.data_dir),
                chunk_words=app_config.index.chunk_words,
                chunk_overlap=app_config.index.chunk_overlap,
                top_k=app_config.index.top_k,
            ),
        )

    @app.get("/api/models", response_model=ModelsResponse)
    def get_models() -> ModelsResponse:
        models_dir = app_root() / "models"
        models = [
            _model_info(path, chat.model_path)
            for path in sorted(models_dir.glob("*.gguf"))
        ]
        return ModelsResponse(models=models, selected_model=str(chat.model_path))

    @app.get("/api/model-catalog", response_model=ModelCatalogResponse)
    def model_catalog() -> ModelCatalogResponse:
        return ModelCatalogResponse(models=MODEL_CATALOG)

    @app.get("/api/model-filesystem", response_model=FilesystemResponse)
    def model_filesystem(path: str | None = Query(default=None)) -> FilesystemResponse:
        try:
            return _filesystem_response(path, file_extensions={".gguf"})
        except (FileNotFoundError, NotADirectoryError, PermissionError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/runtime", response_model=RuntimeResponse)
    def get_runtime() -> RuntimeResponse:
        return RuntimeResponse(**chat.runtime_status())

    @app.post("/api/runtime", response_model=RuntimeResponse)
    def update_runtime(request: RuntimeUpdateRequest) -> RuntimeResponse:
        gpu_layers = _runtime_gpu_layers(request.gpu_layers)
        chat.set_gpu_layers(gpu_layers)
        return RuntimeResponse(**chat.runtime_status())

    @app.post("/api/model", response_model=ModelSelectResponse)
    def select_model(request: ModelSelectRequest) -> ModelSelectResponse:
        model_path = Path(request.path).expanduser()
        if not model_path.is_absolute():
            model_path = app_root() / model_path
        if model_path.suffix.lower() != ".gguf":
            raise HTTPException(status_code=400, detail="Model path must point to a GGUF file.")
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model file does not exist: {model_path}")
        chat.set_model_path(model_path)
        return ModelSelectResponse(
            status="selected",
            runtime=RuntimeResponse(**chat.runtime_status()),
        )

    @app.post("/api/model/download", response_model=ModelDownloadResponse)
    def download_model(request: ModelDownloadRequest) -> ModelDownloadResponse:
        catalog_item = _catalog_item(request.id)
        models_dir = app_root() / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / catalog_item.filename
        if model_path.suffix.lower() != ".gguf":
            raise HTTPException(status_code=400, detail="Catalog item must be a GGUF model.")

        already_downloaded = model_path.exists()
        if not already_downloaded:
            try:
                _download_file(_huggingface_url(catalog_item), model_path)
            except Exception as exc:
                model_path.unlink(missing_ok=True)
                detail = f"Model download failed: {exc}"
                raise HTTPException(status_code=502, detail=detail) from exc

        chat.set_model_path(model_path)
        return ModelDownloadResponse(
            status="ready" if already_downloaded else "downloaded",
            model=_model_info(model_path, chat.model_path),
            runtime=RuntimeResponse(**chat.runtime_status()),
        )

    @app.post("/api/index", response_model=IndexResponse)
    def index_path(request: IndexRequest) -> IndexResponse:
        target = Path(request.path).expanduser()
        try:
            stats = index.index_path(target)
        except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return IndexResponse(status="indexed", path=str(target), **stats)

    @app.get("/api/filesystem", response_model=FilesystemResponse)
    def filesystem(path: str | None = Query(default=None)) -> FilesystemResponse:
        try:
            return _filesystem_response(path)
        except (FileNotFoundError, NotADirectoryError, PermissionError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/index", response_model=IndexStateResponse)
    def index_state() -> IndexStateResponse:
        return IndexStateResponse(
            roots=[IndexedRootResponse(**root) for root in index.indexed_roots()],
            files=[IndexedFileResponse(**file.__dict__) for file in index.indexed_files()],
            chunks_total=len(index.sources()),
        )

    @app.delete("/api/index", response_model=RemoveIndexResponse)
    def remove_index(request: RemoveIndexRequest) -> RemoveIndexResponse:
        path = Path(request.path).expanduser()
        stats = index.remove_path(path)
        return RemoveIndexResponse(status="removed", path=str(path), **stats)

    @app.get("/api/sources", response_model=SourcesResponse)
    def sources() -> SourcesResponse:
        return SourcesResponse(
            sources=[SourceResponse(**chunk.__dict__) for chunk in index.sources()]
        )

    @app.post("/api/chat", response_model=ChatResponse)
    def chat_endpoint(request: ChatRequest) -> ChatResponse:
        top_k = request.top_k or max(app_config.index.top_k, 8)
        results = index.search(request.message, top_k=top_k)
        return ChatResponse(
            answer=chat.answer(request.message, results),
            sources=[
                SourceResponse(
                    **result.chunk.__dict__,
                    score=result.score,
                )
                for result in results
            ],
        )

    @app.post("/api/shutdown", response_model=ShutdownResponse)
    def shutdown() -> ShutdownResponse:
        _schedule_shutdown()
        return ShutdownResponse(status="shutting_down")

    return app


def _filesystem_response(
    path: str | None,
    file_extensions: set[str] | None = None,
) -> FilesystemResponse:
    active_extensions = file_extensions or SUPPORTED_EXTENSIONS
    if not path:
        return FilesystemResponse(
            current_path=None,
            parent_path=None,
            folders=_filesystem_roots(),
            files=[],
            unsupported_files_count=0,
        )

    current = Path(path).expanduser().resolve()
    if not current.exists():
        raise FileNotFoundError(f"Path does not exist: {current}")
    if not current.is_dir():
        raise NotADirectoryError(f"Path is not a folder: {current}")

    folders: list[FilesystemEntryResponse] = []
    files: list[FilesystemEntryResponse] = []
    unsupported_files_count = 0
    try:
        entries = sorted(current.iterdir(), key=lambda item: item.name.lower())
    except PermissionError as exc:
        raise PermissionError(f"Cannot read folder: {current}") from exc

    for entry in entries:
        if entry.name.startswith("."):
            continue
        try:
            stat = entry.stat()
        except OSError:
            continue
        if entry.is_dir():
            folders.append(
                FilesystemEntryResponse(
                    name=entry.name,
                    path=str(entry),
                    kind="folder",
                    modified_time=stat.st_mtime,
                )
            )
        elif entry.is_file() and entry.suffix.lower() in active_extensions:
            files.append(
                FilesystemEntryResponse(
                    name=entry.name,
                    path=str(entry),
                    kind="file",
                    modified_time=stat.st_mtime,
                )
            )
        elif entry.is_file():
            unsupported_files_count += 1

    parent = current.parent if current.parent != current else None
    return FilesystemResponse(
        current_path=str(current),
        parent_path=str(parent) if parent else None,
        folders=folders,
        files=files,
        unsupported_files_count=unsupported_files_count,
    )


def _filesystem_roots() -> list[FilesystemEntryResponse]:
    roots: list[FilesystemEntryResponse] = []
    home = Path.home()
    if home.exists():
        roots.append(
            FilesystemEntryResponse(
                name=f"Home ({home.name})",
                path=str(home),
                kind="folder",
                modified_time=None,
            )
        )

    for letter in string.ascii_uppercase:
        drive = Path(f"{letter}:\\")
        if drive.exists():
            roots.append(
                FilesystemEntryResponse(
                    name=f"{letter}:",
                    path=str(drive),
                    kind="folder",
                    modified_time=None,
                )
            )

    if not roots:
        roots.append(
            FilesystemEntryResponse(
                name="/",
                path=str(Path("/")),
                kind="folder",
                modified_time=None,
            )
        )
    return roots


def _catalog_item(item_id: str) -> ModelCatalogItem:
    for item in MODEL_CATALOG:
        if item.id == item_id:
            return item
    raise HTTPException(status_code=404, detail=f"Unknown model catalog item: {item_id}")


def _huggingface_url(item: ModelCatalogItem) -> str:
    return f"https://huggingface.co/{item.repo}/resolve/main/{item.filename}?download=true"


def _download_file(url: str, destination: Path) -> None:
    temp_path = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "FerretRAG/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        with temp_path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    temp_path.replace(destination)


def _schedule_shutdown() -> None:
    threading.Timer(0.35, lambda: os._exit(0)).start()


def _runtime_gpu_layers(value: str | int) -> str | int:
    if isinstance(value, int):
        return value

    cleaned = str(value).strip().lower()
    if cleaned == "auto":
        return "auto"
    try:
        return int(cleaned)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="gpu_layers must be 'auto', 0, or a positive integer.",
        ) from None


def _model_info(path: Path, selected_path: Path) -> ModelInfo:
    runtime = LocalChatEngine(path, gpu_layers=0).runtime_status()
    architecture = runtime["architecture"]
    is_compatible = bool(runtime["is_compatible"])
    return ModelInfo(
        name=path.name,
        path=str(path),
        size_bytes=path.stat().st_size,
        architecture=architecture if isinstance(architecture, str) else None,
        is_selected=path.resolve() == selected_path.resolve(),
        is_compatible=is_compatible,
        status="compatible" if is_compatible else str(runtime["message"]),
    )
