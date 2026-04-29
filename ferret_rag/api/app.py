from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ferret_rag.core.config import AppConfig
from ferret_rag.core.paths import app_root, resource_root
from ferret_rag.index.store import LocalIndex
from ferret_rag.llm.engine import LocalChatEngine


class IndexRequest(BaseModel):
    path: str


class ChatRequest(BaseModel):
    message: str
    top_k: int | None = None


class ModelSelectRequest(BaseModel):
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


class RuntimeResponse(BaseModel):
    model_path: str
    model_exists: bool
    architecture: str | None
    llama_cpp_available: bool
    llama_cpp_version: str | None
    is_compatible: bool
    last_error: str | None
    message: str


class ModelSelectResponse(BaseModel):
    status: str
    runtime: RuntimeResponse


def create_app(config: AppConfig | None = None) -> FastAPI:
    app_config = config or AppConfig.load()
    selected_model_path = app_config.model.path
    index = LocalIndex(
        data_dir=app_config.index.data_dir,
        chunk_words=app_config.index.chunk_words,
        overlap=app_config.index.chunk_overlap,
    )
    chat = LocalChatEngine(selected_model_path)
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
                gpu_layers=app_config.model.gpu_layers,
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

    @app.get("/api/runtime", response_model=RuntimeResponse)
    def get_runtime() -> RuntimeResponse:
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

    @app.post("/api/index", response_model=IndexResponse)
    def index_folder(request: IndexRequest) -> IndexResponse:
        folder = Path(request.path).expanduser()
        try:
            stats = index.index_folder(folder)
        except (FileNotFoundError, NotADirectoryError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return IndexResponse(status="indexed", path=str(folder), **stats)

    @app.get("/api/sources", response_model=SourcesResponse)
    def sources() -> SourcesResponse:
        return SourcesResponse(
            sources=[SourceResponse(**chunk.__dict__) for chunk in index.sources()]
        )

    @app.post("/api/chat", response_model=ChatResponse)
    def chat_endpoint(request: ChatRequest) -> ChatResponse:
        top_k = request.top_k or app_config.index.top_k
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

    return app


def _model_info(path: Path, selected_path: Path) -> ModelInfo:
    runtime = LocalChatEngine(path).runtime_status()
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
