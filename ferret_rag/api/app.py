from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ferret_rag.core.config import AppConfig
from ferret_rag.index.store import LocalIndex
from ferret_rag.llm.engine import LocalChatEngine
from ferret_rag.llm.gguf import read_gguf_architecture


class IndexRequest(BaseModel):
    path: str


class ChatRequest(BaseModel):
    message: str
    top_k: int | None = None


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


def create_app(config: AppConfig | None = None) -> FastAPI:
    app_config = config or AppConfig.load()
    index = LocalIndex(
        data_dir=app_config.index.data_dir,
        chunk_words=app_config.index.chunk_words,
        overlap=app_config.index.chunk_overlap,
    )
    chat = LocalChatEngine(app_config.model.path)
    ui_dir = Path(__file__).resolve().parents[1] / "ui"
    icons_dir = Path(__file__).resolve().parents[2] / "icons"

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
        return HealthResponse(
            status="ok",
            model_exists=app_config.model.path.exists(),
            model_path=str(app_config.model.path),
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
                path=str(app_config.model.path),
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
        models_dir = Path(__file__).resolve().parents[2] / "models"
        models = [
            _model_info(path, app_config.model.path)
            for path in sorted(models_dir.glob("*.gguf"))
        ]
        return ModelsResponse(models=models)

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
    architecture = read_gguf_architecture(path)
    is_compatible = architecture not in {None, "gemma4"}
    return ModelInfo(
        name=path.name,
        path=str(path),
        size_bytes=path.stat().st_size,
        architecture=architecture,
        is_selected=path.resolve() == selected_path.resolve(),
        is_compatible=is_compatible,
        status="compatible" if is_compatible else "unsupported runtime",
    )
