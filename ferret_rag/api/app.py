from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ferret_rag.core.config import AppConfig
from ferret_rag.index.store import LocalIndex
from ferret_rag.llm.engine import LocalChatEngine


class IndexRequest(BaseModel):
    path: str


class ChatRequest(BaseModel):
    message: str
    top_k: int | None = None


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

    @app.get("/favicon.ico")
    def favicon() -> FileResponse:
        return FileResponse(icons_dir / "icon_round.png")

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "model_exists": app_config.model.path.exists(),
            "chunks": len(index.sources()),
        }

    @app.post("/api/index")
    def index_folder(request: IndexRequest) -> dict[str, object]:
        folder = Path(request.path).expanduser()
        try:
            stats = index.index_folder(folder)
        except (FileNotFoundError, NotADirectoryError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"status": "indexed", "path": str(folder), **stats}

    @app.get("/api/sources")
    def sources() -> dict[str, object]:
        return {"sources": [chunk.__dict__ for chunk in index.sources()]}

    @app.post("/api/chat")
    def chat_endpoint(request: ChatRequest) -> dict[str, object]:
        top_k = request.top_k or app_config.index.top_k
        results = index.search(request.message, top_k=top_k)
        return {
            "answer": chat.answer(request.message, results),
            "sources": [
                {
                    "file_path": result.chunk.file_path,
                    "chunk_id": result.chunk.chunk_id,
                    "text": result.chunk.text,
                    "score": result.score,
                }
                for result in results
            ],
        }

    return app
