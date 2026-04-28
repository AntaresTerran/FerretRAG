from __future__ import annotations

import argparse
import webbrowser

import uvicorn

from ferret_rag.api.app import create_app
from ferret_rag.core.config import AppConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FerretRAG local server.")
    parser.add_argument("--host", default=None, help="Host to bind. Defaults to config value.")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind. Defaults to config value.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the browser on startup.",
    )
    args = parser.parse_args()

    config = AppConfig.load()
    host = args.host or config.server.host
    port = args.port or config.server.port
    app = create_app(config)

    if config.server.open_browser and not args.no_browser:
        webbrowser.open(f"http://{host}:{port}")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
