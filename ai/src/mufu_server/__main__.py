"""
From ai/: PYTHONPATH=src python -m mufu_server

Reads `MUFU_*` environment variables. Exits 1 if the model fails to load.
"""

import logging
import sys

import uvicorn

from mufu_server.app import create_app
from mufu_server.config import Settings


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    settings = Settings.from_env()
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port, proxy_headers=False)
    return 1 if app.state.load_error else 0


if __name__ == "__main__":
    sys.exit(main())
