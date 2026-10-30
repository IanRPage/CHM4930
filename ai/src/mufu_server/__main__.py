"""
From ai/: PYTHONPATH=src python -m mufu_server

Reads `MUFU_*` environment variables. Exits 1 if the settings or key file are invalid
or the model fails to load.
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
    server = None

    # stop uvicorn directly: on SIGTERM it reraises signal and exits 143. systemd treats
    # as clean stop and won't restart
    def stop(exc: BaseException) -> None:
        server.should_exit = True

    app = create_app(settings, on_load_error=stop)
    server = uvicorn.Server(
        uvicorn.Config(app, host=settings.host, port=settings.port, proxy_headers=False)
    )
    server.run()
    return 1 if app.state.load_error else 0


if __name__ == "__main__":
    sys.exit(main())
