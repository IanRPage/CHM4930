"""
HTTP server around `Predictor.infer`.

Every path except `/health` needs the `X-API-Key` header and counts toward the rate
limit. The model loads in a background thread, so `/health` reports `"loading"` until
it's ready.
"""

import asyncio
import logging
import secrets
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException

from mufu_server.config import Settings
from mufu_server.infer import InvalidSmiles, load_predictor
from mufu_server.limits import BodyLimit, SlidingWindowLimiter, retry_after

log = logging.getLogger(__name__)

OPEN_PATHS = ("/health",)


class PredictRequest(BaseModel):
    smiles: str


def error(status: int, message: str, **headers: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status, headers=headers)


def create_app(settings: Settings, load=load_predictor, on_load_error=None):
    limiter = SlidingWindowLimiter(settings.rate_limit_per_minute)
    # one inference at a time; Predictor serializes them anyway
    busy = threading.Lock()

    def load_in_background(app: FastAPI) -> None:
        try:
            app.state.predictor = load(settings)
        except Exception as exc:
            log.exception("failed to load the model")
            app.state.load_error = exc
            if on_load_error:
                on_load_error(exc)
        else:
            log.info("model %s ready", app.state.predictor.model_version)
        finally:
            app.state.loaded.set()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        threading.Thread(
            target=load_in_background, args=(app,), name="load-model", daemon=True
        ).start()
        yield

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.predictor = None
    app.state.load_error = None
    app.state.loaded = threading.Event()

    app.add_middleware(BodyLimit, max_bytes=settings.max_body_bytes)

    @app.middleware("http")
    async def gate(request: Request, call_next):
        if request.url.path in OPEN_PATHS:
            return await call_next(request)
        key = request.headers.get("x-api-key", "")
        if not secrets.compare_digest(key.encode(), settings.api_key.encode()):
            return error(401, "unauthorized")
        if wait := limiter.acquire():
            return error(429, "rate_limited", **{"Retry-After": retry_after(wait)})
        return await call_next(request)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        return error(exc.status_code, str(exc.detail).lower().replace(" ", "_"))

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        return error(422, "invalid_request")

    @app.exception_handler(Exception)
    async def internal_error(request: Request, exc: Exception):
        log.error("unhandled error", exc_info=exc)
        return error(500, "internal_error")

    @app.get("/health")
    def health():
        predictor = app.state.predictor
        return {
            "status": "ready" if predictor else "loading",
            "model_version": predictor and predictor.model_version,
        }

    @app.post("/predict")
    async def predict(body: PredictRequest):
        predictor = app.state.predictor
        if predictor is None:
            return error(503, "loading")
        smiles = body.smiles
        if (
            not smiles
            or len(smiles) > settings.max_smiles_length
            or any(c.isspace() for c in smiles)
        ):
            return error(422, "invalid_smiles")
        if not busy.acquire(blocking=False):
            return error(503, "busy")

        # released when inference finishes, even after a timeout
        def run():
            try:
                return predictor.infer(smiles)
            finally:
                busy.release()

        start = time.perf_counter()
        job = asyncio.get_running_loop().run_in_executor(None, run)
        try:
            result = await asyncio.wait_for(
                asyncio.shield(job), settings.request_timeout_s
            )
        except TimeoutError:
            log.warning("inference timed out after %.0fs", settings.request_timeout_s)
            return error(503, "timeout")
        except InvalidSmiles:
            return error(422, "invalid_smiles")
        except Exception:
            log.exception("inference failed")
            return error(500, "internal_error")
        log.info("predict ok in %.0f ms", (time.perf_counter() - start) * 1000)
        return result

    return app
