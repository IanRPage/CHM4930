import logging
import os
import socket
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mufu_server.app import create_app
from mufu_server.config import Settings
from mufu_server.infer import InvalidSmiles
from mufu_server.keys import add_key, hash_key

KEY = "test-key"
SRC = Path(__file__).resolve().parents[2] / "src"
AUTH = {"X-API-Key": KEY}


class StubPredictor:
    model_version = "stub@test"

    def __init__(self, infer=None):
        self._infer = infer or (lambda smiles: {"smiles": smiles})

    def infer(self, smiles):
        return self._infer(smiles)


@contextmanager
def serve(predictor=None, *, load=None, wait=True, on_load_error=None, **settings):
    app = create_app(
        Settings(checkpoint_dir=Path("unused"), **{"api_key": KEY, **settings}),
        load=load or (lambda _: predictor or StubPredictor()),
        on_load_error=on_load_error,
    )
    with TestClient(app) as client:
        if wait:
            assert app.state.loaded.wait(5)
        yield client


def test_health_reports_loading_then_ready():
    release = threading.Event()

    def slow_load(_):
        release.wait(5)
        return StubPredictor()

    with serve(load=slow_load, wait=False) as client:
        assert client.get("/health").json() == {
            "status": "loading",
            "model_version": None,
        }
        response = client.post("/predict", json={"smiles": "CCO"}, headers=AUTH)
        assert response.status_code == 503
        assert response.json() == {"error": "loading"}
        release.set()
        client.app.state.loaded.wait(5)
        assert client.get("/health").json() == {
            "status": "ready",
            "model_version": "stub@test",
        }


def test_predict_returns_the_predictor_response():
    with serve() as client:
        response = client.post(
            "/predict", json={"smiles": "CCO", "extra": 1}, headers=AUTH
        )
    assert response.status_code == 200
    assert response.json() == {"smiles": "CCO"}


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}, AUTH])
def test_key_is_checked_before_the_body(tmp_path, headers):
    keys_file = tmp_path / "api_keys"
    alex = add_key(keys_file, "alex")
    with serve(api_key="", api_keys_file=keys_file) as client:
        response = client.post("/predict", content=b"not json", headers=headers)
        allowed = client.post(
            "/predict", content=b"not json", headers={"X-API-Key": alex}
        )
    assert response.status_code == 401
    assert response.json() == {"error": "unauthorized"}
    assert allowed.status_code == 422


@pytest.mark.parametrize(
    ("text", "mode", "api_key", "match"),
    [
        ("", 0o600, "", "no keys"),
        ("", 0o600, KEY, "no keys"),
        (f"alex {hash_key('other')}\n", 0o620, KEY, "writable"),
        (f"default {hash_key('other')}\n", 0o600, KEY, "'default'"),
        (f"alex {hash_key(KEY)}\n", 0o600, KEY, "also in"),
    ],
)
def test_bad_key_setup_fails_at_startup(tmp_path, text, mode, api_key, match):
    keys_file = tmp_path / "api_keys"
    keys_file.write_text(text)
    keys_file.chmod(mode)
    with pytest.raises(ValueError, match=match):
        create_app(Settings(Path("unused"), api_key=api_key, api_keys_file=keys_file))


@pytest.mark.parametrize(
    ("body", "error"),
    [
        ({"smiles": "C1CC"}, "invalid_smiles"),
        ({"smiles": ""}, "invalid_smiles"),
        ({"smiles": "CC O"}, "invalid_smiles"),
        ({"smiles": "C" * 11}, "invalid_smiles"),
        ({"smiles": 5}, "invalid_request"),
        ({}, "invalid_request"),
    ],
)
def test_bad_input_is_422(body, error):
    def infer(smiles):
        raise InvalidSmiles(smiles)

    with serve(StubPredictor(infer), max_smiles_length=10) as client:
        response = client.post("/predict", json=body, headers=AUTH)
    assert response.status_code == 422
    assert response.json() == {"error": error}


def chunked(body: bytes):
    # a generator body makes httpx send Transfer-Encoding: chunked, no Content-Length
    yield from (body[i : i + 16] for i in range(0, len(body), 16))


def test_body_limit_with_and_without_content_length():
    big = b'{"smiles": "' + b"C" * 100 + b'"}'
    headers = {**AUTH, "Content-Type": "application/json"}
    with serve(max_body_bytes=64) as client:
        sized = client.post("/predict", content=big, headers=headers)
        streamed = client.post("/predict", content=chunked(big), headers=headers)
        small = client.post(
            "/predict", content=chunked(b'{"smiles": "CCO"}'), headers=headers
        )
    for response in (sized, streamed):
        assert response.status_code == 413
        assert response.json() == {"error": "body_too_large"}
    assert "content-length" not in streamed.request.headers
    assert small.status_code == 200 and small.json() == {"smiles": "CCO"}


def test_each_key_has_its_own_rate_limit(tmp_path, caplog):
    keys_file = tmp_path / "api_keys"
    alex = add_key(keys_file, "alex")
    caplog.set_level(logging.INFO)
    with serve(rate_limit_per_minute=2, api_keys_file=keys_file) as client:
        statuses = [
            client.post("/predict", json={"smiles": "CCO"}, headers=AUTH)
            for _ in range(3)
        ]
        other = client.post(
            "/predict", json={"smiles": "CCO"}, headers={"X-API-Key": alex}
        )
        assert client.get("/health").status_code == 200
    assert [r.status_code for r in statuses] == [200, 200, 429]
    assert statuses[2].json() == {"error": "rate_limited"}
    assert 1 <= int(statuses[2].headers["Retry-After"]) <= 60
    assert other.status_code == 200
    assert "predict ok for alex" in caplog.text
    assert "predict ok for default" in caplog.text


def test_busy_and_timeout_are_503():
    started, release = threading.Event(), threading.Event()

    def blocking(smiles):
        started.set()
        release.wait(5)
        return {"smiles": smiles}

    with serve(StubPredictor(blocking), request_timeout_s=0.2) as client:
        with ThreadPoolExecutor(1) as pool:
            first = pool.submit(
                client.post, "/predict", json={"smiles": "CCO"}, headers=AUTH
            )
            assert started.wait(5)
            busy = client.post("/predict", json={"smiles": "CCN"}, headers=AUTH)
            timed_out = first.result(5)
        release.set()

    assert busy.status_code == 503 and busy.json() == {"error": "busy"}
    assert timed_out.status_code == 503 and timed_out.json() == {"error": "timeout"}


def test_predictor_failure_is_500_and_key_is_never_logged(caplog):
    def broken(smiles):
        raise RuntimeError("boom")

    caplog.set_level(logging.DEBUG)
    with serve(StubPredictor(broken)) as client:
        response = client.post("/predict", json={"smiles": "CCO"}, headers=AUTH)
        missing = client.get("/nope", headers=AUTH)
    assert response.status_code == 500
    assert response.json() == {"error": "internal_error"}
    assert missing.status_code == 404 and missing.json() == {"error": "not_found"}
    assert KEY not in caplog.text


def test_main_exits_1_when_the_model_fails_to_load(tmp_path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = {
        **os.environ,
        "PYTHONPATH": str(SRC),
        "MUFU_CHECKPOINT_DIR": str(tmp_path),
        "MUFU_API_KEY": KEY,
        "MUFU_PORT": str(port),
    }
    result = subprocess.run(
        [sys.executable, "-m", "mufu_server"],
        env=env,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 1


def test_load_failure_is_reported():
    errors = []

    def failing_load(_):
        raise FileNotFoundError("missing checkpoints")

    with serve(load=failing_load, on_load_error=errors.append) as client:
        assert client.get("/health").json()["status"] == "loading"
    assert len(errors) == 1 and isinstance(errors[0], FileNotFoundError)
