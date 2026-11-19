"""function/handler.py over a stand-in backend: llama.cpp and the image are the deploy's to
test. Everything else, the routes, the status codes and bodies, the settings, the load and the
model prefetch, is the handler's own."""

from __future__ import annotations

import base64
import importlib
import io
import itertools
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import ClassVar

import pytest
from PIL import Image
from pydantic import ValidationError

from conftest import FakeBackend, FakeImageBackend, write_assets
from strands_decider.hidden_engine import HiddenStateEngine

FUNCTION = Path(__file__).resolve().parents[1] / "function"
REQUEST = {
    "state": "My renewal failed and I leave on Friday!",
    "questions": {
        "urgent": {"type": "noul", "instructions": "Is this urgent?"},
        "team": {
            "type": "choice",
            "instructions": "Which team?",
            "criteria": {"circulation": "renewals", "billing": "fines"},
        },
    },
}


class LlamaBackend(FakeBackend):
    """Stands in for backend.LlamaBackend, recording every backend the handler makes."""

    made: ClassVar[list[FakeBackend]] = []

    def __init__(self, lib_path: str, model_path: str, threads: int):
        super().__init__()
        self.args = (lib_path, model_path, threads)
        LlamaBackend.made.append(self)


class LlamaVisionBackend(FakeImageBackend):
    def __init__(self, lib_path: str, model_path: str, mmproj_path: str, threads: int):
        super().__init__()
        self.args = (lib_path, model_path, threads)
        self.mmproj_path = mmproj_path
        LlamaBackend.made.append(self)


@pytest.fixture
def model_dir(tmp_path):
    """What the image holds in /opt/model: the GGUF (any bytes here) and the assets."""
    write_assets(tmp_path / "assets")
    (tmp_path / "decider.gguf").write_bytes(os.urandom(1000))
    return tmp_path


@pytest.fixture
def load(model_dir, monkeypatch):
    """Import the handler afresh, as a new execution environment does, with `env` set."""

    def _load(**env: str):
        monkeypatch.setenv("DECIDER_MODEL_DIR", str(model_dir))
        monkeypatch.delenv("AWS_LAMBDA_INITIALIZATION_TYPE", raising=False)
        monkeypatch.delenv("DECIDER_MAX_REQUEST_TOKENS", raising=False)
        monkeypatch.delenv("DECIDER_IMAGE_LONG_SIDE", raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        monkeypatch.setitem(
            sys.modules,
            "backend",
            types.SimpleNamespace(LlamaBackend=LlamaBackend, LlamaVisionBackend=LlamaVisionBackend),
        )
        monkeypatch.syspath_prepend(str(FUNCTION))
        monkeypatch.delitem(sys.modules, "handler", raising=False)
        LlamaBackend.made.clear()
        return importlib.import_module("handler")

    yield _load
    sys.modules.pop("handler", None)


def url_event(method: str, path: str, body=None, *, raw: str | None = None, b64: bool = False):
    """A Function URL event, as Lambda passes it."""
    text = raw if raw is not None else (json.dumps(body) if body is not None else None)
    if b64 and text is not None:
        text = base64.b64encode(text.encode()).decode()
    return {
        "rawPath": path,
        "requestContext": {"http": {"method": method}},
        "body": text,
        "isBase64Encoded": b64,
    }


def call(handler, event):
    response = handler.handler(event, None)
    assert response["headers"] == {"content-type": "application/json"}
    return response["statusCode"], json.loads(response["body"])


def test_loads_the_model_from_the_image(load, model_dir):
    handler = load()
    backend = handler._engine.result().backend
    lib_path, model_path, threads = backend.args
    assert lib_path == str(FUNCTION / "libdecider.so")
    assert model_path == str(model_dir / "decider.gguf")
    assert threads == os.cpu_count()
    assert len(backend.calls) == 1  # the warm-up, before any request


def test_health_reports_the_model_and_settings(load):
    assert call(load(), url_event("GET", "/health")) == (
        200,
        {
            "status": "ok",
            "model": "tiny-q8_0",
            "max_length": 512,
            "temperature": 1.0,
            "max_request_tokens": None,
            "vision": False,
            "image_long_side": 448,
        },
    )
    handler = load(DECIDER_MAX_REQUEST_TOKENS="24576")
    _, body = call(handler, url_event("GET", "/health"))
    assert body["max_request_tokens"] == 24576


@pytest.mark.parametrize("b64", [False, True], ids=["text", "base64"])
def test_answers_as_the_engine_does(load, model_dir, b64):
    handler = load()
    status, body = call(handler, url_event("POST", "/v1/systemone", REQUEST, b64=b64))
    engine = HiddenStateEngine.from_assets(str(model_dir / "assets"), FakeBackend())
    expected = engine.evaluate(REQUEST).model_dump()
    assert status == 200
    assert body.pop("latency_ms") > 0
    assert body == expected


def test_direct_invocation_returns_the_response(load):
    handler = load()
    response = handler.handler(REQUEST, None)
    assert set(response) == {"model", "answers", "usage", "latency_ms"}
    assert set(response["answers"]) == {"urgent", "team"}
    with pytest.raises(ValidationError):
        handler.handler({"state": "x", "questions": {}}, None)


def test_unknown_routes(load):
    handler = load()
    assert call(handler, url_event("GET", "/nope")) == (404, {"detail": "Not Found"})
    assert call(handler, url_event("GET", "/v1/systemone")) == (
        405,
        {"detail": "Method Not Allowed"},
    )
    assert call(handler, url_event("POST", "/health")) == (405, {"detail": "Method Not Allowed"})


def test_invalid_requests_get_fastapis_422(load):
    handler = load()
    status, body = call(
        handler, url_event("POST", "/v1/systemone", {"state": "x", "questions": {}})
    )
    assert status == 422
    assert [(e["type"], e["loc"]) for e in body["detail"]] == [("too_short", ["body", "questions"])]
    status, body = call(handler, url_event("POST", "/v1/systemone", raw="{nope"))
    assert status == 422
    assert [(e["type"], e["loc"]) for e in body["detail"]] == [("json_invalid", ["body"])]


def test_requests_the_engine_refuses_get_422_before_the_forward(load):
    handler = load(DECIDER_MAX_REQUEST_TOKENS="10")
    backend = handler._engine.result().backend
    backend.calls.clear()
    status, body = call(handler, url_event("POST", "/v1/systemone", REQUEST))
    assert status == 422 and "exceeds the limit of 10 tokens per request" in body["detail"]
    status, body = call(
        handler, url_event("POST", "/v1/systemone", {**REQUEST, "images": ["aGk="]})
    )
    assert status == 422 and "images are not supported" in body["detail"]
    assert not backend.calls


def test_on_demand_init_loads_in_the_background(load):
    handler = load(AWS_LAMBDA_INITIALIZATION_TYPE="on-demand")
    status, body = call(handler, url_event("POST", "/v1/systemone", REQUEST))
    assert status == 200 and set(body["answers"]) == {"urgent", "team"}


def test_a_failed_background_load_exits(load, model_dir, monkeypatch):
    (model_dir / "assets" / "head_config.json").unlink()
    exits: list[int] = []
    monkeypatch.setattr(os, "_exit", exits.append)
    handler = load(AWS_LAMBDA_INITIALIZATION_TYPE="on-demand")
    with pytest.raises(FileNotFoundError):
        handler._engine.result()
    for _ in range(100):  # the done-callback runs just after result() returns
        if exits:
            break
        time.sleep(0.01)
    assert exits == [1]


def test_a_negative_request_limit_fails_the_init_before_the_load(load):
    with pytest.raises(ValueError, match="must be 0 \\(no limit\\) or more, not -5"):
        load(DECIDER_MAX_REQUEST_TOKENS="-5")
    assert not LlamaBackend.made  # refused before the model loaded


def test_a_failed_load_at_import_fails_the_init(load, model_dir):
    (model_dir / "assets" / "head_config.json").unlink()
    with pytest.raises(FileNotFoundError):
        load()


def test_prefetch_reads_every_byte_once(load, model_dir, monkeypatch):
    handler = load()
    read: list[tuple[int, int]] = []
    pread = os.pread

    def recording_pread(fd, n, offset):
        data = pread(fd, n, offset)
        read.append((offset, len(data)))
        return data

    monkeypatch.setattr(os, "pread", recording_pread)
    handler._prefetch(str(model_dir / "decider.gguf"), chunk=64, readers=4)
    covered = sorted(read)
    assert covered[0][0] == 0
    assert all(a + n == b for (a, n), (b, _) in itertools.pairwise(covered))
    assert sum(n for _, n in covered) == 1000


def _png(width: int, height: int) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (40, 120, 200)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_an_image_with_a_vision_tower_answers_over_images(load, model_dir):
    (model_dir / "vision.gguf").write_bytes(os.urandom(500))
    handler = load()
    backend = handler._engine.result().backend
    assert isinstance(backend, LlamaVisionBackend)
    assert backend.mmproj_path == str(model_dir / "vision.gguf")
    # The image warm-up: 64 px, raised to the processor grid's least area, 256 x 256.
    assert [i["sizes"] for i in backend.images] == [[(256, 256)]]
    assert call(handler, url_event("GET", "/health"))[1]["vision"] is True
    status, body = call(
        handler, url_event("POST", "/v1/systemone", {**REQUEST, "images": [_png(600, 400)]})
    )
    assert status == 200 and set(body["answers"]) == {"urgent", "team"}
    assert backend.images[-1]["sizes"] == [(448, 288)]  # 448 x 299, on the processor's grid


def test_an_image_without_a_vision_tower_refuses_images(load):
    handler = load()
    assert not isinstance(handler._engine.result().backend, LlamaVisionBackend)
    status, body = call(
        handler, url_event("POST", "/v1/systemone", {**REQUEST, "images": [_png(64, 64)]})
    )
    assert status == 422 and "images are not supported" in body["detail"]


def test_a_fault_in_the_function_is_a_500_with_its_traceback_logged(load, monkeypatch, caplog):
    handler = load()

    def broken_forward(self, state, questions, rows):
        raise RuntimeError("dh_forward failed with 4")

    monkeypatch.setattr(FakeBackend, "forward", broken_forward)
    with caplog.at_level("ERROR", logger="decider"):
        status, body = call(handler, url_event("POST", "/v1/systemone", REQUEST))
    assert (status, body) == (500, {"detail": "Internal Server Error"})
    assert "dh_forward failed with 4" in caplog.text  # the traceback, in the function's log
    with pytest.raises(RuntimeError, match="dh_forward failed"):
        handler.handler(REQUEST, None)  # a direct invocation fails, as Lambda reports it


def test_the_image_size_setting_reaches_the_engine(load, model_dir):
    (model_dir / "vision.gguf").write_bytes(os.urandom(500))
    handler = load(DECIDER_IMAGE_LONG_SIDE="768")
    assert call(handler, url_event("GET", "/health"))[1]["image_long_side"] == 768
    status, _ = call(
        handler, url_event("POST", "/v1/systemone", {**REQUEST, "images": [_png(1200, 800)]})
    )
    assert status == 200
    assert handler._engine.result().backend.images[-1]["sizes"] == [
        (768, 512)
    ]  # 768 x 512, already on the grid


def test_an_image_size_under_the_grid_fails_the_init_before_the_load(load):
    with pytest.raises(ValueError, match="DECIDER_IMAGE_LONG_SIDE must be 32 or more, not 16"):
        load(DECIDER_IMAGE_LONG_SIDE="16")
    assert not LlamaBackend.made
