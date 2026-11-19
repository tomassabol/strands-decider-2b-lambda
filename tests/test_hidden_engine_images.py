from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from PIL import Image

from conftest import FakeBackend, FakeImageBackend
from strands_decider.hidden_engine import HiddenStateEngine
from strands_decider.images import IMAGE_PAD, smart_resize, state_segments, to_grid

QUESTIONS = {
    "paid": {"type": "noul", "instructions": "Did the payment succeed?"},
    "next": {
        "type": "choice",
        "instructions": "Which control next?",
        "criteria": {"retry": "the retry button", "cancel": "the cancel link"},
    },
}


def _png(width: int, height: int) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (40, 120, 200)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture
def engine(assets):
    def make(backend=None, **config) -> HiddenStateEngine:
        return HiddenStateEngine.from_assets(
            str(assets), backend or FakeImageBackend(), **{"max_length": 4096, **config}
        )

    return make


def _request(
    n_images: int = 1, state: str = "The checkout page after the user tapped Pay.", size=(320, 320)
):
    return {
        "state": state,
        "questions": QUESTIONS,
        "images": [_png(*size) for _ in range(n_images)],
    }


@pytest.mark.parametrize(
    ("height", "width", "grid"),
    [(448, 448, (448, 448)), (64, 64, (256, 256)), (299, 448, (288, 448)), (149, 224, (224, 320))],
)
def test_smart_resize_fits_the_grid(height, width, grid):
    """Sizes from transformers' Qwen2-VL smart_resize."""
    assert smart_resize(height, width) == grid


def test_smart_resize_keeps_sides_on_the_grid_and_the_area_in_bounds():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        h, w = int(rng.integers(8, 5000)), int(rng.integers(8, 5000))
        if max(h, w) / min(h, w) > 200:
            continue
        gh, gw = smart_resize(h, w)
        assert gh % 32 == 0 and gw % 32 == 0
        assert 65536 <= gh * gw <= 16777216


def test_smart_resize_refuses_an_aspect_ratio_over_200_as_the_processor_does():
    assert smart_resize(1, 200) == (32, 3648)  # exactly 200 is allowed, as transformers allows it
    with pytest.raises(
        ValueError, match=r"^absolute aspect ratio must be smaller than 200, got 224\.0$"
    ):
        smart_resize(2, 448)


def test_to_grid_resizes_and_counts_tokens():
    rgb, w, h, tokens = to_grid(Image.new("RGB", (448, 299)))
    assert (w, h, tokens, len(rgb)) == (448, 288, 14 * 9, 448 * 288 * 3)


def test_state_segments_put_each_image_between_two_segments():
    segments = state_segments("The page.", 2)
    assert len(segments) == 3 and segments[-1].endswith("The page.\n</state>\n")


def test_state_segments_refuse_a_state_with_an_image_pad_of_its_own():
    with pytest.raises(ValueError, match=r"^the state holds 1 image placeholder\(s\) of its own$"):
        state_segments(f"see {IMAGE_PAD} here", 1)


def test_four_images_are_answered_and_five_refused(engine):
    backend = FakeImageBackend()
    response = engine(backend).evaluate(_request(4))
    assert set(response.answers) == {"paid", "next"}
    assert (
        backend.images[0]["sizes"] == [(320, 320)] * 4 and len(backend.images[0]["segments"]) == 5
    )
    with pytest.raises(ValueError, match="5 images; this server takes at most 4"):
        engine(backend).evaluate(_request(5))
    assert len(backend.images) == 1


def test_images_are_sized_with_the_engines_settings(engine):
    backend = FakeImageBackend()
    engine(backend, image_long_side=224).evaluate(_request(1, size=(600, 400)))
    # 600 x 400 is 224 x 149 at a long side of 224, which the grid's least area raises to 320 x 224.
    assert backend.images[0]["sizes"] == [(320, 224)]
    with pytest.raises(ValueError, match="at most 10,000 pixels"):
        engine(backend, max_image_pixels=10_000).evaluate(_request(1, size=(101, 100)))
    assert len(backend.images) == 1


def test_the_request_limit_counts_the_images(engine):
    backend = FakeImageBackend()
    usage = engine(backend).evaluate(_request(1)).usage.input_tokens
    image_tokens = 10 * 10 + 2  # 320 px square: 10 x 10 on the grid, with its start and end
    engine(backend, max_request_tokens=usage).evaluate(_request(1))
    with pytest.raises(ValueError, match=f"request of {usage} tokens .* the limit of {usage - 1}"):
        engine(backend, max_request_tokens=usage - 1).evaluate(_request(1))
    assert len(backend.images) == 2
    text_only = engine(FakeBackend()).evaluate({**_request(0)}).usage.input_tokens
    assert usage - text_only == image_tokens


def test_a_prompt_over_the_window_is_refused_with_its_parts(engine):
    request = {**_request(1), "questions": {"paid": QUESTIONS["paid"]}}
    exact = engine().evaluate(request).usage.input_tokens
    engine(max_length=exact).evaluate(request)
    with pytest.raises(
        ValueError,
        match=rf"^prompt of {exact} tokens \(.* state \+ 100 image \+ .* question\) exceeds the context window of {exact - 1} tokens$",
    ):
        engine(max_length=exact - 1).evaluate(request)


@pytest.mark.parametrize(
    ("request_", "message"),
    [
        (_request(1, size=(2000, 8)), "absolute aspect ratio must be smaller than 200"),
        (_request(1, state=f"see {IMAGE_PAD} here"), "image placeholder\\(s\\) of its own"),
    ],
    ids=["aspect ratio", "image pad in the state"],
)
def test_requests_vision_refuses_are_refused_before_the_backend(engine, request_, message):
    backend = FakeImageBackend()
    with pytest.raises(ValueError, match=message):
        engine(backend).evaluate(request_)
    assert not backend.images


def test_images_need_a_backend_that_reads_them(engine):
    with pytest.raises(ValueError, match="no vision tower"):
        engine(FakeBackend()).evaluate(_request(1))


def test_a_backend_that_sizes_images_otherwise_is_named(engine):
    with pytest.raises(RuntimeError, match=r"backend made \[103\] image tokens; expected \[102\]"):
        engine(FakeImageBackend(extra_tokens=1)).evaluate(_request(1))
