"""A tiny model: a word-level tokenizer, a random pointer head, and a backend that draws hidden
states from the request's shape, so the same request gets the same ones."""

from __future__ import annotations

import numpy as np
import pytest
from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers

from strands_decider.hidden_engine import save_assets
from strands_decider.images import IMAGE_PAD, VISION_END, VISION_START

HIDDEN, POINTER = 8, 4


def write_assets(path, **settings):
    tok = Tokenizer(models.WordLevel({"<unk>": 0}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens(
        [AddedToken(t, normalized=False) for t in (VISION_START, IMAGE_PAD, VISION_END)]
    )
    rng = np.random.default_rng(0)
    head = {
        "norm.weight": rng.normal(size=HIDDEN),
        "norm.bias": rng.normal(size=HIDDEN),
        "q.weight": rng.normal(size=(POINTER, HIDDEN)),
        "q.bias": rng.normal(size=POINTER),
        "k.weight": rng.normal(size=(POINTER, HIDDEN)),
        "k.bias": rng.normal(size=POINTER),
    }
    defaults = {
        "model_name": "tiny-q8_0",
        "pointer_dim": POINTER,
        "eps": 1e-5,
        "max_length": 512,
        "temperature": 1.0,
        "temperature_by_kind": {},
        "ordinal_smoothing": 0.0,
    }
    save_assets(str(path), tok, head, **{**defaults, **settings})
    return path


@pytest.fixture
def assets(tmp_path):
    return write_assets(tmp_path / "assets")


class FakeBackend:
    def __init__(self):
        self.calls: list[tuple[list[int], list[list[int]], list[list[int]]]] = []

    def forward(self, state, questions, rows):
        self.calls.append((state, questions, rows))
        rng = np.random.default_rng(len(state))
        return [
            rng.normal(size=(len(r), HIDDEN)) + len(q) for q, r in zip(questions, rows, strict=True)
        ]


class FakeImageBackend(FakeBackend):
    """Sizes each image as the processor's grid does, plus `extra_tokens`."""

    def __init__(self, extra_tokens: int = 0):
        super().__init__()
        self.extra_tokens = extra_tokens
        self.images: list[dict] = []

    def forward_images(self, segments, images, questions, rows):
        self.images.append({"segments": segments, "sizes": [(w, h) for _, w, h in images]})
        tokens = [(w // 32) * (h // 32) + 2 + self.extra_tokens for _, w, h in images]
        return self.forward([t for s in segments for t in s], questions, rows), tokens
