"""A System One engine without torch: a backend runs the torso, numpy the pointer head.

The engine renders the prompts, fits them to the window and reads the answers with the
functions in prompting.py. It asks the backend (llama.cpp, in backend.py) only for the hidden
states the head reads: at each option's last token, and at the question's last token,
`<answer>`. With a backend that also reads images (`ImageBackend`), it answers over images;
images.py decodes and sizes them.

`usage.input_tokens` counts the state once, then every question.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from tokenizers import Tokenizer

from .prompting import (
    RenderedQuestion,
    option_token_index,
    render_question,
    render_state,
    to_answer,
)
from .schema import Answer, SystemOneRequest, SystemOneResponse, Usage


class Backend(Protocol):
    def forward(
        self, state: list[int], questions: list[list[int]], rows: list[list[int]]
    ) -> list[np.ndarray]:
        """Run each question as a continuation of the state, seeing no other question.
        For question i, return its final (post-norm) hidden states at the positions in
        rows[i], counted from the question's first token, in the order of rows[i] (each
        option's last token, then `<answer>`): an array [len(rows[i]), hidden]."""
        ...


class ImageBackend(Backend, Protocol):
    def forward_images(
        self,
        segments: list[list[int]],
        images: list[tuple[bytes, int, int]],
        questions: list[list[int]],
        rows: list[list[int]],
    ) -> tuple[list[np.ndarray], list[int]]:
        """forward over a state that holds images. The state is `segments`, its text as
        token ids around the images, with image i between segment i and segment i + 1;
        each image is (RGB bytes, width, height), already at the processor's grid size
        (images.to_grid). Returns forward's hidden states, and each image's token count,
        its `<|vision_start|>` and `<|vision_end|>` included."""
        ...


class NumpyPointerHead:
    """The checkpoint's pointer head: a LayerNorm, then a query from `<answer>`'s hidden state
    against a key from each option's, scaled by 1/sqrt(pointer_dim)."""

    def __init__(self, weights: Mapping[str, np.ndarray], pointer_dim: int, eps: float):
        self.w = {k: np.asarray(v, dtype=np.float64) for k, v in weights.items()}
        self.scale = pointer_dim**-0.5
        self.eps = eps

    def _norm(self, x: np.ndarray) -> np.ndarray:
        mu = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)
        return (x - mu) / np.sqrt(var + self.eps) * self.w["norm.weight"] + self.w["norm.bias"]

    def logits(self, decide: np.ndarray, options: np.ndarray) -> np.ndarray:
        """decide [hidden], options [K, hidden] -> logits [K]."""
        q = self._norm(decide) @ self.w["q.weight"].T + self.w["q.bias"]
        k = self._norm(options) @ self.w["k.weight"].T + self.w["k.bias"]
        out: np.ndarray = (k @ q) * self.scale
        return out


@dataclass
class HiddenEngineConfig:
    max_length: int
    model_name: str
    temperature: float = 1.0
    temperature_by_kind: dict[str, float] = field(default_factory=dict)
    ordinal_smoothing: float = 0.0
    # The most tokens one request may run through the backend, the state once plus every
    # question; a larger request is refused before the backend runs. None: no limit.
    max_request_tokens: int | None = None
    image_long_side: int = 448
    max_images: int = 4
    max_image_pixels: int = 4096 * 4096


class HiddenStateEngine:
    def __init__(
        self,
        tokenizer: Tokenizer,
        head: NumpyPointerHead,
        config: HiddenEngineConfig,
        backend: Backend,
    ):
        tokenizer.no_padding()
        tokenizer.no_truncation()
        self.tok = tokenizer
        self.head = head
        self.cfg = config
        self.backend = backend

    @property
    def vision(self) -> bool:
        return hasattr(self.backend, "forward_images")

    @classmethod
    def from_assets(cls, path: str, backend: Backend, **config: Any) -> HiddenStateEngine:
        """The engine from a save_assets directory. `config` sets HiddenEngineConfig
        fields, over the saved ones."""
        from safetensors.numpy import load_file

        with open(os.path.join(path, "head_config.json"), encoding="utf-8") as f:
            saved = json.load(f)
        weights = load_file(os.path.join(path, "head.safetensors"))
        head = NumpyPointerHead(weights, saved.pop("pointer_dim"), saved.pop("eps"))
        tok = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
        return cls(tok, head, HiddenEngineConfig(**{**saved, **config}), backend)

    def evaluate(self, request: SystemOneRequest | Mapping[str, Any]) -> SystemOneResponse:
        req = (
            request
            if isinstance(request, SystemOneRequest)
            else SystemOneRequest.model_validate(request)
        )
        names = list(req.questions)
        rendered = [render_question(req.questions[n]) for n in names]
        texts = [rq.text for rq in rendered]
        if not req.images:
            (s,), q, offs = self._tokenize([render_state(req.state)], [], texts)
            n_input = self._check_limit(len(s), q)
            rows = self._rows(rendered, q, offs)
            return self._read(names, rendered, self.backend.forward(s, q, rows), rows, n_input)

        if not self.vision:
            raise ValueError("this engine has no vision tower; images are not supported")
        if len(req.images) > self.cfg.max_images:
            raise ValueError(
                f"{len(req.images)} images; this server takes at most {self.cfg.max_images}"
            )
        from .images import decode_image, fit_image, state_segments, to_grid

        long_side, max_pixels = self.cfg.image_long_side, self.cfg.max_image_pixels
        pictures = [to_grid(fit_image(decode_image(b, max_pixels), long_side)) for b in req.images]
        image_tokens = [n + 2 for *_, n in pictures]  # with <|vision_start|> and <|vision_end|>
        segments, q, offs = self._tokenize(
            state_segments(req.state, len(pictures)), image_tokens, texts
        )
        n_input = self._check_limit(sum(len(x) for x in segments) + sum(image_tokens), q)
        rows = self._rows(rendered, q, offs)
        hidden, made = self.backend.forward_images(  # type: ignore[attr-defined]
            segments, [(rgb, w, h) for rgb, w, h, _ in pictures], q, rows
        )
        if list(made) != image_tokens:
            # The window and the token count were computed for a prompt it did not run.
            raise RuntimeError(f"backend made {list(made)} image tokens; expected {image_tokens}")
        return self._read(names, rendered, hidden, rows, n_input)

    def _tokenize(
        self, segments: list[str], image_tokens: list[int], questions: list[str]
    ) -> tuple[list[list[int]], list[list[int]], list[list[tuple[int, int]]]]:
        """The state's text segments (with image i between segment i and i + 1, of
        image_tokens[i] tokens) and the questions as token ids, with each question's character
        offsets. A prompt of the state and the longest question over the window is refused."""
        state = [self.tok.encode(t, add_special_tokens=i == 0).ids for i, t in enumerate(segments)]
        encodings = self.tok.encode_batch(questions, add_special_tokens=False)
        q = [e.ids for e in encodings]
        longest = max(len(x) for x in q)
        n_text, n_image = sum(len(x) for x in state), sum(image_tokens) - 2 * len(image_tokens)
        n_prompt = n_text + sum(image_tokens) + longest
        if n_prompt > self.cfg.max_length:
            parts = (
                f"{n_prompt - n_image - longest} state + {n_image} image + {longest} question"
                if image_tokens
                else f"{n_text} state + {longest} question"
            )
            raise ValueError(
                f"prompt of {n_prompt} tokens ({parts}) exceeds the context window "
                f"of {self.cfg.max_length} tokens"
            )
        return state, q, [list(e.offsets) for e in encodings]

    def _check_limit(self, n_state: int, q: list[list[int]]) -> int:
        """The request's tokens, the state once then every question, within the limit."""
        n_input = n_state + sum(len(x) for x in q)
        if self.cfg.max_request_tokens and n_input > self.cfg.max_request_tokens:
            raise ValueError(
                f"request of {n_input} tokens (the state's {n_state} once, then its questions) "
                f"exceeds the limit of {self.cfg.max_request_tokens} tokens per request"
            )
        return n_input

    @staticmethod
    def _rows(
        rendered: list[RenderedQuestion], q: list[list[int]], offs: list[list[tuple[int, int]]]
    ) -> list[list[int]]:
        """Each question's rows for the backend: its options' last tokens, then `<answer>`."""
        return [
            [*option_token_index(o, rq.option_spans), len(x) - 1]
            for rq, o, x in zip(rendered, offs, q, strict=True)
        ]

    def _read(
        self,
        names: list[str],
        rendered: list[RenderedQuestion],
        hidden: list[np.ndarray],
        rows: list[list[int]],
        n_input: int,
    ) -> SystemOneResponse:
        """The answers from the backend's hidden states, checked against what was asked."""
        if len(hidden) != len(names):
            raise ValueError(f"backend returned {len(hidden)} arrays for {len(names)} questions")
        answers: dict[str, Answer] = {}
        for name, rq, h, r in zip(names, rendered, hidden, rows, strict=True):
            h = np.asarray(h, dtype=np.float64)
            if h.ndim != 2 or h.shape[0] != len(r):
                raise ValueError(
                    f"backend returned shape {h.shape} for question {name!r}; expected "
                    f"({len(r)}, hidden): one row per option, then <answer>"
                )
            t = float(self.cfg.temperature_by_kind.get(rq.kind, self.cfg.temperature))
            logits = self.head.logits(h[-1], h[:-1]) / max(t, 1e-6)
            p = np.exp(logits - np.max(logits))
            answers[name] = to_answer(
                rq, (p / p.sum()).tolist(), ordinal_smoothing=self.cfg.ordinal_smoothing
            )
        return SystemOneResponse(
            model=self.cfg.model_name,
            answers=answers,
            usage=Usage(input_tokens=n_input, output_tokens=len(names)),
        )


def save_assets(
    path: str,
    tokenizer: Tokenizer,
    head: Mapping[str, np.ndarray],
    *,
    model_name: str,
    pointer_dim: int,
    eps: float,
    max_length: int,
    temperature: float,
    temperature_by_kind: dict[str, float],
    ordinal_smoothing: float,
) -> None:
    """Write what from_assets reads: the tokenizer, the pointer head's weights and its settings
    (tokenizer.json, head.safetensors, head_config.json). The torso is the backend's."""
    from safetensors.numpy import save_file

    os.makedirs(path, exist_ok=True)
    tokenizer.save(os.path.join(path, "tokenizer.json"))
    save_file(
        {k: np.asarray(v, dtype=np.float32) for k, v in head.items()},
        os.path.join(path, "head.safetensors"),
    )
    saved = {
        "model_name": model_name,
        "pointer_dim": pointer_dim,
        "eps": eps,
        "max_length": max_length,
        "temperature": temperature,
        "temperature_by_kind": temperature_by_kind,
        "ordinal_smoothing": ordinal_smoothing,
    }
    with open(os.path.join(path, "head_config.json"), "w", encoding="utf-8") as f:
        json.dump(saved, f, indent=2)
