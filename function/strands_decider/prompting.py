"""Turning a (state, question) pair into text, and the head's distribution into an answer.

The pointer head has no fixed meaning per option: it scores whatever option is listed at
position k of the prompt, from the hidden state at the end of that option's line. The prompt
alone binds a position to `billing` in one request and `Calm` in the next, so the options are
numbered explicitly, one per line.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from .schema import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    Content,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
    derive_confidence,
    derive_score_confidence,
)

NOUL_SLOT_LABELS = ("false", "true")
NOUL_DEFAULT_CRITERIA = {
    "false": "the statement does not hold for this state",
    "true": "the statement holds for this state",
}


def render_content(content: Content | None) -> str:
    """Flatten a state or instruction into text: dicts and lists as indented JSON, in the
    order given, as the model was trained to read them."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    return json.dumps(content, indent=2, ensure_ascii=False, sort_keys=False)


@dataclass(frozen=True)
class RenderedQuestion:
    """A question as prompt text, with what reads the head's distribution back."""

    text: str
    slot_labels: tuple[str, ...]
    slot_descriptions: tuple[str, ...]
    kind: str  # "noul" | "choice" | "score"
    # Each option's line, as (start, end) characters in `text`, for the tokenizer's offsets.
    option_spans: tuple[tuple[int, int], ...]


def _option_block(pairs: Sequence[tuple[str, str]]) -> tuple[str, list[tuple[int, int]]]:
    """The options numbered from 1, one line each, and each line's (start, end) characters."""
    lines: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for i, (name, desc) in enumerate(pairs):
        desc = " ".join((desc or "").split())
        line = f"{i + 1}. {name}" + (f" \u2014 {desc}" if desc else "")
        lines.append(line)
        spans.append((cursor, cursor + len(line)))
        cursor += len(line) + 1
    return "\n".join(lines), spans


def render_question(question: Question) -> RenderedQuestion:
    instructions = render_content(question.instructions)

    if isinstance(question, NoulQuestion):
        crit = {**NOUL_DEFAULT_CRITERIA, **(question.criteria or {})}
        pairs = [(lbl, render_content(crit[lbl])) for lbl in NOUL_SLOT_LABELS]
        kind = "noul"
        header = "Decide whether the statement is true of the state."
    elif isinstance(question, ChoiceQuestion):
        pairs = [(name, render_content(desc)) for name, desc in question.criteria.items()]
        kind = "choice"
        header = "Select exactly one option."
    elif isinstance(question, ScoreQuestion):
        pairs = [(str(i), desc) for i, desc in enumerate(question.criteria)]
        kind = "score"
        header = "Rate the state against the ordered levels below (lowest first)."
    else:  # pragma: no cover - guarded by pydantic unions
        raise TypeError(f"unknown question type: {type(question)!r}")

    block, block_spans = _option_block(pairs)
    prefix = f'<question type="{kind}">\n{header}\n{instructions}\n<options>\n'
    text = prefix + block + "\n</options>\n</question>\n<answer>"
    base = len(prefix)
    return RenderedQuestion(
        text=text,
        slot_labels=tuple(n for n, _ in pairs),
        slot_descriptions=tuple((d or "").strip() for _, d in pairs),
        kind=kind,
        option_spans=tuple((base + s, base + e) for s, e in block_spans),
    )


def render_state(state: Content) -> str:
    """The prefix every question of a request continues."""
    return f"<state>\n{render_content(state)}\n</state>\n"


def option_token_index(
    offsets: Sequence[Sequence[int]], spans: Sequence[Sequence[int]]
) -> list[int]:
    """The index of the last token inside each option's line: the token that has just read
    the whole option under causal attention. An option no token lies inside is refused, as
    scoring it from a neighbour's token would be silently wrong."""
    out: list[int] = []
    for a, b in spans:
        inside = [j for j, (lo, hi) in enumerate(offsets) if lo < hi and lo >= a and hi <= b]
        if not inside:
            raise ValueError(f"option span ({a},{b}) has no token of its own")
        out.append(inside[-1])
    return out


def to_answer(
    rq: RenderedQuestion, probs: list[float], *, ordinal_smoothing: float = 0.0
) -> Answer:
    by_label = {name: float(p) for name, p in zip(rq.slot_labels, probs, strict=True)}
    if rq.kind == "noul":
        return NoulAnswer(noul=round(by_label["true"], 4))
    rounded = {k: round(v, 4) for k, v in by_label.items()}
    if rq.kind == "choice":
        return ChoiceAnswer(
            choice=max(by_label, key=lambda k: by_label[k]),
            probabilities=rounded,
            confidence=round(derive_confidence(list(by_label.values())), 4),
        )
    # A score's labels are its level indices, in order.
    return ScoreAnswer(
        score=round(sum(i * p for i, p in enumerate(by_label.values())), 4),
        legend=dict(zip(rq.slot_labels, rq.slot_descriptions, strict=True)),
        probabilities=rounded,
        confidence=round(
            derive_score_confidence(list(by_label.values()), ordinal_smoothing=ordinal_smoothing),
            4,
        ),
    )
