from __future__ import annotations

import pytest

from conftest import FakeBackend, write_assets
from strands_decider.hidden_engine import HiddenStateEngine

POLICY = (
    "Loans run 21 days and renew twice online unless another patron holds the item. "
    "Accounts with fines above ten dollars cannot renew until the fine is paid. "
)
QUESTIONS = {
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
            "circulation": "renewals and holds",
            "billing": "fines",
            "reference": "research help",
        },
    },
    "urgent": {"type": "noul", "instructions": "Does the patron convey urgency?"},
    "mood": {
        "type": "score",
        "instructions": "How frustrated is the patron?",
        "criteria": ["calm", "annoyed", "angry"],
    },
}
REQUEST = {"state": POLICY + "My renewal failed and I leave Friday!", "questions": QUESTIONS}


def _engine(assets, backend=None, **config) -> HiddenStateEngine:
    return HiddenStateEngine.from_assets(str(assets), backend or FakeBackend(), **config)


def test_the_saved_settings_load_and_config_overrides_them(tmp_path):
    assets = write_assets(tmp_path / "assets", eps=0.01, temperature_by_kind={"noul": 2.0})
    engine = _engine(assets, max_length=1024)
    assert (engine.cfg.model_name, engine.cfg.max_length) == ("tiny-q8_0", 1024)
    assert engine.cfg.temperature_by_kind == {"noul": 2.0}
    assert engine.head.eps == 0.01


def test_answers_each_question_type(assets):
    backend = FakeBackend()
    response = _engine(assets, backend).evaluate(REQUEST)
    ((state, questions, _),) = backend.calls
    assert response.model == "tiny-q8_0"
    assert response.usage.input_tokens == len(state) + sum(len(q) for q in questions)
    assert response.usage.output_tokens == 3
    team, urgent, mood = (response.answers[n] for n in ("team", "urgent", "mood"))
    assert team.choice == max(team.probabilities, key=team.probabilities.get)
    assert sum(team.probabilities.values()) == pytest.approx(1, abs=1e-3)
    assert 0 <= urgent.noul <= 1
    assert mood.legend == {"0": "calm", "1": "annoyed", "2": "angry"}
    assert mood.score == pytest.approx(
        sum(int(k) * p for k, p in mood.probabilities.items()), abs=1e-3
    )


def test_each_kind_gets_its_own_temperature(tmp_path):
    assets = write_assets(tmp_path / "assets", temperature_by_kind={"choice": 1e9})
    answers = _engine(assets).evaluate(REQUEST).answers
    assert set(answers["team"].probabilities.values()) == {0.3333}
    assert set(answers["mood"].probabilities.values()) != {0.3333}


def test_backend_is_asked_for_the_option_ends_then_the_last_token(assets):
    backend = FakeBackend()
    _engine(assets, backend).evaluate(REQUEST)
    ((_, questions, rows),) = backend.calls
    for q, r, n_options in zip(questions, rows, (3, 2, 3), strict=True):
        assert len(r) == n_options + 1
        assert r[-1] == len(q) - 1 and r[:-1] == sorted(r[:-1])


def test_a_prompt_over_the_window_is_refused_before_the_forward(assets):
    backend = FakeBackend()
    with pytest.raises(ValueError, match="exceeds the context window of 512 tokens"):
        _engine(assets, backend).evaluate({**REQUEST, "state": POLICY * 30})
    assert not backend.calls


def test_request_limit_and_images_are_refused_before_the_forward(assets):
    backend = FakeBackend()
    with pytest.raises(ValueError, match="exceeds the limit of 100 tokens"):
        _engine(assets, backend, max_request_tokens=100).evaluate(REQUEST)
    with pytest.raises(ValueError, match="images are not supported"):
        _engine(assets, backend).evaluate({**REQUEST, "images": ["aGk="]})
    assert not backend.calls


class _WrongRows(FakeBackend):
    def __init__(self, mistake: str):
        super().__init__()
        self.mistake = mistake

    def forward(self, state, questions, rows):
        if self.mistake == "every position":
            rows = [list(range(len(q))) for q in questions]
        out = super().forward(state, questions, rows)
        if self.mistake == "a row too few":
            return [h[:-1] for h in out]
        if self.mistake == "an array too few":
            return out[:-1]
        return out


@pytest.mark.parametrize(
    ("mistake", "message"),
    [
        ("every position", r"backend returned shape \(\d+, 8\) for question 'team'; expected"),
        (
            "a row too few",
            r"backend returned shape \(3, 8\) for question 'team'; expected \(4, hidden\)",
        ),
        ("an array too few", "backend returned 2 arrays for 3 questions"),
    ],
)
def test_a_backend_returning_the_wrong_rows_is_named(assets, mistake, message):
    with pytest.raises(ValueError, match=message):
        _engine(assets, _WrongRows(mistake)).evaluate(REQUEST)
