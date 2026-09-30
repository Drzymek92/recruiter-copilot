"""The question matcher (D16): it proposes, never applies — and the two bugs the bench found."""

from __future__ import annotations

import pytest

from recruiter_copilot.matcher import (
    QuestionIndex,
    content_tokens,
    looks_like_question,
    parse_llm_answer,
    propose_all,
    propose_for_line,
    propose_lexical,
    sentences,
)
from recruiter_copilot.models import Question, QuestionState, Speaker, TranscriptLine


def _bank() -> list[Question]:
    return [
        Question(
            "q1",
            {
                "pl": "Opowiedz o ostatnim projekcie produkcyjnym w Pythonie.",
                "en": "Tell me about your last production Python project.",
            },
            intent="production Python depth",
        ),
        Question(
            "q2",
            {
                "pl": "Jaką funkcję opartą na modelach językowych wdrożyłeś?",
                "en": "Which LLM-backed feature did you ship?",
            },
        ),
        Question(
            "q3",
            {
                "en": "Describe a time you explained a technical trade-off to a "
                "non-technical stakeholder."
            },
        ),
    ]


def _line(text: str, lang: str = "pl", speaker: Speaker = Speaker.INTERVIEWER) -> TranscriptLine:
    return TranscriptLine(0.0, 5.0, text, speaker, lang)


# ── regression: the interrogative gate must look at every SENTENCE ────────────────────────


def test_question_after_a_greeting_is_still_a_question() -> None:
    """Measured miss: a whole-turn gate saw only 'Dzień dobry. Zacznijmy od...' and rejected it."""
    assert looks_like_question(
        "Dzień dobry. Zacznijmy od Pana doświadczenia. "
        "Opowiedz o ostatnim projekcie produkcyjnym w Pythonie"
    )


def test_question_after_an_english_preamble_is_still_a_question() -> None:
    assert looks_like_question(
        "Let me switch to English for this one. Describe a time you had to explain a trade-off"
    )


def test_a_plain_statement_is_not_a_question() -> None:
    assert not looks_like_question("Rozumiem, dziękuję za odpowiedź.")
    assert not looks_like_question("That makes sense to me.")


def test_a_question_mark_anywhere_is_enough() -> None:
    assert looks_like_question("Right. So, the architecture?")


def test_sentences_splits_on_terminators() -> None:
    assert sentences("A. B? C! ") == ["A", "B", "C"]


# ── regression: per-language variants, not one blended bag ────────────────────────────────


def test_a_polish_line_is_not_penalised_by_the_english_wording() -> None:
    """Measured: blending both wordings into one vector halved the cosine (recall 0.17)."""
    index = QuestionIndex.build(_bank())
    score = dict(index.score("Opowiedz o ostatnim projekcie produkcyjnym w Pythonie", "pl"))["q1"]
    assert score > 0.8, f"a near-verbatim Polish match scored only {score:.2f}"


def test_an_english_line_matches_an_english_only_bank_question() -> None:
    index = QuestionIndex.build(_bank())
    ranked = index.score("Describe a time you explained a technical trade-off", "en")
    assert ranked[0][0] == "q3" and ranked[0][1] > 0.6


def test_a_question_asked_in_the_other_language_still_matches() -> None:
    """Language is a preference, not a filter — the interviewer may switch mid-call."""
    index = QuestionIndex.build(_bank())
    ranked = index.score("Tell me about your last production Python project", "pl")
    assert ranked[0][0] == "q1"


def test_empty_and_stopword_only_text_scores_nothing() -> None:
    index = QuestionIndex.build(_bank())
    assert index.score("") == []
    assert index.score("i to jest") == []


def test_content_tokens_drop_stopwords_but_keep_accents() -> None:
    tokens = content_tokens("Czy to jest wdrożenie modelu?")
    assert "wdrożenie" in tokens and "czy" not in tokens and "to" not in tokens


# ── proposals ─────────────────────────────────────────────────────────────────────────────


def test_candidate_lines_never_produce_a_proposal() -> None:
    index = QuestionIndex.build(_bank())
    line = _line("Opowiedz o ostatnim projekcie produkcyjnym w Pythonie", speaker=Speaker.CANDIDATE)
    assert propose_lexical(line, 0, index, 0.45) is None


def test_below_threshold_raises_nothing() -> None:
    index = QuestionIndex.build(_bank())
    assert propose_lexical(_line("Jak się Pan dziś czuje?"), 0, index, 0.9) is None


def test_a_proposal_carries_its_evidence_and_margin() -> None:
    index = QuestionIndex.build(_bank())
    p = propose_lexical(
        _line("Opowiedz o ostatnim projekcie produkcyjnym w Pythonie"), 3, index, 0.45
    )
    assert p is not None
    assert p.question_id == "q1" and p.line_index == 3 and p.route == "lexical"
    assert p.evidence and p.margin > 0
    assert "question_id" in p.as_dict()


def test_answered_questions_are_not_proposed_again() -> None:
    bank = _bank()
    bank[0].state = QuestionState.ANSWERED
    index = QuestionIndex.build(bank)
    line = _line("Opowiedz o ostatnim projekcie produkcyjnym w Pythonie")
    assert propose_lexical(line, 0, index, 0.45) is None


def test_skipped_questions_can_still_be_proposed() -> None:
    bank = _bank()
    bank[0].state = QuestionState.SKIPPED
    index = QuestionIndex.build(bank)
    p = propose_lexical(
        _line("Opowiedz o ostatnim projekcie produkcyjnym w Pythonie"), 0, index, 0.45
    )
    assert p is not None and p.question_id == "q1"


def test_propose_all_never_mutates_question_state() -> None:
    """D16: the matcher proposes. Only the interviewer's click changes state."""
    bank = _bank()
    before = [q.state for q in bank]
    lines = [
        _line("Opowiedz o ostatnim projekcie produkcyjnym w Pythonie"),
        _line("Jaką funkcję opartą na modelach językowych wdrożyłeś?"),
    ]
    proposals = propose_all(lines, bank, route="lexical", min_confidence=0.45)
    assert len(proposals) == 2
    assert [q.state for q in bank] == before


def test_propose_all_does_not_repeat_one_question_across_similar_lines() -> None:
    bank = _bank()
    text = "Opowiedz o ostatnim projekcie produkcyjnym w Pythonie"
    proposals = propose_all([_line(text), _line(text)], bank, min_confidence=0.45)
    assert [p.question_id for p in proposals] == ["q1"]


def test_route_off_proposes_nothing() -> None:
    index = QuestionIndex.build(_bank())
    line = _line("Opowiedz o ostatnim projekcie produkcyjnym w Pythonie")
    assert propose_for_line(line, 0, index, "off", 0.45) is None


def test_unknown_route_is_an_error() -> None:
    index = QuestionIndex.build(_bank())
    with pytest.raises(ValueError, match="unknown matcher route"):
        propose_for_line(_line("co?"), 0, index, "magic", 0.45, chat=object())


# ── the model routes ──────────────────────────────────────────────────────────────────────


class FakeChat:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def complete(self, prompt, system=None, max_tokens=None):  # noqa: ARG002
        self.prompts.append(prompt)
        return type("R", (), {"text": self.answer})()


def test_llm_route_uses_the_id_the_model_names() -> None:
    index = QuestionIndex.build(_bank())
    chat = FakeChat("q2")
    p = propose_for_line(_line("A co wdrożyłeś ostatnio?"), 0, index, "llm", 0.45, chat=chat)
    assert p is not None and p.question_id == "q2" and p.route == "llm"


def test_llm_route_respects_none() -> None:
    index = QuestionIndex.build(_bank())
    p = propose_for_line(_line("Jak minął weekend?"), 0, index, "llm", 0.45, chat=FakeChat("NONE"))
    assert p is None


def test_a_model_failure_never_breaks_the_transcript() -> None:
    class Broken:
        def complete(self, prompt, system=None, max_tokens=None):
            raise RuntimeError("model down")

    index = QuestionIndex.build(_bank())
    assert propose_for_line(_line("co?"), 0, index, "llm", 0.45, chat=Broken()) is None


def test_hybrid_shortlists_before_asking() -> None:
    index = QuestionIndex.build(_bank())
    chat = FakeChat("q1")
    p = propose_for_line(
        _line("Opowiedz o ostatnim projekcie produkcyjnym w Pythonie"),
        0,
        index,
        "hybrid",
        0.45,
        chat=chat,
    )
    assert p is not None and p.route == "hybrid"
    assert len(chat.prompts) == 1


def test_parse_llm_answer_rejects_invented_ids() -> None:
    assert parse_llm_answer("q2", {"q1", "q2"}) == "q2"
    assert parse_llm_answer("The answer is q1.", {"q1"}) == "q1"
    assert parse_llm_answer("q99", {"q1"}) is None
    assert parse_llm_answer("NONE", {"q1"}) is None
    assert parse_llm_answer("", {"q1"}) is None
