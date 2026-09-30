"""Match what the interviewer just said to a question in the bank (D16 — it only PROPOSES).

Nothing here ever changes a question's state. A ``Proposal`` is shown to the interviewer, who
applies it with one click or ignores it. That constraint comes from a measurement in the sibling
project: a purely syntactic question detector fired on 24 turns of a real call of which only ~8
were substantive, because an interview is full of logistics and rapport that parse as questions.
A wrong auto-transition mid-interview costs attention; a wrong proposal costs one glance.

Three routes are implemented so they can be **benchmarked against each other** on a labelled
transcript (``scripts/bench_matcher.py``), rather than one being chosen by assertion:

* ``lexical`` — IDF-weighted token overlap against the bank. Deterministic, no model, no GPU,
  runs in microseconds. IDF matters: without it "what", "you", "czy", "jak" dominate every score.
* ``llm`` — asks the resident chat model to name the question or say NONE.
* ``hybrid`` — lexical shortlist, then the model picks among the top candidates. Costs one call
  but only over a handful of options.

**An embedding route is deliberately absent.** The sibling project built and measured it, and it
failed for reasons that apply here: ``nomic-embed-text`` could not read Polish (1/7 top-1 against
4/4 on English translations of the same probes), and an interview is topically saturated, so
salient and non-salient turns scored 0.510 vs 0.518 cosine — signal absent and slightly inverted.
Re-adding it needs a multilingual embedding model and a fresh measurement, not an assumption.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field

from .models import Question, QuestionState, Speaker, TranscriptLine

# A word that carries no topic signal in either language of a typical bilingual interview.
# Kept small and explicit: IDF already suppresses bank-wide common words, and an over-long
# stoplist would strip the domain nouns the match depends on.
STOPWORDS = frozenset("""
    a an the and or but if then of to in on at by for with about from as is are was were be been
    do does did you your yours we our i me my it its that this these those how what which who whom
    when where why can could would should will shall may might must have has had not no yes so
    tell me about
    i w z na do nie tak jest są był była było czy jak co kto gdzie kiedy dlaczego który która które
    to ten ta te o od za pod nad przez dla oraz albo lub ale się jego jej ich mnie mi ty twoje
    opowiedz powiedz
    """.split())

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def normalize(text: str) -> list[str]:
    """Lowercase word tokens, accents preserved (they are meaningful in Polish and many others)."""
    text = unicodedata.normalize("NFC", text or "")
    return [w.casefold() for w in _WORD.findall(text)]


def content_tokens(text: str) -> list[str]:
    return [t for t in normalize(text) if t not in STOPWORDS and len(t) > 1]


@dataclass
class Proposal:
    """A suggestion that the interviewer just asked ``question_id``. Never auto-applied (D16)."""

    question_id: str
    confidence: float
    route: str
    line_index: int
    evidence: str = ""
    runner_up: str | None = None
    margin: float = 0.0

    def as_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "confidence": round(self.confidence, 3),
            "route": self.route,
            "line_index": self.line_index,
            "evidence": self.evidence,
            "runner_up": self.runner_up,
            "margin": round(self.margin, 3),
        }


def _unit(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    vec: dict[str, float] = {}
    for t in tokens:
        vec[t] = vec.get(t, 0.0) + idf.get(t, 0.0)
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {t: v / norm for t, v in vec.items()}


@dataclass
class QuestionIndex:
    """IDF-weighted bag of words per question, **one vector per language variant**.

    Measured (mock interview, 6 labelled questions): blending every language's wording into one
    vector per question scores recall **0.17** — a Polish line can only match the Polish half of
    a pl+en bag, so the cosine is roughly halved and the threshold rejects nearly everything.
    Scoring against the best single variant instead lifts recall to **1.00** at unchanged
    precision. The interviewer asks in one language at a time; the comparison must too.

    The intent and expected-signals text is kept as an extra variant rather than mixed into the
    wordings, so a paraphrase can still match without diluting the literal wording.
    """

    questions: list[Question]
    idf: dict[str, float] = field(default_factory=dict)
    variants: dict[str, list[tuple[str, dict[str, float]]]] = field(default_factory=dict)

    @classmethod
    def build(cls, questions: list[Question]) -> QuestionIndex:
        raw: dict[str, list[tuple[str, list[str]]]] = {}
        for q in questions:
            items: list[tuple[str, list[str]]] = []
            for lang, wording in q.text.items():
                tokens = content_tokens(wording)
                if tokens:
                    items.append((lang, tokens))
            gloss = " ".join([q.intent, *q.expected_signals]).strip()
            if gloss:
                tokens = content_tokens(gloss)
                if tokens:
                    items.append(("_intent", tokens))
            raw[q.id] = items

        # IDF is computed over the whole bank so a term common to every question is damped
        # wherever it appears; a variant is one document.
        docs = [tokens for items in raw.values() for _, tokens in items]
        n = max(1, len(docs))
        df: dict[str, int] = {}
        for tokens in docs:
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1
        idf = {t: math.log((n + 1) / (c + 0.5)) for t, c in df.items()}
        variants = {
            qid: [(lang, _unit(tokens, idf)) for lang, tokens in items]
            for qid, items in raw.items()
        }
        return cls(questions=questions, idf=idf, variants=variants)

    def score(self, text: str, lang: str = "") -> list[tuple[str, float]]:
        """Cosine against each question's best-matching variant, best question first.

        ``lang`` is a preference, not a filter: an interviewer who asks a Polish-only bank
        question in English must still match, so every variant is scored and the best wins.
        """
        tokens = content_tokens(text)
        if not tokens:
            return []
        vec = _unit(tokens, self.idf)
        scored: list[tuple[str, float]] = []
        for qid, items in self.variants.items():
            best = 0.0
            for variant_lang, qvec in items:
                sim = sum(w * qvec.get(t, 0.0) for t, w in vec.items())
                # A tiny nudge for the language actually spoken breaks ties between two
                # wordings of the same question; it cannot promote a different question.
                if lang and variant_lang == lang:
                    sim *= 1.05
                best = max(best, sim)
            scored.append((qid, best))
        return sorted(scored, key=lambda kv: kv[1], reverse=True)


INTERROGATIVES = frozenset("""
    what how why when where which who whom whose can could would should do does did
    tell describe explain walk give share
    co jak dlaczego kiedy gdzie który która które kto czy jaki jaka jakie jaką jakim
    opowiedz powiedz opisz wyjaśnij podaj przedstaw
    """.split())

_SENTENCE = re.compile(r"[.?!…]+")

# How many opening words of a SENTENCE are examined for an interrogative.
QUESTION_OPENER_WORDS = 6


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text or "") if s.strip()]


def looks_like_question(text: str) -> bool:
    """Syntactic gate: does this turn contain an interrogative **sentence**?

    Applied per sentence, not to the first words of the whole turn. Measured on the mock
    interview: a whole-turn gate missed 2 of 6 questions — "Dzień dobry. Zacznijmy od Pana
    doświadczenia. Opowiedz o..." and "Let me switch to English for this one. Describe a
    time..." — because the interrogative sits past the greeting or preamble that real
    interviewers put in front of it. Both turns already scored 0.93 and 1.00 against the right
    bank question, so the gate, not the scoring, was rejecting them.

    Cheap and deliberately permissive: it removes obvious non-questions before any scoring, and
    is NOT a salience judgement — that is what the interviewer's click is for (D16).
    """
    if "?" in (text or ""):
        return True
    return any(
        w in INTERROGATIVES
        for sentence in sentences(text)
        for w in normalize(sentence)[:QUESTION_OPENER_WORDS]
    )


def propose_lexical(
    line: TranscriptLine,
    line_index: int,
    index: QuestionIndex,
    min_confidence: float,
    open_only: bool = True,
) -> Proposal | None:
    """Route 1: IDF cosine. Deterministic, no model."""
    if line.speaker is Speaker.CANDIDATE or not looks_like_question(line.text):
        return None
    scored = index.score(line.text, line.lang or "")
    if open_only:
        states = {q.id: q.state for q in index.questions}
        scored = [
            (qid, s)
            for qid, s in scored
            if states.get(qid) in (QuestionState.PENDING, QuestionState.SKIPPED)
        ]
    if not scored:
        return None
    best_id, best = scored[0]
    runner_up, second = scored[1] if len(scored) > 1 else (None, 0.0)
    if best < min_confidence:
        return None
    return Proposal(
        question_id=best_id,
        confidence=best,
        route="lexical",
        line_index=line_index,
        evidence=line.text,
        runner_up=runner_up,
        margin=best - second,
    )


LLM_SYSTEM = (
    "You match an interviewer's spoken line to a question from a prepared bank. "
    "Answer with the question id alone, or NONE if the line does not ask any of them. "
    "Never explain. Never invent an id."
)


def build_llm_prompt(line_text: str, candidates: list[Question], lang: str) -> str:
    rows = []
    for q in candidates:
        wording = q.wording(lang)
        rows.append(f"{q.id}: {wording}" + (f"  [intent: {q.intent}]" if q.intent else ""))
    bank = "\n".join(rows)
    return (
        f"Question bank:\n{bank}\n\n"
        f'Interviewer said: "{line_text}"\n\n'
        "Which bank question is the interviewer asking? Reply with the id, or NONE."
    )


def parse_llm_answer(answer: str, valid_ids: set[str]) -> str | None:
    """Take the first valid id the model names; anything else (including NONE) is no match.

    Trailing sentence punctuation is stripped before comparing: a question id may itself contain
    a dot or a dash, so those characters stay inside the token, which means a model that ends
    with "...is q1." would otherwise have a perfectly good answer rejected.
    """
    for token in re.findall(r"[A-Za-z0-9_.-]+", answer or ""):
        for candidate in (token, token.rstrip(".,;:-")):
            if candidate in valid_ids:
                return candidate
        if token.rstrip(".,;:-").upper() == "NONE":
            return None
    return None


def propose_llm(
    line: TranscriptLine,
    line_index: int,
    index: QuestionIndex,
    chat: object,
    min_confidence: float,
    shortlist: int = 0,
    route_name: str = "llm",
) -> Proposal | None:
    """Route 2/3: ask the model. ``shortlist>0`` makes it the hybrid route.

    The confidence a model route reports is not a probability — it is a fixed, honest constant
    meaning "the model named this id". Dressing a single categorical answer up as a calibrated
    score would be a lie the UI would then rank by.
    """
    if line.speaker is Speaker.CANDIDATE or not looks_like_question(line.text):
        return None
    states = {q.id: q.state for q in index.questions}
    open_questions = [
        q
        for q in index.questions
        if states.get(q.id) in (QuestionState.PENDING, QuestionState.SKIPPED)
    ]
    if not open_questions:
        return None
    candidates = open_questions
    lexical_score = 0.0
    if shortlist > 0:
        ranked = [
            (qid, s)
            for qid, s in index.score(line.text, line.lang or "")
            if qid in {q.id for q in open_questions}
        ]
        top = ranked[:shortlist]
        if not top:
            return None
        lexical_score = top[0][1]
        keep = {qid for qid, _ in top}
        candidates = [q for q in open_questions if q.id in keep]

    prompt = build_llm_prompt(line.text, candidates, line.lang or "")
    try:
        reply = chat.complete(prompt, system=LLM_SYSTEM, max_tokens=16)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 — a matcher failure must never stop a transcript
        return None
    matched = parse_llm_answer(reply.text, {q.id for q in candidates})
    if matched is None:
        return None
    # A named id is a categorical answer; the hybrid route can additionally report how much
    # lexical support it had, which is what makes its proposals sortable next to route 1's.
    confidence = max(min_confidence, lexical_score) if shortlist else 0.75
    return Proposal(
        question_id=matched,
        confidence=confidence,
        route=route_name,
        line_index=line_index,
        evidence=line.text,
    )


def propose_for_line(
    line: TranscriptLine,
    line_index: int,
    index: QuestionIndex,
    route: str,
    min_confidence: float,
    chat: object | None = None,
) -> Proposal | None:
    """Single entry point used by the pipeline and the cockpit."""
    if route == "off":
        return None
    if route == "lexical" or chat is None:
        return propose_lexical(line, line_index, index, min_confidence)
    if route == "llm":
        return propose_llm(line, line_index, index, chat, min_confidence)
    if route == "hybrid":
        return propose_llm(
            line, line_index, index, chat, min_confidence, shortlist=4, route_name="hybrid"
        )
    raise ValueError(f"unknown matcher route {route!r}")


def propose_all(
    lines: list[TranscriptLine],
    questions: list[Question],
    route: str = "lexical",
    min_confidence: float = 0.45,
    chat: object | None = None,
) -> list[Proposal]:
    """Run the matcher over a whole transcript (post-hoc path).

    Question states are *simulated* forward — once a question has been proposed it stops being a
    candidate — so a bank question is not proposed for every similar line in the call. This
    mirrors what the live cockpit sees after the interviewer clicks, without mutating the
    session (D16: the matcher never writes state).
    """
    index = QuestionIndex.build(questions)
    original = {q.id: q.state for q in questions}
    proposed: set[str] = set()
    out: list[Proposal] = []
    try:
        for i, line in enumerate(lines):
            proposal = propose_for_line(line, i, index, route, min_confidence, chat)
            if proposal is None:
                continue
            out.append(proposal)
            proposed.add(proposal.question_id)
            for q in questions:
                if q.id == proposal.question_id:
                    q.state = QuestionState.ASKED
    finally:
        for q in questions:  # restore: propose_all is read-only from the caller's point of view
            q.state = original[q.id]
    return out
