"""Candidate questions answered from the recruiter's info pack, verbatim (D26, F9).

The recruiter may put an optional ``docs/job_info.md`` in the session bundle (D15 amendment):
benefits, salary band, remote policy, team, stack, process, company basics. When the CANDIDATE
asks a question, the cockpit shows the interviewer the matching pack section **verbatim** —
heading and body exactly as written — or "Not in your pack". Nothing is generated, no model runs
in this path, and nothing here changes any question state (D16): the interviewer reads the section
and speaks the answer. The pack is never evidence about the candidate, so ``analysis.py`` and the
report never read it (D11, D17 — pinned by tests).

Contract (D15): the pack is split into sections by ``##`` heading. Text before the first ``##``
(a ``# Title`` line, an intro) is a preamble and is **ignored**; ``###`` and deeper headings stay
inside their section's body; a section whose body is empty is dropped (a bare heading answers
nothing). Absent or empty pack = the feature is off.

Matching (D21 machinery, shared — ``matcher.build_variant_vectors`` / ``score_variants``): IDF
cosine of the question against each section's best variant. A section's variants are its heading,
the whole section, and each body line — the per-language idea of D21 carried over, because a pack
line is written in one language and a Polish question must not be diluted by the English half of a
bilingual section. Only CANDIDATE lines are considered, and only the sentences that pass the
per-sentence ``matcher.looks_like_question`` gate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .matcher import (
    Variants,
    build_variant_vectors,
    content_tokens,
    looks_like_question,
    score_variants,
)
from .models import Speaker, TranscriptLine

# Tokens are cut to this many characters before IDF/cosine. Polish inflects nouns heavily
# ("benefity" / "benefitów" / "benefitach"), and a pack written in one case is asked about in
# another; a fixed prefix is the cheapest deterministic stemmer. Measured, not guessed —
# ``scripts/bench_info_pack.py --prefix`` compares 0 (off) against prefixes (see README / D26).
PACK_TOKEN_PREFIX = 6

_HEADING = re.compile(r"^##(?!#)\s*(.*?)\s*#*\s*$")
_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
# A sentence WITH its terminator, so a per-sentence gate still sees the "?".
_SENTENCE_WITH_END = re.compile(r"[^.?!…]+[.?!…]*")


@dataclass(frozen=True)
class PackSection:
    """One ``##`` section of the info pack. ``heading``/``body`` are the file's text, verbatim."""

    id: str
    heading: str
    body: str

    def as_dict(self) -> dict[str, str]:
        return {"id": self.id, "heading": self.heading, "body": self.body}


def split_sections(text: str) -> list[PackSection]:
    """Split a markdown pack into ``##`` sections (D15). Preamble before the first ``##`` ignored.

    The body keeps its lines exactly (only blank lines at its start and end are trimmed), so the
    cockpit can show it verbatim. Section ids are positional (``s1``, ``s2``, ...).
    """
    sections: list[PackSection] = []
    heading: str | None = None
    body: list[str] = []

    def close() -> None:
        if heading is None:
            return
        text_body = "\n".join(body).strip("\n")
        if text_body.strip():
            sections.append(PackSection(f"s{len(sections) + 1}", heading, text_body.rstrip()))

    for line in (text or "").splitlines():
        m = _HEADING.match(line)
        if m:
            close()
            heading, body = m.group(1), []
        elif heading is not None:
            body.append(line)
    close()
    return sections


def pack_tokens(text: str, prefix: int = PACK_TOKEN_PREFIX) -> list[str]:
    """``matcher.content_tokens`` cut to ``prefix`` characters (0 = no cut)."""
    tokens = content_tokens(text)
    return [t[:prefix] for t in tokens] if prefix > 0 else tokens


def section_variants(section: PackSection) -> list[tuple[str, str]]:
    """The texts a section can be matched by: heading, whole section, each body line."""
    out = [("heading", section.heading), ("section", f"{section.heading}\n{section.body}")]
    for line in section.body.splitlines():
        line = _BULLET.sub("", line).strip().lstrip("#").strip()
        if line:
            out.append(("line", line))
    return out


@dataclass
class InfoPackIndex:
    """IDF-cosine index over the pack's sections (D21 scoring, shared with ``QuestionIndex``)."""

    sections: list[PackSection]
    prefix: int = PACK_TOKEN_PREFIX
    idf: dict[str, float] = field(default_factory=dict)
    variants: Variants = field(default_factory=dict)

    @classmethod
    def build(cls, sections: list[PackSection], prefix: int = PACK_TOKEN_PREFIX) -> InfoPackIndex:
        raw: dict[str, list[tuple[str, list[str]]]] = {}
        for s in sections:
            items = [(tag, pack_tokens(text, prefix)) for tag, text in section_variants(s)]
            raw[s.id] = [(tag, tokens) for tag, tokens in items if tokens]
        idf, variants = build_variant_vectors(raw)
        return cls(sections=sections, prefix=prefix, idf=idf, variants=variants)

    @classmethod
    def from_text(cls, text: str, prefix: int = PACK_TOKEN_PREFIX) -> InfoPackIndex:
        return cls.build(split_sections(text), prefix)

    def section(self, section_id: str) -> PackSection:
        return next(s for s in self.sections if s.id == section_id)

    def score(self, text: str) -> list[tuple[str, float]]:
        """Best section first. Variants carry no language tag, so there is no language nudge."""
        return score_variants(pack_tokens(text, self.prefix), self.idf, self.variants)


@dataclass
class PackAnswer:
    """One candidate question and what the pack says about it (``section`` None = not in pack)."""

    key: str  # "<line_index>:<sentence_index>" — stable per transcript line, used to dismiss
    line_index: int
    t_start: float
    question: str
    section: PackSection | None
    score: float

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "line_index": self.line_index,
            "t_start": self.t_start,
            "question": self.question,
            "found": self.section is not None,
            "section_id": self.section.id if self.section else None,
            "heading": self.section.heading if self.section else "",
            "body": self.section.body if self.section else "",
            "score": round(self.score, 3),
        }


def question_sentences(text: str) -> list[str]:
    """The sentences of a line that pass the per-sentence question gate (D21), terminator kept."""
    sentences = [m.group(0).strip() for m in _SENTENCE_WITH_END.finditer(text or "")]
    return [s for s in sentences if s and looks_like_question(s)]


def answer_line(
    line: TranscriptLine, line_index: int, index: InfoPackIndex, min_confidence: float
) -> list[PackAnswer]:
    """Pack answers for one transcript line: CANDIDATE lines only, one per question sentence.

    A question sentence with no content word at all ("Right?", "Co?") is not matched — there is
    nothing to look up. One whose best section scores below ``min_confidence`` is "not in pack".
    """
    if line.speaker is not Speaker.CANDIDATE or not looks_like_question(line.text):
        return []
    out: list[PackAnswer] = []
    for j, sentence in enumerate(question_sentences(line.text)):
        if not content_tokens(sentence):
            continue
        scored = index.score(sentence)
        best_id, best = scored[0] if scored else ("", 0.0)
        section = index.section(best_id) if best_id and best >= min_confidence else None
        out.append(
            PackAnswer(
                key=f"{line_index}:{j}",
                line_index=line_index,
                t_start=line.t_start,
                question=sentence,
                section=section,
                score=best,
            )
        )
    return out


def answer_transcript(
    lines: list[TranscriptLine],
    index: InfoPackIndex,
    min_confidence: float,
    dismissed: set[str] | None = None,
) -> list[PackAnswer]:
    """Every pack answer over a transcript, in call order, minus the keys the interviewer waved off."""
    dismissed = dismissed or set()
    out: list[PackAnswer] = []
    for i, line in enumerate(lines):
        out.extend(a for a in answer_line(line, i, index, min_confidence) if a.key not in dismissed)
    return out
