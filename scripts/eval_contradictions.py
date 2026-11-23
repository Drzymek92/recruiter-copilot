"""Measure the analyser's contradiction RECALL + PRECISION against the seeded ground truth (#965).

The mock interview (``tests/fixtures/mock_interview.json``) plants THREE cross-document
contradictions, each against a distinct source document:

    a) cv               — "two years of LLM work" said, but the CV dates the first LLM project to 2025
    b) previous_summary — "Sam zbudowałem cały ten system" (sole ownership), but the HR screen
                           recorded a team effort of three
    c) cover_letter     — the cover letter claims "I led its evaluation programme", but the answer
                           describes only informal user impressions

This harness runs the REAL analyser (``analyse``) against a chosen model + transcript + prompt
variant, then scores the surviving contradictions with a PURE, deterministic scorer:

    RECALL    = (distinct seeded contradictions caught) / 3
    PRECISION = (distinct seeded contradictions caught) / (total contradictions returned)

The scorer never calls an LLM (fw:D2 determinism-first): a returned contradiction matches a seed
only when its ``source_doc`` matches AND its text carries one of that seed's discriminating anchors
(verbatim substring or a light-noise fuzzy hit). The anchors are the load-bearing part — a spurious
finding that merely cites the right document (e.g. an English-fluency claim against the cover letter)
does NOT match the evaluation seed and is scored as a false positive.

Usage (the live model call needs a reachable Ollama; run it through the GPU lease board).
``--model`` defaults to qwen3:14b, the analyser's baseline (D23); llama3.1:8b is the older
comparison model:

    python scripts/eval_contradictions.py --model qwen3:14b   --transcript ground_truth
    python scripts/eval_contradictions.py --model qwen3:14b   --transcript stt --json
    python scripts/eval_contradictions.py --model llama3.1:8b --transcript ground_truth  # older
    python scripts/eval_contradictions.py --gen-stt tests/fixtures/stt   # one-time: transcribe WAV
    python scripts/eval_contradictions.py --negatives offline   # #1994 known non-conflicts, no model
    python scripts/eval_contradictions.py --negatives live      # same cases, real analyser

Every run starts from an EMPTY Ollama prompt cache by default (#2011): at temperature 0 the findings
depend on the cache state (#2006), so the model is unloaded before each run (and once before the
``--negatives live`` batch). ``--warm`` restores the old behaviour; the mode is part of the
checkpoint key and is recorded in each run's metadata.

Every multi-run sweep also records a PAIRED ablation of the #1994 non-conflict filter (the same
runs scored with the filter off = kept + filter-rejected), so before/after is free of run noise.

The pure scorer (``score``, ``match_contradiction``, ``SEEDED``, ``build_ground_truth_lines``) is
imported and unit-tested offline in ``tests/test_eval_contradictions.py`` — no model, no GPU.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from recruiter_copilot.analysis import (  # noqa: E402
    ANALYSIS_SYSTEM,
    THINK_OPEN,
    RejectedContradiction,
    _extract_json,
    _normalize,
    _partial_ratio,
    analyse,
    analyse_question,
    filter_non_conflicts,
    strip_think_blocks,
    verify_contradiction,
)
from recruiter_copilot.llm import ChatReply  # noqa: E402
from recruiter_copilot.models import Contradiction, Question, Speaker, TranscriptLine  # noqa: E402
from recruiter_copilot.store import load_session  # noqa: E402

FIXTURE_JSON = PROJECT / "tests" / "fixtures" / "mock_interview.json"
FIXTURE_WAV = PROJECT / "tests" / "fixtures" / "mock_interview.wav"
SAMPLE_SESSION = PROJECT / "examples" / "sample_session"
DEFAULT_STT_DIR = PROJECT / "tests" / "fixtures" / "stt"

# Fuzzy threshold for an anchor that is not a verbatim substring of the finding text (light STT
# noise / dropped diacritics). Tight enough that an unrelated phrase does not clear it.
ANCHOR_FUZZY_THRESHOLD = 0.85


# ── the seeded ground truth (pure data) ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class SeededContradiction:
    """One planted contradiction. ``anchors`` are discriminating phrases; a returned finding
    matches this seed only when its ``source_doc`` matches AND its text carries one anchor."""

    key: str
    source_doc: str
    description: str
    anchors: tuple[str, ...]


SEEDED: tuple[SeededContradiction, ...] = (
    SeededContradiction(
        key="cv_two_years",
        source_doc="cv",
        description="Claims 2 years of LLM work; CV dates the first LLM project to 2025",
        anchors=(
            "od dwóch lat",
            "od dwoch lat",
            "dwóch lat",
            "dwoch lat",
            "dwa lata",
            "two years",
            "2 years",
            "first llm project in 2025",
            "2025",
        ),
    ),
    SeededContradiction(
        key="previous_summary_sole_ownership",
        source_doc="previous_summary",
        description="Claims sole ownership; HR screen recorded a team effort of three",
        anchors=(
            "sam zbudowałem",
            "sam zbudowalem",
            "cały ten system",
            "team effort of three",
            "team of three",
            "effort of three",
            "sole ownership",
            "on my own",
            "by myself",
            "solo",
        ),
    ),
    SeededContradiction(
        key="cover_letter_evaluation",
        source_doc="cover_letter",
        description="Cover letter claims leading an evaluation programme; answer is informal only",
        anchors=(
            "led its evaluation programme",
            "evaluation programme",
            "evaluation program",
            "led the evaluation",
            "opinie użytkowników",
            "opinie uzytkownikow",
            "user impressions",
            "user feedback",
        ),
    ),
)


# ── known NEGATIVE cases (#1994): non-conflicts the analyser reported on a real call ──────────
#
# Each is a SYNTHETIC paraphrase of a false positive seen on the first real two-person call (the
# real candidate's lines stay in the gitignored sessions/ folder — D18/SI1). Any contradiction the
# analyser reports for one of these is a FALSE POSITIVE by construction. They are benched two ways:
#   offline (``check_negatives_offline``) — the canned model finding goes through the real D17
#     verifier + the #1994 non-conflict filter; it must ground (so the filter, not D17, is what
#     rejects it) and must be rejected. Pure, no model; unit-tested.
#   live (``--negatives``) — the real analyser runs on each case's mini transcript + docs; every
#     surviving contradiction counts as FP, and the pre-filter count shows what the filter caught.


@dataclass(frozen=True)
class NegativeCase:
    key: str
    description: str
    question_id: str
    question_text: str
    answer: tuple[tuple[float, float, str], ...]  # (t_start, t_end, candidate text)
    docs: dict[str, str]
    finding: dict  # the model-shaped false finding (paraphrase of the real one)
    expected_rule: str


NEGATIVE_CASES: tuple[NegativeCase, ...] = (
    NegativeCase(
        key="omission_test_set_size",
        description="Doc names the eval-set size; the answer simply does not repeat it (absence)",
        question_id="n1",
        question_text="Jak oceniałeś jakość systemu przed wdrożeniem?",
        answer=(
            (
                10.0,
                18.0,
                "Mierzyliśmy trafność odpowiedzi, ich spójność i czas odpowiedzi systemu.",
            ),
        ),
        docs={
            "previous_summary": "Candidate described an offline evaluation run on a set of "
            "200 test questions before each release.",
        },
        finding={
            "claim": "Size of the test set (200 questions) is in the documents but not mentioned "
            "in the answer",
            "source_doc": "previous_summary",
            "source_quote": "an offline evaluation run on a set of 200 test questions",
            "transcript_quote": {
                "quote": "Mierzyliśmy trafność odpowiedzi, ich spójność i czas odpowiedzi systemu.",
                "t_start": 10.0,
                "t_end": 18.0,
            },
            "severity": "medium",
            "explanation": "The summary records a 200-question set; the answer omits it.",
        },
        expected_rule="omission",
    ),
    NegativeCase(
        key="aspiration_vs_no_experience",
        description="A learning goal now vs 'no experience' earlier — consistent, not a conflict",
        question_id="n2",
        question_text="Czego chciałbyś się nauczyć na tym stanowisku?",
        answer=(
            (
                30.0,
                38.0,
                "Liczę tu na zdobycie umiejętności uruchamiania modeli na Kubernetesie.",
            ),
        ),
        docs={
            "earlier_answer:n0": "Szczerze mówiąc, nie mam doświadczenia z uruchamianiem modeli "
            "na Kubernetesie.",
        },
        finding={
            "claim": "Kubernetes deployment experience",
            "source_doc": "earlier_answer:n0",
            "source_quote": "nie mam doświadczenia z uruchamianiem modeli na Kubernetesie",
            "transcript_quote": {
                "quote": "zdobycie umiejętności uruchamiania modeli na Kubernetesie",
                "t_start": 30.0,
                "t_end": 38.0,
            },
            "severity": "high",
            "explanation": "Earlier admitted no Kubernetes experience, now claims the skill.",
        },
        expected_rule="aspiration_vs_no_experience",
    ),
)


def negative_case_lines(case: NegativeCase) -> list[TranscriptLine]:
    return [
        TranscriptLine(t_start=t0, t_end=t1, text=text, speaker=Speaker.CANDIDATE, lang="pl")
        for t0, t1, text in case.answer
    ]


def check_negatives_offline() -> list[dict]:
    """Pure: run each canned false finding through D17 grounding + the #1994 filter.

    Per case: ``grounded`` (D17 kept it — so the filter is what must catch it), ``rejected_by``
    (the filter rule that dropped it, or ``None``), and ``fp`` (it would reach the report).
    """
    out: list[dict] = []
    for case in NEGATIVE_CASES:
        c = verify_contradiction(case.finding, negative_case_lines(case), case.docs)
        rule: str | None = None
        if c is not None:
            _kept, rejected = filter_non_conflicts([c], case.question_id)
            rule = rejected[0].rule if rejected else None
        out.append(
            {
                "key": case.key,
                "grounded": c is not None,
                "rejected_by": rule,
                "expected_rule": case.expected_rule,
                "fp": c is not None and rule is None,
            }
        )
    return out


def _anchor_hit(anchor: str, blob: str) -> bool:
    """A verbatim (normalized) substring, or a light-noise fuzzy hit above the threshold."""
    na = _normalize(anchor)
    if not na:
        return False
    if na in blob:
        return True
    return _partial_ratio(na, blob) >= ANCHOR_FUZZY_THRESHOLD


def _finding_blob(c: Contradiction) -> str:
    """All the free text a contradiction carries, normalized into one searchable string."""
    parts = [
        c.claim,
        c.source_quote,
        c.transcript_quote.quote if c.transcript_quote else "",
        c.explanation,
    ]
    return _normalize(" ".join(p for p in parts if p))


def match_contradiction(c: Contradiction, seed: SeededContradiction) -> bool:
    """Pure: does returned contradiction ``c`` match the seeded contradiction ``seed``?

    Requires the source document to match AND at least one discriminating anchor to appear in the
    finding's text. Source-doc alone is not enough — a finding may cite the right document about the
    wrong thing (e.g. English fluency vs the cover letter), which must not be scored as a catch.
    """
    if _normalize(c.source_doc) != _normalize(seed.source_doc):
        return False
    blob = _finding_blob(c)
    return any(_anchor_hit(a, blob) for a in seed.anchors)


# ── the pure scorer ────────────────────────────────────────────────────────────────────────


@dataclass
class EvalScore:
    total_findings: int
    matched_seeds: list[str] = field(default_factory=list)
    missed_seeds: list[str] = field(default_factory=list)
    true_positive_findings: int = 0
    false_positive_findings: int = 0
    per_finding: list[dict] = field(default_factory=list)

    @property
    def recall(self) -> float:
        return round(len(self.matched_seeds) / len(SEEDED), 3)

    @property
    def precision(self) -> float | None:
        """Task #965 definition: (distinct seeded caught) / (total findings)."""
        if self.total_findings == 0:
            return None
        return round(len(self.matched_seeds) / self.total_findings, 3)

    def as_dict(self) -> dict:
        return {
            "recall": self.recall,
            "precision": self.precision,
            "seeded_total": len(SEEDED),
            "caught": len(self.matched_seeds),
            "total_findings": self.total_findings,
            "true_positive_findings": self.true_positive_findings,
            "false_positive_findings": self.false_positive_findings,
            "matched_seeds": self.matched_seeds,
            "missed_seeds": self.missed_seeds,
            "per_finding": self.per_finding,
        }


def score(contradictions: list[Contradiction]) -> EvalScore:
    """Pure, deterministic scoring of the analyser's contradictions against ``SEEDED``.

    RECALL = distinct seeds matched / 3. PRECISION = distinct seeds matched / total findings.
    A finding is a true positive if it matches ANY seed; a false positive if it matches none.
    """
    matched: dict[str, bool] = {}
    tp = fp = 0
    per_finding: list[dict] = []
    for c in contradictions:
        hit_keys = [s.key for s in SEEDED if match_contradiction(c, s)]
        if hit_keys:
            tp += 1
            for k in hit_keys:
                matched[k] = True
        else:
            fp += 1
        per_finding.append(
            {
                "source_doc": c.source_doc,
                "claim": c.claim,
                "transcript_quote": c.transcript_quote.quote if c.transcript_quote else "",
                "matched_seeds": hit_keys,
                "is_true_positive": bool(hit_keys),
            }
        )
    matched_keys = [s.key for s in SEEDED if matched.get(s.key)]
    missed_keys = [s.key for s in SEEDED if not matched.get(s.key)]
    return EvalScore(
        total_findings=len(contradictions),
        matched_seeds=matched_keys,
        missed_seeds=missed_keys,
        true_positive_findings=tp,
        false_positive_findings=fp,
        per_finding=per_finding,
    )


# ── multi-run aggregation (pure, no LLM, no I/O) ───────────────────────────────────────────────


@dataclass
class AggregateScore:
    """Aggregate of N independent ``EvalScore`` runs of the SAME config (model/transcript/prompt).

    Summarises temp-0 noise (±1 contradiction) without hand-looping: mean recall/precision plus
    their spread, how often each seed was caught across the runs, and the false-positive spread.
    Precision is undefined for a run that returned zero findings; those runs are excluded from the
    precision statistics (``precision_runs`` says how many contributed).
    """

    n_runs: int
    mean_recall: float
    min_recall: float
    max_recall: float
    stddev_recall: float
    precision_runs: int
    mean_precision: float | None
    min_precision: float | None
    max_precision: float | None
    stddev_precision: float | None
    per_seed_caught: dict[str, int]
    per_seed_catch_freq: dict[str, float]
    mean_false_positives: float
    min_false_positives: int
    max_false_positives: int
    mean_true_positives: float
    mean_total_findings: float

    def as_dict(self) -> dict:
        return {
            "n_runs": self.n_runs,
            "seeded_total": len(SEEDED),
            "mean_recall": self.mean_recall,
            "recall": {
                "mean": self.mean_recall,
                "min": self.min_recall,
                "max": self.max_recall,
                "stddev": self.stddev_recall,
            },
            "mean_precision": self.mean_precision,
            "precision": {
                "runs": self.precision_runs,
                "mean": self.mean_precision,
                "min": self.min_precision,
                "max": self.max_precision,
                "stddev": self.stddev_precision,
            },
            "per_seed_caught": self.per_seed_caught,
            "per_seed_catch_freq": self.per_seed_catch_freq,
            "false_positives": {
                "mean": self.mean_false_positives,
                "min": self.min_false_positives,
                "max": self.max_false_positives,
            },
            "mean_true_positives": self.mean_true_positives,
            "mean_total_findings": self.mean_total_findings,
        }


def aggregate(scores: list[EvalScore]) -> AggregateScore:
    """Pure, deterministic aggregation of a list of per-run ``EvalScore`` objects.

    No LLM, no I/O — feed it synthetic ``EvalScore`` lists to unit-test it. Example: runs catching
    ``{a,b}``, ``{a,c}``, ``{a,c}`` → mean recall 0.667; per-seed frequency a=3/3, b=1/3, c=2/3.
    """
    n = len(scores)
    if n == 0:
        raise ValueError("aggregate() needs at least one EvalScore")

    recalls = [s.recall for s in scores]
    precisions = [s.precision for s in scores if s.precision is not None]
    fps = [s.false_positive_findings for s in scores]
    tps = [s.true_positive_findings for s in scores]
    totals = [s.total_findings for s in scores]

    per_seed_caught = {seed.key: 0 for seed in SEEDED}
    for sc in scores:
        for key in sc.matched_seeds:
            if key in per_seed_caught:
                per_seed_caught[key] += 1
    per_seed_catch_freq = {k: round(v / n, 3) for k, v in per_seed_caught.items()}

    have_prec = len(precisions) > 0
    return AggregateScore(
        n_runs=n,
        mean_recall=round(statistics.fmean(recalls), 3),
        min_recall=min(recalls),
        max_recall=max(recalls),
        stddev_recall=round(statistics.pstdev(recalls), 3),
        precision_runs=len(precisions),
        mean_precision=round(statistics.fmean(precisions), 3) if have_prec else None,
        min_precision=min(precisions) if have_prec else None,
        max_precision=max(precisions) if have_prec else None,
        stddev_precision=round(statistics.pstdev(precisions), 3) if have_prec else None,
        per_seed_caught=per_seed_caught,
        per_seed_catch_freq=per_seed_catch_freq,
        mean_false_positives=round(statistics.fmean(fps), 3),
        min_false_positives=min(fps),
        max_false_positives=max(fps),
        mean_true_positives=round(statistics.fmean(tps), 3),
        mean_total_findings=round(statistics.fmean(totals), 3),
    )


# ── checkpoint / resume (survive a lease-wall timeout mid-sweep, #1000) ────────────────────────

# A long qwen3:14b sweep (~115-190 s/run) can overrun a 10-min GPU-lease wall and lose the whole
# run set. The checkpoint STREAMS each completed run to a JSONL file the instant it finishes, keyed
# by the sweep's identifying params (model/transcript/prompt); a later invocation reuses the finished
# runs and only executes the remainder. The default (no checkpoint) path is unchanged.

DEFAULT_CKPT_DIR = PROJECT / "scripts" / "outputs"


class CheckpointMismatch(Exception):
    """A checkpoint file exists but was written for a different sweep config (model/transcript/prompt).

    Refuse to silently mix runs from two different configs into one aggregate — the caller either
    points at the right file or passes ``--fresh`` to overwrite it.
    """

    def __init__(self, path: Path, wanted: dict, found: dict) -> None:
        self.path = path
        self.wanted = wanted
        self.found = found
        super().__init__(
            f"checkpoint {path} was written for "
            f"model={found.get('model')!r} transcript={found.get('transcript')!r} "
            f"prompt={found.get('prompt')!r} cache={found.get('cache')!r}, but this run wants "
            f"model={wanted.get('model')!r} transcript={wanted.get('transcript')!r} "
            f"prompt={wanted.get('prompt')!r} cache={wanted.get('cache')!r}. "
            f"Point at a different --checkpoint, or pass --fresh to overwrite it."
        )


def cache_mode(cold: bool) -> str:
    """The Ollama prompt-cache mode label a run/sweep is recorded under (#2011)."""
    return "cold" if cold else "warm"


def sweep_params(
    model: str,
    transcript_kind: str,
    prompt: str,
    cold: bool = True,
    reasoning_effort: str = "",
    second_pass: bool = False,
) -> dict:
    """The identifying params of a sweep — what a checkpoint is keyed on.

    The cache mode is part of the key (#2011): a cold and a warm sweep measure different things
    (#2006), so a checkpoint from one mode is never resumed in the other. A checkpoint written
    before #2011 carries no ``cache`` key and is therefore refused too (``CheckpointMismatch``).
    A set ``reasoning_effort`` (#2012) is added to the key; unset, the key is exactly as before, so
    existing thinking-on checkpoints still resume and a thinking-off one never mixes with them.
    The #999 ``second_pass`` arm is keyed the same way: added only when on.
    """
    params: dict[str, object] = {
        "model": model,
        "transcript": transcript_kind,
        "prompt": prompt,
        "cache": cache_mode(cold),
    }
    if reasoning_effort:
        params["reasoning_effort"] = reasoning_effort
    if second_pass:
        params["second_pass"] = True
    return params


def default_checkpoint_path(
    model: str,
    transcript_kind: str,
    prompt: str,
    cold: bool = True,
    reasoning_effort: str = "",
    second_pass: bool = False,
) -> Path:
    """A stable per-config checkpoint path under ``scripts/outputs/`` (used by ``--resume``).

    The cache mode is in the name (#2011), so a cold and a warm ``--resume`` never share a file;
    a set ``reasoning_effort`` (#2012) or an on ``second_pass`` (#999) is appended, so the default
    name is unchanged.
    """
    raw = f"{model}_{transcript_kind}_{prompt}_{cache_mode(cold)}"
    if reasoning_effort:
        raw += f"_re-{reasoning_effort}"
    if second_pass:
        raw += "_second-pass"
    slug = re.sub(r"[^A-Za-z0-9]+", "-", raw).strip("-")
    return DEFAULT_CKPT_DIR / f"eval_ckpt_{slug}.jsonl"


def _evalscore_to_dict(sc: EvalScore) -> dict:
    """The persisted fields of an ``EvalScore`` (``recall``/``precision`` are recomputed on load)."""
    return {
        "total_findings": sc.total_findings,
        "matched_seeds": sc.matched_seeds,
        "missed_seeds": sc.missed_seeds,
        "true_positive_findings": sc.true_positive_findings,
        "false_positive_findings": sc.false_positive_findings,
        "per_finding": sc.per_finding,
    }


def _evalscore_from_dict(d: dict) -> EvalScore:
    return EvalScore(
        total_findings=d["total_findings"],
        matched_seeds=list(d.get("matched_seeds", [])),
        missed_seeds=list(d.get("missed_seeds", [])),
        true_positive_findings=d.get("true_positive_findings", 0),
        false_positive_findings=d.get("false_positive_findings", 0),
        per_finding=list(d.get("per_finding", [])),
    )


def append_checkpoint(path: Path, run_index: int, params: dict, sc: EvalScore, meta: dict) -> None:
    """Append ONE finished run to the checkpoint and flush immediately (stream, do not buffer)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "run_index": run_index,
        "params": params,
        "score": _evalscore_to_dict(sc),
        "meta": meta,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()


def load_checkpoint(path: Path, params: dict) -> list[tuple[int, EvalScore, dict]]:
    """Read completed runs from ``path``, ordered by run index (last write wins per index).

    Returns ``[]`` when the file is absent (fresh sweep). Raises ``CheckpointMismatch`` if any
    record was written for a different config, so a stale checkpoint is never silently reused.
    """
    if not path.exists():
        return []
    by_index: dict[int, tuple[EvalScore, dict]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        rec = json.loads(raw)
        if rec.get("params") != params:
            raise CheckpointMismatch(path, params, rec.get("params", {}))
        by_index[int(rec["run_index"])] = (
            _evalscore_from_dict(rec["score"]),
            rec.get("meta", {}),
        )
    return [(idx, sc, meta) for idx, (sc, meta) in sorted(by_index.items())]


# ── transcripts ──────────────────────────────────────────────────────────────────────────────


def build_ground_truth_lines() -> list[TranscriptLine]:
    """The clean transcript built straight from the fixture's ground-truth turns (no acoustic model)."""
    turns = json.loads(FIXTURE_JSON.read_text(encoding="utf-8"))["turns"]
    return [
        TranscriptLine(
            t_start=turn["t_start"],
            t_end=turn["t_end"],
            text=turn["text"],
            speaker=Speaker(turn["speaker"]),
            lang=turn["lang"],
        )
        for turn in turns
    ]


def load_stt_lines(folder: Path) -> list[TranscriptLine]:
    """Load a previously generated STT transcript (pipeline ``transcript.json`` format)."""
    from recruiter_copilot.pipeline import load_transcript  # noqa: PLC0415

    return load_transcript(folder)


# ── prompt variants ─────────────────────────────────────────────────────────────────────────

# The precision-tuned SYSTEM prompt. As of #965 (apply) this was FOLDED INTO analysis.py's default
# ANALYSIS_SYSTEM, so `--prompt default` and `--prompt tight` now run the same text — the copy is
# kept here as an explicit benchmark of what was landed. Aims at the precision problem: report a
# contradiction only for a genuine factual conflict, not for vagueness, omission, or elaboration.
TIGHT_SYSTEM = (
    "You analyse ONE interview answer against the candidate's own documents for an interviewer. "
    "You are decision support, not a decision maker. Return STRICT JSON only — no prose, no code "
    "fences. Ground EVERY claim in verbatim quotes: copy transcript text exactly as shown, keeping "
    "its t_start and t_end numbers. NEVER invent a quote. "
    "Report a contradiction ONLY when a specific statement in the answer DIRECTLY and FACTUALLY "
    "conflicts with a specific statement in a source document — a different number, date, role, "
    "scope, or ownership. Do NOT report vagueness, a missing detail, weak evidence, elaboration, "
    "or a topic the documents simply do not mention: those are NOT contradictions and must be left "
    "out. If in doubt, do not report it. For a contradiction, name the source document (one of: "
    "cv, cover_letter, previous_summary, job, earlier_answer:<qid>) and quote BOTH sides verbatim. "
    "If the answer supports nothing, return empty lists and score null."
)

# Completeness experiments (#982). The co-located q2 miss (cv-timeline + previous_summary ownership
# in ONE answer) is a per-answer salience problem, not a missing category — ANALYSIS_SYSTEM already
# names "ownership". Two deltas were benched:
#
#   COMPLETE_SYSTEM (heavy): "report EVERY conflict" + a strong ownership example. It SEESAWED — it
#     drove previous_summary to 5/5 but crashed cv_two_years to 0/5 and cover_letter to 1/5 and
#     precision to ~0.37 (FP ~2). The heavy ownership emphasis just relocated the blind spot and the
#     "report every" clause manufactured false positives. Kept only as a documented negative control.
#
#   MULTI_SYSTEM (light): one balanced sentence — if an answer holds more than one distinct conflict,
#     report each separately — with ownership named only as a parenthetical peer of number/date, and
#     NO "report every" language. Benched to see whether a minimal nudge lifts the co-located seed
#     without the seesaw.
COMPLETE_SYSTEM = ANALYSIS_SYSTEM + (
    " A single answer may contain SEVERAL distinct contradictions: examine every factual claim in "
    "the answer separately and report EVERY genuine conflict you find, not only the first or most "
    "obvious one. In particular, an ownership or authorship claim — the answer says the candidate "
    "built or did something alone or by themselves, while a document records a team or shared effort "
    "— is a contradiction in its own right, and must still be reported even when the same answer "
    "already contains another (e.g. a number or date) contradiction. Report each on its own; do not "
    "merge two distinct conflicts into one finding."
)

MULTI_SYSTEM = ANALYSIS_SYSTEM + (
    " If a single answer contains more than one distinct factual conflict (for example a wrong number "
    "or date AND a mismatched ownership or role claim), report each as its own separate contradiction "
    "rather than only the most obvious one."
)

PROMPT_VARIANTS = {
    "default": ANALYSIS_SYSTEM,
    "tight": TIGHT_SYSTEM,
    "complete": COMPLETE_SYSTEM,
    "multi": MULTI_SYSTEM,
}

# `strip_think_blocks` lives in recruiter_copilot.analysis (#2020): production `_ask_model` strips
# too, and this wrapper reuses that one copy (MOD). It stays re-exported here for the eval's tests.


class _SystemOverrideProvider:
    """Wrap a ChatProvider: force a chosen SYSTEM prompt and strip reasoning ``<think>`` blocks.

    Lets the eval try a prompt variant WITHOUT editing analysis.py (the analyser keeps calling
    ``complete(prompt, system=ANALYSIS_SYSTEM, ...)``; we substitute). The strip here duplicates the
    one ``_ask_model`` now does (#2020) — harmless (idempotent) and it is what feeds the
    ``think_blocks_seen`` / ``parse_failures`` counters.
    """

    def __init__(self, inner: object, system: str) -> None:
        self._inner = inner
        self._system = system
        self.name = getattr(inner, "name", "unknown")
        self.model_name = getattr(inner, "model_name", "unknown")
        self.think_blocks_seen = 0
        # #2012 cost/robustness counters: output tokens per call (thinking included — Ollama counts
        # it) and replies the analyser's JSON extractor could not parse (→ an empty finding).
        self.output_tokens: list[int] = []
        self.parse_failures = 0

    def complete(
        self, prompt: str, system: str | None = None, max_tokens: int | None = None
    ):  # noqa: ARG002
        reply = self._inner.complete(prompt, system=self._system, max_tokens=max_tokens)
        text = getattr(reply, "text", "")
        if THINK_OPEN in text:
            self.think_blocks_seen += 1
        cleaned = strip_think_blocks(text)
        self.output_tokens.append(int(getattr(reply, "output_tokens", 0) or 0))
        if not _extract_json(cleaned):
            self.parse_failures += 1
        if isinstance(reply, ChatReply):
            reply.text = cleaned
            return reply
        return ChatReply(text=cleaned, model=self.model_name, provider="wrapped")


# ── live orchestration (the only place a model is called) ─────────────────────────────────────


def build_provider(model: str, reasoning_effort: str = ""):
    """A local Ollama provider pinned to ``model`` (config default is not touched).

    ``reasoning_effort`` (#2012) is passed through only when set; unset, the provider sends exactly
    the request it always did.
    """
    from recruiter_copilot.config import load_settings  # noqa: PLC0415
    from recruiter_copilot.llm import OllamaProvider  # noqa: PLC0415

    overrides: dict[str, object] = {"ollama_model": model}
    if reasoning_effort:
        overrides["ollama_reasoning_effort"] = reasoning_effort
    settings = load_settings(overrides=overrides, env_file=Path("/dev/null"))
    return OllamaProvider(settings)


def findings_signature(sc: EvalScore) -> str:
    """Pure: an order-independent fingerprint of a run's scored findings (#2012 identical-runs)."""
    rows = sorted(
        json.dumps([f["source_doc"], f["claim"], f["transcript_quote"]], ensure_ascii=False)
        for f in sc.per_finding
    )
    return "|".join(rows)


def _token_stats(provider: _SystemOverrideProvider) -> dict:
    """Per-run cost counters from the wrapper (#2012)."""
    toks = provider.output_tokens
    return {
        "calls": len(toks),
        "output_tokens_total": sum(toks),
        "output_tokens_mean": round(statistics.fmean(toks), 1) if toks else None,
        "output_tokens_max": max(toks) if toks else None,
        "parse_failures": provider.parse_failures,
    }


def reset_cache_if_cold(provider: object, cold: bool) -> bool | None:
    """Start the next analyse batch from an EMPTY Ollama prompt cache when ``cold`` (#2011).

    Reuses ``OllamaProvider.reset_prompt_cache`` (#2006; never raises). Returns its result
    (True = model unloaded / was not loaded, False = reset failed, run is NOT guaranteed cold), or
    ``None`` when warm mode skipped the reset or the provider has no reset hook.
    """
    reset = getattr(provider, "reset_prompt_cache", None)
    if not cold or reset is None:
        return None
    ok = bool(reset())
    if not ok:
        print(
            "[cache] warning: prompt-cache reset failed; this run may not be cold", file=sys.stderr
        )
    return ok


def _run_once(
    model: str,
    transcript_kind: str,
    prompt: str,
    stt_dir: Path = DEFAULT_STT_DIR,
    cold: bool = True,
    reasoning_effort: str = "",
    second_pass: bool = False,
) -> tuple[EvalScore, dict]:
    """LIVE single run: run the real analyser once, returning the pure ``EvalScore`` + run metadata.

    ``second_pass`` (#999) turns on the analyser's exclusion re-ask; it is recorded in the metadata.

    With ``cold`` (the default, #2011) the model is unloaded first, so every run starts from an
    empty prompt cache and N runs of one config are comparable; ``cold=False`` keeps the old
    behaviour (the cache state is whatever the previous run left — #2006). ``reasoning_effort``
    (#2012) is sent only when set (``"none"`` = qwen3 thinking off).
    """
    session = load_session(SAMPLE_SESSION)
    if transcript_kind == "ground_truth":
        lines = build_ground_truth_lines()
    elif transcript_kind == "stt":
        lines = load_stt_lines(stt_dir)
    else:
        raise ValueError(f"unknown transcript kind {transcript_kind!r}")

    base = build_provider(model, reasoning_effort=reasoning_effort)
    system = PROMPT_VARIANTS[prompt]
    provider = _SystemOverrideProvider(base, system)
    cache_reset = reset_cache_if_cold(base, cold)

    t0 = time.perf_counter()
    result = analyse(session, lines, provider, second_pass=second_pass)
    elapsed = time.perf_counter() - t0

    scored = score(result.contradictions())
    meta = {
        "model": model,
        "transcript": transcript_kind,
        "prompt": prompt,
        "cache": cache_mode(cold),
        "cache_reset": cache_reset,
        "reasoning_effort": reasoning_effort,
        "second_pass": second_pass,
        "seconds": round(elapsed, 2),
        "transcript_lines": len(lines),
        # #999: which question each surviving contradiction came from (where a seed is missed).
        "per_question": [
            [qa.question_id, c.source_doc] for qa in result.analyses for c in qa.contradictions
        ],
        "think_blocks_seen": provider.think_blocks_seen,
        "analyses": len(result.analyses),
        "no_answer_summaries": sum(
            1 for a in result.analyses if "no answer" in (a.summary or "").lower()
        ),
    }
    meta["findings_signature"] = findings_signature(scored)
    meta.update(_token_stats(provider))
    meta.update(filter_ablation(result.contradictions(), result.rejected_contradictions))
    return scored, meta


def filter_ablation(kept: list[Contradiction], rejected: list[RejectedContradiction]) -> dict:
    """Pure: score the SAME run with the #1994 non-conflict filter off (kept + rejected) — a paired
    ablation, free of temp-0 run-to-run noise — and list what the filter dropped, by rule."""
    unfiltered = score(kept + [r.contradiction for r in rejected])
    return {
        "unfiltered": {
            "recall": unfiltered.recall,
            "precision": unfiltered.precision,
            "false_positive_findings": unfiltered.false_positive_findings,
            "matched_seeds": unfiltered.matched_seeds,
        },
        "filter_rejected": [
            {
                "rule": r.rule,
                "question_id": r.question_id,
                "source_doc": r.contradiction.source_doc,
                "claim": r.contradiction.claim,
                "was_seed": any(match_contradiction(r.contradiction, s) for s in SEEDED),
            }
            for r in rejected
        ],
    }


def run_negatives_live(
    model: str,
    prompt: str,
    cold: bool = True,
    reasoning_effort: str = "",
    second_pass: bool = False,
) -> list[dict]:
    """LIVE: run the real analyser on each NEGATIVE case. Every reported contradiction is an FP;
    ``raw`` (grounded, pre-filter) shows what the #1994 filter had to catch. Like ``analyse`` in
    the CLI, the cache is reset ONCE before the batch when ``cold`` (#2011)."""
    base = build_provider(model, reasoning_effort=reasoning_effort)
    provider = _SystemOverrideProvider(base, PROMPT_VARIANTS[prompt])
    reset_cache_if_cold(base, cold)
    # #999: passed only when on, so the default call is exactly the pre-#999 one.
    extra: dict[str, object] = {"second_pass": True} if second_pass else {}
    out: list[dict] = []
    for case in NEGATIVE_CASES:
        question = Question(id=case.question_id, text={"pl": case.question_text})
        rejected: list[RejectedContradiction] = []
        qa = analyse_question(
            question,
            negative_case_lines(case),
            case.docs,
            provider,
            "pl",
            rejected=rejected,
            **extra,
        )
        out.append(
            {
                "key": case.key,
                "raw_grounded": len(qa.contradictions) + len(rejected),
                "rejected": [r.rule for r in rejected],
                "fp": len(qa.contradictions),
                "reported_claims": [c.claim for c in qa.contradictions],
            }
        )
    return out


def run_eval(
    model: str,
    transcript_kind: str,
    prompt: str,
    stt_dir: Path = DEFAULT_STT_DIR,
    cold: bool = True,
    reasoning_effort: str = "",
    second_pass: bool = False,
) -> dict:
    """LIVE: run the real analyser on ``model`` + transcript + prompt variant, then score it."""
    scored, meta = _run_once(
        model,
        transcript_kind,
        prompt,
        stt_dir=stt_dir,
        cold=cold,
        reasoning_effort=reasoning_effort,
        second_pass=second_pass,
    )
    out = scored.as_dict()
    out.update(meta)
    return out


def run_eval_multi(
    model: str,
    transcript_kind: str,
    prompt: str,
    runs: int,
    stt_dir: Path = DEFAULT_STT_DIR,
    checkpoint: Path | None = None,
    run_once: Callable[..., tuple[EvalScore, dict]] = _run_once,
    cold: bool = True,
    reasoning_effort: str = "",
    second_pass: bool = False,
) -> tuple[AggregateScore, list[dict]]:
    """LIVE: run the same config ``runs`` times and aggregate the per-run scores.

    Each run starts from an empty Ollama prompt cache unless ``cold=False`` (#2011), so the N runs
    are comparable. With ``checkpoint`` set, each finished run is streamed to that JSONL file as
    soon as it lands, and an existing matching checkpoint is RESUMED — its completed runs are
    reused and only the remainder is executed (so a sweep survives a lease-wall timeout /
    interruption). A checkpoint written for a different config — including the other cache mode —
    raises ``CheckpointMismatch`` (never silently reused). The final aggregate spans the full set
    (checkpointed + new). ``run_once`` is injectable for offline tests.
    """
    params = sweep_params(
        model,
        transcript_kind,
        prompt,
        cold=cold,
        reasoning_effort=reasoning_effort,
        second_pass=second_pass,
    )
    # Passed only when set, so an injected ``run_once`` written before #2012/#999 keeps working.
    extra: dict[str, object] = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
    if second_pass:
        extra["second_pass"] = True
    scores: list[EvalScore] = []
    metas: list[dict] = []

    resumed = 0
    if checkpoint is not None:
        for _idx, sc, meta in load_checkpoint(checkpoint, params):  # may raise CheckpointMismatch
            scores.append(sc)
            metas.append(meta)
        resumed = len(scores)
        if resumed:
            print(f"[checkpoint] resuming {checkpoint}: {resumed} run(s) reused")

    for i in range(resumed, runs):
        sc, meta = run_once(model, transcript_kind, prompt, stt_dir=stt_dir, cold=cold, **extra)
        scores.append(sc)
        metas.append(meta)
        if checkpoint is not None:
            append_checkpoint(checkpoint, i, params, sc, meta)
    return aggregate(scores), metas


def gen_stt(out_dir: Path, model: str | None = None, bias: bool = False) -> Path:
    """One-time: transcribe the fixture WAV with faster-whisper and save a pipeline transcript.

    Produces ``<out_dir>/transcript.json`` (+ .txt), which ``--transcript stt`` then reads. This is
    a live GPU step — run it through the lease board. ``bias=True`` turns on the #976 term-biasing
    (``STT_BIAS_PROMPT``): the decode is primed with the session's vocabulary + ownership phrasing.
    """
    from recruiter_copilot.config import load_settings  # noqa: PLC0415
    from recruiter_copilot.pipeline import save_transcript, transcribe_recording  # noqa: PLC0415

    overrides: dict[str, object] = {}
    if model:
        overrides["stt_model"] = model
    if bias:
        overrides["stt_bias_prompt"] = True
    settings = load_settings(overrides=overrides, env_file=Path("/dev/null"))
    session = load_session(SAMPLE_SESSION)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines, stats, _decoded = transcribe_recording(session, FIXTURE_WAV, settings)
    paths = save_transcript(out_dir, lines, stats, [])
    print(f"STT transcript written: {paths['json']}  ({len(lines)} lines, {stats.stt_provider})")
    for ln in lines:
        print(f"  [{ln.t_start:6.2f}-{ln.t_end:6.2f}] {ln.speaker.value:11} ({ln.lang}): {ln.text}")
    return paths["json"]


def aggregate_ablation(metas: list[dict]) -> dict | None:
    """Pure: the filter-OFF side of the paired ablation across runs (``None`` for runs recorded
    before #1994, whose metadata carries no ``unfiltered`` block)."""
    runs = [m for m in metas if "unfiltered" in m]
    if not runs:
        return None
    precisions = [m["unfiltered"]["precision"] for m in runs]
    precisions = [p for p in precisions if p is not None]
    by_rule: dict[str, int] = {}
    for m in runs:
        for r in m.get("filter_rejected", []):
            by_rule[r["rule"]] = by_rule.get(r["rule"], 0) + 1
    return {
        "runs": len(runs),
        "unfiltered_mean_recall": round(
            statistics.fmean(m["unfiltered"]["recall"] for m in runs), 3
        ),
        "unfiltered_mean_precision": (
            round(statistics.fmean(precisions), 3) if precisions else None
        ),
        "unfiltered_mean_false_positives": round(
            statistics.fmean(m["unfiltered"]["false_positive_findings"] for m in runs), 3
        ),
        "rejected_by_rule": by_rule,
        "rejected_seeds": sum(
            1 for m in runs for r in m.get("filter_rejected", []) if r.get("was_seed")
        ),
    }


def aggregate_cost(metas: list[dict]) -> dict | None:
    """Pure: output tokens per call + parse failures across runs (``None`` for runs recorded
    before #2012, whose metadata carries no token counters)."""
    runs = [m for m in metas if "output_tokens_total" in m]
    if not runs:
        return None
    calls = sum(m["calls"] for m in runs)
    return {
        "runs": len(runs),
        "calls": calls,
        "output_tokens_per_call_mean": (
            round(sum(m["output_tokens_total"] for m in runs) / calls, 1) if calls else None
        ),
        "output_tokens_per_call_max": max(
            (m["output_tokens_max"] for m in runs if m["output_tokens_max"] is not None),
            default=None,
        ),
        "parse_failures": sum(m["parse_failures"] for m in runs),
        "no_answer_summaries": sum(m.get("no_answer_summaries", 0) for m in runs),
    }


def identical_runs(metas: list[dict]) -> bool | None:
    """Pure: True when every run produced the same per-finding records (``None`` if not recorded).

    Compares the scored findings (source doc, claim, TP/FP), which is what the eval grades.
    """
    sigs = [m.get("findings_signature") for m in metas]
    if any(s is None for s in sigs):
        return None
    return all(s == sigs[0] for s in sigs)


def _emit_aggregate(agg: AggregateScore, metas: list[dict], as_json: bool) -> None:
    """Print (or JSON-dump) the multi-run aggregate. Only reached when ``--runs`` > 1."""
    model = metas[0]["model"]
    transcript = metas[0]["transcript"]
    prompt = metas[0]["prompt"]
    cache = metas[0].get("cache", "unrecorded")  # runs checkpointed before #2011 carry no mode
    mean_seconds = round(statistics.fmean(m["seconds"] for m in metas), 2)
    ablation = aggregate_ablation(metas)
    cost = aggregate_cost(metas)
    effort = metas[0].get("reasoning_effort") or "unset"

    if as_json:
        out = agg.as_dict()
        out.update(
            {
                "model": model,
                "transcript": transcript,
                "prompt": prompt,
                "cache": cache,
                "reasoning_effort": effort,
                "second_pass": bool(metas[0].get("second_pass")),
                "per_question_by_run": [m.get("per_question") for m in metas],
                "cache_reset_failures": sum(1 for m in metas if m.get("cache_reset") is False),
                "mean_seconds": mean_seconds,
                "cost": cost,
                "identical_runs": identical_runs(metas),
                "filter_ablation": ablation,
            }
        )
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    def _fmt(v: float | None) -> str:
        return "n/a" if v is None else f"{v:.3f}"

    print(
        f"model={model}  transcript={transcript}  prompt={prompt}  cache={cache}  "
        f"reasoning_effort={effort}  x{agg.n_runs} runs ({mean_seconds}s mean)"
    )
    if cost is not None:
        print(
            f"  output tokens/call mean={cost['output_tokens_per_call_mean']}  "
            f"max={cost['output_tokens_per_call_max']}  parse failures={cost['parse_failures']}  "
            f"'no answer' summaries={cost['no_answer_summaries']}"
        )
    print(
        f"  RECALL     mean={agg.mean_recall:.3f}  min={agg.min_recall:.3f} "
        f"max={agg.max_recall:.3f}  sd={agg.stddev_recall:.3f}"
    )
    print(
        f"  PRECISION  mean={_fmt(agg.mean_precision)}  min={_fmt(agg.min_precision)} "
        f"max={_fmt(agg.max_precision)}  sd={_fmt(agg.stddev_precision)}  "
        f"(defined in {agg.precision_runs}/{agg.n_runs} runs)"
    )
    print("  per-seed catch frequency:")
    for seed in SEEDED:
        caught = agg.per_seed_caught[seed.key]
        freq = agg.per_seed_catch_freq[seed.key]
        print(f"    {seed.key:34} {caught}/{agg.n_runs} ({freq:.3f})")
    print(
        f"  false positives:  mean={agg.mean_false_positives:.3f}  "
        f"min={agg.min_false_positives}  max={agg.max_false_positives}"
    )
    print(
        f"  (mean TP={agg.mean_true_positives:.3f}  "
        f"mean total findings={agg.mean_total_findings:.3f})"
    )
    if ablation is not None:
        print(
            f"  filter OFF (same runs): recall={ablation['unfiltered_mean_recall']:.3f}  "
            f"precision={_fmt(ablation['unfiltered_mean_precision'])}  "
            f"FP={ablation['unfiltered_mean_false_positives']:.3f}  "
            f"rejected={ablation['rejected_by_rule']}  seeds rejected={ablation['rejected_seeds']}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Contradiction recall/precision eval (#965)")
    ap.add_argument("--model", default="qwen3:14b", help="Ollama model tag")
    ap.add_argument("--transcript", choices=["ground_truth", "stt"], default="ground_truth")
    ap.add_argument("--prompt", choices=list(PROMPT_VARIANTS), default="default")
    ap.add_argument(
        "--stt-dir", default=str(DEFAULT_STT_DIR), help="folder holding STT transcript.json"
    )
    ap.add_argument(
        "--gen-stt", metavar="OUT_DIR", default=None, help="transcribe the WAV and exit"
    )
    ap.add_argument("--stt-model", default=None, help="STT model tag for --gen-stt")
    ap.add_argument(
        "--stt-bias",
        action="store_true",
        help="enable #976 term-biasing when generating the STT transcript (--gen-stt)",
    )
    ap.add_argument(
        "--runs",
        type=int,
        default=1,
        help="evaluate the chosen config N times and report the mean/spread (default 1)",
    )
    ap.add_argument(
        "--checkpoint",
        metavar="PATH",
        default=None,
        help="stream each finished run to this JSONL file and RESUME from it if it already holds "
        "runs for this exact model/transcript/prompt (reuse them, run only the remainder). Lets a "
        "long sweep survive a lease-wall timeout. A checkpoint from a different config is refused.",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="shorthand for --checkpoint <auto>: use a stable per-config checkpoint path under "
        "scripts/outputs/ (resume it if present, else create it)",
    )
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="ignore and overwrite any existing checkpoint for this config (start the sweep clean)",
    )
    ap.add_argument(
        "--negatives",
        choices=["offline", "live"],
        default=None,
        help="bench the #1994 known NEGATIVE cases instead of the seeded fixture: 'offline' runs "
        "the canned false findings through D17 + the non-conflict filter (no model); 'live' runs "
        "the real analyser on each case (every reported contradiction is an FP)",
    )
    ap.add_argument(
        "--warm",
        action="store_true",
        help="do NOT unload the model before each run (pre-#2011 behaviour). By default every run "
        "(and the --negatives live batch) starts from an empty Ollama prompt cache, because at "
        "temperature 0 the findings depend on the cache state (#2006). The mode is part of the "
        "checkpoint key: a checkpoint from the other mode, or one written before #2011 (no mode "
        "recorded), is refused with a mismatch error — pass --fresh or a new --checkpoint.",
    )
    ap.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high"],
        default="",
        help="send reasoning_effort to the model (#2012): 'none' turns qwen3 thinking OFF on "
        "Ollama's OpenAI endpoint; omitted = not sent (today's request, thinking on). Part of the "
        "checkpoint key when set.",
    )
    ap.add_argument(
        "--second-pass",
        action="store_true",
        help="turn on the analyser's #999 exclusion re-ask (after an answer's first call finds a "
        "contradiction, ask once more without the docs it cites). Part of the checkpoint key when "
        "set; omitted = today's one call per answer.",
    )
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    cold = not args.warm
    effort = args.reasoning_effort
    second = args.second_pass

    if args.gen_stt:
        gen_stt(Path(args.gen_stt), model=args.stt_model, bias=args.stt_bias)
        return 0

    if args.negatives:
        rows = (
            check_negatives_offline()
            if args.negatives == "offline"
            else run_negatives_live(
                args.model, args.prompt, cold=cold, reasoning_effort=effort, second_pass=second
            )
        )
        total_fp = sum(int(r["fp"]) for r in rows)
        if args.json:
            print(json.dumps({"negatives": rows, "false_positives": total_fp}, indent=2))
        else:
            for r in rows:
                print(f"  {r['key']:32} {json.dumps({k: v for k, v in r.items() if k != 'key'})}")
            print(f"negatives={len(rows)}  FP={total_fp}")
        return 0 if total_fp == 0 else 1

    if args.runs < 1:
        ap.error("--runs must be >= 1")

    checkpoint_path: Path | None = None
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    elif args.resume:
        checkpoint_path = default_checkpoint_path(
            args.model,
            args.transcript,
            args.prompt,
            cold=cold,
            reasoning_effort=effort,
            second_pass=second,
        )

    if checkpoint_path is not None and args.fresh and checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"[checkpoint] --fresh: removed existing {checkpoint_path}")

    # A checkpoint implies a sweep: route through the aggregating path even for a single run.
    if args.runs > 1 or checkpoint_path is not None:
        try:
            agg, metas = run_eval_multi(
                args.model,
                args.transcript,
                args.prompt,
                args.runs,
                stt_dir=Path(args.stt_dir),
                checkpoint=checkpoint_path,
                cold=cold,
                reasoning_effort=effort,
                second_pass=second,
            )
        except CheckpointMismatch as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        _emit_aggregate(agg, metas, as_json=args.json)
        return 0

    out = run_eval(
        args.model,
        args.transcript,
        args.prompt,
        stt_dir=Path(args.stt_dir),
        cold=cold,
        reasoning_effort=effort,
        second_pass=second,
    )
    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0

    prec = "n/a" if out["precision"] is None else f"{out['precision']:.3f}"
    print(
        f"model={out['model']}  transcript={out['transcript']}  prompt={out['prompt']}  "
        f"({out['seconds']}s, {out['transcript_lines']} lines)"
    )
    print(
        f"  RECALL={out['recall']:.3f} ({out['caught']}/{out['seeded_total']})   "
        f"PRECISION={prec} ({out['caught']}/{out['total_findings']} findings)   "
        f"TP={out['true_positive_findings']} FP={out['false_positive_findings']}"
    )
    if out["think_blocks_seen"]:
        print(f"  (stripped <think> blocks from {out['think_blocks_seen']} model replies)")
    print(f"  caught:  {out['matched_seeds']}")
    print(f"  missed:  {out['missed_seeds']}")
    for f in out["per_finding"]:
        mark = "TP" if f["is_true_positive"] else "FP"
        print(f"    [{mark}] {f['source_doc']:16} :: {f['claim'][:70]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
