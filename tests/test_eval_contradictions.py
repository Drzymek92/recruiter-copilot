"""Tests for the #965 contradiction eval harness.

The PURE scorer (match/score/anchors/think-stripping) is tested offline with no LLM and no GPU.
The live-model run is a single opt-in test gated behind ``RECRUITER_LIVE_LLM`` (same pattern as
``test_analysis.py``), so the normal ``pytest`` stays offline and this one skips.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import eval_contradictions as ev  # noqa: E402

from recruiter_copilot.models import Contradiction, Evidence, Severity  # noqa: E402


def _contradiction(
    source_doc: str,
    claim: str,
    source_quote: str,
    transcript_quote: str,
    explanation: str = "",
) -> Contradiction:
    return Contradiction(
        claim=claim,
        source_doc=source_doc,
        source_quote=source_quote,
        transcript_quote=Evidence(quote=transcript_quote, t_start=28.73, t_end=39.7),
        severity=Severity.MEDIUM,
        explanation=explanation,
    )


# ── the three real seeded findings each match their seed ──────────────────────────────────────


def test_cv_two_years_finding_matches_its_seed():
    c = _contradiction(
        "cv",
        "Says two years of LLM experience",
        "the first LLM project in 2025",
        "Pracuję z modelami językowymi od dwóch lat.",
    )
    seed = next(s for s in ev.SEEDED if s.key == "cv_two_years")
    assert ev.match_contradiction(c, seed) is True


def test_sole_ownership_finding_matches_its_seed():
    c = _contradiction(
        "previous_summary",
        "Claims sole ownership of the system",
        "a team effort of three",
        "Sam zbudowałem cały ten system",
    )
    seed = next(s for s in ev.SEEDED if s.key == "previous_summary_sole_ownership")
    assert ev.match_contradiction(c, seed) is True


def test_evaluation_finding_matches_its_seed():
    c = _contradiction(
        "cover_letter",
        "Claims to have led an evaluation programme",
        "I led its evaluation programme",
        "Patrzyliśmy głównie na opinie użytkowników.",
    )
    seed = next(s for s in ev.SEEDED if s.key == "cover_letter_evaluation")
    assert ev.match_contradiction(c, seed) is True


def test_noised_stt_two_years_still_matches():
    """Dropped diacritics on the Polish cue must still match (fuzzy anchor)."""
    c = _contradiction(
        "cv",
        "two years claimed",
        "first LLM project in 2025",
        "pracuje z modelami jezykowymi od dwoch lat",
    )
    seed = next(s for s in ev.SEEDED if s.key == "cv_two_years")
    assert ev.match_contradiction(c, seed) is True


# ── discrimination: right document, wrong topic is NOT a catch ────────────────────────────────


def test_cover_letter_english_fluency_finding_is_not_the_evaluation_seed():
    """A finding citing the cover letter about English fluency must not match the evaluation seed."""
    c = _contradiction(
        "cover_letter",
        "Claims fluent English but the CV self-assesses B2",
        "I am fluent in English",
        "Byłem starszym inżynierem i prowadziłem zespół.",
    )
    seed = next(s for s in ev.SEEDED if s.key == "cover_letter_evaluation")
    assert ev.match_contradiction(c, seed) is False


def test_wrong_source_doc_never_matches():
    c = _contradiction(
        "job",
        "Says two years",
        "3+ years production Python",
        "Pracuję z modelami językowymi od dwóch lat.",
    )
    for seed in ev.SEEDED:
        assert ev.match_contradiction(c, seed) is False


# ── the aggregate scorer ──────────────────────────────────────────────────────────────────────


def test_score_perfect_recall_and_precision():
    findings = [
        _contradiction(
            "cv",
            "two years",
            "first LLM project in 2025",
            "Pracuję z modelami językowymi od dwóch lat.",
        ),
        _contradiction(
            "previous_summary",
            "sole ownership",
            "a team effort of three",
            "Sam zbudowałem cały ten system",
        ),
        _contradiction(
            "cover_letter",
            "led evaluation",
            "I led its evaluation programme",
            "Patrzyliśmy głównie na opinie użytkowników.",
        ),
    ]
    s = ev.score(findings)
    assert s.recall == 1.0
    assert s.precision == 1.0
    assert s.total_findings == 3
    assert s.true_positive_findings == 3 and s.false_positive_findings == 0
    assert set(s.matched_seeds) == {
        "cv_two_years",
        "previous_summary_sole_ownership",
        "cover_letter_evaluation",
    }


def test_score_over_firing_drops_precision_recall_partial():
    """2/3 caught + spurious over-firing: recall 0.667, precision = 2 seeded / 5 findings."""
    findings = [
        _contradiction("cv", "two years", "first LLM project in 2025", "od dwóch lat"),
        _contradiction(
            "cover_letter",
            "led evaluation",
            "I led its evaluation programme",
            "opinie użytkowników",
        ),
        # spurious 1: right doc, wrong topic
        _contradiction(
            "cover_letter",
            "claims fluent English",
            "I am fluent in English",
            "prowadziłem zespół",
        ),
        # spurious 2 + 3: unrelated docs
        _contradiction("job", "x", "3+ years production Python", "cache"),
        _contradiction("cv", "junior role", "Company Y — Junior Developer", "starszym inżynierem"),
    ]
    s = ev.score(findings)
    assert s.recall == round(2 / 3, 3)
    assert s.precision == round(2 / 5, 3)
    assert s.total_findings == 5
    assert s.true_positive_findings == 2 and s.false_positive_findings == 3
    assert set(s.matched_seeds) == {"cv_two_years", "cover_letter_evaluation"}
    assert s.missed_seeds == ["previous_summary_sole_ownership"]


def test_score_no_findings_precision_is_none():
    s = ev.score([])
    assert s.recall == 0.0
    assert s.precision is None
    assert s.total_findings == 0


# ── multi-run aggregation (pure, no LLM) ──────────────────────────────────────────────────────


def _evalscore(caught: set[str], total_findings: int | None = None, fp: int = 0) -> ev.EvalScore:
    """Build a synthetic EvalScore for a run that caught the seeds in ``caught``.

    ``fp`` spurious findings are added; ``total_findings`` defaults to true + false positives.
    """
    matched = [s.key for s in ev.SEEDED if s.key in caught]
    missed = [s.key for s in ev.SEEDED if s.key not in caught]
    tp = len(matched)
    if total_findings is None:
        total_findings = tp + fp
    return ev.EvalScore(
        total_findings=total_findings,
        matched_seeds=matched,
        missed_seeds=missed,
        true_positive_findings=tp,
        false_positive_findings=fp,
    )


A = "cv_two_years"
B = "previous_summary_sole_ownership"
C = "cover_letter_evaluation"


def test_aggregate_mean_recall_and_per_seed_frequency():
    """Task-spec example: runs catch {a,b}, {a,c}, {a,c} → mean recall 0.667,
    per-seed frequency a=3/3, b=1/3, c=2/3."""
    scores = [_evalscore({A, B}), _evalscore({A, C}), _evalscore({A, C})]
    agg = ev.aggregate(scores)
    assert agg.n_runs == 3
    assert agg.mean_recall == 0.667
    assert agg.per_seed_caught == {A: 3, B: 1, C: 2}
    assert agg.per_seed_catch_freq == {A: 1.0, B: 0.333, C: 0.667}


def test_aggregate_recall_spread():
    """Spread reported as min/max and (population) stddev over the runs."""
    scores = [_evalscore(set()), _evalscore({A, B, C})]  # recall 0.0 and 1.0
    agg = ev.aggregate(scores)
    assert agg.min_recall == 0.0
    assert agg.max_recall == 1.0
    assert agg.mean_recall == 0.5
    assert agg.stddev_recall == 0.5  # pstdev([0.0, 1.0]) == 0.5


def test_aggregate_false_positive_distribution():
    scores = [_evalscore({A}, fp=0), _evalscore({A}, fp=2), _evalscore({A}, fp=4)]
    agg = ev.aggregate(scores)
    assert agg.min_false_positives == 0
    assert agg.max_false_positives == 4
    assert agg.mean_false_positives == 2.0


def test_aggregate_precision_ignores_undefined_runs():
    """A run with zero findings has precision None and is excluded from the precision stats."""
    scores = [
        _evalscore({A, B, C}),  # 3/3 findings all TP -> precision 1.0
        _evalscore(set(), total_findings=0),  # no findings -> precision None
    ]
    agg = ev.aggregate(scores)
    assert agg.precision_runs == 1
    assert agg.mean_precision == 1.0
    assert agg.min_precision == 1.0 and agg.max_precision == 1.0


def test_aggregate_all_precision_undefined_is_none():
    scores = [_evalscore(set(), total_findings=0), _evalscore(set(), total_findings=0)]
    agg = ev.aggregate(scores)
    assert agg.precision_runs == 0
    assert agg.mean_precision is None
    assert agg.min_precision is None and agg.max_precision is None
    assert agg.stddev_precision is None


def test_aggregate_single_run_has_zero_spread():
    agg = ev.aggregate([_evalscore({A, C}, fp=1)])
    assert agg.n_runs == 1
    assert agg.mean_recall == round(2 / 3, 3)
    assert agg.stddev_recall == 0.0
    assert agg.min_false_positives == agg.max_false_positives == 1


def test_aggregate_empty_raises():
    with pytest.raises(ValueError):
        ev.aggregate([])


def test_aggregate_as_dict_shape():
    agg = ev.aggregate([_evalscore({A, B}), _evalscore({A, C})])
    d = agg.as_dict()
    assert d["n_runs"] == 2
    assert set(d["per_seed_catch_freq"]) == {A, B, C}
    assert "mean_recall" in d and "mean_precision" in d
    assert "false_positives" in d and set(d["false_positives"]) >= {
        "mean",
        "min",
        "max",
    }


# ── reasoning-model think-block stripping (qwen3) ─────────────────────────────────────────────


def test_strip_think_blocks_removes_reasoning_and_keeps_json():
    raw = '<think>The candidate said X { but } really...</think>\n{"summary": "ok", "score": 3}'
    cleaned = ev.strip_think_blocks(raw)
    assert "<think>" not in cleaned and "really" not in cleaned
    assert cleaned.startswith("{") and cleaned.endswith("}")


def test_strip_think_blocks_handles_unterminated_block():
    raw = "prefix\n<think>reasoning that never closed because the reply was truncated"
    assert ev.strip_think_blocks(raw) == "prefix"


def test_strip_think_blocks_noop_without_blocks():
    raw = '{"summary": "ok"}'
    assert ev.strip_think_blocks(raw) == raw


# ── ground-truth transcript builder ───────────────────────────────────────────────────────────


def test_build_ground_truth_lines_matches_the_fixture():
    lines = ev.build_ground_truth_lines()
    assert len(lines) == 12  # 6 interviewer + 6 candidate turns in the fixture
    assert any("Sam zbudowałem" in ln.text for ln in lines)


# ── prompt variants (pure; #982 benched negative controls) ─────────────────────────────────────


def test_prompt_variants_registered():
    # default/tight are the #965 pair; complete/multi are the #982 completeness benchmarks.
    assert set(ev.PROMPT_VARIANTS) == {"default", "tight", "complete", "multi"}
    assert ev.PROMPT_VARIANTS["default"] is ev.ANALYSIS_SYSTEM


def test_completeness_variants_extend_the_default_prompt():
    # Both #982 variants are ANALYSIS_SYSTEM plus an appended completeness clause — they never
    # weaken the tight precision gate, they only add text after it. (They still seesawed the other
    # seeds in eval, so neither was promoted to the default; they stay as documented controls.)
    for key in ("complete", "multi"):
        text = ev.PROMPT_VARIANTS[key]
        assert text.startswith(ev.ANALYSIS_SYSTEM)
        assert len(text) > len(ev.ANALYSIS_SYSTEM)


# ── checkpoint / resume mechanics (pure, no LLM/GPU — #1000) ───────────────────────────────────


def _fake_run_once(calls: list, per_run: list) -> object:
    """A stub for ``_run_once``: records each call and returns the next canned (EvalScore, meta).

    Injected via ``run_eval_multi(..., run_once=...)`` so the resume mechanics are proven with NO
    Ollama and NO GPU. Indexing ``per_run`` by call count makes an unexpected extra call IndexError.
    """

    def _fake(model, transcript_kind, prompt, stt_dir=ev.DEFAULT_STT_DIR, cold=True):
        idx = len(calls)
        calls.append((model, transcript_kind, prompt))
        return per_run[idx], {
            "model": model,
            "transcript": transcript_kind,
            "prompt": prompt,
            "cache": ev.cache_mode(cold),
            "seconds": 1.0,
        }

    return _fake


def test_checkpoint_streams_each_finished_run(tmp_path):
    ckpt = tmp_path / "ck.jsonl"
    calls: list = []
    per_run = [_evalscore({A, B}), _evalscore({A, C}), _evalscore({A, C})]
    agg, metas = ev.run_eval_multi(
        "m", "ground_truth", "default", 3, checkpoint=ckpt, run_once=_fake_run_once(calls, per_run)
    )
    assert len(calls) == 3
    assert agg.n_runs == 3 and len(metas) == 3
    recs = [json.loads(ln) for ln in ckpt.read_text().splitlines() if ln.strip()]
    assert [r["run_index"] for r in recs] == [0, 1, 2]
    assert all(
        r["params"]
        == {"model": "m", "transcript": "ground_truth", "prompt": "default", "cache": "cold"}
        for r in recs
    )


def test_resume_reuses_checkpointed_runs_and_executes_only_remainder(tmp_path):
    ckpt = tmp_path / "ck.jsonl"
    # First sweep: 2 runs completed and streamed to the checkpoint.
    ev.run_eval_multi(
        "m",
        "ground_truth",
        "default",
        2,
        checkpoint=ckpt,
        run_once=_fake_run_once([], [_evalscore({A, B}), _evalscore({A, C})]),
    )
    # Resume toward 3 runs: exactly ONE new run should touch the model; runs 0-1 are reused.
    calls2: list = []
    agg, metas = ev.run_eval_multi(
        "m",
        "ground_truth",
        "default",
        3,
        checkpoint=ckpt,
        run_once=_fake_run_once(calls2, [_evalscore({A, C})]),
    )
    assert len(calls2) == 1  # only the remaining run executed
    assert agg.n_runs == 3  # aggregate spans reused + new
    assert len(metas) == 3
    assert len([ln for ln in ckpt.read_text().splitlines() if ln.strip()]) == 3


def test_fully_checkpointed_sweep_executes_nothing(tmp_path):
    ckpt = tmp_path / "ck.jsonl"
    ev.run_eval_multi(
        "m",
        "ground_truth",
        "default",
        2,
        checkpoint=ckpt,
        run_once=_fake_run_once([], [_evalscore({A}), _evalscore({A, B})]),
    )
    calls: list = []
    agg, _metas = ev.run_eval_multi(
        "m", "ground_truth", "default", 2, checkpoint=ckpt, run_once=_fake_run_once(calls, [])
    )
    assert calls == []  # every run reused, model never called
    assert agg.n_runs == 2


def test_mismatched_checkpoint_is_rejected_not_reused(tmp_path):
    ckpt = tmp_path / "ck.jsonl"
    ev.run_eval_multi(
        "modelA",
        "ground_truth",
        "default",
        2,
        checkpoint=ckpt,
        run_once=_fake_run_once([], [_evalscore({A}), _evalscore({A})]),
    )
    # A different model against the same file must refuse and execute nothing.
    calls: list = []
    with pytest.raises(ev.CheckpointMismatch):
        ev.run_eval_multi(
            "modelB",
            "ground_truth",
            "default",
            2,
            checkpoint=ckpt,
            run_once=_fake_run_once(calls, [_evalscore({A}), _evalscore({A})]),
        )
    assert calls == []


def test_mismatched_prompt_also_rejected(tmp_path):
    ckpt = tmp_path / "ck.jsonl"
    params = ev.sweep_params("m", "ground_truth", "default")
    ev.append_checkpoint(ckpt, 0, params, _evalscore({A}), {"seconds": 1.0})
    with pytest.raises(ev.CheckpointMismatch):
        ev.load_checkpoint(ckpt, ev.sweep_params("m", "ground_truth", "tight"))


def test_load_checkpoint_absent_returns_empty(tmp_path):
    params = ev.sweep_params("m", "ground_truth", "default")
    assert ev.load_checkpoint(tmp_path / "nope.jsonl", params) == []


def test_evalscore_checkpoint_roundtrip(tmp_path):
    ckpt = tmp_path / "ck.jsonl"
    params = ev.sweep_params("m", "stt", "tight")
    sc = _evalscore({A, C}, total_findings=4, fp=2)
    ev.append_checkpoint(ckpt, 0, params, sc, {"seconds": 3.0})
    loaded = ev.load_checkpoint(ckpt, params)
    assert len(loaded) == 1
    idx, sc2, meta = loaded[0]
    assert idx == 0
    assert sc2.matched_seeds == sc.matched_seeds
    assert sc2.total_findings == sc.total_findings
    assert sc2.false_positive_findings == sc.false_positive_findings
    assert sc2.recall == sc.recall and sc2.precision == sc.precision
    assert meta["seconds"] == 3.0


def test_load_checkpoint_dedups_by_run_index_last_write_wins(tmp_path):
    ckpt = tmp_path / "ck.jsonl"
    params = ev.sweep_params("m", "ground_truth", "default")
    ev.append_checkpoint(ckpt, 0, params, _evalscore({A}), {"seconds": 1.0})
    ev.append_checkpoint(ckpt, 0, params, _evalscore({A, B}), {"seconds": 2.0})  # rewrite idx 0
    loaded = ev.load_checkpoint(ckpt, params)
    assert len(loaded) == 1
    _idx, sc, meta = loaded[0]
    assert set(sc.matched_seeds) == {A, B}  # last write wins
    assert meta["seconds"] == 2.0


def test_default_checkpoint_path_is_config_specific(tmp_path):
    p1 = ev.default_checkpoint_path("qwen3:14b", "ground_truth", "default")
    p2 = ev.default_checkpoint_path("llama3.2:3b", "ground_truth", "default")
    p3 = ev.default_checkpoint_path("qwen3:14b", "ground_truth", "tight")
    assert p1 != p2 and p1 != p3
    assert p1.suffix == ".jsonl"
    assert "qwen3-14b" in p1.name  # ':' is slugified out of the filename


# ── cold-cache eval runs (#2011, pure — fake provider + fake analyser, no Ollama/GPU) ──────────


class _FakeResettableProvider:
    """Stands in for ``OllamaProvider``: records every cache reset into a shared event log."""

    name = "fake"
    model_name = "fake-model"

    def __init__(self, events: list) -> None:
        self._events = events

    def reset_prompt_cache(self) -> bool:
        self._events.append("reset")
        return True

    def complete(self, prompt, system=None, max_tokens=None):  # pragma: no cover — never reached
        raise AssertionError("the fake analyser must not call the model")


@pytest.fixture
def fake_live(monkeypatch):
    """Patch the live seams of the eval (provider + analyser) and return the shared event log."""
    from types import SimpleNamespace

    from recruiter_copilot.analysis import AnalysisResult

    events: list = []
    monkeypatch.setattr(ev, "build_provider", lambda model, **_kw: _FakeResettableProvider(events))

    def _fake_analyse(session, lines, provider, **_kw):
        events.append("analyse")
        return AnalysisResult()

    def _fake_analyse_question(question, lines, docs, provider, lang, rejected=None):
        events.append("analyse_question")
        return SimpleNamespace(contradictions=[])

    monkeypatch.setattr(ev, "analyse", _fake_analyse)
    monkeypatch.setattr(ev, "analyse_question", _fake_analyse_question)
    return events


def test_eval_resets_cache_once_per_run_by_default(fake_live):
    agg, metas = ev.run_eval_multi("m", "ground_truth", "default", 3)
    assert agg.n_runs == 3
    # Exactly one reset per run, each immediately BEFORE that run's analyse batch.
    assert fake_live == ["reset", "analyse"] * 3
    assert all(m["cache"] == "cold" and m["cache_reset"] is True for m in metas)


def test_eval_warm_mode_never_resets(fake_live):
    _agg, metas = ev.run_eval_multi("m", "ground_truth", "default", 2, cold=False)
    assert fake_live == ["analyse", "analyse"]
    assert all(m["cache"] == "warm" and m["cache_reset"] is None for m in metas)


def test_single_run_eval_is_cold_by_default(fake_live):
    out = ev.run_eval("m", "ground_truth", "default")
    assert fake_live == ["reset", "analyse"]
    assert out["cache"] == "cold"


def test_live_negatives_reset_once_before_the_batch_and_respect_warm(fake_live):
    ev.run_negatives_live("m", "default")
    n = len(ev.NEGATIVE_CASES)
    assert fake_live == ["reset"] + ["analyse_question"] * n
    fake_live.clear()
    ev.run_negatives_live("m", "default", cold=False)
    assert fake_live == ["analyse_question"] * n


# ── #2012: reasoning_effort arm + cost counters (offline) ──────────────────────────────────────


def test_sweep_params_unchanged_when_reasoning_effort_unset():
    """Existing thinking-on checkpoints must still resume: the unset key is exactly pre-#2012."""
    assert ev.sweep_params("m", "ground_truth", "default") == {
        "model": "m",
        "transcript": "ground_truth",
        "prompt": "default",
        "cache": "cold",
    }
    off = ev.sweep_params("m", "ground_truth", "default", reasoning_effort="none")
    assert off["reasoning_effort"] == "none"


def test_checkpoint_from_the_other_thinking_arm_is_rejected(tmp_path):
    ckpt = tmp_path / "ckpt.jsonl"
    ev.run_eval_multi(
        "m",
        "ground_truth",
        "default",
        1,
        checkpoint=ckpt,
        run_once=_fake_run_once([], [_evalscore({A})]),
    )
    with pytest.raises(ev.CheckpointMismatch):
        ev.run_eval_multi(
            "m",
            "ground_truth",
            "default",
            1,
            checkpoint=ckpt,
            run_once=_fake_run_once([], [_evalscore({A})]),
            reasoning_effort="none",
        )


def test_default_checkpoint_path_unchanged_unset_and_distinct_when_set():
    base = ev.default_checkpoint_path("qwen3:14b", "ground_truth", "default")
    assert base.name == "eval_ckpt_qwen3-14b-ground-truth-default-cold.jsonl"
    off = ev.default_checkpoint_path(
        "qwen3:14b", "ground_truth", "default", reasoning_effort="none"
    )
    assert off != base and "re-none" in off.name


def test_run_eval_multi_passes_reasoning_effort_to_each_run_only_when_set():
    seen: list = []

    def run_once(model, transcript_kind, prompt, stt_dir=None, cold=True, **kw):
        seen.append(kw)
        return _evalscore({A}), {
            "model": model,
            "transcript": transcript_kind,
            "prompt": prompt,
            "seconds": 1.0,
        }

    ev.run_eval_multi("m", "ground_truth", "default", 1, run_once=run_once)
    ev.run_eval_multi("m", "ground_truth", "default", 1, run_once=run_once, reasoning_effort="none")
    assert seen == [{}, {"reasoning_effort": "none"}]


# ── #999: second-pass arm (offline) ─────────────────────────────────────────────────────────────


def test_second_pass_arm_keys_checkpoint_only_when_on():
    base = ev.sweep_params("m", "ground_truth", "default")
    assert "second_pass" not in base  # pre-#999 checkpoints still resume
    assert ev.sweep_params("m", "ground_truth", "default", second_pass=True) == {
        **base,
        "second_pass": True,
    }
    off_path = ev.default_checkpoint_path("qwen3:14b", "ground_truth", "default")
    on_path = ev.default_checkpoint_path("qwen3:14b", "ground_truth", "default", second_pass=True)
    assert off_path.name == "eval_ckpt_qwen3-14b-ground-truth-default-cold.jsonl"
    assert on_path != off_path and "second-pass" in on_path.name


def test_checkpoint_from_the_single_pass_arm_is_rejected_by_the_second_pass_arm(tmp_path):
    ckpt = tmp_path / "ckpt.jsonl"
    ev.run_eval_multi(
        "m",
        "ground_truth",
        "default",
        1,
        checkpoint=ckpt,
        run_once=_fake_run_once([], [_evalscore({A})]),
    )
    with pytest.raises(ev.CheckpointMismatch):
        ev.run_eval_multi(
            "m",
            "ground_truth",
            "default",
            1,
            checkpoint=ckpt,
            run_once=_fake_run_once([], [_evalscore({A})]),
            second_pass=True,
        )


def test_run_eval_multi_passes_second_pass_only_when_on():
    seen: list = []

    def run_once(model, transcript_kind, prompt, stt_dir=None, cold=True, **kw):
        seen.append(kw)
        return _evalscore({A}), {"model": model, "transcript": transcript_kind, "prompt": prompt}

    ev.run_eval_multi("m", "ground_truth", "default", 1, run_once=run_once)
    ev.run_eval_multi("m", "ground_truth", "default", 1, run_once=run_once, second_pass=True)
    assert seen == [{}, {"second_pass": True}]


def test_build_provider_wires_reasoning_effort_and_leaves_default_unset():
    assert ev.build_provider("qwen3:14b").reasoning_effort == ""
    assert ev.build_provider("qwen3:14b", reasoning_effort="none").reasoning_effort == "none"


def test_eval_run_records_reasoning_effort_and_routes_it_to_the_provider(fake_live, monkeypatch):
    efforts: list = []
    inner = ev.build_provider  # the fixture's fake

    def build(model, **kw):
        efforts.append(kw.get("reasoning_effort", ""))
        return inner(model, **kw)

    monkeypatch.setattr(ev, "build_provider", build)
    _agg, metas = ev.run_eval_multi("m", "ground_truth", "default", 1, reasoning_effort="none")
    assert efforts == ["none"]
    assert metas[0]["reasoning_effort"] == "none"
    assert metas[0]["calls"] == 0 and metas[0]["parse_failures"] == 0


def test_wrapper_counts_output_tokens_and_parse_failures():
    class _Inner:
        name = "fake"
        model_name = "fake"

        def __init__(self):
            self.replies = [
                ev.ChatReply('{"summary": "ok"}', "m", "p", output_tokens=700),
                ev.ChatReply("not json at all", "m", "p", output_tokens=50),
            ]

        def complete(self, prompt, system=None, max_tokens=None):
            return self.replies.pop(0)

    w = ev._SystemOverrideProvider(_Inner(), "SYS")
    w.complete("a")
    w.complete("b")
    assert w.output_tokens == [700, 50]
    assert w.parse_failures == 1
    stats = ev._token_stats(w)
    assert stats == {
        "calls": 2,
        "output_tokens_total": 750,
        "output_tokens_mean": 375.0,
        "output_tokens_max": 700,
        "parse_failures": 1,
    }


def test_wrapper_counts_think_blocks_and_strips_them_via_the_production_function():
    """#2020: the strip moved to analysis.py; the wrapper's counter + cleaned text still work."""

    class _Inner:
        name = "fake"
        model_name = "fake"

        def __init__(self):
            self.replies = [
                ev.ChatReply('<think>{"bogus": 1}</think>{"summary": "ok"}', "m", "p"),
                ev.ChatReply('{"summary": "plain"}', "m", "p"),
            ]

        def complete(self, prompt, system=None, max_tokens=None):
            return self.replies.pop(0)

    w = ev._SystemOverrideProvider(_Inner(), "SYS")
    assert w.complete("a").text == '{"summary": "ok"}'
    assert w.complete("b").text == '{"summary": "plain"}'
    assert w.think_blocks_seen == 1
    assert w.parse_failures == 0


def test_aggregate_cost_and_identical_runs():
    metas = [
        {
            "calls": 6,
            "output_tokens_total": 600,
            "output_tokens_max": 150,
            "parse_failures": 0,
            "no_answer_summaries": 1,
            "findings_signature": "x",
        },
        {
            "calls": 6,
            "output_tokens_total": 1200,
            "output_tokens_max": 400,
            "parse_failures": 2,
            "findings_signature": "x",
        },
    ]
    cost = ev.aggregate_cost(metas)
    assert cost["calls"] == 12 and cost["output_tokens_per_call_mean"] == 150.0
    assert cost["output_tokens_per_call_max"] == 400 and cost["parse_failures"] == 2
    assert cost["no_answer_summaries"] == 1
    assert ev.identical_runs(metas) is True
    metas[1]["findings_signature"] = "y"
    assert ev.identical_runs(metas) is False
    assert ev.identical_runs([{}]) is None
    assert ev.aggregate_cost([{"seconds": 1}]) is None


def test_findings_signature_is_order_independent_and_content_sensitive():
    f1 = {"source_doc": "cv", "claim": "two years", "transcript_quote": "q1"}
    f2 = {"source_doc": "cover_letter", "claim": "timeline", "transcript_quote": "q2"}
    a = ev.EvalScore(total_findings=2, matched_seeds=[], missed_seeds=[], per_finding=[f1, f2])
    b = ev.EvalScore(total_findings=2, matched_seeds=[], missed_seeds=[], per_finding=[f2, f1])
    c = ev.EvalScore(total_findings=1, matched_seeds=[], missed_seeds=[], per_finding=[f1])
    assert ev.findings_signature(a) == ev.findings_signature(b)
    assert ev.findings_signature(a) != ev.findings_signature(c)


def test_sweep_params_carry_the_cache_mode():
    assert ev.sweep_params("m", "ground_truth", "default")["cache"] == "cold"
    assert ev.sweep_params("m", "ground_truth", "default", cold=False)["cache"] == "warm"


def test_default_checkpoint_path_separates_cold_and_warm():
    cold = ev.default_checkpoint_path("qwen3:14b", "ground_truth", "default")
    warm = ev.default_checkpoint_path("qwen3:14b", "ground_truth", "default", cold=False)
    assert cold != warm and "cold" in cold.name and "warm" in warm.name


def test_checkpoint_from_the_other_cache_mode_is_rejected(tmp_path):
    ckpt = tmp_path / "ck.jsonl"
    ev.run_eval_multi(
        "m",
        "ground_truth",
        "default",
        2,
        checkpoint=ckpt,
        cold=False,
        run_once=_fake_run_once([], [_evalscore({A}), _evalscore({A})]),
    )
    calls: list = []
    with pytest.raises(ev.CheckpointMismatch) as exc:
        ev.run_eval_multi(
            "m",
            "ground_truth",
            "default",
            2,
            checkpoint=ckpt,  # cold (default) must refuse to resume a warm sweep
            run_once=_fake_run_once(calls, [_evalscore({A}), _evalscore({A})]),
        )
    assert calls == []
    assert "cache='warm'" in str(exc.value) and "cache='cold'" in str(exc.value)


def test_pre_2011_checkpoint_without_cache_mode_is_rejected(tmp_path):
    ckpt = tmp_path / "ck.jsonl"
    legacy = {"model": "m", "transcript": "ground_truth", "prompt": "default"}  # no "cache" key
    ev.append_checkpoint(ckpt, 0, legacy, _evalscore({A}), {"seconds": 1.0})
    with pytest.raises(ev.CheckpointMismatch):
        ev.load_checkpoint(ckpt, ev.sweep_params("m", "ground_truth", "default"))


# ── #1994 known negative cases + the paired filter ablation (pure, offline) ────────────────────


def test_negative_cases_ground_and_are_rejected_by_their_rule():
    """Each canned false finding passes D17 (so it is the FILTER that catches it) and is then
    rejected by the expected rule — i.e. the bench would count it as an FP if it were reported."""
    rows = ev.check_negatives_offline()
    assert [r["key"] for r in rows] == [c.key for c in ev.NEGATIVE_CASES]
    assert {r["key"] for r in rows} == {"omission_test_set_size", "aspiration_vs_no_experience"}
    for r in rows:
        assert r["grounded"], r
        assert r["rejected_by"] == r["expected_rule"], r
        assert r["fp"] is False


def test_negative_cases_do_not_match_any_seed():
    """A negative-case finding must never be scored as a caught seed (it is an FP by construction)."""
    from recruiter_copilot.analysis import verify_contradiction

    for case in ev.NEGATIVE_CASES:
        c = verify_contradiction(case.finding, ev.negative_case_lines(case), case.docs)
        assert c is not None
        sc = ev.score([c])
        assert sc.false_positive_findings == 1 and sc.matched_seeds == []


def test_filter_ablation_scores_the_same_run_with_the_filter_off():
    from recruiter_copilot.analysis import RejectedContradiction

    tp = _contradiction(
        "cv", "Says two years", "the CV shows the first LLM project in 2025", "od dwóch lat"
    )
    fp = _contradiction("earlier_answer:q4", "Cloud deployment experience", "nie mam", "zdobycie")
    out = ev.filter_ablation([tp], [RejectedContradiction("q5", "aspiration_vs_no_experience", fp)])
    assert out["unfiltered"]["false_positive_findings"] == 1
    assert out["unfiltered"]["precision"] == 0.5
    assert out["unfiltered"]["matched_seeds"] == ["cv_two_years"]
    assert out["filter_rejected"] == [
        {
            "rule": "aspiration_vs_no_experience",
            "question_id": "q5",
            "source_doc": "earlier_answer:q4",
            "claim": "Cloud deployment experience",
            "was_seed": False,
        }
    ]


def test_aggregate_ablation_handles_pre_1994_runs_and_counts_rules():
    assert ev.aggregate_ablation([{"model": "m"}]) is None
    metas = [
        {
            "unfiltered": {"recall": 0.667, "precision": 0.5, "false_positive_findings": 2},
            "filter_rejected": [{"rule": "omission", "was_seed": False}],
        },
        {
            "unfiltered": {"recall": 0.333, "precision": None, "false_positive_findings": 0},
            "filter_rejected": [],
        },
    ]
    agg = ev.aggregate_ablation(metas)
    assert agg == {
        "runs": 2,
        "unfiltered_mean_recall": 0.5,
        "unfiltered_mean_precision": 0.5,
        "unfiltered_mean_false_positives": 1.0,
        "rejected_by_rule": {"omission": 1},
        "rejected_seeds": 0,
    }


# ── live model path (guarded) ─────────────────────────────────────────────────────────────────


def _ollama_reachable() -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(
    not os.environ.get("RECRUITER_LIVE_LLM"),
    reason="live-LLM eval; opt in with RECRUITER_LIVE_LLM=1 (also needs a reachable Ollama)",
)
def test_live_eval_runs_and_scores():
    """Opt-in: `RECRUITER_LIVE_LLM=1 pytest`. Runs the real analyser and asserts the score SHAPE,
    not a recall number (that is measured and reported, not gated)."""
    if not _ollama_reachable():
        pytest.skip("Ollama not reachable on localhost:11434")
    model = os.environ.get("RECRUITER_EVAL_MODEL", "qwen3:14b")
    out = ev.run_eval(model, "ground_truth", "default")
    assert 0.0 <= out["recall"] <= 1.0
    assert out["precision"] is None or 0.0 <= out["precision"] <= 1.0
    assert out["seeded_total"] == 3
    assert out["analyses"] == 6  # the sample session has six questions
