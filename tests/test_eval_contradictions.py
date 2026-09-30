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

    def _fake(model, transcript_kind, prompt, stt_dir=ev.DEFAULT_STT_DIR):
        idx = len(calls)
        calls.append((model, transcript_kind, prompt))
        return per_run[idx], {
            "model": model,
            "transcript": transcript_kind,
            "prompt": prompt,
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
        r["params"] == {"model": "m", "transcript": "ground_truth", "prompt": "default"}
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
