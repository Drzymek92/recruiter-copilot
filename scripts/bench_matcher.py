"""Benchmark the question-matcher routes against a labelled transcript (G1 / P1).

The routes exist so they can be compared, not asserted about. This scores each one on the
synthetic mock interview, whose ground truth says which bank question each interviewer turn was
asking, and prints precision / recall / F1 plus wall-clock cost per line.

    python scripts/bench_matcher.py                      # lexical only (no model needed)
    python scripts/bench_matcher.py --routes lexical,llm,hybrid
    python scripts/bench_matcher.py --transcript <dir>   # score a real saved transcript instead

A route is only allowed to win on numbers printed by this script.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from recruiter_copilot.config import load_settings  # noqa: E402
from recruiter_copilot.matcher import propose_all  # noqa: E402
from recruiter_copilot.models import Speaker, TranscriptLine  # noqa: E402
from recruiter_copilot.pipeline import load_transcript  # noqa: E402
from recruiter_copilot.store import load_session  # noqa: E402

FIXTURE_JSON = PROJECT / "tests" / "fixtures" / "mock_interview.json"
SAMPLE_SESSION = PROJECT / "examples" / "sample_session"


def truth_by_line(lines: list[TranscriptLine], turns: list[dict]) -> dict[int, str]:
    """Map each transcript line index to the question id its turn was asking.

    Matched on time overlap, not on order, so a segmenter that merges or splits a turn is
    scored honestly rather than silently shifting every label by one.
    """
    out: dict[int, str] = {}
    for i, line in enumerate(lines):
        if line.speaker is Speaker.CANDIDATE:
            continue
        best, best_overlap = None, 0.0
        for turn in turns:
            if turn.get("speaker") != "interviewer" or not turn.get("question_id"):
                continue
            overlap = min(line.t_end, turn["t_end"]) - max(line.t_start, turn["t_start"])
            if overlap > best_overlap:
                best, best_overlap = turn["question_id"], overlap
        if best and best_overlap > 0.5:
            out[i] = best
    return out


def score(proposals, truth: dict[int, str], n_lines: int) -> dict:
    """Per-line scoring: a proposal is correct only if it names the right question on that line."""
    by_line = {p.line_index: p for p in proposals}
    tp = fp = fn = 0
    wrong_id = 0
    for i in range(n_lines):
        expected = truth.get(i)
        got = by_line.get(i)
        if expected and got and got.question_id == expected:
            tp += 1
        elif expected and got:
            fp += 1
            wrong_id += 1
            fn += 1
        elif expected and not got:
            fn += 1
        elif got and not expected:
            fp += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "wrong_id": wrong_id,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", default="lexical")
    ap.add_argument("--transcript", default=None, help="folder holding transcript.json")
    ap.add_argument("--session", default=str(SAMPLE_SESSION))
    ap.add_argument("--min-confidence", type=float, default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    settings = load_settings(env_file=Path("/dev/null"))
    min_conf = (
        args.min_confidence if args.min_confidence is not None else settings.matcher_min_confidence
    )
    session = load_session(Path(args.session))

    if args.transcript:
        lines = load_transcript(Path(args.transcript))
    else:
        raise SystemExit(
            "--transcript <folder> is required (run `recruiter-copilot analyse` first)"
        )

    truth = truth_by_line(lines, json.loads(FIXTURE_JSON.read_text(encoding="utf-8"))["turns"])
    results = {}
    chat = None
    for route in [r.strip() for r in args.routes.split(",") if r.strip()]:
        if route in ("llm", "hybrid") and chat is None:
            from recruiter_copilot.llm import build_chat_provider  # noqa: PLC0415

            chat = build_chat_provider(settings)
        t0 = time.perf_counter()
        proposals = propose_all(
            lines, session.questions, route=route, min_confidence=min_conf, chat=chat
        )
        elapsed = time.perf_counter() - t0
        r = score(proposals, truth, len(lines))
        r["seconds_total"] = round(elapsed, 3)
        r["ms_per_line"] = round(1000 * elapsed / max(1, len(lines)), 1)
        r["proposals"] = [p.as_dict() for p in proposals]
        results[route] = r

    if args.json:
        print(
            json.dumps(
                {"labelled_questions": len(truth), "routes": results}, indent=2, ensure_ascii=False
            )
        )
        return 0

    print(f"transcript: {len(lines)} lines, {len(truth)} labelled interviewer questions")
    print(f"threshold: min_confidence={min_conf}")
    print(f"{'route':10} {'P':>6} {'R':>6} {'F1':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'ms/line':>8}")
    for route, r in results.items():
        print(
            f"{route:10} {r['precision']:6.2f} {r['recall']:6.2f} {r['f1']:6.2f} "
            f"{r['tp']:4d} {r['fp']:4d} {r['fn']:4d} {r['ms_per_line']:8.1f}"
        )
    print()
    for route, r in results.items():
        print(f"-- {route} proposals --")
        for p in r["proposals"]:
            expected = truth.get(p["line_index"], "-")
            mark = "OK " if p["question_id"] == expected else "XX "
            print(
                f"  {mark}line {p['line_index']:2d} → {p['question_id']} "
                f"(expected {expected}) conf {p['confidence']:.2f} margin {p['margin']:.2f}"
            )
        missed = [i for i in truth if i not in {p["line_index"] for p in r["proposals"]}]
        for i in missed:
            print(f'  MISS line {i:2d} → expected {truth[i]}: "{lines[i].text[:60]}"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
