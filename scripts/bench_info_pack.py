"""Benchmark the D26 candidate-question info-pack matcher and pick its threshold (G10).

Sibling of ``bench_matcher.py``. Scores ``info_pack.answer_line`` over a labelled synthetic bench
(``tests/fixtures/info_pack_bench.json`` against ``info_pack_bench.md``): candidate questions the
pack covers (in pl and en, literal and paraphrased), questions it does NOT cover, and plain
candidate answers that should show nothing. Deterministic — no model, no GPU.

    python scripts/bench_info_pack.py                     # sweep thresholds at the default prefix
    python scripts/bench_info_pack.py --prefix 0,5,6,7    # also compare token-prefix lengths
    python scripts/bench_info_pack.py --json

Scoring, per labelled line (D26: a wrong section shown verbatim is the costly error):
  * covered line  — TP when a shown section is the expected one, FN when it is not shown;
                    every other section shown on it is an FP.
  * uncovered line — every section shown is an FP ("Not in your pack" is the right answer).
  * answer line   — not a question; every section shown is an FP, and every "Not in your pack"
                    item is counted as detector noise (``nip_noise``).
P = TP/(TP+FP) over all lines, R = TP/covered. The threshold is only chosen on numbers printed here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from recruiter_copilot.info_pack import (  # noqa: E402
    PACK_TOKEN_PREFIX,
    InfoPackIndex,
    answer_line,
)
from recruiter_copilot.models import Speaker, TranscriptLine  # noqa: E402

FIXTURE = PROJECT / "tests" / "fixtures" / "info_pack_bench.json"


def threshold_grid(spec: str) -> list[float]:
    """``start:stop:step`` (inclusive) → thresholds, e.g. ``0.05:0.80:0.05``."""
    start, stop, step = (float(x) for x in spec.split(":"))
    n = int(round((stop - start) / step))
    return [round(start + i * step, 3) for i in range(n + 1)]


def load_bench(path: Path) -> tuple[str, list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    pack = (path.parent / data["pack"]).read_text(encoding="utf-8")
    return pack, data["lines"]


def evaluate(pack: str, rows: list[dict], threshold: float, prefix: int) -> dict:
    index = InfoPackIndex.from_text(pack, prefix=prefix)
    tp = fp = fn = nip_noise = gated_answers = 0
    misses: list[str] = []
    for i, row in enumerate(rows):
        line = TranscriptLine(
            t_start=float(i), t_end=float(i) + 1, text=row["text"], speaker=Speaker.CANDIDATE
        )
        items = answer_line(line, i, index, threshold)
        shown = [a.section.heading for a in items if a.section is not None]
        if row["kind"] == "answer":
            gated_answers += 1 if items else 0
            nip_noise += sum(1 for a in items if a.section is None)
        expect = row.get("expect")
        if row["kind"] == "covered":
            if expect in shown:
                tp += 1
            else:
                fn += 1
                misses.append(f"MISS {row['text']!r} → {shown or 'not in pack'}")
            fp += sum(1 for h in shown if h != expect)
        else:
            fp += len(shown)
            if shown:
                misses.append(f"FP   {row['text']!r} → {shown}")
    covered = sum(1 for r in rows if r["kind"] == "covered")
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / covered if covered else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": threshold,
        "prefix": prefix,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "answer_lines_gated": gated_answers,
        "nip_noise": nip_noise,
        "errors": misses,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default=str(FIXTURE))
    ap.add_argument("--prefix", default=str(PACK_TOKEN_PREFIX), help="comma list, 0 = no cut")
    ap.add_argument("--thresholds", default="0.05:0.80:0.05", help="start:stop:step, inclusive")
    ap.add_argument("--show-errors", type=float, default=None, help="list errors at this threshold")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    pack, rows = load_bench(Path(args.fixture))
    prefixes = [int(p) for p in args.prefix.split(",") if p.strip()]
    grid = threshold_grid(args.thresholds)
    results = {p: [evaluate(pack, rows, t, p) for t in grid] for p in prefixes}
    if args.json:
        print(json.dumps({str(p): r for p, r in results.items()}, indent=2, ensure_ascii=False))
        return 0

    kinds = {k: sum(1 for r in rows if r["kind"] == k) for k in ("covered", "uncovered", "answer")}
    print(f"bench: {len(rows)} candidate lines — {kinds}")
    for p, rs in results.items():
        print(f"\nprefix={p} (0 = no cut)")
        print(
            f"{'thr':>5} {'P':>6} {'R':>6} {'F1':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'nipNoise':>9}"
        )
        for r in rs:
            print(
                f"{r['threshold']:5.2f} {r['precision']:6.2f} {r['recall']:6.2f} {r['f1']:6.2f} "
                f"{r['tp']:4d} {r['fp']:4d} {r['fn']:4d} {r['nip_noise']:9d}"
            )
        print(
            f"answer lines passing the question gate: {rs[0]['answer_lines_gated']}/{kinds['answer']}"
        )
        if args.show_errors is not None:
            hit = next((r for r in rs if abs(r["threshold"] - args.show_errors) < 1e-9), None)
            for e in (hit or {}).get("errors", []):
                print("  " + e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
