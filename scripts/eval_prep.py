"""Measure D29 interview-prep DRAFT quality (``prep.draft_job_prep``) on the real chat model (#2042).

Runs the real drafter on the committed SYNTHETIC offers in ``tests/fixtures/prep_eval/`` and
scores every draft with PURE, deterministic scorers (fw:D2 — no LLM judges the draft):

- first-try valid / repair used / wall time (from ``PrepStats``)
- grounded % (requirements whose ``source_quote`` is in the offer), language coverage, plan sum
- ``nonlang_in_lc``  — questions placed in "language check" without ``assesses_language``
- ``hypothetical``   — hypothetical phrasing anywhere in a wording, per language:
  en ``how would you|what would you|imagine`` (+ ``how you would|how you'd``, the embedded form
  "walk us through how you would …"); pl ``jak byś|jakbyś|jak by(ś|łby)|co byś|wyobraź``
- ``closed``         — a yes/no OPENER (start of the wording): en ``do|have|are|can|did you``,
  pl ``czy``
- ``unlinked``       — requirements no question links; ``unprobed_musts`` — the 'must' ones
- prompt tokens of the first and the repair call (each reply's usage, recorded per call by a thin
  wrapper around the provider) and the truncation signals that fired

A question is counted once even when several of its wordings match. Judgement-only quality (does a
question really probe the requirement it links?) is NOT scored here — read the saved drafts.

Usage (the live model needs Ollama; run it through the GPU lease board from the repo root):

    python -m commons.coordination.gpu run --vram 10500 --label "sc prep quality" -- \\
        timeout 1800 python projects/recruiter_copilot/scripts/eval_prep.py --runs 2 --out <dir>
    python scripts/eval_prep.py --offers en_data --runs 1        # one offer
    python scripts/eval_prep.py --rescore <drafts_dir>           # score saved drafts, no model

Every run starts from an EMPTY Ollama prompt cache (D23; ``--warm`` opts out). Output: one JSONL row
per run + ``summary.md`` + the drafts, in ``--out`` (default ``scripts/outputs/prep_eval_<stamp>/``).
The scorers are unit-tested offline in ``tests/test_eval_prep.py`` — no model, no GPU.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from recruiter_copilot import prep  # noqa: E402
from recruiter_copilot import workspace as ws  # noqa: E402
from recruiter_copilot.models import Languages, to_dict  # noqa: E402

FIXTURES = PROJECT / "tests" / "fixtures" / "prep_eval"

# key → (job title, offer file, session languages). All synthetic.
OFFERS: dict[str, tuple[str, str, Languages]] = {
    "pl_backend": ("Programista Backend (Python)", "offer_pl_backend.md", Languages("pl", "en")),
    "en_data": ("Data Engineer", "offer_en_data.md", Languages("en")),
    "mixed_support": (
        "Technical Support Engineer L2",
        "offer_mixed_support.md",
        Languages("pl", "en", assess_language="en"),
    ),
}

HYPOTHETICAL: dict[str, re.Pattern[str]] = {
    "en": re.compile(
        r"\b(how would you|what would you|how you would|how you'd|imagine)\b", re.IGNORECASE
    ),
    "pl": re.compile(r"\b(jak ?byś|jak by(ś|łby)|co byś|wyobraź)", re.IGNORECASE),
}
CLOSED: dict[str, re.Pattern[str]] = {
    "en": re.compile(r"^\W*(do|have|are|can|did) you\b", re.IGNORECASE),
    "pl": re.compile(r"^\W*czy\b", re.IGNORECASE),
}

COUNT_KEYS = (
    "nonlang_in_lc",
    "hypothetical",
    "closed",
    "unlinked",
    "unprobed_musts",
    "unlinked_questions",
    "nice_missing",
)


# ── pure scorers (on the draft.json dict shape) ─────────────────────────────────────────────


def _wordings(question: dict[str, Any]) -> dict[str, str]:
    text = question.get("text") or {}
    return {k: v for k, v in text.items() if isinstance(v, str)}


def _matching(draft: dict[str, Any], patterns: dict[str, re.Pattern[str]]) -> list[str]:
    """Question ids with at least one wording matched by its language's pattern."""
    out: list[str] = []
    for q in draft.get("questions", []):
        for lang, wording in _wordings(q).items():
            pattern = patterns.get(lang)
            if pattern is not None and pattern.search(wording.strip()):
                out.append(q.get("id", ""))
                break
    return out


def hypothetical_questions(draft: dict[str, Any]) -> list[str]:
    return _matching(draft, HYPOTHETICAL)


def closed_questions(draft: dict[str, Any]) -> list[str]:
    return _matching(draft, CLOSED)


def nonlang_in_language_check(draft: dict[str, Any]) -> list[str]:
    """Question ids in the "language check" section that do not have ``assesses_language``."""
    assesses = {q.get("id"): bool(q.get("assesses_language")) for q in draft.get("questions", [])}
    out: list[str] = []
    for s in (draft.get("plan") or {}).get("sections", []):
        if prep.normalise_section_name(str(s.get("name", ""))) == prep.LANGUAGE_CHECK:
            out.extend(qid for qid in s.get("question_ids", []) if not assesses.get(qid, False))
    return out


def unlinked_requirements(draft: dict[str, Any], kind: str | None = None) -> list[str]:
    """Requirement ids (optionally of one ``kind``) that no question lists in requirement_ids."""
    linked = {rid for q in draft.get("questions", []) for rid in q.get("requirement_ids", [])}
    return [
        r.get("id", "")
        for r in draft.get("requirements", [])
        if r.get("id") not in linked and (kind is None or r.get("kind") == kind)
    ]


def unlinked_questions(draft: dict[str, Any]) -> list[str]:
    """Question ids with an empty requirement_ids list."""
    return [q.get("id", "") for q in draft.get("questions", []) if not q.get("requirement_ids")]


_NICE_HEADING = re.compile(r"^#+\s*(nice to have|mile widziane)\s*$", re.IGNORECASE)


def nice_lines_without_requirement(draft: dict[str, Any], offer: str) -> list[str]:
    """Bullet lines under the offer's '## Nice to have' / '## Mile widziane' heading that no
    requirement's source_quote overlaps (the fixtures' markdown shape; independent of prep's
    own heading parser so the scorer does not grade the code with itself)."""
    lines: list[str] = []
    in_nice = False
    for line in offer.splitlines():
        if line.startswith("#"):
            in_nice = bool(_NICE_HEADING.match(line.strip()))
        elif in_nice and line.startswith("- "):
            lines.append(line[2:].strip())
    quotes = [prep.normalise_text(r.get("source_quote") or "") for r in draft["requirements"]]

    def covered(line: str) -> bool:
        n = prep.normalise_text(line)
        return any(q and (q in n or n in q) for q in quotes)

    return [ln for ln in lines if not covered(ln)]


def score_draft(draft: dict[str, Any], offer: str | None = None) -> dict[str, Any]:
    """All deterministic quality numbers for one draft (draft.json shape); ``offer`` enables the
    nice-to-have coverage count."""
    langs = list(draft.get("languages") or [])
    qs = draft.get("questions", [])
    reqs = draft.get("requirements", [])
    covered = sum(all((_wordings(q).get(lang) or "").strip() for lang in langs) for q in qs)
    plan = draft.get("plan") or {}
    sections = plan.get("sections", [])
    total = sum(int(s.get("minutes", 0)) for s in sections)
    target = plan.get("target_minutes", draft.get("target_minutes", 0))
    return {
        "n_requirements": len(reqs),
        "n_questions": len(qs),
        "grounded_pct": round(
            100 * sum(bool(r.get("grounded")) for r in reqs) / max(1, len(reqs)), 1
        ),
        "languages": ",".join(langs),
        "lang_coverage": f"{covered}/{len(qs)}",
        "plan_sum": f"{total}/{target}",
        "sections": [f"{s.get('name')}:{s.get('minutes')}" for s in sections],
        "assesses_language": sum(bool(q.get("assesses_language")) for q in qs),
        "nonlang_in_lc": nonlang_in_language_check(draft),
        "hypothetical": hypothetical_questions(draft),
        "closed": closed_questions(draft),
        "unlinked": unlinked_requirements(draft),
        "unprobed_musts": unlinked_requirements(draft, "must"),
        "unlinked_questions": unlinked_questions(draft),
        "nice_missing": nice_lines_without_requirement(draft, offer) if offer else [],
    }


def stats_row(stats: dict[str, Any] | None, calls: list[dict[str, Any]]) -> dict[str, Any]:
    """How the draft was obtained: PrepStats fields + per-call prompt tokens from the wrapper."""
    s = stats or {}
    return {
        "first_try_valid": bool(s.get("first_try_valid")),
        "repair_used": bool(s.get("repair_used")),
        "rescaled": bool(s.get("rescaled")),
        "calls": len(calls) or int(s.get("calls", 0)),
        "wall_s": s.get("wall_seconds", 0.0),
        "prompt_tok_first": calls[0]["input_tokens"] if calls else None,
        "prompt_tok_repair": calls[1]["input_tokens"] if len(calls) > 1 else None,
        "out_tok": [c["output_tokens"] for c in calls] or [int(s.get("output_tokens", 0))],
        "truncation": sorted({sig for c in calls for sig in c["truncation"]}),
        "first_try_errors": list(s.get("first_try_errors", []))[:6],
    }


def _mean(values: list[float]) -> str:
    return f"{statistics.mean(values):.1f}" if values else "-"


def summarise(rows: list[dict[str, Any]]) -> str:
    """One markdown table row per offer (counts are means over the runs)."""
    head = (
        "| offer | runs | valid 1st | repair | time s | grounded % | langs | plan sum | "
        "non-lang in LC | hypothetical | closed | unlinked | unprobed musts | q w/o req | "
        "nice lines w/o req | n req | out tok | prompt tok 1st/repair | truncation |\n|"
        + "---|" * 19
    )
    lines = [head]
    for offer in dict.fromkeys(r["offer"] for r in rows):
        rs = [r for r in rows if r["offer"] == offer]
        ok = [r for r in rs if r.get("valid")]
        counts = {k: _mean([len(r[k]) for r in ok]) for k in COUNT_KEYS}
        tok1 = sorted({r.get("prompt_tok_first") for r in rs if r.get("prompt_tok_first")})
        tok2 = sorted({r.get("prompt_tok_repair") for r in rs if r.get("prompt_tok_repair")})
        trunc = sorted({t for r in rs for t in r.get("truncation", [])})
        lines.append(
            f"| {offer} | {len(rs)} | {sum(r.get('first_try_valid', False) for r in rs)}/{len(rs)} "
            f"| {sum(r.get('repair_used', False) for r in rs)}/{len(rs)} "
            f"| {_mean([r.get('wall_s') or 0.0 for r in rs])} "
            f"| {_mean([r['grounded_pct'] for r in ok])} "
            f"| {' '.join(sorted({r['lang_coverage'] for r in ok})) or '-'} "
            f"| {' '.join(sorted({r['plan_sum'] for r in ok})) or '-'} "
            f"| {counts['nonlang_in_lc']} | {counts['hypothetical']} | {counts['closed']} "
            f"| {counts['unlinked']} | {counts['unprobed_musts']} "
            f"| {counts['unlinked_questions']} | {counts['nice_missing']} "
            f"| {_mean([r['n_requirements'] for r in ok])} "
            f"| {_mean([sum(r.get('out_tok') or [0]) for r in rs])} "
            f"| {','.join(map(str, tok1)) or '-'}/{','.join(map(str, tok2)) or '-'} "
            f"| {','.join(trunc) or '-'} |"
        )
    return "\n".join(lines)


# ── the live run ────────────────────────────────────────────────────────────────────────────


@dataclass
class RecordingChat:
    """Wraps a provider and records each reply's usage, so first and repair calls are separate."""

    inner: Any
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __getattr__(self, name: str) -> Any:  # model_name, context_window, reset_prompt_cache …
        return getattr(self.inner, name)

    def complete(self, prompt: str, system: str | None = None, max_tokens: int | None = None):
        reply = self.inner.complete(prompt, system=system, max_tokens=max_tokens)
        self.calls.append(
            {
                "input_tokens": int(getattr(reply, "input_tokens", 0) or 0),
                "output_tokens": int(getattr(reply, "output_tokens", 0) or 0),
                "truncation": list(getattr(reply, "truncation_signals", ()) or ()),
                "sent_chars": len(prompt) + len(system or ""),
            }
        )
        return reply


def run_one(
    root: Path, offer_key: str, chat: RecordingChat, drafts_dir: Path, run: int, warm: bool
) -> dict[str, Any]:
    title, fname, langs = OFFERS[offer_key]
    job = ws.create_job(root, title, langs)
    ws.add_offer_docs(job, [FIXTURES / fname])
    cold = None if warm else bool(getattr(chat.inner, "reset_prompt_cache", lambda: False)())
    chat.calls = []
    row: dict[str, Any] = {"offer": offer_key, "run": run, "cold_reset": cold}
    t0 = time.perf_counter()
    try:
        prep.draft_job_prep(job, chat)
    except prep.PrepError as e:
        row.update(valid=False, reasons=e.reasons[:10])
        row.update(stats_row(to_dict(e.stats) if e.stats else None, chat.calls))
    else:
        saved = drafts_dir / f"{offer_key}_run{run}.json"
        shutil.copyfile(job / prep.DRAFT_FILE, saved)
        as_dict = json.loads(saved.read_text(encoding="utf-8"))
        row.update(valid=True, **stats_row(as_dict.get("stats"), chat.calls))
        row.update(score_draft(as_dict, (FIXTURES / fname).read_text(encoding="utf-8")))
        row["warnings"] = as_dict.get("warnings", [])
        row["language_moves"] = (as_dict.get("stats") or {}).get("language_moves", [])
        row["repair_cuts"] = (as_dict.get("stats") or {}).get("repair_cuts", [])
        row["added_nice"] = (as_dict.get("stats") or {}).get("added_nice", [])
        row["linked_questions"] = (as_dict.get("stats") or {}).get("linked_questions", [])
    row["total_s"] = round(time.perf_counter() - t0, 1)
    return row


def rescore(folder: Path) -> list[dict[str, Any]]:
    """Score saved drafts (``<offer>_run<n>.json``) without a model."""
    rows: list[dict[str, Any]] = []
    for path in sorted(folder.glob("*_run*.json")):
        offer, _, run = path.stem.rpartition("_run")
        if offer not in OFFERS:
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        row = {"offer": offer, "run": int(run), "valid": True}
        row.update(stats_row(d.get("stats"), []))
        row.update(score_draft(d, (FIXTURES / OFFERS[offer][1]).read_text(encoding="utf-8")))
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="D29 interview-prep draft quality eval (#2042)")
    ap.add_argument("--runs", type=int, default=2, help="runs per offer (default 2)")
    ap.add_argument("--offers", default=",".join(OFFERS), help=f"comma list of {', '.join(OFFERS)}")
    ap.add_argument("--warm", action="store_true", help="skip the cold prompt-cache reset (D23)")
    ap.add_argument("--out", default=None, help="output folder (must not exist or be empty)")
    ap.add_argument("--rescore", metavar="DRAFTS_DIR", default=None, help="score saved drafts")
    args = ap.parse_args(argv)

    if args.rescore:
        rows = rescore(Path(args.rescore))
        for row in rows:
            print(json.dumps(row, ensure_ascii=False))
        print(summarise(rows))
        return 0

    offers = [o.strip() for o in args.offers.split(",") if o.strip()]
    unknown = [o for o in offers if o not in OFFERS]
    if unknown or args.runs < 1:
        print(f"unknown offer(s) {unknown} or --runs < 1", file=sys.stderr)
        return 2
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"{stamp}_{uuid.uuid4().hex[:6]}"
    out = Path(args.out) if args.out else PROJECT / "scripts" / "outputs" / f"prep_eval_{stamp}"
    if out.exists() and any(out.iterdir()):
        print(f"{out} is not empty — pick a new --out", file=sys.stderr)
        return 2
    drafts_dir = out / "drafts"
    drafts_dir.mkdir(parents=True)
    results = out / "results.jsonl"

    from recruiter_copilot.config import load_settings  # noqa: PLC0415
    from recruiter_copilot.llm import build_chat_provider  # noqa: PLC0415

    settings = load_settings(overrides={}, env_file=None)
    chat = RecordingChat(build_chat_provider(settings))
    print(
        f"run_id {run_id} START model {chat.inner.name} context "
        f"{settings.ollama_context_tokens} {'warm' if args.warm else 'cold'} → {out}",
        flush=True,
    )
    rows: list[dict[str, Any]] = []
    try:
        for offer in offers:
            for run in range(1, args.runs + 1):
                row = run_one(out / "ws", offer, chat, drafts_dir, run, args.warm)
                row["run_id"] = run_id
                rows.append(row)
                with results.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        reset = getattr(chat.inner, "reset_prompt_cache", None)
        if reset is not None:
            reset()  # leave the GPU idle
    table = summarise(rows)
    (out / "summary.md").write_text(f"run_id {run_id}\n\n{table}\n", encoding="utf-8")
    print(table)
    print(f"run_id {run_id} END", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
