"""Offline tests for scripts/eval_prep.py's deterministic scorers (#2042) — no model, no GPU.

Every draft here is hand-made and synthetic."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import eval_prep as ev  # noqa: E402


def _q(qid: str, en: str = "", pl: str = "", reqs: list[str] | None = None, lang: bool = False):
    text = {k: v for k, v in (("en", en), ("pl", pl)) if v}
    return {"id": qid, "text": text, "requirement_ids": reqs or [], "assesses_language": lang}


def _draft(questions: list[dict], sections: list[dict], languages=("en",)) -> dict:
    return {
        "languages": list(languages),
        "target_minutes": 45,
        "requirements": [
            {"id": "r1", "kind": "must", "grounded": True},
            {"id": "r2", "kind": "must", "grounded": True},
            {"id": "r3", "kind": "nice", "grounded": False},
        ],
        "questions": questions,
        "plan": {"target_minutes": 45, "sections": sections},
    }


def test_hypothetical_matches_anywhere_per_language() -> None:
    d = _draft(
        [
            _q("q1", en="Walk us through how you would model a star schema."),
            _q("q2", en="Imagine the pipeline fails at 3 am."),
            _q("q3", en="Tell me about a time you modelled a star schema."),
            _q("q4", pl="Jak byś zaprojektował API?", en="Tell me about an API you built."),
            _q("q5", pl="Co byś zrobił, gdyby baza padła?"),
            _q("q6", pl="Wyobraź sobie awarię."),
            _q("q7", pl="Jakbyś opisał swój zespół?"),
            _q("q8", pl="Opowiedz o sytuacji, gdy optymalizowałeś zapytanie."),
        ],
        [],
        languages=("pl", "en"),
    )
    assert ev.hypothetical_questions(d) == ["q1", "q2", "q4", "q5", "q6", "q7"]


def test_language_patterns_apply_only_to_their_language() -> None:
    # a Polish wording is never checked against the English pattern and vice versa
    d = _draft([_q("q1", pl="imagine — to słowo po angielsku"), _q("q2", en="Czy is Polish")], [])
    assert ev.hypothetical_questions(d) == []
    assert ev.closed_questions(d) == []


def test_closed_counts_only_yes_no_openers() -> None:
    d = _draft(
        [
            _q("q1", en="Do you use Airflow?"),
            _q("q2", en="Have you ever led a migration?"),
            _q("q3", en="What tools do you use for monitoring?"),  # not an opener
            _q("q4", pl="Czy posiadaliście znajomość niemieckiego?"),
            _q("q5", pl="Opisz, czy i jak testowałeś kod."),  # 'czy' mid-sentence
            _q("q6", en="  Can you describe a SQL query you tuned?"),
            _q("q7", pl="Czyściłeś kiedyś dane?"),  # 'Czy' only as a prefix of a word
        ],
        [],
        languages=("pl", "en"),
    )
    assert ev.closed_questions(d) == ["q1", "q2", "q4", "q6"]


def test_nonlang_in_language_check() -> None:
    d = _draft(
        [_q("q1", en="a", lang=True), _q("q2", en="b"), _q("q3", en="c")],
        [
            {"name": "technical", "minutes": 20, "question_ids": ["q3"]},
            {"name": "Language_Check", "minutes": 5, "question_ids": ["q1", "q2"]},
        ],
    )
    assert ev.nonlang_in_language_check(d) == ["q2"]


def test_unlinked_and_unprobed_musts() -> None:
    d = _draft([_q("q1", en="a", reqs=["r1"])], [])
    assert ev.unlinked_requirements(d) == ["r2", "r3"]
    assert ev.unlinked_requirements(d, "must") == ["r2"]


def test_score_draft_numbers() -> None:
    d = _draft(
        [_q("q1", en="Do you know SQL?", reqs=["r1"]), _q("q2", pl="tylko po polsku", reqs=["r2"])],
        [
            {"name": "intro", "minutes": 5, "question_ids": []},
            {"name": "technical", "minutes": 40, "question_ids": ["q1", "q2"]},
        ],
        languages=("pl", "en"),
    )
    s = ev.score_draft(d)
    assert s["grounded_pct"] == 66.7
    assert s["lang_coverage"] == "0/2"  # neither question has both wordings
    assert s["plan_sum"] == "45/45"
    assert s["closed"] == ["q1"]
    assert s["unprobed_musts"] == [] and s["unlinked"] == ["r3"]


def test_stats_row_splits_first_and_repair_calls() -> None:
    calls = [
        {"input_tokens": 760, "output_tokens": 2000, "truncation": []},
        {"input_tokens": 2300, "output_tokens": 2100, "truncation": ["context"]},
    ]
    row = ev.stats_row({"first_try_valid": False, "repair_used": True, "wall_seconds": 99}, calls)
    assert row["prompt_tok_first"] == 760 and row["prompt_tok_repair"] == 2300
    assert row["truncation"] == ["context"] and row["calls"] == 2
    assert ev.stats_row(None, [])["prompt_tok_first"] is None


def test_recording_chat_records_each_call_and_delegates() -> None:
    class Inner:
        name = "stub"
        model_name = "m"
        context_window = 4096

        def complete(self, prompt, system=None, max_tokens=None):
            return SimpleNamespace(
                text="{}", input_tokens=len(prompt), output_tokens=1, truncation_signals=()
            )

    chat = ev.RecordingChat(Inner())
    chat.complete("abc", system="s")
    chat.complete("abcdef")
    assert [c["input_tokens"] for c in chat.calls] == [3, 6]
    assert chat.context_window == 4096 and chat.model_name == "m"


def test_summary_and_rescore(tmp_path: Path) -> None:
    d = _draft(
        [_q("q1", en="How would you test it?", reqs=["r1", "r2"])],
        [{"name": "technical", "minutes": 45, "question_ids": ["q1"]}],
    )
    d["stats"] = {"first_try_valid": True, "wall_seconds": 40.0}
    (tmp_path / "en_data_run1.json").write_text(json.dumps(d), encoding="utf-8")
    (tmp_path / "unknown_run1.json").write_text(json.dumps(d), encoding="utf-8")  # skipped
    rows = ev.rescore(tmp_path)
    assert [(r["offer"], r["run"]) for r in rows] == [("en_data", 1)]
    table = ev.summarise(rows)
    assert "| en_data | 1 | 1/1 | 0/1 | 40.0 |" in table
    assert ev.main(["--rescore", str(tmp_path)]) == 0


def test_unlinked_questions_and_nice_lines_without_requirement() -> None:
    d = _draft([_q("q1", en="a", reqs=["r1"]), _q("q2", en="b")], [])
    assert ev.unlinked_questions(d) == ["q2"]
    offer = (
        "# Job\n## Must have\n- SQL\n## Nice to have\n- dbt\n- Terraform (IaC)\n## Benefits\n- x\n"
    )
    d["requirements"] = [{"id": "r1", "kind": "nice", "source_quote": "Terraform"}]
    assert ev.nice_lines_without_requirement(d, offer) == ["dbt"]
    assert ev.score_draft(d, offer)["nice_missing"] == ["dbt"]
    assert ev.score_draft(d)["nice_missing"] == []  # no offer → not scored


def test_main_rejects_unknown_offer(tmp_path: Path) -> None:
    assert ev.main(["--offers", "nope", "--out", str(tmp_path / "x")]) == 2
