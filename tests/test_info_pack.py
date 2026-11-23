"""D26 / D15 amendment: candidate questions answered from the recruiter's info pack, verbatim.

Everything here is synthetic and deterministic (no model, no GPU, no network): the ``##`` section
split, the candidate-only per-sentence question gate, the IDF-cosine match (shared D21 machinery),
the feature being OFF without a pack, the pack never reaching the analysis or the report, old
bundles loading unchanged, the threshold's CFG precedence, and the bench numbers the threshold was
picked on (G10).
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from recruiter_copilot.config import Settings, SettingsError, load_settings
from recruiter_copilot.info_pack import (
    PACK_TOKEN_PREFIX,
    InfoPackIndex,
    answer_line,
    answer_transcript,
    question_sentences,
    split_sections,
)
from recruiter_copilot.models import Consent, Job, Session, Speaker, TranscriptLine, from_dict
from recruiter_copilot.server import (
    compute_pack_answers,
    create_app,
    pack_view,
    session_state,
)
from recruiter_copilot.store import bundle_summary, load_session, save_session

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE = PROJECT_ROOT / "examples" / "sample_session"
DEMO = PROJECT_ROOT / "examples" / "demo_session"

PACK = """# Synthetic pack title
Preamble text that must be ignored: salary is secret.

## Benefity / Benefits
- Prywatna opieka medyczna, karta sportowa.
- Private medical care, sports card.

### Extra detail
Kept inside the Benefits body.

## Empty heading

## Praca zdalna / Remote policy
- Hybrid: 2 days a week in the office, the rest remote.
- Model hybrydowy: 2 dni w biurze, reszta zdalnie.
"""


def _cand(text: str, t: float = 0.0, speaker: Speaker = Speaker.CANDIDATE) -> TranscriptLine:
    return TranscriptLine(t_start=t, t_end=t + 1.0, text=text, speaker=speaker)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return load_settings(env_file=tmp_path / "none")


@pytest.fixture
def index() -> InfoPackIndex:
    return InfoPackIndex.from_text(PACK)


# ── contract: the ## split (D15) ───────────────────────────────────────────────────────────


def test_split_ignores_preamble_keeps_subheadings_and_drops_empty_sections() -> None:
    sections = split_sections(PACK)
    assert [s.heading for s in sections] == ["Benefity / Benefits", "Praca zdalna / Remote policy"]
    assert [s.id for s in sections] == ["s1", "s2"]
    assert all("Preamble" not in s.body and "secret" not in s.body for s in sections)
    # ### stays inside its ## section; the body is the file's lines verbatim
    assert sections[0].body == (
        "- Prywatna opieka medyczna, karta sportowa.\n"
        "- Private medical care, sports card.\n"
        "\n"
        "### Extra detail\n"
        "Kept inside the Benefits body."
    )


def test_split_of_empty_or_headingless_text_is_no_sections() -> None:
    assert split_sections("") == []
    assert split_sections("# Only a title\nand prose, no ## headings") == []


# ── detector: candidate lines only, per sentence ────────────────────────────────────────────


def test_detector_only_fires_on_candidate_lines(index: InfoPackIndex) -> None:
    text = "Jakie benefity oferujecie?"
    assert answer_line(_cand(text, speaker=Speaker.INTERVIEWER), 0, index, 0.36) == []
    assert answer_line(_cand(text, speaker=Speaker.UNKNOWN), 0, index, 0.36) == []
    got = answer_line(_cand(text), 0, index, 0.36)
    assert len(got) == 1 and got[0].section is not None


def test_detector_is_per_sentence_and_skips_contentless_questions(index: InfoPackIndex) -> None:
    line = _cand("Tak, pracowałem z Kafką dwa lata. A jak wygląda u was praca zdalna? Tak?")
    got = answer_line(line, 3, index, 0.36)
    assert [a.question for a in got] == ["A jak wygląda u was praca zdalna?"]
    assert got[0].key == "3:0"
    assert got[0].section is not None and got[0].section.heading == "Praca zdalna / Remote policy"
    # a plain answer passes no question gate → nothing at all, not even "Not in your pack"
    assert answer_line(_cand("Zbudowałem system rekomendacji."), 0, index, 0.36) == []
    assert question_sentences("I built it. Where is it? Fine.") == ["Where is it?"]


# ── match: covered → the right section, uncovered → none ───────────────────────────────────


def test_covered_question_returns_its_section_verbatim(index: InfoPackIndex) -> None:
    (a,) = answer_line(_cand("What benefits do you offer?", 12.0), 5, index, 0.36)
    assert a.section is not None and a.section.heading == "Benefity / Benefits"
    view = a.as_dict()
    assert view["found"] is True and view["t_start"] == 12.0
    assert view["body"] == split_sections(PACK)[0].body  # verbatim, not a summary


def test_uncovered_question_is_not_in_pack(index: InfoPackIndex) -> None:
    (a,) = answer_line(_cand("Do you sponsor visas?"), 0, index, 0.36)
    assert a.section is None
    assert a.as_dict()["found"] is False and a.as_dict()["body"] == ""


def test_threshold_decides_between_section_and_not_in_pack(index: InfoPackIndex) -> None:
    line = _cand("Czy mogę pracować zdalnie?")
    (low,) = answer_line(line, 0, index, 0.0)
    (high,) = answer_line(line, 0, index, 0.99)
    assert low.section is not None and high.section is None
    assert low.score == high.score


def test_pack_index_reuses_the_d21_scoring_not_a_copy() -> None:
    import inspect

    from recruiter_copilot import info_pack, matcher

    src = inspect.getsource(info_pack)
    assert "build_variant_vectors" in src and "score_variants" in src
    assert "math.log" not in src  # the IDF formula lives once, in matcher
    # and QuestionIndex goes through the same two functions
    assert "build_variant_vectors(raw)" in inspect.getsource(matcher.QuestionIndex.build)


def test_answer_transcript_drops_dismissed_keys(index: InfoPackIndex) -> None:
    lines = [_cand("What benefits do you offer?", 1.0), _cand("Do you sponsor visas?", 5.0)]
    assert [a.key for a in answer_transcript(lines, index, 0.36)] == ["0:0", "1:0"]
    assert [a.key for a in answer_transcript(lines, index, 0.36, {"0:0"})] == ["1:0"]


# ── feature off without a pack; on with one; no state change ───────────────────────────────


def test_absent_pack_is_feature_off(settings: Settings) -> None:
    session = load_session(DEMO)  # demo_session ships no docs/job_info.md
    assert session.job.info_pack == ""
    lines = [_cand("What benefits do you offer?")]
    assert compute_pack_answers(session, lines, settings) == []
    assert pack_view(session, lines, settings) is None
    assert session_state(session, lines, settings)["pack"] is None


def test_absent_pack_serves_without_errors_and_page_hides_the_panel(settings: Settings) -> None:
    session = load_session(DEMO)
    client = TestClient(create_app(settings, session))
    r = client.get("/session")
    assert r.status_code == 200 and r.json()["pack"] is None
    page = client.get("/").text
    assert '<h2 id="pack-h" hidden>' in page
    assert 'id="pack-answers" aria-labelledby="pack-h" aria-live="polite" hidden' in page


def _consented_sample() -> Session:
    session = load_session(SAMPLE)
    session.consent = Consent(
        informed_by="tester",
        informed_at="2026-11-23T10:00:00",
        method="test",
        candidate_agreed=True,
    )
    return session


def test_snapshot_lists_newest_first_and_dismiss_changes_no_question_state(
    settings: Settings,
) -> None:
    session = _consented_sample()
    assert session.job.info_pack  # the demo pack ships with sample_session
    client = TestClient(create_app(settings, session))
    cockpit = client.app.state.cockpit
    cockpit.transcript[:] = [
        _cand("Jakie benefity oferujecie?", 1.0),
        _cand("Mam pytanie. Czy jest możliwość relokacji?", 9.0),
        _cand("Thanks, that is clear.", 12.0),
    ]
    before = session.counts()
    pack = client.get("/session").json()["pack"]
    assert pack["sections"] == 7
    # keys are "<line>:<n-th question sentence of the line>"
    assert [(a["key"], a["found"]) for a in pack["answers"]] == [("1:0", False), ("0:0", True)]
    assert pack["answers"][0]["question"] == "Czy jest możliwość relokacji?"
    assert pack["answers"][1]["heading"] == "Benefity / Benefits"
    r = client.post("/pack/0:0/dismiss")
    assert r.status_code == 200 and r.json()["dismissed"] == ["0:0"]
    assert [a["key"] for a in client.get("/session").json()["pack"]["answers"]] == ["1:0"]
    assert session.counts() == before  # D16: the panel never moves a question


def test_pack_dismiss_is_consent_gated(settings: Settings) -> None:
    client = TestClient(create_app(settings, load_session(SAMPLE)))  # empty consent block
    assert client.post("/pack/0:0/dismiss").status_code == 403


def test_panel_is_capped_to_the_latest_questions(settings: Settings) -> None:
    from recruiter_copilot.server import PACK_PANEL_MAX

    session = _consented_sample()
    lines = [_cand("What benefits do you offer?", float(i)) for i in range(PACK_PANEL_MAX + 3)]
    answers = pack_view(session, lines, settings)["answers"]
    assert len(answers) == PACK_PANEL_MAX
    assert answers[0]["line_index"] == PACK_PANEL_MAX + 2  # newest first


# ── isolation: the pack never reaches the analysis prompts or the report (D11/D17/D26) ──────

MARKER = "ZETA-PACK-MARKER-7731"


class _RecordingChat:
    name = "stub"
    model_name = "stub"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, prompt: str, system: str | None = None, max_tokens: int | None = None):
        self.prompts.append(f"{system or ''}\n{prompt}")

        class _Reply:
            text = "{}"
            truncated = False
            truncation_signals: list[str] = []

        return _Reply()


def _mock_lines() -> list[TranscriptLine]:
    turns = json.loads((FIXTURES / "mock_interview.json").read_text(encoding="utf-8"))["turns"]
    return [
        TranscriptLine(
            t_start=t["t_start"],
            t_end=t["t_end"],
            text=t["text"],
            speaker=Speaker(t["speaker"]),
            lang=t["lang"],
        )
        for t in turns
    ]


def test_pack_never_enters_analysis_prompts_or_report() -> None:
    from recruiter_copilot.analysis import _source_docs, analyse
    from recruiter_copilot.report import build_report, render_html, render_markdown

    session = load_session(SAMPLE)
    session.job.info_pack += f"\n## Secret perk\n- {MARKER}\n"
    assert MARKER in session.job.info_pack
    chat = _RecordingChat()
    result = analyse(session, _mock_lines(), chat)
    assert chat.prompts, "the stub must have been asked at least once"
    assert not any(MARKER in p for p in chat.prompts)
    assert not any("Benefity / Benefits" in p for p in chat.prompts)
    assert not any(MARKER in v for v in _source_docs(session, {}).values())
    report = build_report(session, result, Settings(), generated_at="2026-11-23T10:00:00")
    assert MARKER not in render_markdown(report)
    assert MARKER not in render_html(report)


# ── old bundles load unchanged; the pack is additive (schema stays v1) ──────────────────────


def test_old_bundle_without_info_pack_key_loads_unchanged() -> None:
    raw = json.loads((DEMO / "session.json").read_text(encoding="utf-8"))
    assert "info_pack" not in raw["job"]  # an existing pre-D26 bundle
    session = from_dict(Session, raw)
    assert session.job.info_pack == "" and session.schema_version == 1
    job = from_dict(Job, {"title": "t", "raw_text": "r"})
    assert job.info_pack == "" and job.raw_text == "r"
    # the loader leaves a folder without docs/job_info.md as it was (read only, never written)
    loaded = load_session(DEMO)
    assert loaded.job.info_pack == ""
    assert [q.id for q in loaded.questions] == [q["id"] for q in raw["questions"]]
    assert bundle_summary(loaded)["docs"]["job_info"] is False


def test_job_info_doc_folds_into_the_bundle_and_json_value_wins(tmp_path: Path) -> None:
    session = load_session(SAMPLE)
    assert session.job.info_pack.startswith("# AI Engineer")
    assert bundle_summary(session)["docs"]["job_info"] is True
    folder = tmp_path / "s"
    shutil.copytree(SAMPLE, folder)
    session.job.info_pack = "## Hand edited\n- kept"
    save_session(session, folder)
    assert load_session(folder).job.info_pack == "## Hand edited\n- kept"


# ── CFG: CLI > env > config > default; out-of-range refused ─────────────────────────────────


def test_threshold_precedence_and_validation(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("INFO_PACK_MIN_CONFIDENCE=0.5\n", encoding="utf-8")
    assert load_settings(env_file=tmp_path / "none", environ={}).info_pack_min_confidence == 0.36
    assert load_settings(env_file=env_file, environ={}).info_pack_min_confidence == 0.5
    env = {"INFO_PACK_MIN_CONFIDENCE": "0.4"}
    assert load_settings(env_file=env_file, environ=env).info_pack_min_confidence == 0.4
    cli = load_settings({"info_pack_min_confidence": 0.3}, env_file=env_file, environ=env)
    assert cli.info_pack_min_confidence == 0.3
    with pytest.raises(SettingsError):
        load_settings({"info_pack_min_confidence": 1.5}, env_file=tmp_path / "none", environ={})


def test_serve_cli_exposes_the_threshold_flag() -> None:
    from recruiter_copilot.cli import build_parser

    args = build_parser().parse_args(
        ["serve", "x", "--info-pack-min-confidence", "0.42", "--env-file", "/dev/null"]
    )
    assert args.info_pack_min_confidence == 0.42


# ── G10: the bench the default threshold was picked on still says so ────────────────────────


def _bench_module():
    spec = importlib.util.spec_from_file_location(
        "bench_info_pack", PROJECT_ROOT / "scripts" / "bench_info_pack.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_bench_at_the_default_threshold_matches_the_measurement() -> None:
    bench = _bench_module()
    pack, rows = bench.load_bench(FIXTURES / "info_pack_bench.json")
    kinds = [r["kind"] for r in rows]
    assert kinds.count("covered") >= 20 and kinds.count("uncovered") >= 8
    r = bench.evaluate(pack, rows, Settings().info_pack_min_confidence, PACK_TOKEN_PREFIX)
    assert (r["tp"], r["fp"], r["fn"]) == (25, 1, 3)
    assert r["precision"] >= 0.95 and r["recall"] >= 0.85
