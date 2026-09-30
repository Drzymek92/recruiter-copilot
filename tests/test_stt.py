"""The STT seam: language policy from the session (D20), decode planning, provider switching."""

from __future__ import annotations

import numpy as np
import pytest

from recruiter_copilot.audio import Segment
from recruiter_copilot.config import Settings
from recruiter_copilot.models import (
    Candidate,
    Job,
    Languages,
    Profile,
    Question,
    Requirement,
    Session,
    Speaker,
)
from recruiter_copilot.stt import (
    OWNERSHIP_SEEDS,
    ApiSttProvider,
    DecodedSegment,
    LanguagePolicy,
    LanguageRun,
    LanguageWindow,
    LocalWhisperProvider,
    build_bias_prompt,
    build_bias_prompts,
    build_stt_provider,
    language_runs,
    spans_from_runs,
    transcribe_segments,
    weighted_avg_logprob,
    write_transcript,
)


class FakeWhisper:
    """Stands in for WhisperModel.

    ``detect_langs`` is consumed in call order, repeating the last entry once exhausted. The
    call order the planner produces is: **1 whole-segment vote**, then (if the session is
    bilingual and the buffer long enough) **2 end probes**, then the full sliding scan.
    """

    def __init__(self, detect: tuple[str, float] = ("pl", 0.99), detect_langs=None) -> None:
        self.detect_result = detect
        self.detect_langs = list(detect_langs or [])
        self.calls: list[tuple[int, str]] = []
        self.detect_calls = 0

    def transcribe(self, audio, beam_size, language, vad_filter):  # noqa: ARG002
        self.calls.append((len(audio), language))
        piece = type(
            "S", (), {"start": 0.0, "end": 1.0, "text": f"[{language}]", "avg_logprob": -0.3}
        )
        return iter([piece()]), None

    def detect_language(self, audio):  # noqa: ARG002
        i = self.detect_calls
        self.detect_calls += 1
        if self.detect_langs:
            return self.detect_langs[min(i, len(self.detect_langs) - 1)], 0.95, None
        return (*self.detect_result, None)


def _settings(**kw) -> Settings:
    return Settings(**kw)


def test_policy_comes_from_the_session_not_a_constant() -> None:
    p = LanguagePolicy.from_languages(Languages("de", "fr"))
    assert p.primary == "de" and p.candidates == ("de", "fr") and p.is_bilingual
    mono = LanguagePolicy.from_languages(Languages("en"))
    assert not mono.is_bilingual and mono.candidates == ("en",)
    dup = LanguagePolicy.from_languages(Languages("en", "en"))
    assert dup.candidates == ("en",), "a duplicated secondary must not create a second candidate"


def test_uncertain_detection_stays_in_the_primary_language() -> None:
    """The measured rule: a short/unsure segment is never gambled on the secondary."""
    s = _settings(stt_language_min_prob=0.7)
    provider = LocalWhisperProvider(s, model=FakeWhisper(detect=("en", 0.55)))
    policy = LanguagePolicy.from_languages(Languages("pl", "en"))
    lang, prob = provider.decide_language(np.zeros(16000, np.float32), policy)
    assert lang == "pl" and prob == pytest.approx(0.55)


def test_confident_secondary_detection_is_accepted() -> None:
    s = _settings(stt_language_min_prob=0.7)
    provider = LocalWhisperProvider(s, model=FakeWhisper(detect=("en", 0.95)))
    policy = LanguagePolicy.from_languages(Languages("pl", "en"))
    assert provider.decide_language(np.zeros(16000, np.float32), policy)[0] == "en"


def test_a_language_outside_the_session_pair_is_never_used() -> None:
    s = _settings()
    provider = LocalWhisperProvider(s, model=FakeWhisper(detect=("de", 0.99)))
    policy = LanguagePolicy.from_languages(Languages("pl", "en"))
    assert provider.decide_language(np.zeros(16000, np.float32), policy)[0] == "pl"


def test_detection_failure_falls_back_and_never_raises() -> None:
    class Broken(FakeWhisper):
        def detect_language(self, audio):
            raise RuntimeError("probe exploded")

    provider = LocalWhisperProvider(_settings(), model=Broken())
    lang, prob = provider.decide_language(
        np.zeros(16000, np.float32), LanguagePolicy.from_languages(Languages("pl", "en"))
    )
    assert lang == "pl" and prob == 0.0


def test_monolingual_session_never_runs_the_switch_detector() -> None:
    model = FakeWhisper()
    provider = LocalWhisperProvider(_settings(), model=model)
    plan = provider.plan_decode(
        np.zeros(16000 * 20, np.float32), LanguagePolicy.from_languages(Languages("en"))
    )
    assert plan.mode == "off" and not plan.switch


def test_short_segment_skips_the_switch_detector() -> None:
    provider = LocalWhisperProvider(_settings(stt_codeswitch_min_seconds=10.0), model=FakeWhisper())
    plan = provider.plan_decode(
        np.zeros(16000 * 3, np.float32), LanguagePolicy.from_languages(Languages("pl", "en"))
    )
    assert plan.mode == "off"


def test_agreeing_ends_skip_the_expensive_full_scan() -> None:
    """The two-stage detector: a full sliding scan costs ~10x the two end probes."""
    # whole-segment vote, then head and tail probes — all pl, so no escalation.
    model = FakeWhisper(detect_langs=["pl", "pl", "pl"])
    provider = LocalWhisperProvider(_settings(), model=model)
    plan = provider.plan_decode(
        np.zeros(16000 * 20, np.float32), LanguagePolicy.from_languages(Languages("pl", "en"))
    )
    assert plan.mode == "off"
    assert model.detect_calls == 3, "1 whole-segment vote + 2 end probes, and no full scan"


def test_disagreeing_ends_escalate_to_a_split() -> None:
    # vote=pl, head=pl, tail=en → disagree → full scan votes pl,pl,en,en,... → two runs → split
    model = FakeWhisper(detect_langs=["pl", "pl", "en", "pl", "pl", "en", "en"])
    provider = LocalWhisperProvider(_settings(stt_codeswitch_mode="split"), model=model)
    plan = provider.plan_decode(
        np.zeros(16000 * 30, np.float32), LanguagePolicy.from_languages(Languages("pl", "en"))
    )
    assert plan.switch and plan.mode == "split"
    assert {sp.language for sp in plan.spans} == {"pl", "en"}
    assert model.detect_calls > 3, "a disagreement must escalate to the full scan"


def test_codeswitch_mode_off_restores_the_single_vote() -> None:
    model = FakeWhisper(detect_langs=["pl", "pl", "en", "pl", "en"])
    provider = LocalWhisperProvider(_settings(stt_codeswitch_mode="off"), model=model)
    plan = provider.plan_decode(
        np.zeros(16000 * 30, np.float32), LanguagePolicy.from_languages(Languages("pl", "en"))
    )
    assert plan.mode == "off" and not plan.switch


def test_split_decodes_each_side_in_its_own_language() -> None:
    model = FakeWhisper(detect_langs=["pl", "pl", "en", "pl", "pl", "en", "en"])
    provider = LocalWhisperProvider(_settings(stt_codeswitch_mode="split"), model=model)
    result = provider.decode(
        np.zeros(16000 * 30, np.float32), LanguagePolicy.from_languages(Languages("pl", "en"))
    )
    assert result.code_switch and result.decode_passes >= 2
    assert set(result.languages) == {"pl", "en"}
    decoded_languages = {lang for _, lang in model.calls}
    assert decoded_languages == {"pl", "en"}


def test_weighted_avg_logprob_weights_by_duration_not_count() -> None:
    """An unweighted mean lets a 0.3 s interjection outvote a 20 s sentence."""
    pieces = [(0.0, 20.0, -0.1), (20.0, 20.3, -5.0)]
    assert weighted_avg_logprob(pieces) == pytest.approx(-0.172, abs=1e-3)
    assert weighted_avg_logprob([]) == 0.0


def test_language_runs_drop_single_window_blips() -> None:
    windows = [
        LanguageWindow(0, 3, "pl", 0.9),
        LanguageWindow(3, 6, "pl", 0.9),
        LanguageWindow(6, 9, "en", 0.9),  # one dissenting window = a detector blip
        LanguageWindow(9, 12, "pl", 0.9),
        LanguageWindow(12, 15, "pl", 0.9),
    ]
    assert [r.language for r in language_runs(windows, min_windows=2)] == ["pl", "pl"]
    assert len(language_runs(windows, min_windows=1)) == 3


def test_spans_cover_the_whole_buffer_and_cut_at_gap_midpoints() -> None:
    runs = [LanguageRun("pl", 0.0, 10.0, 3), LanguageRun("en", 14.0, 30.0, 4)]
    spans = spans_from_runs(runs, 30.0)
    assert spans[0].t0 == 0.0 and spans[-1].t1 == 30.0
    assert spans[0].t1 == pytest.approx(12.0)  # midpoint of the 10→14 gap
    assert [s.language for s in spans] == ["pl", "en"]
    assert spans_from_runs([], 30.0) == []


class FakeProvider:
    name = "fake"

    def decode(self, audio, policy):  # noqa: ARG002
        return DecodedSegment("hello", policy.primary, 0.9, 0.01, len(audio) / 16000)


def test_transcribe_segments_preserves_timing_and_speaker() -> None:
    segs = [
        Segment(1, 0.0, 2.0, b"\x00\x00" * 100, 2.0, Speaker.INTERVIEWER),
        Segment(2, 3.0, 5.0, b"\x00\x00" * 100, 2.0, Speaker.CANDIDATE),
    ]
    lines, decoded = transcribe_segments(
        segs, FakeProvider(), LanguagePolicy.from_languages(Languages("pl", "en"))
    )
    assert [line.speaker for line in lines] == [Speaker.INTERVIEWER, Speaker.CANDIDATE]
    assert lines[0].t_start == 0.0 and lines[1].t_end == 5.0
    assert all(line.lang == "pl" for line in lines) and len(decoded) == 2


def test_empty_decodes_produce_no_transcript_line() -> None:
    class Silent(FakeProvider):
        def decode(self, audio, policy):  # noqa: ARG002
            return DecodedSegment("", policy.primary, 0.0, 0.01, 1.0)

    segs = [Segment(1, 0.0, 2.0, b"\x00\x00" * 100, 2.0)]
    lines, decoded = transcribe_segments(
        segs, Silent(), LanguagePolicy.from_languages(Languages("pl"))
    )
    assert lines == [] and len(decoded) == 1


def test_write_transcript_is_readable(tmp_path) -> None:
    segs = [Segment(1, 65.0, 70.0, b"\x00\x00" * 100, 5.0, Speaker.INTERVIEWER)]
    lines, _ = transcribe_segments(
        segs, FakeProvider(), LanguagePolicy.from_languages(Languages("pl"))
    )
    text = write_transcript(lines, tmp_path / "t.txt").read_text(encoding="utf-8")
    assert "[01:05] interviewer (pl): hello" in text


def test_provider_switch_follows_the_profile() -> None:
    assert isinstance(build_stt_provider(Settings()), LocalWhisperProvider)
    api = Settings(profile=Profile.API, openai_api_key="k")
    assert isinstance(build_stt_provider(api), ApiSttProvider)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        ApiSttProvider(Settings(profile=Profile.API))


# ── term-biasing (#976) — pure logic, no model ─────────────────────────────────────────────


def _bias_session() -> Session:
    """A minimal bilingual session bundle for the bias-prompt builders."""
    return Session(
        id="s",
        created_at="2026-01-01T00:00:00+00:00",
        job=Job(title="AI Engineer", requirements=[Requirement(id="r1", text="production Python")]),
        candidate=Candidate(display_name="X"),
        languages=Languages(primary="pl", secondary="en"),
        questions=[
            Question(id="q1", text={"pl": "Opowiedz o projekcie.", "en": "Tell me about it."}),
            Question(id="q4", text={"en": "Describe a trade-off."}),  # en-only
        ],
    )


def test_bias_prompt_default_is_ownership_seeds_only() -> None:
    # measured on the fixture: the domain vocabulary looped the decoder; seeds-only is what ships
    prompt = build_bias_prompt(_bias_session(), "pl", max_chars=1000)
    assert prompt == " ".join(OWNERSHIP_SEEDS["pl"])
    assert "Opowiedz o projekcie." not in prompt  # no question wordings by default
    assert "AI Engineer" not in prompt  # no job nouns by default


def test_bias_prompt_include_vocab_prepends_domain_and_keeps_seeds_last() -> None:
    prompt = build_bias_prompt(_bias_session(), "pl", max_chars=1000, include_vocab=True)
    # in-language question wording is present; the en-only question is NOT mixed in
    assert "Opowiedz o projekcie." in prompt
    assert "Describe a trade-off." not in prompt
    # job/domain nouns lead
    assert prompt.startswith("AI Engineer")
    # the load-bearing ownership seeds still sit at the very END (the half Whisper keeps)
    assert prompt.rstrip().endswith(OWNERSHIP_SEEDS["pl"][-1])


def test_bias_prompt_does_not_leak_the_target_phrase() -> None:
    # honesty guard: the generic seeds must not hand the decoder the fixture's expected sentence
    for lang in ("pl", "en"):
        prompt = build_bias_prompt(_bias_session(), lang, max_chars=1000).lower()
        assert "sam zbudowałem cały ten system" not in prompt


def test_bias_prompt_keeps_the_tail_when_capped() -> None:
    prompt = build_bias_prompt(_bias_session(), "pl", max_chars=40)
    assert len(prompt) <= 40
    # truncation keeps the tail, so an ownership seed still survives the cap
    assert prompt.rstrip().endswith(OWNERSHIP_SEEDS["pl"][-1])


def test_build_bias_prompts_covers_each_candidate_language() -> None:
    session = _bias_session()
    policy = LanguagePolicy.from_languages(session.languages)
    prompts = build_bias_prompts(session, policy, Settings(stt_bias_max_chars=1000))
    assert set(prompts) == {"pl", "en"}
    assert "Ja to zaprojektowałem." in prompts["pl"]
    assert "I designed it." in prompts["en"]


class _CapturingWhisper(FakeWhisper):
    """Records the kwargs each transcribe call receives, to assert initial_prompt threading."""

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.transcribe_kwargs: list[dict] = []

    def transcribe(self, audio, **kwargs):
        self.transcribe_kwargs.append(kwargs)
        piece = type("S", (), {"start": 0.0, "end": 1.0, "text": "ok", "avg_logprob": -0.3})
        return iter([piece()]), None


def test_decode_passes_initial_prompt_only_when_biased() -> None:
    audio = np.zeros(16000, dtype=np.float32)
    policy = LanguagePolicy.from_languages(Languages("pl"))

    biased = LocalWhisperProvider(
        _settings(), model=_CapturingWhisper(), bias_prompts={"pl": "Sam wdrożyłem ten moduł."}
    )
    biased.decode(audio, policy)
    assert biased._model.transcribe_kwargs[0].get("initial_prompt") == "Sam wdrożyłem ten moduł."

    plain = LocalWhisperProvider(_settings(), model=_CapturingWhisper())
    plain.decode(audio, policy)
    # unbiased path threads no initial_prompt at all (pre-#976 behaviour byte-for-byte)
    assert "initial_prompt" not in plain._model.transcribe_kwargs[0]


def test_build_stt_provider_biases_only_when_flag_and_session_present() -> None:
    session = _bias_session()
    off = build_stt_provider(Settings(), session=session)
    assert isinstance(off, LocalWhisperProvider) and off.bias_prompts == {}
    on = build_stt_provider(Settings(stt_bias_prompt=True), session=session)
    assert isinstance(on, LocalWhisperProvider) and on.bias_prompts
    # flag on but no session → nothing to build from, so no biasing
    no_session = build_stt_provider(Settings(stt_bias_prompt=True))
    assert isinstance(no_session, LocalWhisperProvider) and no_session.bias_prompts == {}
