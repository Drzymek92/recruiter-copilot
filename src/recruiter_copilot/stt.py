"""Speech-to-text providers (D12: one interface, two profiles).

``LocalWhisperProvider`` runs faster-whisper on the GPU (or CPU) and is the primary path;
``ApiSttProvider`` posts audio to an OpenAI-compatible transcription endpoint and is the
keyless-machine fallback. Both return the same ``list[TranscriptLine]`` so nothing downstream
knows which ran.

Vendored from interview_copilot (D19) with one substantive change: the language pair is no
longer the constant ``pl,en`` — it comes from the session's ``Languages`` (D20), which is what
makes the app usable for any language pair rather than the author's own.

Three measured behaviours are preserved, and undoing any of them regresses a real call:

* **Detection is biased to the primary language.** A short or uncertain segment stays primary
  rather than being gambled on the secondary; only a candidate language clearing
  ``stt_language_min_prob`` flips it. Whisper forced onto the wrong language *translates*.
* **A segment that straddles a language switch cannot be served by one language.** The
  whole-segment vote was right on 108 of 110 real segments and wrong on both that straddled a
  switch — including one where it was 99% confident. ``split`` cuts the *audio* at the switch
  and decodes each side in its own language; segmentation is untouched.
* **The switch detector is two-stage.** A full sliding scan costs ~1.4 s on a 30 s segment, so
  the default probes only the first and last window and escalates only when they disagree.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from .audio import Segment
from .config import Settings
from .models import Languages, Session, Speaker, TranscriptLine

logger = logging.getLogger("recruiter_copilot.stt")


@dataclass
class LanguagePolicy:
    """What a session's ``Languages`` means for the decoder (D20)."""

    primary: str
    candidates: tuple[str, ...]

    @classmethod
    def from_languages(cls, languages: Languages) -> LanguagePolicy:
        candidates = [languages.primary]
        if languages.secondary and languages.secondary != languages.primary:
            candidates.append(languages.secondary)
        return cls(primary=languages.primary, candidates=tuple(candidates))

    @property
    def is_bilingual(self) -> bool:
        return len(self.candidates) > 1


@dataclass
class DecodedSegment:
    """A decoded segment plus the measurements that justify the decode path taken."""

    text: str
    language: str
    language_probability: float
    latency_seconds: float
    audio_seconds: float
    code_switch: bool = False
    languages: tuple[str, ...] = ()
    decode_passes: int = 1
    avg_logprob: float = 0.0

    @property
    def realtime_factor(self) -> float:
        return self.latency_seconds / self.audio_seconds if self.audio_seconds else 0.0


class SttProvider(Protocol):
    """The seam every caller uses. Implementations must not know about audio devices."""

    name: str

    def decode(self, audio: np.ndarray, policy: LanguagePolicy) -> DecodedSegment: ...


# ── language-planning helpers (pure; tested without a model) ──────────────────────────────


@dataclass
class LanguageWindow:
    t0: float
    t1: float
    language: str
    probability: float


@dataclass
class LanguageRun:
    language: str
    t0: float
    t1: float
    n_windows: int


@dataclass
class LanguageSpan:
    t0: float
    t1: float
    language: str


@dataclass
class DecodePlan:
    mode: str  # "off" | "rescore" | "split"
    language: str
    probability: float
    switch: bool = False
    languages: tuple[str, ...] = ()
    spans: list[LanguageSpan] = field(default_factory=list)


def weighted_avg_logprob(pieces: list[tuple[float, float, float]]) -> float:
    """Duration-weighted mean logprob over ``(start, end, avg_logprob)`` triples.

    Weighted by duration, not by count: an unweighted mean lets a 0.3 s interjection outvote a
    20 s sentence, and two decodes of the same audio do not agree on how many pieces it holds
    (measured: 12 under one language vs 6 under the other for the same 30 s).
    """
    total = sum(max(0.0, e - s) for s, e, _ in pieces)
    if not pieces or total <= 0:
        return 0.0
    return sum(lp * max(0.0, e - s) for s, e, lp in pieces) / total


def language_runs(windows: list[LanguageWindow], min_windows: int) -> list[LanguageRun]:
    """Collapse per-window votes into runs, dropping runs shorter than ``min_windows``.

    A genuine switch holds for several windows; a single dissenting window is a detector blip.
    """
    runs: list[LanguageRun] = []
    for w in windows:
        if runs and runs[-1].language == w.language:
            runs[-1].t1 = w.t1
            runs[-1].n_windows += 1
        else:
            runs.append(LanguageRun(w.language, w.t0, w.t1, 1))
    return [r for r in runs if r.n_windows >= max(1, min_windows)]


def spans_from_runs(runs: list[LanguageRun], total_seconds: float) -> list[LanguageSpan]:
    """Contiguous decode spans covering the whole buffer; boundaries at gap midpoints.

    Neither side is cut inside the other's speech, the first span starts at 0.0 and the last
    ends at the buffer's end, so the split drops no audio.
    """
    if not runs:
        return []
    spans: list[LanguageSpan] = []
    for i, run in enumerate(runs):
        t0 = 0.0 if i == 0 else spans[-1].t1
        t1 = total_seconds if i == len(runs) - 1 else (run.t1 + runs[i + 1].t0) / 2.0
        if t1 > t0:
            spans.append(LanguageSpan(t0, t1, run.language))
    if spans:
        spans[-1].t1 = total_seconds
    return spans


# ── term-biasing (pure; tested without a model) — #976 ─────────────────────────────────────

# First-person / sole-ownership phrasing, per language. This repairs a specific corruption class:
# faster-whisper drops a near-homophone short function word that carries an OWNERSHIP cue — on the
# fixture "Sam zbudowałem cały ten system" (I ALONE built) decoded as "Tam zbudowałem" (THERE I
# built), erasing the sole-ownership signal the analyser needs. Priming the decoder's prompt with
# the domain's expected first-person ownership vocabulary biases it away from that drop.
#
# These are GENERIC ownership phrasings, deliberately NOT the fixture's target sentence — biasing
# with the literal expected text would rig the measurement. "Sam" appears here only as ordinary
# first-person emphatic vocabulary ("Sam wdrożyłem ten moduł"), never as the target phrase.
OWNERSHIP_SEEDS: dict[str, tuple[str, ...]] = {
    "pl": ("Ja to zaprojektowałem.", "Zrobiłem to sam.", "Sam wdrożyłem ten moduł."),
    "en": ("I designed it.", "I did it alone.", "I built the module myself."),
}


def _dedup_keep_order(items: list[str]) -> list[str]:
    """Trimmed, non-empty, case-insensitively de-duplicated, order preserved."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = item.strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def build_bias_prompt(
    session: Session,
    language: str,
    *,
    max_chars: int = 120,
    include_vocab: bool = False,
    ownership_seeds: dict[str, tuple[str, ...]] | None = None,
) -> str:
    """Assemble a faster-whisper ``initial_prompt`` for decoding ``language`` in this session (#976).

    Pure — no model, no I/O. By default the prompt is ONLY the first-person ownership seeds for
    ``language`` — that is the content measured to help (see below). With ``include_vocab=True`` the
    session's domain vocabulary (job title + requirement text, plus the question-bank wordings that
    exist in this language) is prepended; the seeds always come LAST because Whisper keeps only the
    tail of an over-long ``initial_prompt``, and when the text exceeds ``max_chars`` the tail is what
    survives. Returns ``""`` when there is nothing to bias with.

    Measured on the fixture (#976, faster-whisper large-v3-turbo, 2026-09-13): seeds-only recovered
    the dropped ownership cue ("Tam zbudowałem" → "Sam zbudowałem") with the segment otherwise
    intact. ``include_vocab=True`` (a ~400-char list of question wordings) instead drove the Polish
    decoder into repetition loops that DESTROYED whole segments and did NOT recover the cue — hence
    the domain vocabulary is opt-in and off by default. Keep prompts short.
    """
    seeds = ownership_seeds or OWNERSHIP_SEEDS
    parts: list[str] = []
    if include_vocab:
        if session.job.title:
            parts.append(session.job.title)
        parts.extend(r.text for r in session.job.requirements if r.text)
        # only wordings actually present in this language — mixing languages is avoided
        for q in session.questions:
            wording = q.text.get(language, "")
            if wording.strip():
                parts.append(wording)
    parts.extend(seeds.get(language, ()))

    prompt = " ".join(_dedup_keep_order(parts))
    if max_chars > 0 and len(prompt) > max_chars:
        prompt = prompt[-max_chars:].lstrip()  # keep the tail — the half Whisper itself keeps
    return prompt


def build_bias_prompts(
    session: Session, policy: LanguagePolicy, settings: Settings
) -> dict[str, str]:
    """One ``initial_prompt`` per session candidate language (pure); empty prompts are dropped."""
    out: dict[str, str] = {}
    for lang in policy.candidates:
        prompt = build_bias_prompt(
            session,
            lang,
            max_chars=settings.stt_bias_max_chars,
            include_vocab=settings.stt_bias_include_vocab,
        )
        if prompt:
            out[lang] = prompt
    return out


# ── local provider ────────────────────────────────────────────────────────────────────────


class LocalWhisperProvider:
    """faster-whisper on CUDA (or CPU). Audio never leaves the machine (SI1)."""

    def __init__(
        self,
        settings: Settings,
        model: object | None = None,
        bias_prompts: dict[str, str] | None = None,
    ) -> None:
        self.settings = settings
        self.device = settings.resolved_stt_device()
        self.compute_type = settings.resolved_stt_compute_type()
        self.name = f"faster-whisper:{settings.stt_model}[{self.device}/{self.compute_type}]"
        self._model = model
        # language → initial_prompt (#976). Empty by default: an unbiased decode passes no
        # initial_prompt at all, so behaviour is byte-for-byte the pre-#976 path.
        self.bias_prompts: dict[str, str] = dict(bias_prompts or {})

    @property
    def model(self) -> object:
        if self._model is None:
            from faster_whisper import WhisperModel  # noqa: PLC0415 — optional [local] extra

            logger.info(
                "loading faster-whisper model=%s device=%s compute_type=%s",
                self.settings.stt_model,
                self.device,
                self.compute_type,
            )
            t0 = time.perf_counter()
            self._model = WhisperModel(
                self.settings.stt_model, device=self.device, compute_type=self.compute_type
            )
            logger.info("model loaded in %.2fs", time.perf_counter() - t0)
        return self._model

    def decode(self, audio: np.ndarray, policy: LanguagePolicy) -> DecodedSegment:
        audio = np.asarray(audio, dtype=np.float32).flatten()
        sr = self.settings.sample_rate
        audio_seconds = len(audio) / sr if sr else 0.0

        t0 = time.perf_counter()
        plan = self.plan_decode(audio, policy)
        if plan.mode == "rescore":
            pieces, language, passes = self._decode_rescored(audio, plan.languages)
        elif plan.mode == "split":
            pieces, language, passes = self._decode_split(audio, plan.spans, policy)
        else:
            pieces = self._decode_once(audio, plan.language)
            language, passes = plan.language, 1
        latency = time.perf_counter() - t0

        text = " ".join(p[3] for p in pieces if p[3]).strip()
        used: list[str] = []
        for p in pieces:
            if p[4] and p[4] not in used:
                used.append(p[4])
        return DecodedSegment(
            text=text,
            language=language,
            language_probability=plan.probability,
            latency_seconds=latency,
            audio_seconds=audio_seconds,
            code_switch=plan.switch,
            languages=tuple(used) or (language,),
            decode_passes=passes,
            avg_logprob=weighted_avg_logprob([(p[0], p[1], p[2]) for p in pieces]),
        )

    # -- decoding primitives; a piece is (start, end, avg_logprob, text, language) --
    def _decode_once(
        self, audio: np.ndarray, language: str, offset: float = 0.0
    ) -> list[tuple[float, float, float, str, str]]:
        kwargs: dict[str, object] = {
            "beam_size": self.settings.stt_beam_size,
            "language": language,
            "vad_filter": True,
        }
        prompt = self.bias_prompts.get(language)
        if prompt:  # only when biasing is on and there is a prompt for this language (#976)
            kwargs["initial_prompt"] = prompt
        segments_iter, _info = self.model.transcribe(audio, **kwargs)  # type: ignore[attr-defined]
        return [
            (s.start + offset, s.end + offset, float(s.avg_logprob), s.text.strip(), language)
            for s in segments_iter
        ]

    def _decode_rescored(
        self, audio: np.ndarray, languages: tuple[str, ...]
    ) -> tuple[list[tuple[float, float, float, str, str]], str, int]:
        """Decode the whole buffer once per language and keep the better score.

        Kept because it is the honest comparison for ``split``, and it repairs a straddling
        segment — but it still picks ONE language, so on 70%-secondary audio it translates the
        primary half. ``split`` is the default for that reason.
        """
        best: list[tuple[float, float, float, str, str]] = []
        best_lang, best_score, passes = languages[0], -float("inf"), 0
        for lang in languages:
            pieces = self._decode_once(audio, lang)
            passes += 1
            score = weighted_avg_logprob([(p[0], p[1], p[2]) for p in pieces])
            if score > best_score:
                best, best_lang, best_score = pieces, lang, score
        return best, best_lang, passes

    def _decode_split(
        self, audio: np.ndarray, spans: list[LanguageSpan], policy: LanguagePolicy
    ) -> tuple[list[tuple[float, float, float, str, str]], str, int]:
        """Cut the AUDIO at the detected switch and decode each side in its own language.

        Segmentation is not touched — the caller's segment keeps its start/end and stays one
        transcript line. The reported language is whichever covers the most audio.
        """
        sr = self.settings.sample_rate
        out: list[tuple[float, float, float, str, str]] = []
        passes = 0
        for span in spans:
            chunk = audio[int(span.t0 * sr) : int(span.t1 * sr)]
            if chunk.size < int(0.5 * sr):
                continue
            out.extend(self._decode_once(chunk, span.language, offset=span.t0))
            passes += 1
        dominant = max(spans, key=lambda sp: sp.t1 - sp.t0).language if spans else policy.primary
        return out, dominant, max(1, passes)

    # -- language planning --
    def plan_decode(self, audio: np.ndarray, policy: LanguagePolicy) -> DecodePlan:
        """Decide HOW to decode: one language, a rescore, or a split.

        The whole-segment vote stays the baseline; the switch detector only ever adds a second
        opinion, and only when sub-windows of the same audio confidently disagree.
        """
        language, probability = self.decide_language(audio, policy)
        base = DecodePlan("off", language, probability)
        s = self.settings
        if s.stt_codeswitch_mode == "off" or not s.stt_detect_language or not policy.is_bilingual:
            return base
        seconds = len(audio) / s.sample_rate if s.sample_rate else 0.0
        if seconds < s.stt_codeswitch_min_seconds:
            return base
        if s.stt_codeswitch_scan == "ends" and not self._ends_disagree(audio, policy):
            return base

        runs = language_runs(self._detect_windows(audio, policy), s.stt_codeswitch_min_windows)
        if len({r.language for r in runs}) < 2:
            return base
        spans = spans_from_runs(runs, seconds)
        ordered: list[str] = []
        for sp in spans:
            if sp.language not in ordered:
                ordered.append(sp.language)
        logger.info(
            "code-switch in %.1fs of audio (whole-segment vote %s p=%.2f): %s",
            seconds,
            language,
            probability,
            " ".join(f"{sp.t0:.1f}-{sp.t1:.1f}:{sp.language}" for sp in spans),
        )
        return DecodePlan(
            s.stt_codeswitch_mode,
            language,
            probability,
            switch=True,
            languages=tuple(ordered),
            spans=spans,
        )

    def _ends_disagree(self, audio: np.ndarray, policy: LanguagePolicy) -> bool:
        """Cheap first stage: does this buffer END in a different language than it STARTS?

        Two encoder passes against the ten a full scan costs. A switch of the utterance's
        language by definition leaves the two ends on opposite sides of it. Both ends must be
        confident; an unsure end is not evidence of anything.
        """
        sr = self.settings.sample_rate
        span = int(self.settings.stt_codeswitch_window_seconds * sr)
        if len(audio) < 2 * span:
            return False
        head = self._window_vote(audio[:span], 0.0, policy)
        tail = self._window_vote(audio[-span:], (len(audio) - span) / sr, policy)
        if head is None or tail is None:
            return False
        return head.language != tail.language

    def _window_vote(
        self, chunk: np.ndarray, t0: float, policy: LanguagePolicy
    ) -> LanguageWindow | None:
        """One confident sub-window vote, or None. A failed probe is never fatal."""
        try:
            lang, prob, _all = self.model.detect_language(chunk)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 — a failed probe must never end a decode
            logger.exception("sub-window language detection failed — skipping window")
            return None
        if lang not in policy.candidates or prob < self.settings.stt_language_min_prob:
            return None
        sr = self.settings.sample_rate
        return LanguageWindow(t0, t0 + len(chunk) / sr, lang, float(prob))

    def _detect_windows(self, audio: np.ndarray, policy: LanguagePolicy) -> list[LanguageWindow]:
        """Sliding ``detect_language`` votes; only confident candidate languages count."""
        sr = self.settings.sample_rate
        span = int(self.settings.stt_codeswitch_window_seconds * sr)
        hop = max(1, int(self.settings.stt_codeswitch_hop_seconds * sr))
        floor = int(1.0 * sr)
        out: list[LanguageWindow] = []
        for offset in range(0, max(1, len(audio)), hop):
            chunk = audio[offset : offset + span]
            if len(chunk) < floor:
                break
            vote = self._window_vote(chunk, offset / sr, policy)
            if vote is not None:
                out.append(vote)
        return out

    def decide_language(self, audio: np.ndarray, policy: LanguagePolicy) -> tuple[str, float]:
        """Choose the language to decode in — biased to the session's PRIMARY (D20).

        Detection is accepted only when it names a session candidate *and* clears the
        probability floor; otherwise the segment stays primary. A short utterance the model is
        unsure about is therefore never gambled on the secondary language.
        """
        if not self.settings.stt_detect_language or not policy.is_bilingual:
            return policy.primary, 1.0
        try:
            lang, prob, _ = self.model.detect_language(audio)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 — detection must never end a decode
            logger.exception("language detection failed — falling back to %s", policy.primary)
            return policy.primary, 0.0
        if lang in policy.candidates and prob >= self.settings.stt_language_min_prob:
            return lang, float(prob)
        logger.info(
            "language detection %s (p=%.2f) not trusted — decoding as %s",
            lang,
            prob,
            policy.primary,
        )
        return policy.primary, float(prob)


# ── api provider ──────────────────────────────────────────────────────────────────────────


class ApiSttProvider:
    """OpenAI-compatible transcription endpoint (``PROFILE=api``).

    **This is an egress (SI1).** Audio for the segment is uploaded to the configured host. The
    caller is responsible for having announced it; ``Settings.data_leaves_machine()`` is what the
    UI and the report read.
    """

    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ValueError("PROFILE=api transcription needs OPENAI_API_KEY")
        self.settings = settings
        self.model_name = settings.openai_stt_model or "whisper-1"
        self.name = f"openai-compatible:{self.model_name}"

    def decode(self, audio: np.ndarray, policy: LanguagePolicy) -> DecodedSegment:
        import io  # noqa: PLC0415
        import wave  # noqa: PLC0415

        from openai import OpenAI  # noqa: PLC0415 — core dependency, lazy so local never loads it

        sr = self.settings.sample_rate
        audio_seconds = len(audio) / sr if sr else 0.0
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
        buf.name = "segment.wav"
        buf.seek(0)

        client = OpenAI(
            api_key=self.settings.openai_api_key, base_url=self.settings.openai_base_url
        )
        t0 = time.perf_counter()
        # The endpoint takes ONE language hint, so the primary is sent: the same bias the local
        # provider applies. Per-segment code-switch splitting is a local-profile capability.
        result = client.audio.transcriptions.create(
            model=self.model_name, file=buf, language=policy.primary
        )
        latency = time.perf_counter() - t0
        language = getattr(result, "language", "") or policy.primary
        return DecodedSegment(
            text=(result.text or "").strip(),
            language=language if language in policy.candidates else policy.primary,
            language_probability=0.0,  # the endpoint reports no probability
            latency_seconds=latency,
            audio_seconds=audio_seconds,
        )


def build_stt_provider(settings: Settings, session: Session | None = None) -> SttProvider:
    """The D12 switch: one setting picks the provider, nothing downstream changes.

    When ``settings.stt_bias_prompt`` is set and a ``session`` is supplied, the local provider is
    primed with per-language ``initial_prompt``s built from the session bundle (#976). The api
    endpoint takes no prompt hint, so biasing is a local-profile capability.
    """
    from .models import Profile  # noqa: PLC0415 — avoids a cycle at import time

    if settings.profile is Profile.API:
        return ApiSttProvider(settings)
    bias_prompts: dict[str, str] | None = None
    if settings.stt_bias_prompt and session is not None:
        policy = LanguagePolicy.from_languages(session.languages)
        bias_prompts = build_bias_prompts(session, policy, settings)
    return LocalWhisperProvider(settings, bias_prompts=bias_prompts)


def transcribe_segments(
    segments: list[Segment],
    provider: SttProvider,
    policy: LanguagePolicy,
    on_line: object | None = None,
) -> tuple[list[TranscriptLine], list[DecodedSegment]]:
    """Decode every segment into transcript lines, preserving timing and speaker tags."""
    lines: list[TranscriptLine] = []
    decoded: list[DecodedSegment] = []
    for seg in segments:
        result = provider.decode(seg.to_float32(), policy)
        decoded.append(result)
        if not result.text:
            continue
        line = TranscriptLine(
            t_start=round(seg.start, 3),
            t_end=round(seg.end, 3),
            text=result.text,
            speaker=seg.speaker if isinstance(seg.speaker, Speaker) else Speaker.UNKNOWN,
            lang=result.language,
        )
        lines.append(line)
        if callable(on_line):
            on_line(line)
    return lines, decoded


def write_transcript(lines: list[TranscriptLine], path: Path) -> Path:
    """Human-readable transcript next to the machine-readable session state."""
    out = []
    for line in lines:
        stamp = f"[{int(line.t_start // 60):02d}:{int(line.t_start % 60):02d}]"
        tag = line.speaker.value if line.speaker else "unknown"
        lang = f" ({line.lang})" if line.lang else ""
        out.append(f"{stamp} {tag}{lang}: {line.text}")
    path.write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8")
    return path
