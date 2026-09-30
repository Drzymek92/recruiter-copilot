"""The post-hoc pipeline (F8): a recording in, a transcript and proposals out.

This is the path that needs no microphone, no browser and no live call, which makes it three
things at once: the offline mode an interviewer uses when they recorded the call another way,
the demo in the README, and the only path CI can exercise. M4's live loop feeds the *same*
segmenter and the *same* providers, so what this proves about them transfers.

Nothing here writes question state — the matcher's output is a list of proposals the interviewer
confirms (D16).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from dataclasses import fields as dataclass_fields
from datetime import datetime, timezone
from pathlib import Path

from .audio import AudioSource, read_wav, segment_source
from .config import Settings
from .matcher import Proposal, propose_all
from .models import Session, TranscriptLine, Speaker, to_dict
from .stt import (
    DecodedSegment,
    LanguagePolicy,
    SttProvider,
    build_stt_provider,
    transcribe_segments,
    write_transcript,
)

logger = logging.getLogger("recruiter_copilot.pipeline")

TRANSCRIPT_JSON = "transcript.json"
TRANSCRIPT_TXT = "transcript.txt"


@dataclass
class PipelineStats:
    """What the run actually cost and did — printed, and stored beside the transcript."""

    audio_seconds: float = 0.0
    segments: int = 0
    lines: int = 0
    decode_seconds: float = 0.0
    total_seconds: float = 0.0
    languages: dict[str, int] = field(default_factory=dict)
    speakers: dict[str, int] = field(default_factory=dict)
    code_switch_segments: int = 0
    decode_passes: int = 0
    stt_provider: str = ""

    @property
    def realtime_factor(self) -> float:
        return self.decode_seconds / self.audio_seconds if self.audio_seconds else 0.0


@dataclass
class PipelineResult:
    lines: list[TranscriptLine]
    proposals: list[Proposal]
    stats: PipelineStats
    decoded: list[DecodedSegment] = field(default_factory=list)


def transcribe_recording(
    session: Session,
    audio_path: Path,
    settings: Settings,
    provider: SttProvider | None = None,
    channel_map: str = "interviewer,candidate",
    on_line: object | None = None,
) -> tuple[list[TranscriptLine], PipelineStats, list[DecodedSegment]]:
    """Recording → segments → decoded transcript lines."""
    started = time.perf_counter()
    source = read_wav(Path(audio_path), channel_map=channel_map)
    if source.sample_rate != settings.sample_rate:
        source = resample(source, settings.sample_rate)
    logger.info(
        "audio: %.1fs at %d Hz, %s",
        source.duration,
        source.sample_rate,
        "two channels (speakers attributable)" if source.has_channels else "mono (no speaker tags)",
    )

    segments = segment_source(source, settings)
    logger.info("segmented into %d speech segments", len(segments))

    provider = provider or build_stt_provider(settings, session=session)
    policy = LanguagePolicy.from_languages(session.languages)
    lines, decoded = transcribe_segments(segments, provider, policy, on_line=on_line)

    stats = PipelineStats(
        audio_seconds=round(source.duration, 2),
        segments=len(segments),
        lines=len(lines),
        decode_seconds=round(sum(d.latency_seconds for d in decoded), 2),
        total_seconds=round(time.perf_counter() - started, 2),
        code_switch_segments=sum(1 for d in decoded if d.code_switch),
        decode_passes=sum(d.decode_passes for d in decoded),
        stt_provider=getattr(provider, "name", "unknown"),
    )
    for line in lines:
        stats.languages[line.lang or "?"] = stats.languages.get(line.lang or "?", 0) + 1
        key = line.speaker.value if isinstance(line.speaker, Speaker) else "unknown"
        stats.speakers[key] = stats.speakers.get(key, 0) + 1
    return lines, stats, decoded


def resample(source: AudioSource, target_rate: int) -> AudioSource:
    """Linear resample to the model's rate. Feeding audio at the wrong rate pitch-shifts it."""
    import numpy as np  # noqa: PLC0415

    if source.sample_rate == target_rate:
        return source
    ratio = target_rate / source.sample_rate

    def _rs(track):
        if track is None:
            return None
        n_out = int(len(track) * ratio)
        if n_out <= 0:
            return track[:0]
        x_old = np.arange(len(track), dtype=np.float64)
        x_new = np.linspace(0, len(track) - 1, n_out)
        return np.interp(x_new, x_old, track).astype(np.float32)

    logger.info("resampling %d Hz → %d Hz", source.sample_rate, target_rate)
    return AudioSource(
        mono=_rs(source.mono),
        sample_rate=target_rate,
        interviewer=_rs(source.interviewer),
        candidate=_rs(source.candidate),
    )


def save_transcript(
    folder: Path, lines: list[TranscriptLine], stats: PipelineStats, proposals: list[Proposal]
) -> dict[str, Path]:
    """Write the machine-readable and human-readable transcripts into the session folder."""
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stats": asdict(stats),
        "lines": [to_dict(line) for line in lines],
        "proposals": [p.as_dict() for p in proposals],
    }
    json_path = folder / TRANSCRIPT_JSON
    tmp = json_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(json_path)
    txt_path = write_transcript(lines, folder / TRANSCRIPT_TXT)
    return {"json": json_path, "txt": txt_path}


def load_transcript(folder: Path) -> list[TranscriptLine]:
    """Read back a saved transcript (M2's analyser entry point)."""
    from .models import from_dict  # noqa: PLC0415

    path = folder / TRANSCRIPT_JSON
    if not path.is_file():
        raise FileNotFoundError(str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [from_dict(TranscriptLine, row) for row in payload.get("lines", [])]


def load_transcript_payload(folder: Path) -> dict | None:
    """The raw ``transcript.json`` payload (lines + stats + proposals), or ``None`` if absent.

    Used by the live cockpit to ACCUMULATE a second capture onto an existing transcript (#994):
    the prior lines and their stats are read back, merged with the new capture and re-written into
    the SAME file, so the bundle stays one purgeable ``transcript.json`` (D18).
    """
    path = folder / TRANSCRIPT_JSON
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def stats_from_payload(payload: dict | None) -> PipelineStats:
    """Reconstruct ``PipelineStats`` from a persisted payload (empty stats when absent)."""
    if not payload:
        return PipelineStats()
    raw = payload.get("stats") or {}
    fields = {f.name for f in dataclass_fields(PipelineStats)}
    return PipelineStats(**{k: v for k, v in raw.items() if k in fields})


def run_posthoc(
    session: Session,
    folder: Path,
    audio_path: Path,
    settings: Settings,
    provider: SttProvider | None = None,
    chat: object | None = None,
    channel_map: str = "interviewer,candidate",
) -> PipelineResult:
    """The whole F8 path: recording → transcript → question proposals → files on disk."""
    lines, stats, decoded = transcribe_recording(
        session, audio_path, settings, provider=provider, channel_map=channel_map
    )
    proposals = propose_all(
        lines,
        session.questions,
        route=settings.matcher_route,
        min_confidence=settings.matcher_min_confidence,
        chat=chat,
    )
    logger.info(
        "%d transcript lines, %d question proposal(s) via route=%s",
        len(lines),
        len(proposals),
        settings.matcher_route,
    )
    save_transcript(folder, lines, stats, proposals)
    if not settings.keep_audio:
        logger.info(
            "KEEP_AUDIO=0: the recording at %s is the interviewer's own file and is left alone; "
            "no copy was made inside the session folder (D18)",
            audio_path,
        )
    return PipelineResult(lines=lines, proposals=proposals, stats=stats, decoded=decoded)
