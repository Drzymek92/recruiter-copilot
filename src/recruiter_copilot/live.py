"""Live capture plumbing (M4, D13/D22): browser PCM frames in, transcript lines out.

The browser (``static/cockpit.html``) captures the interviewer's microphone with ``getUserMedia``
and the call tab's audio with ``getDisplayMedia``, resamples both to 16 kHz mono PCM16, and streams
each to the server over **its own websocket** — ``ws:/audio?role=interviewer`` for the mic,
``ws:/audio?role=candidate`` for the tab (route A: one socket per role). A JSON ``hello`` opens a
stream, binary frames carry audio, a JSON ``end`` closes it and flushes the last segment.

Why one socket per role and one segmenter per role: the two browser sources run on independent
clocks (a ``MediaStream`` per device) and arrive in independently sized chunks, so mixing them into
the single mono frame the post-hoc path uses would need sample-level alignment the browser cannot
promise. Instead each role drives its **own** ``ChannelGate`` + ``VadSegmenter`` (the D22 seam,
unchanged) with a fixed verdict for the other role, so every segment is owned by the role that
produced it — which is exactly the headphones assumption (G5): the mic hears the interviewer, the
tab hears the candidate. Timestamps are stamped on one shared live clock so the two roles'
lines interleave in the transcript by time (``insert_line``), and save-answer / proposals work on
the live transcript exactly as they do on replay.

MOD split — everything above the websocket is pure and unit-tested with no server and no GPU:
``parse_role``, ``parse_hello``, ``FrameAccumulator``, ``RoleStream``, ``insert_line``.
``LiveCapture`` is the asyncio coordinator: it owns the per-role streams, a single decode queue and
the STT worker (``asyncio.to_thread`` — STT is blocking, the event loop never is). The websocket
endpoint itself lives in ``server.py``.
"""

from __future__ import annotations

import asyncio
import bisect
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .audio import BYTES_PER_SAMPLE, ChannelGate, Segment, VadSegmenter
from .config import Settings
from .models import Session, Speaker, TranscriptLine
from .stt import LanguagePolicy, SttProvider, transcribe_segments

logger = logging.getLogger("recruiter_copilot.live")

ROLES: dict[str, Speaker] = {
    "interviewer": Speaker.INTERVIEWER,
    "candidate": Speaker.CANDIDATE,
}
PCM_FORMAT = "pcm16"
# #1003: silence left between a prior capture's last line and the first second of a new serve
# process's live clock, so a cross-process 2nd capture lands strictly AFTER the saved transcript.
CAPTURE_GAP_SECONDS = 1.0


class LiveProtocolError(ValueError):
    """A client broke the ``ws:/audio`` contract (bad role, bad hello, wrong sample rate)."""


# ── pure helpers ────────────────────────────────────────────────────────────────────────────


def parse_role(role: str | None) -> Speaker:
    """``?role=interviewer|candidate`` → the Speaker that owns every segment on that socket."""
    key = (role or "").strip().lower()
    if key not in ROLES:
        raise LiveProtocolError(
            f"role must be one of {sorted(ROLES)}, got {role!r} (ws:/audio?role=...)"
        )
    return ROLES[key]


def frame_bytes(settings: Settings) -> int:
    """Size of one VAD frame in bytes at the server's rate (20 ms @ 16 kHz mono PCM16 = 640)."""
    return int(settings.sample_rate * settings.vad_frame_ms / 1000) * BYTES_PER_SAMPLE


def parse_hello(text: str, settings: Settings) -> dict[str, Any]:
    """Validate the first (JSON) message on an audio socket.

    ``{"type": "hello", "sample_rate": 16000, "format": "pcm16", "channels": 1}`` — the rate must
    equal the server's (the browser resamples; the server never does, so a mismatch would
    pitch-shift the decode), the format must be PCM16, mono. Returns the normalised hello.
    """
    try:
        data = json.loads(text)
    except (TypeError, ValueError) as e:
        raise LiveProtocolError(f"hello must be JSON: {e}") from e
    if not isinstance(data, dict) or data.get("type") != "hello":
        raise LiveProtocolError('first message must be {"type": "hello", ...}')
    rate = int(data.get("sample_rate", 0) or 0)
    if rate != settings.sample_rate:
        raise LiveProtocolError(
            f"sample_rate must be {settings.sample_rate} (resample in the browser), got {rate}"
        )
    fmt = str(data.get("format", PCM_FORMAT)).lower()
    if fmt != PCM_FORMAT:
        raise LiveProtocolError(f"format must be {PCM_FORMAT!r}, got {fmt!r}")
    channels = int(data.get("channels", 1) or 1)
    if channels != 1:
        raise LiveProtocolError(f"audio must be mono, got {channels} channels")
    return {"type": "hello", "sample_rate": rate, "format": fmt, "channels": 1}


@dataclass
class FrameAccumulator:
    """Re-chunk arbitrary-size PCM16 byte chunks into fixed VAD frames; the remainder carries."""

    frame_bytes: int
    buffer: bytearray = field(default_factory=bytearray)

    def push(self, chunk: bytes) -> list[bytes]:
        if len(chunk) % BYTES_PER_SAMPLE:
            raise LiveProtocolError(f"PCM16 chunk has an odd byte length ({len(chunk)})")
        self.buffer.extend(chunk)
        out: list[bytes] = []
        while len(self.buffer) >= self.frame_bytes:
            out.append(bytes(self.buffer[: self.frame_bytes]))
            del self.buffer[: self.frame_bytes]
        return out

    def flush(self) -> bytes | None:
        """The trailing partial frame zero-padded to a full one (``None`` when nothing is pending)."""
        if not self.buffer:
            return None
        frame = bytes(self.buffer) + b"\x00" * (self.frame_bytes - len(self.buffer))
        self.buffer.clear()
        return frame


class RoleStream:
    """One role's live audio: frames in (any chunking), closed ``Segment``s out, on the live clock.

    ``offset`` is where this stream's first sample sits on the shared live clock (seconds since
    the live session opened), so a role that joins late — or reconnects — still lands its lines at
    the right point of the transcript. The segmenter is fed this role's speech verdict and a
    constant ``False`` for the other role, so ``Segment.speaker`` is always this role (G5).
    """

    def __init__(
        self, role: Speaker, settings: Settings, offset: float = 0.0, vad: object | None = None
    ) -> None:
        if role not in (Speaker.INTERVIEWER, Speaker.CANDIDATE):
            raise LiveProtocolError(f"a live stream needs a role, got {role!r}")
        self.role = role
        self.settings = settings
        self.offset = offset
        self.gate = ChannelGate(settings, role.value, vad=vad)
        self.segmenter = VadSegmenter(settings)
        self.accumulator = FrameAccumulator(frame_bytes(settings))
        self.frame_samples = int(settings.sample_rate * settings.vad_frame_ms / 1000)
        self.samples = 0
        self.frames = 0
        self.segments = 0
        self.connected = False

    @property
    def seconds(self) -> float:
        """Audio received so far, in seconds (a sample-count clock, immune to network jitter)."""
        return self.samples / self.settings.sample_rate

    @property
    def clock(self) -> float:
        return self.offset + self.seconds

    def _push_frame(self, frame: bytes) -> Segment | None:
        self.samples += self.frame_samples
        self.frames += 1
        speech = self.gate.is_speech(frame)
        is_interviewer = self.role is Speaker.INTERVIEWER
        segment = self.segmenter.push(
            frame, speech and is_interviewer, speech and not is_interviewer, self.clock
        )
        if segment is not None:
            self.segments += 1
        return segment

    def push(self, chunk: bytes) -> list[Segment]:
        """Feed one binary websocket message; returns every segment that closed because of it."""
        out: list[Segment] = []
        for frame in self.accumulator.push(chunk):
            segment = self._push_frame(frame)
            if segment is not None:
                out.append(segment)
        return out

    def flush(self) -> list[Segment]:
        """End of stream: pad the partial frame, then close whatever speech is still open."""
        out: list[Segment] = []
        tail = self.accumulator.flush()
        if tail is not None:
            segment = self._push_frame(tail)
            if segment is not None:
                out.append(segment)
        final = self.segmenter.flush(self.clock)
        if final is not None:
            self.segments += 1
            out.append(final)
        return out

    def view(self) -> dict[str, Any]:
        return {
            "connected": self.connected,
            "seconds": round(self.seconds, 2),
            "offset": round(self.offset, 2),
            "frames": self.frames,
            "segments": self.segments,
        }


def insert_line(transcript: list[TranscriptLine], line: TranscriptLine) -> int:
    """Insert a decoded line by ``t_start`` so two roles' lines interleave in time; returns index.

    Segments decode in arrival order, which is not time order when both roles talk close
    together — a stable insert keeps the transcript readable and keeps ``build_answer_span``'s
    window semantics identical to the post-hoc path.
    """
    keys = [ln.t_start for ln in transcript]
    index = bisect.bisect_right(keys, line.t_start)
    transcript.insert(index, line)
    return index


def line_identity(line: TranscriptLine) -> tuple[Any, ...]:
    """A stable identity for a transcript line, for deduping accumulated captures (#994).

    Two lines with the same timing, speaker, language and text are indistinguishable, so treating
    them as one is what makes re-persisting a capture idempotent (the finish + shutdown double
    write must not double the lines).
    """
    speaker = line.speaker.value if isinstance(line.speaker, Speaker) else str(line.speaker)
    return (line.t_start, line.t_end, speaker, line.lang, line.text)


def merge_transcript_lines(
    existing: list[TranscriptLine], new: list[TranscriptLine]
) -> list[TranscriptLine]:
    """Union of an already-persisted transcript and a fresh capture (#994) — pure.

    Preserves the existing (already time-ordered) lines and inserts each genuinely new line by
    ``t_start`` via ``insert_line``, dropping any whose identity already appears. So a second live
    capture ACCUMULATES into the same transcript, stays time-ordered, and re-persisting the same
    lines never duplicates them.
    """
    merged = list(existing)
    seen = {line_identity(ln) for ln in merged}
    for line in new:
        ident = line_identity(line)
        if ident in seen:
            continue
        insert_line(merged, line)
        seen.add(ident)
    return merged


def capture_clock_base(
    prior_lines: list[TranscriptLine], gap: float = CAPTURE_GAP_SECONDS
) -> float:
    """Where a new serve process's live clock starts, given the transcript already on disk (#1003).

    A fresh ``serve`` restarts ``time.monotonic``-based offsets at ~0, so without a base a second
    capture from a SEPARATE process would interleave with the saved one (``merge_transcript_lines``
    orders by ``t_start``). Seeding the clock at ``max(t_end) + gap`` puts every line, asked-mark
    and saved answer span of the new capture on ONE shifted timeline, strictly after the prior
    capture. ``0.0`` when nothing is on disk (the first capture is unchanged). Pure.
    """
    if not prior_lines:
        return 0.0
    return round(max(ln.t_end for ln in prior_lines) + gap, 3)


def live_view(
    streams: dict[str, RoleStream], pending: int, lines: int, provider: str | None
) -> dict[str, Any]:
    """The ``live`` block of the snapshot payload: per-role status + decode backlog."""
    return {
        "roles": {name: stream.view() for name, stream in streams.items()},
        "pending_segments": pending,
        "lines": lines,
        "stt_provider": provider,
        "active": any(s.connected for s in streams.values()),
    }


# ── asyncio coordinator (owns the streams, the decode queue, the STT worker) ─────────────────


OnLine = Callable[[TranscriptLine], Awaitable[None]]


class LiveCapture:
    """Per-role streams → one serialised decode queue → ``on_line`` for every transcript line.

    One worker, one queue: the local provider is one model on one GPU, so decodes are serialised
    in arrival order; the blocking decode runs in ``asyncio.to_thread`` so the event loop (and
    therefore ``/events`` broadcasts and every REST click) never stalls behind Whisper.
    """

    def __init__(
        self,
        settings: Settings,
        session: Session,
        provider_factory: Callable[[], SttProvider],
        on_line: OnLine,
        vad: object | None = None,
        clock_base: Callable[[], float] | None = None,
    ) -> None:
        self.settings = settings
        # #1003: read ONCE, when this process's live clock starts (the first ``open_stream``), so
        # a re-Start within the same process — or this process's own persist having written
        # transcript.json in between — never re-seeds or double-shifts the clock.
        self._clock_base_source = clock_base
        self.clock_base = 0.0
        self.session = session
        self.policy = LanguagePolicy.from_languages(session.languages)
        self.provider_factory = provider_factory
        self.on_line = on_line
        self.vad = vad
        self.provider: SttProvider | None = None
        self.streams: dict[str, RoleStream] = {}
        self.queue: asyncio.Queue[Segment] = asyncio.Queue()
        self.worker: asyncio.Task | None = None
        self.started_at: float | None = None
        self.lines = 0
        self.decode_errors = 0
        # #994: audio a role contributed BEFORE its stream was replaced by a later capture, so the
        # persisted per-role seconds/segments reflect every capture in this process, not just the
        # last one. ``open_stream`` folds the outgoing stream in here before it is replaced.
        self.retired_samples = 0
        self.retired_segments = 0

    # -- lifecycle ---------------------------------------------------------------------------

    async def ensure_provider(self) -> SttProvider:
        """Build the STT provider off the loop (a local model load can take seconds)."""
        if self.provider is None:
            self.provider = await asyncio.to_thread(self.provider_factory)
            logger.info("live STT provider ready: %s", getattr(self.provider, "name", "?"))
        return self.provider

    def _ensure_worker(self) -> None:
        if self.worker is None or self.worker.done():
            self.worker = asyncio.create_task(self._decode_loop(), name="live-stt-worker")

    def open_stream(self, role_name: str) -> RoleStream:
        """(Re)open a role's stream at the current point of the live clock."""
        role = parse_role(role_name)
        now = time.monotonic()
        if self.started_at is None:
            self.started_at = now
            if self._clock_base_source is not None:  # #1003: after any capture already on disk
                self.clock_base = float(self._clock_base_source())
        offset = self.clock_base + (now - self.started_at)
        previous = self.streams.get(role.value)
        if previous is not None:  # #994: a re-Start for this role — keep its audio in the totals
            self.retired_samples += previous.samples
            self.retired_segments += previous.segments
        stream = RoleStream(role, self.settings, offset=offset, vad=self.vad)
        stream.connected = True
        self.streams[role.value] = stream
        return stream

    @property
    def total_audio_seconds(self) -> float:
        """Audio decoded across every capture in this process (retired + still-open streams)."""
        live = sum(s.samples for s in self.streams.values())
        return round((self.retired_samples + live) / self.settings.sample_rate, 2)

    @property
    def total_segments(self) -> int:
        """Segments closed across every capture in this process (retired + still-open streams)."""
        return self.retired_segments + sum(s.segments for s in self.streams.values())

    async def feed(self, role_name: str, chunk: bytes) -> int:
        """One binary frame from the browser → segments queued for decode; returns how many."""
        stream = self.streams[parse_role(role_name).value]
        segments = stream.push(chunk)
        for segment in segments:
            await self.queue.put(segment)
        if segments:
            self._ensure_worker()
        return len(segments)

    async def close_stream(self, role_name: str) -> int:
        """End of a role's socket: flush its last segment(s) into the queue; returns how many."""
        stream = self.streams.get(parse_role(role_name).value)
        if stream is None:
            return 0
        segments = stream.flush()
        stream.connected = False
        for segment in segments:
            await self.queue.put(segment)
        if segments:
            self._ensure_worker()
        return len(segments)

    async def drain(self) -> None:
        """Wait until every queued segment has been decoded (tests, and the smoke script)."""
        await self.queue.join()

    @property
    def active(self) -> bool:
        """True while at least one role socket is open."""
        return any(s.connected for s in self.streams.values())

    # -- the worker --------------------------------------------------------------------------

    async def _decode_loop(self) -> None:
        provider = await self.ensure_provider()
        while True:
            segment = await self.queue.get()
            try:
                lines, _decoded = await asyncio.to_thread(
                    transcribe_segments, [segment], provider, self.policy
                )
                for line in lines:
                    self.lines += 1
                    await self.on_line(line)
            except Exception:  # one bad decode must not kill the live loop
                self.decode_errors += 1
                logger.exception("live decode failed for segment %s", segment.index)
            finally:
                self.queue.task_done()

    # -- view --------------------------------------------------------------------------------

    def view(self) -> dict[str, Any]:
        provider = getattr(self.provider, "name", None) if self.provider is not None else None
        view = live_view(self.streams, self.queue.qsize(), self.lines, provider)
        view["decode_errors"] = self.decode_errors
        view["clock_base"] = self.clock_base
        return view
