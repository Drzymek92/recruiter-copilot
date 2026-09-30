"""Voice-activity segmentation over an audio stream (vendored from interview_copilot, D19).

What was vendored and why it is trusted: `ChannelGate` (webrtcvad + an adaptive per-channel
noise floor) and `VadSegmenter` (close a segment on a **pause**, never on a clock) were measured
on a real 42-minute bilingual call. Two findings are baked in and must not be undone:

* **webrtcvad alone labels steady microphone hiss as speech.** Without the amplitude gate every
  segment runs to the max-length cap and the decoder returns garbage.
* **A fixed-clock window cuts both edges of a phrase.** Segments must open on a real speech onset
  (hence the pre-roll) and close on silence.

What was deliberately **left behind**: all the `parec` / `pactl` / PipeWire capture code. This
module never touches a device — it consumes frames somebody else produced (a WAV file today, a
browser websocket at M4), which is what makes the app cross-platform (D13).

Speaker attribution (D20 note): the two channels are *roles*, not devices. Whichever role's gate
saw more speech frames owns the segment. A mono source cannot attribute and yields
``Speaker.UNKNOWN`` rather than a guess.
"""

from __future__ import annotations

import wave
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import Settings
from .models import Speaker

BYTES_PER_SAMPLE = 2  # int16 PCM


@dataclass
class Segment:
    """One closed stretch of speech, ready to decode."""

    index: int
    start: float
    end: float
    pcm: bytes
    speech_seconds: float
    speaker: Speaker = Speaker.UNKNOWN
    continued: bool = False  # produced by a max-length force-cut; text may overlap the previous

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_float32(self) -> np.ndarray:
        """Mono float32 in [-1, 1] — what faster-whisper expects."""
        return np.frombuffer(self.pcm, dtype=np.int16).astype(np.float32) / 32768.0


class ChannelGate:
    """webrtcvad plus an adaptive noise floor, for ONE channel.

    A frame counts as speech only when it is meaningfully louder than this channel's *own*
    recent noise floor **and** the VAD agrees. Per channel, because two sources sit at very
    different levels.
    """

    def __init__(self, settings: Settings, label: str, vad: object | None = None) -> None:
        self.settings = settings
        self.label = label
        self.vad = vad if vad is not None else _make_vad(settings)
        frames_per_window = int(
            settings.vad_noise_window_seconds / (settings.vad_frame_ms / 1000.0)
        )
        self.history: deque[float] = deque(maxlen=max(50, frames_per_window))
        self.floor = 0.0

    def is_speech(self, frame: bytes) -> bool:
        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
        self.history.append(rms)
        # Below a full window the floor is not trustworthy yet — fall back to the absolute
        # minimum so the loop still works in its first seconds.
        self.floor = float(np.percentile(self.history, 10)) if len(self.history) >= 50 else 0.0
        threshold = max(
            self.settings.vad_speech_min_rms, self.floor * self.settings.vad_speech_rms_mult
        )
        if rms < threshold:
            return False
        return bool(self.vad.is_speech(frame, self.settings.sample_rate))  # type: ignore[attr-defined]


def _make_vad(settings: Settings) -> object:
    import webrtcvad  # noqa: PLC0415 — core dependency, imported lazily so tests can inject

    return webrtcvad.Vad(settings.vad_aggressiveness)


@dataclass
class VadSegmenter:
    """Close a segment on a PAUSE, not on a clock (the measured constraint)."""

    settings: Settings
    frames: list[bytes] = field(default_factory=list)
    preroll: deque[bytes] = field(default_factory=deque)
    in_speech: bool = False
    speech_frames: int = 0
    silence_run: int = 0
    start_time: float = 0.0
    index: int = 0
    carry: list[bytes] = field(default_factory=list)
    continued: bool = False
    interviewer_votes: int = 0
    candidate_votes: int = 0

    def __post_init__(self) -> None:
        self.frame_seconds = self.settings.vad_frame_ms / 1000.0
        self.preroll = deque(
            maxlen=max(1, int(self.settings.segment_preroll_seconds / self.frame_seconds))
        )

    def push(
        self,
        frame: bytes,
        interviewer_speech: bool,
        candidate_speech: bool,
        now: float,
    ) -> Segment | None:
        """Feed one frame of mixed mono audio plus each role's speech verdict."""
        is_speech = interviewer_speech or candidate_speech
        if not self.in_speech:
            self.preroll.append(frame)
            if not is_speech:
                return None
            # Speech onset: open with the pre-roll (and any carry from a force-cut) so the
            # segment never starts mid-word.
            self.in_speech = True
            self.frames = [*self.carry, *self.preroll]
            self.start_time = max(0.0, now - len(self.frames) * self.frame_seconds)
            self.preroll.clear()
            self.carry = []
            self.speech_frames = 1
            self.silence_run = 0
            self.interviewer_votes = 1 if interviewer_speech else 0
            self.candidate_votes = 1 if candidate_speech else 0
            return None

        self.frames.append(frame)
        if is_speech:
            self.speech_frames += 1
            self.silence_run = 0
            self.interviewer_votes += 1 if interviewer_speech else 0
            self.candidate_votes += 1 if candidate_speech else 0
        else:
            self.silence_run += 1

        duration = len(self.frames) * self.frame_seconds
        silence = self.silence_run * self.frame_seconds
        s = self.settings

        if silence >= s.segment_silence_seconds and duration >= s.segment_min_seconds:
            return self._emit(now)
        if silence >= s.segment_max_silence_seconds:
            return self._emit(now)
        if duration >= s.segment_max_seconds:
            return self._emit(now, force_cut=True)
        return None

    def _emit(self, now: float, force_cut: bool = False) -> Segment | None:
        frames, continued = self.frames, self.continued
        speech_seconds = self.speech_frames * self.frame_seconds
        speaker = self._speaker()
        self.frames, self.in_speech, self.speech_frames, self.silence_run = [], False, 0, 0
        self.interviewer_votes = self.candidate_votes = 0
        self.preroll.clear()

        if force_cut:
            # Mid-speech cut: carry the tail into the next segment so a bisected phrase
            # survives whole somewhere.
            n_carry = int(self.settings.segment_carryover_seconds / self.frame_seconds)
            self.carry = frames[-n_carry:] if n_carry else []
            self.continued = True
        else:
            self.carry, self.continued = [], False

        if speech_seconds < self.settings.segment_min_speech_seconds:
            return None
        self.index += 1
        return Segment(
            index=self.index,
            start=self.start_time,
            end=now,
            pcm=b"".join(frames),
            speech_seconds=speech_seconds,
            speaker=speaker,
            continued=continued,
        )

    def _speaker(self) -> Speaker:
        """Dominant role owns the segment; a mono source (no votes at all) stays UNKNOWN."""
        if self.interviewer_votes == 0 and self.candidate_votes == 0:
            return Speaker.UNKNOWN
        if self.candidate_votes > self.interviewer_votes:
            return Speaker.CANDIDATE
        if self.interviewer_votes > self.candidate_votes:
            return Speaker.INTERVIEWER
        return Speaker.UNKNOWN

    def flush(self, now: float) -> Segment | None:
        return self._emit(now) if self.in_speech and self.frames else None


# ── audio sources ─────────────────────────────────────────────────────────────────────────


@dataclass
class AudioSource:
    """Decoded audio ready to segment.

    ``interviewer`` / ``candidate`` are per-role mono float32 tracks when the source is
    two-channel; both are ``None`` for a mono recording, in which case only ``mono`` is used
    and no speaker can be attributed.
    """

    mono: np.ndarray
    sample_rate: int
    interviewer: np.ndarray | None = None
    candidate: np.ndarray | None = None

    @property
    def duration(self) -> float:
        return len(self.mono) / self.sample_rate if self.sample_rate else 0.0

    @property
    def has_channels(self) -> bool:
        return self.interviewer is not None and self.candidate is not None


def read_wav(path: Path, channel_map: str = "interviewer,candidate") -> AudioSource:
    """Read a PCM WAV. Stereo is treated as two roles (default: left interviewer, right candidate).

    ``channel_map="candidate,interviewer"`` swaps them; ``channel_map="mono"`` forces a downmix
    even for a stereo file (use when both sides share one track).
    """
    with wave.open(str(path), "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    if sampwidth not in dtype_map:
        raise ValueError(f"unsupported WAV sample width: {sampwidth} bytes")
    data = np.frombuffer(raw, dtype=dtype_map[sampwidth]).astype(np.float32)
    data /= float(np.iinfo(dtype_map[sampwidth]).max)

    if n_channels == 1 or channel_map == "mono":
        mono = data.reshape(-1, n_channels).mean(axis=1) if n_channels > 1 else data
        return AudioSource(mono=mono, sample_rate=sample_rate)

    frames = data.reshape(-1, n_channels)
    left, right = frames[:, 0], frames[:, 1]
    if channel_map == "candidate,interviewer":
        interviewer, candidate = right, left
    else:
        interviewer, candidate = left, right
    return AudioSource(
        mono=frames[:, :2].mean(axis=1),
        sample_rate=sample_rate,
        interviewer=interviewer,
        candidate=candidate,
    )


def _to_pcm16(track: np.ndarray) -> bytes:
    return (np.clip(track, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


def segment_source(
    source: AudioSource,
    settings: Settings,
    segmenter: VadSegmenter | None = None,
    gates: tuple[ChannelGate, ChannelGate] | None = None,
) -> list[Segment]:
    """Run the live segmentation loop over a finished recording (F8 post-hoc path).

    This is the *same* segmenter the live loop will drive at M4 — frames are fed in order with
    a synthetic clock — so what post-hoc mode proves about segmentation transfers to live.
    """
    if source.sample_rate != settings.sample_rate:
        raise ValueError(
            f"audio is {source.sample_rate} Hz but settings expect {settings.sample_rate} Hz; "
            "resample before segmenting"
        )
    segmenter = segmenter or VadSegmenter(settings)
    frame_samples = int(settings.sample_rate * settings.vad_frame_ms / 1000)
    frame_seconds = settings.vad_frame_ms / 1000.0

    if gates is None:
        gates = (ChannelGate(settings, "interviewer"), ChannelGate(settings, "candidate"))
    interviewer_gate, candidate_gate = gates

    mono_pcm = _to_pcm16(source.mono)
    int_pcm = _to_pcm16(source.interviewer) if source.interviewer is not None else None
    cand_pcm = _to_pcm16(source.candidate) if source.candidate is not None else None

    out: list[Segment] = []
    step = frame_samples * BYTES_PER_SAMPLE
    n_frames = len(mono_pcm) // step
    for i in range(n_frames):
        lo, hi = i * step, (i + 1) * step
        frame = mono_pcm[lo:hi]
        now = (i + 1) * frame_seconds
        if int_pcm is not None and cand_pcm is not None:
            says_interviewer = interviewer_gate.is_speech(int_pcm[lo:hi])
            says_candidate = candidate_gate.is_speech(cand_pcm[lo:hi])
        else:
            # Mono: one gate decides speech, and no role can be attributed. Both flags are set
            # so the segmenter's tie rule yields UNKNOWN rather than inventing a speaker.
            speaking = interviewer_gate.is_speech(frame)
            says_interviewer = says_candidate = speaking
        segment = segmenter.push(frame, says_interviewer, says_candidate, now)
        if segment is not None:
            out.append(segment)
    final = segmenter.flush(n_frames * frame_seconds)
    if final is not None:
        out.append(final)
    return out
