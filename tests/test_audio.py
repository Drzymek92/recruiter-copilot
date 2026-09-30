"""Segmentation, the noise-floor gate, and speaker attribution (vendored seam, D19/D20)."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from recruiter_copilot.audio import (
    ChannelGate,
    Segment,
    VadSegmenter,
    read_wav,
    segment_source,
)
from recruiter_copilot.config import Settings
from recruiter_copilot.models import Speaker


class FakeVad:
    """Always agrees. Isolates the amplitude gate, which is the part with a measured bug history."""

    def is_speech(self, frame: bytes, rate: int) -> bool:  # noqa: ARG002
        return True


@pytest.fixture
def settings() -> Settings:
    return Settings()


def _frame(settings: Settings, amplitude: float) -> bytes:
    n = int(settings.sample_rate * settings.vad_frame_ms / 1000)
    rng = np.random.default_rng(0)
    return (rng.standard_normal(n) * amplitude * 32767).astype(np.int16).tobytes()


def test_gate_rejects_quiet_frames_even_when_the_vad_says_speech(settings: Settings) -> None:
    """The measured failure: webrtcvad alone calls steady hiss speech."""
    gate = ChannelGate(settings, "test", vad=FakeVad())
    assert gate.is_speech(_frame(settings, 0.0001)) is False
    assert gate.is_speech(_frame(settings, 0.3)) is True


def test_gate_floor_adapts_to_a_noisy_channel(settings: Settings) -> None:
    gate = ChannelGate(settings, "noisy", vad=FakeVad())
    for _ in range(200):  # establish a high noise floor
        gate.is_speech(_frame(settings, 0.02))
    assert gate.floor > 0
    # A frame at the noise level must no longer count as speech once the floor has risen.
    assert gate.is_speech(_frame(settings, 0.02)) is False


def test_segmenter_closes_on_a_pause_not_a_clock(settings: Settings) -> None:
    seg = VadSegmenter(settings)
    frame = _frame(settings, 0.3)
    fs = settings.vad_frame_ms / 1000.0
    now = 0.0
    out = None
    for i in range(int(6.0 / fs)):  # 6 s of speech: past segment_min_seconds, no pause yet
        now = (i + 1) * fs
        out = seg.push(frame, True, False, now)
        assert out is None, "a segment closed while speech was still running"
    for i in range(int(1.2 / fs)):  # then a pause longer than segment_silence_seconds
        now += fs
        out = seg.push(frame, False, False, now)
        if out:
            break
    assert out is not None and out.speech_seconds >= settings.segment_min_seconds


def test_segmenter_force_cuts_at_the_safety_cap_and_carries_the_tail(settings: Settings) -> None:
    seg = VadSegmenter(settings)
    frame = _frame(settings, 0.3)
    fs = settings.vad_frame_ms / 1000.0
    out = None
    for i in range(int((settings.segment_max_seconds + 1) / fs)):
        out = seg.push(frame, True, False, (i + 1) * fs)
        if out:
            break
    assert out is not None
    assert out.duration >= settings.segment_max_seconds - 1
    assert seg.carry, "a force-cut must carry its tail into the next segment"
    assert seg.continued


def test_speaker_is_the_dominant_channel(settings: Settings) -> None:
    seg = VadSegmenter(settings)
    frame = _frame(settings, 0.3)
    fs = settings.vad_frame_ms / 1000.0
    for i in range(int(6.0 / fs)):
        seg.push(frame, False, True, (i + 1) * fs)  # candidate speaking
    out = seg.flush(6.5)
    assert out is not None and out.speaker is Speaker.CANDIDATE


def test_a_tie_is_unknown_never_a_guess(settings: Settings) -> None:
    seg = VadSegmenter(settings)
    frame = _frame(settings, 0.3)
    fs = settings.vad_frame_ms / 1000.0
    for i in range(int(6.0 / fs)):
        seg.push(frame, True, True, (i + 1) * fs)  # both channels: mono case
    out = seg.flush(6.5)
    assert out is not None and out.speaker is Speaker.UNKNOWN


def test_too_little_speech_emits_nothing(settings: Settings) -> None:
    seg = VadSegmenter(settings)
    fs = settings.vad_frame_ms / 1000.0
    seg.push(_frame(settings, 0.3), True, False, fs)
    assert seg.flush(2 * fs) is None  # below segment_min_speech_seconds


def _write_wav(path: Path, tracks: list[np.ndarray], rate: int) -> None:
    stereo = np.stack(tracks, axis=1) if len(tracks) > 1 else tracks[0].reshape(-1, 1)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(len(tracks))
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes((np.clip(stereo, -1, 1) * 32767).astype(np.int16).tobytes())


def test_read_wav_maps_channels_to_roles(tmp_path: Path) -> None:
    left = np.full(1000, 0.5, dtype=np.float32)
    right = np.full(1000, -0.25, dtype=np.float32)
    path = tmp_path / "s.wav"
    _write_wav(path, [left, right], 16000)

    src = read_wav(path)
    assert src.has_channels
    assert float(src.interviewer.mean()) > 0 and float(src.candidate.mean()) < 0

    swapped = read_wav(path, channel_map="candidate,interviewer")
    assert float(swapped.interviewer.mean()) < 0

    mono = read_wav(path, channel_map="mono")
    assert not mono.has_channels and mono.interviewer is None


def test_mono_source_yields_no_speaker(tmp_path: Path, settings: Settings) -> None:
    rng = np.random.default_rng(1)
    loud = rng.standard_normal(16000 * 6).astype(np.float32) * 0.3
    quiet = rng.standard_normal(16000 * 2).astype(np.float32) * 0.0001
    path = tmp_path / "m.wav"
    _write_wav(path, [np.concatenate([loud, quiet])], 16000)
    segments = segment_source(read_wav(path), settings)
    assert segments
    assert all(s.speaker is Speaker.UNKNOWN for s in segments)


def test_segment_source_refuses_a_rate_mismatch(tmp_path: Path, settings: Settings) -> None:
    path = tmp_path / "r.wav"
    _write_wav(path, [np.zeros(8000, dtype=np.float32)], 8000)
    with pytest.raises(ValueError, match="resample"):
        segment_source(read_wav(path), settings)


def test_segment_to_float32_round_trips() -> None:
    pcm = np.array([0, 16384, -16384], dtype=np.int16).tobytes()
    seg = Segment(1, 0.0, 1.0, pcm, 1.0)
    audio = seg.to_float32()
    assert audio.dtype == np.float32
    assert abs(float(audio[1]) - 0.5) < 0.01
