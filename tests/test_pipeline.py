"""The post-hoc pipeline (F8) end to end, with a stub decoder so CI needs no GPU or network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recruiter_copilot.audio import AudioSource
from recruiter_copilot.config import Settings
from recruiter_copilot.models import Speaker
from recruiter_copilot.pipeline import (
    load_transcript,
    resample,
    run_posthoc,
    save_transcript,
    transcribe_recording,
)
from recruiter_copilot.stt import DecodedSegment
from recruiter_copilot.store import load_session

FIXTURE = Path(__file__).parent / "fixtures" / "mock_interview.wav"
TRUTH = Path(__file__).parent / "fixtures" / "mock_interview.json"

pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(),
    reason="run scripts/make_fixture_audio.py to generate the synthetic interview",
)


class ScriptedProvider:
    """Returns the fixture's own ground-truth text for whichever turn a segment overlaps.

    This exercises every stage except the acoustic model itself, which is what lets the whole
    pipeline be tested without a GPU. The real decoder is proven separately by running
    `recruiter-copilot analyse` on the same file.
    """

    name = "scripted"

    def __init__(self) -> None:
        self.turns = json.loads(TRUTH.read_text(encoding="utf-8"))["turns"]
        self.cursor = 0

    def decode(self, audio, policy):  # noqa: ARG002
        turn = self.turns[min(self.cursor, len(self.turns) - 1)]
        self.cursor += 1
        return DecodedSegment(
            text=turn["text"],
            language=turn["lang"],
            language_probability=0.95,
            latency_seconds=0.01,
            audio_seconds=len(audio) / 16000,
        )


def test_segmentation_finds_every_turn_and_attributes_both_speakers() -> None:
    session = load_session(Path(__file__).parents[1] / "examples" / "sample_session")
    lines, stats, _ = transcribe_recording(
        session, FIXTURE, Settings(), provider=ScriptedProvider()
    )
    expected_turns = len(json.loads(TRUTH.read_text(encoding="utf-8"))["turns"])
    assert stats.segments == expected_turns
    assert stats.speakers == {"interviewer": 6, "candidate": 6}
    assert lines[0].speaker is Speaker.INTERVIEWER
    assert stats.audio_seconds > 100


def test_mono_channel_map_drops_speaker_tags_but_still_segments() -> None:
    session = load_session(Path(__file__).parents[1] / "examples" / "sample_session")
    _, stats, _ = transcribe_recording(
        session, FIXTURE, Settings(), provider=ScriptedProvider(), channel_map="mono"
    )
    assert stats.segments > 0
    assert set(stats.speakers) == {"unknown"}


def test_run_posthoc_writes_both_transcripts_and_proposals(tmp_path: Path) -> None:
    session = load_session(Path(__file__).parents[1] / "examples" / "sample_session")
    result = run_posthoc(session, tmp_path, FIXTURE, Settings(), provider=ScriptedProvider())
    assert (tmp_path / "transcript.json").is_file()
    assert (tmp_path / "transcript.txt").is_file()
    assert not (tmp_path / "transcript.json.tmp").exists()

    # every bank question is asked once in the fixture, so a perfect matcher finds all six
    assert {p.question_id for p in result.proposals} == {"q1", "q2", "q3", "q4", "q5", "q6"}
    assert all(p.confidence >= Settings().matcher_min_confidence for p in result.proposals)

    payload = json.loads((tmp_path / "transcript.json").read_text(encoding="utf-8"))
    assert payload["stats"]["lines"] == len(result.lines)
    assert len(payload["proposals"]) == len(result.proposals)


def test_saved_transcript_round_trips(tmp_path: Path) -> None:
    session = load_session(Path(__file__).parents[1] / "examples" / "sample_session")
    result = run_posthoc(session, tmp_path, FIXTURE, Settings(), provider=ScriptedProvider())
    assert load_transcript(tmp_path) == result.lines


def test_run_posthoc_does_not_change_question_state(tmp_path: Path) -> None:
    """D16 again, at the pipeline level: proposals are not applications."""
    session = load_session(Path(__file__).parents[1] / "examples" / "sample_session")
    before = session.counts()
    run_posthoc(session, tmp_path, FIXTURE, Settings(), provider=ScriptedProvider())
    assert session.counts() == before


def test_audio_is_never_copied_into_the_session_folder(tmp_path: Path) -> None:
    """D18: the recording stays where the interviewer put it; nothing is silently retained."""
    session = load_session(Path(__file__).parents[1] / "examples" / "sample_session")
    run_posthoc(session, tmp_path, FIXTURE, Settings(), provider=ScriptedProvider())
    assert not list(tmp_path.rglob("*.wav"))


def test_load_transcript_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_transcript(tmp_path)


def test_resample_changes_length_proportionally() -> None:
    import numpy as np

    src = AudioSource(mono=np.zeros(32000, dtype=np.float32), sample_rate=32000)
    out = resample(src, 16000)
    assert out.sample_rate == 16000 and len(out.mono) == 16000
    assert resample(out, 16000) is out


def test_save_transcript_handles_an_empty_run(tmp_path: Path) -> None:
    from recruiter_copilot.pipeline import PipelineStats

    save_transcript(tmp_path, [], PipelineStats(), [])
    assert (tmp_path / "transcript.txt").read_text(encoding="utf-8") == ""
