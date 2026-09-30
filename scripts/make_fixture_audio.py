"""Generate the synthetic bilingual mock-interview recording used by tests and the README demo.

Why synthetic: a real interview recording is another person's voice and personal data, and it
could never ship in a public portfolio repo. This produces a deterministic stand-in that
exercises every property the pipeline claims — two languages, a mid-call code switch, two
speakers on two channels, pauses that segmentation must find, and answers that deliberately
contradict the sample bundle's CV and cover letter so M2's analyser has a real target.

Requires ``piper-tts`` and two voices (a dev-only dependency; the app itself never needs a TTS):

    pip install piper-tts
    python -m piper.download_voices en_US-lessac-medium pl_PL-darkman-medium \\
        --download-dir .cache/piper_voices
    python scripts/make_fixture_audio.py

Output: ``tests/fixtures/mock_interview.wav`` (16 kHz, stereo — left interviewer, right candidate)
and ``tests/fixtures/mock_interview.json`` (the ground-truth script the bench scores against).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
VOICE_DIR = PROJECT / ".cache" / "piper_voices"
OUT_WAV = PROJECT / "tests" / "fixtures" / "mock_interview.wav"
OUT_JSON = PROJECT / "tests" / "fixtures" / "mock_interview.json"
TARGET_RATE = 16000

VOICES = {"pl": "pl_PL-darkman-medium", "en": "en_US-lessac-medium"}

# (speaker, language, text, question_id or None, pause_after_seconds)
# The candidate's answers seed three contradictions against examples/sample_session/docs:
#   * "od dwóch lat" LLM work vs a CV whose first LLM project is 2025
#   * "sam zbudowałem" / sole ownership vs an HR screen recording a team of three
#   * a vague evaluation answer vs a cover letter claiming they led the evaluation programme
SCRIPT: list[tuple[str, str, str, str | None, float]] = [
    (
        "interviewer",
        "pl",
        "Dzień dobry. Zacznijmy od Pana doświadczenia. "
        "Opowiedz o ostatnim projekcie produkcyjnym w Pythonie, co było najtrudniejsze?",
        "q1",
        1.2,
    ),
    (
        "candidate",
        "pl",
        "Dzień dobry. Ostatnio pracowałem nad usługą wyszukiwania dokumentów. "
        "Najtrudniejsze było utrzymanie niskich opóźnień przy dużym ruchu. "
        "Przepisałem warstwę zapytań i dodałem cache.",
        None,
        1.4,
    ),
    (
        "interviewer",
        "pl",
        "Rozumiem. A jaką funkcję opartą na modelach językowych wdrożyłeś "
        "i jak wyglądała architektura?",
        "q2",
        1.2,
    ),
    (
        "candidate",
        "pl",
        "Zbudowałem asystenta opartego na wyszukiwaniu semantycznym. "
        "Pracuję z modelami językowymi od dwóch lat. "
        "Sam zbudowałem cały ten system, od wyszukiwania po interfejs.",
        None,
        1.5,
    ),
    ("interviewer", "pl", "A jak mierzyłeś, że zmiana modelu była lepsza?", "q3", 1.2),
    (
        "candidate",
        "pl",
        "Patrzyliśmy głównie na opinie użytkowników. "
        "Wydawało się, że odpowiedzi są lepsze po zmianie modelu.",
        None,
        1.5,
    ),
    (
        "interviewer",
        "en",
        "Let me switch to English for this one. Describe a time you had to "
        "explain a technical trade-off to a non-technical stakeholder.",
        "q4",
        1.2,
    ),
    (
        "candidate",
        "en",
        "Yes, of course. I explained to our product manager why we needed to "
        "cache the model responses. I told him it would make the system faster but the answers "
        "could be a little bit old. He understood and we agreed on a shorter cache time.",
        None,
        1.5,
    ),
    (
        "interviewer",
        "pl",
        "Dziękuję. Wracając do polskiego. Gdzie i jak wdrażałeś modele " "w chmurze?",
        "q5",
        1.2,
    ),
    (
        "candidate",
        "pl",
        "Korzystaliśmy z Amazon Web Services, konkretnie z kontenerów. "
        "Wdrożenia szły przez zautomatyzowany proces.",
        None,
        1.4,
    ),
    (
        "interviewer",
        "pl",
        "Ostatnie pytanie. W CV piszesz o czterech latach w firmie X. "
        "Jaką rolę pełniłeś w ostatnim roku?",
        "q6",
        1.2,
    ),
    (
        "candidate",
        "pl",
        "Byłem starszym inżynierem i prowadziłem zespół. "
        "Odpowiadałem za całą platformę wyszukiwania.",
        None,
        1.0,
    ),
]


def synthesize(text: str, lang: str) -> tuple[np.ndarray, int]:
    """One turn of speech as float32 mono, via piper."""
    model = VOICE_DIR / f"{VOICES[lang]}.onnx"
    if not model.exists():
        raise SystemExit(
            f"missing voice {model}. Run:\n"
            f"  python -m piper.download_voices {VOICES[lang]} --download-dir {VOICE_DIR}"
        )
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out = Path(tmp.name)
    proc = subprocess.run(
        [sys.executable, "-m", "piper", "-m", str(model), "-f", str(out)],
        input=text.encode("utf-8"),
        capture_output=True,
    )
    if proc.returncode != 0 or not out.exists():
        raise SystemExit(f"piper failed for {lang}: {proc.stderr.decode()[:400]}")
    with wave.open(str(out), "rb") as wf:
        rate = wf.getframerate()
        data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    out.unlink(missing_ok=True)
    return data.astype(np.float32) / 32768.0, rate


def resample(track: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return track
    n_out = int(len(track) * dst / src)
    return np.interp(
        np.linspace(0, len(track) - 1, n_out), np.arange(len(track), dtype=np.float64), track
    ).astype(np.float32)


def main() -> int:
    OUT_WAV.parent.mkdir(parents=True, exist_ok=True)
    left: list[np.ndarray] = []  # interviewer
    right: list[np.ndarray] = []  # candidate
    truth: list[dict] = []
    cursor = 0.0

    # A little room tone on both channels: silence that is *digitally* perfect never happens on a
    # real call, and the adaptive noise floor exists precisely to cope with the real thing. A
    # fixture with pure zeros would let a broken gate pass.
    rng = np.random.default_rng(20260906)

    def tone(n: int) -> np.ndarray:
        return (rng.standard_normal(n) * 0.0015).astype(np.float32)

    for speaker, lang, text, qid, pause in SCRIPT:
        audio, rate = synthesize(text, lang)
        audio = resample(audio, rate, TARGET_RATE)
        n = len(audio)
        start = cursor
        end = cursor + n / TARGET_RATE
        if speaker == "interviewer":
            left.append(audio)
            right.append(tone(n))
        else:
            left.append(tone(n))
            right.append(audio)
        truth.append(
            {
                "speaker": speaker,
                "lang": lang,
                "text": text,
                "question_id": qid,
                "t_start": round(start, 2),
                "t_end": round(end, 2),
            }
        )
        gap = int(pause * TARGET_RATE)
        left.append(tone(gap))
        right.append(tone(gap))
        cursor = end + pause

    l_track, r_track = np.concatenate(left), np.concatenate(right)
    stereo = np.stack([l_track, r_track], axis=1)
    pcm = (np.clip(stereo, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(OUT_WAV), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_RATE)
        wf.writeframes(pcm.tobytes())

    OUT_JSON.write_text(
        json.dumps(
            {
                "description": "Synthetic bilingual mock interview (piper TTS). Left channel = "
                "interviewer, right = candidate. No real person is recorded.",
                "sample_rate": TARGET_RATE,
                "duration": round(cursor, 2),
                "primary_language": "pl",
                "secondary_language": "en",
                "turns": truth,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {OUT_WAV} ({cursor:.1f}s, {len(SCRIPT)} turns)")
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
