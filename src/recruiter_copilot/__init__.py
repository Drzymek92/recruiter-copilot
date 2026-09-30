"""recruiter_copilot — disclosed, local-first interviewer copilot.

Package layout (design/DECISIONS.md D19; module map in
documentation_processed/00_concept_and_requirements.md §4):

    models    — the data contract (D15): session bundle, transcript, analysis, report
    config    — profile switch (D12) + settings precedence CLI > env > config > default
    audio     — VAD segmentation + channel gate (vendored from interview_copilot)
    stt       — SttProvider implementations: faster-whisper (local), OpenAI-compatible (api)
    llm       — ChatProvider implementations: Ollama (local), OpenAI-compatible / Anthropic (api)
    matcher   — proposes "you just asked Q#" from the transcript (D16 — proposes, never applies)
    analysis  — post-call, evidence-cited analysis (D17)
    report    — markdown + self-contained HTML writer with quote verification (D17)
    store     — sessions/<id>/session.json + live.db (D15)
    server    — FastAPI + websocket, loopback only (D13, D14)
    cli       — `recruiter-copilot serve | analyse | purge`
"""

__version__ = "0.0.1"
