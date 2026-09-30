"""Stream a stereo WAV over ``ws:/audio`` as two live roles — the M4 synthetic end-to-end proof.

The left channel plays the interviewer (the browser's mic socket), the right channel the candidate
(the tab-audio socket), exactly as ``cockpit.html`` would stream them. Nothing here touches a
device: it is the replay path for the live plumbing, so the real local STT can be proven on the
fixture without a person at a microphone.

    # against a cockpit you already started (recruiter-copilot serve examples/demo_session)
    python scripts/push_wav_ws.py --wav tests/fixtures/mock_interview.wav --url http://127.0.0.1:8765

    # self-contained: spawn the cockpit on a spare port, stream, report, tear it down
    python scripts/push_wav_ws.py --serve examples/demo_session --port 8799

The server decodes with the REAL local Whisper in the second form, so run it through the GPU lease
board:

    python -m commons.coordination.gpu run --vram 3000 --label "rc M4 smoke" -- \\
        python scripts/push_wav_ws.py --serve examples/demo_session --port 8799

Exit 0 when at least one transcript line arrived, 2 otherwise.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WAV = PROJECT_ROOT / "tests" / "fixtures" / "mock_interview.wav"
ROLES = ("interviewer", "candidate")


def read_roles(path: Path, channel_map: str) -> tuple[int, dict[str, bytes]]:
    """Stereo PCM16 WAV → one mono PCM16 byte string per role (left/right per ``channel_map``)."""
    with wave.open(str(path), "rb") as wf:
        rate, channels, width = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if width != 2:
        raise SystemExit(f"{path}: need 16-bit PCM, got {width * 8}-bit")
    data = np.frombuffer(raw, dtype=np.int16).reshape(-1, channels)
    if channels == 1:
        left = right = data[:, 0]
    else:
        left, right = data[:, 0], data[:, 1]
    if channel_map == "candidate,interviewer":
        left, right = right, left
    return rate, {"interviewer": left.tobytes(), "candidate": right.tobytes()}


def get_json(url: str, timeout: float = 5.0) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 — loopback only
            return json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None


async def stream_role(
    ws_base: str, role: str, pcm: bytes, rate: int, chunk_ms: int, speed: float, log
) -> dict:
    from websockets.asyncio.client import connect  # noqa: PLC0415 — core dep (pyproject)

    chunk_bytes = int(rate * chunk_ms / 1000) * 2
    pace = (chunk_ms / 1000.0) / speed if speed > 0 else 0.0
    sent = 0
    async with connect(f"{ws_base}/audio?role={role}", max_size=None) as ws:
        await ws.send(json.dumps({"type": "hello", "sample_rate": rate, "format": "pcm16"}))
        ready = json.loads(await ws.recv())
        if ready.get("type") != "ready":
            raise SystemExit(f"{role}: server refused the stream: {ready}")
        log(f"{role}: ready (offset {ready['offset']}s, frame {ready['frame_bytes']} B)")
        t0 = time.perf_counter()
        for i in range(0, len(pcm), chunk_bytes):
            await ws.send(pcm[i : i + chunk_bytes])
            sent += 1
            if pace:
                await asyncio.sleep(pace)
        await ws.send(json.dumps({"type": "end"}))
        ended = json.loads(await ws.recv())
        log(
            f"{role}: {sent} chunks ({len(pcm) / 2 / rate:.1f}s audio) in "
            f"{time.perf_counter() - t0:.1f}s → {ended.get('segments')} segment(s)"
        )
        return ended


def wait_for_drain(base: str, timeout: float, log) -> dict:
    """Poll ``/session`` until the decode backlog is empty (and stays empty for one more poll)."""
    deadline = time.monotonic() + timeout
    stable = 0
    state: dict | None = None
    while time.monotonic() < deadline:
        state = get_json(f"{base}/session")
        live = (state or {}).get("live") or {}
        pending = live.get("pending_segments", 1)
        if state is not None and pending == 0:
            stable += 1
            if stable >= 2:
                return state
        else:
            stable = 0
            log(f"  decoding… {pending} segment(s) pending, {len(state['transcript'])} line(s)")
        time.sleep(1.0)
    raise SystemExit(f"decode backlog did not drain within {timeout:.0f}s")


def spawn_server(session_dir: Path, port: int, log) -> subprocess.Popen:
    env = {**os.environ, "PORT": str(port), "RECRUITER_COPILOT_HOST": "127.0.0.1"}
    cmd = [sys.executable, "-c", "from recruiter_copilot.cli import main; main()", "serve"]
    proc = subprocess.Popen(  # noqa: S603 — our own interpreter, fixed argv
        [*cmd, str(session_dir)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    log(f"spawned cockpit pid {proc.pid} on 127.0.0.1:{port} for {session_dir}")
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            out = proc.stdout.read().decode("utf-8", "replace") if proc.stdout else ""
            raise SystemExit(f"cockpit exited early ({proc.returncode}):\n{out}")
        if get_json(f"http://127.0.0.1:{port}/session") is not None:
            return proc
        time.sleep(0.5)
    proc.terminate()
    raise SystemExit("cockpit did not come up within 60s")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--wav", type=Path, default=DEFAULT_WAV)
    ap.add_argument("--url", default="http://127.0.0.1:8765", help="a running cockpit")
    ap.add_argument("--serve", type=Path, help="spawn the cockpit for this session dir instead")
    ap.add_argument("--port", type=int, default=8799, help="port for --serve")
    ap.add_argument("--channel-map", default="interviewer,candidate")
    ap.add_argument("--chunk-ms", type=int, default=100, help="browser-like message size")
    ap.add_argument("--speed", type=float, default=0.0, help="realtime multiplier; 0 = no pacing")
    ap.add_argument("--timeout", type=float, default=600.0, help="max seconds to wait for decode")
    ap.add_argument("--json", action="store_true", help="print the final transcript as JSON")
    args = ap.parse_args(argv)

    def log(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    proc: subprocess.Popen | None = None
    base = args.url.rstrip("/")
    if args.serve:
        base = f"http://127.0.0.1:{args.port}"
        proc = spawn_server(args.serve, args.port, log)
    try:
        if get_json(f"{base}/session") is None:
            raise SystemExit(f"no cockpit answering at {base} (consent complete? server up?)")
        rate, roles = read_roles(args.wav, args.channel_map)
        ws_base = "ws" + base[len("http") :]

        async def run() -> None:
            await asyncio.gather(
                *(
                    stream_role(ws_base, role, roles[role], rate, args.chunk_ms, args.speed, log)
                    for role in ROLES
                )
            )

        asyncio.run(run())
        state = wait_for_drain(base, args.timeout, log)
        lines = state["transcript"]
        live = state["live"]
        print(
            f"live transcript: {len(lines)} line(s) via {live.get('stt_provider')} — "
            f"speakers {dict(_count(lines, 'speaker'))}, languages {dict(_count(lines, 'lang'))}, "
            f"decode errors {live.get('decode_errors', 0)}, proposals {len(state['proposals'])}"
        )
        for ln in lines:
            print(
                f"  [{ln['t_start']:6.2f}–{ln['t_end']:6.2f}] {ln['speaker']:<11} {ln['lang']}: {ln['text']}"
            )
        if args.json:
            print(json.dumps(lines, ensure_ascii=False, indent=2))
        return 0 if lines else 2
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
            log(f"cockpit pid {proc.pid} stopped")


def _count(lines: list[dict], key: str) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for ln in lines:
        counts[ln.get(key) or "?"] = counts.get(ln.get(key) or "?", 0) + 1
    return sorted(counts.items())


if __name__ == "__main__":
    sys.exit(main())
