# Second Chair

**Second Chair is a disclosed, local-first copilot for the person running the interview** — the
second interviewer in the room who keeps notes, tracks questions and checks claims while you
decide. It transcribes a
bilingual call, keeps the question bank in view (asked / answered / left), pins the candidate's
answers to the questions, and after the call writes an **evidence-cited** analysis: contradictions
against the CV, cover letter and earlier conversations, weighted job-fit coverage, and, only when
the role needs it, an indicative language estimate.

![Second Chair cockpit replaying a synthetic bilingual interview: questions move from pending to
answered, and two candidate questions are answered from the info pack — one verbatim, one "Not in
your pack"](docs/demo.gif)

<sub>The demo is synthetic (no real person) and runs with no microphone: serve a copy of
`examples/demo_session` (see its README), then click **Replay demo**.</sub>

Runs in one of two profiles behind a single setting:

| `PROFILE` | Speech-to-text | Reasoning | Needs |
|---|---|---|---|
| `local` | faster-whisper on your GPU (or CPU) | Ollama | no keys, no data leaves the machine |
| `api` | any OpenAI-compatible transcription endpoint | OpenAI-compatible or Anthropic | an API key; the egress is shown on screen and in the report |

Design principles, in order: **local by default · consent-gated recording · decision support, never
decision making · every finding quotes its evidence.**

> **Name (D27):** formerly **recruiter-copilot**, this repository was renamed to `second-chair`.
> The command is now `second-chair`; `recruiter-copilot` still works as an alias for every command
> below, and the Python package is still imported as `recruiter_copilot`. Links to the old
> repository name redirect.

> Status: **early preview (v0.0.1).** Built and working: the post-hoc pipeline (a recording becomes
> a bilingual, speaker-tagged transcript with question proposals), the live cockpit with in-browser
> capture, the evidence-cited post-call analysis, the workspace home with jobs and interview prep,
> and Docker compose on CPU. The GPU compose file builds but is **untested on a GPU**. Design
> reasoning: `design/DECISIONS.md`.

## Install (from a clone)

Needs **Python 3.11+** and **Chrome** (live capture uses the browser's mic + tab-audio sharing).

```bash
# Linux / macOS
git clone https://github.com/Drzymek92/second-chair.git
cd second-chair
python -m venv .venv
source .venv/bin/activate
pip install -e ".[local]"            # GPU/CPU-local profile (add ,dev for pytest)
cp config/.env.example config/.env   # then edit if needed
```

```powershell
# Windows (PowerShell)
git clone https://github.com/Drzymek92/second-chair.git
cd second-chair
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[local]"
copy config\.env.example config\.env
```

**`PROFILE=local` (recommended with an NVIDIA GPU):** install [Ollama](https://ollama.com) and pull
the analyser model once: `ollama pull qwen3:14b` (the default, ~9.3 GB; `OLLAMA_MODEL=llama3.1:8b`
is the measured lighter fallback). No key, nothing leaves the machine. On Windows the GPU
speech-to-text also needs the CUDA 12 cuBLAS + cuDNN 9 runtime DLLs on `PATH` (see the
faster-whisper README); without them set `STT_DEVICE=cpu` and `STT_MODEL=small`.

**`PROFILE=api`:** `pip install -e ".[api]"`, then in `config/.env` set `PROFILE=api` and
`OPENAI_API_KEY` (+ `OPENAI_CHAT_MODEL`, `OPENAI_STT_MODEL`, e.g. `gpt-4o-mini` / `whisper-1`).
Transcription needs an **OpenAI-compatible** key; an Anthropic key alone covers only the analysis.
Prefer exporting the key in your shell over writing it to disk.

Two-person walkthrough with planted contradictions: [`examples/TEST_SCENARIO_two_person.md`](examples/TEST_SCENARIO_two_person.md).

## Install with pipx (recommended)

[pipx](https://pipx.pypa.io) installs the app into its own virtual environment and puts both
commands (`second-chair`, and the alias `recruiter-copilot`) on your `PATH`. From a clone of this
repository:

```bash
# Linux / macOS
pipx install ".[local]"           # GPU/CPU-local profile (faster-whisper + Ollama)
pipx install ".[api]"             # or: API-only profile (no speech model on this machine)
second-chair --version
second-chair info examples/sample_session
```

```powershell
# Windows (PowerShell; install pipx first: py -m pip install --user pipx; py -m pipx ensurepath)
pipx install ".\[local]"
second-chair --version
```

Run the commands from the folder that should hold your `sessions/` and `config/.env` — both are
looked up relative to where you start them (or set `SESSIONS_DIR`). Measured on Linux (Ubuntu
24.04, Python 3.12, pip cache disabled): `pipx install ".[local]"` took **32 s** and the cockpit
was serving **under a second** after that — well inside a five-minute fresh install. The speech
model itself downloads on the first live or `analyse` run, not at install time. The Windows lines
follow pipx's documented usage and have not been run on Windows yet.

## Run with Docker compose (app + a bundled Ollama)

`compose.yaml` runs two containers: the app (the workspace home, `second-chair serve`) and its own
Ollama, which is **not published to the host** — it is reachable only on the compose network, so
an Ollama you already run on the host (port 11434) is left alone. CPU is the default.

```bash
mkdir -p workspace                                  # your jobs, interviews and reports (host folder)
docker compose up -d --build                        # build the image, start both containers
docker compose exec ollama ollama pull qwen3:14b    # the analyser model (D23), once — kept in a volume
# open http://127.0.0.1:8765/  → workspace home; "Open cockpit" opens http://127.0.0.1:8766/ …
docker compose exec app second-chair analyse /workspace/<path-to-interview> --audio /workspace/call.wav
docker compose down                                 # clean stop (see below)
```

- **Where your data lives.** Jobs, interviews, live sessions and reports are written to the host
  folder `./workspace` (bind-mounted at `/workspace`; `SECOND_CHAIR_HOST_WORKSPACE=/some/dir` picks
  another). It must exist before `up` and be writable by UID 1000, the image's non-root user
  (build with `--build-arg APP_UID=$(id -u) --build-arg APP_GID=$(id -g)` for another UID).
  Because the files are on the host, `second-chair purge` (or deleting the folder) really deletes
  them (D18). Ollama models and the speech-model cache are named volumes (`ollama-models`,
  `whisper-cache`); `docker compose down` keeps them, `down -v` deletes them.
- **Models.** `OLLAMA_MODEL` picks the analyser (default `qwen3:14b`, ~9.3 GB; pull it once as
  above). On a small CPU box, `OLLAMA_MODEL=llama3.1:8b` is the measured lighter fallback.
  `llama3.2:3b` (~2 GB) is enough to check the plumbing but finds nothing useful (recall 0). The
  CPU file decodes speech with Whisper `small` (`STT_MODEL` changes it); it downloads on the first
  `analyse` or live capture.
- **Loopback only (D14).** Every port is published on the host's **127.0.0.1 only**: `8765` for
  the workspace home and `8766-8775` for the cockpits it opens (the same numbers inside and out,
  set by `COCKPIT_PORT_FIRST` / `COCKPIT_PORT_COUNT`). Nothing is reachable from your LAN. Inside
  the container the app binds `0.0.0.0`. It may do that only because the image sets
  `SECOND_CHAIR_IN_CONTAINER=1`. Without that flag (any `serve` outside the image), every
  non-loopback host is still refused, and even with the flag a LAN address is refused.
  `tests/test_compose.py` checks that every published port starts with `127.0.0.1:` and that
  Ollama publishes none.
- **Stopping.** `docker compose down` (or `stop`) sends one SIGTERM. The workspace app stops each
  open cockpit the clean way (transcript saved, `.serve.lock` removed) and exits 0. Compose
  allows `stop_grace_period: 30s`, more than serve's 5 s + 4 s bound (see the stop timeouts under
  "Server host").

**GPU.** `compose.gpu.yaml` builds the image's `gpu` target (adds cuBLAS 12 + cuDNN 9 for
faster-whisper on CUDA), sets `STT_DEVICE=cuda` with `large-v3-turbo`, and reserves the NVIDIA GPU
for both containers. It needs the NVIDIA driver and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) on
the host:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml up -d --build
docker compose exec app second-chair info           # should list a CUDA device for STT
docker compose -f compose.yaml -f compose.gpu.yaml down
```

Measured on Linux (Ubuntu 24.04, Docker 29.1, Ryzen 7 9700X, CPU file, cold): image build
**58 s** (image 919 MB), `up` including the Ollama image pull **141 s** (`ollama/ollama` is
9.4 GB), `llama3.2:3b` pull **54 s**. Opening a cockpit from the workspace page took 0.7 s.
`analyse` of the 106 s fixture recording took **91 s**, including the first download of Whisper
`small` (486 MB). The CPU decode itself took 43 s (rtf 0.41).
`stop` took 1.3 s, exit 0, no lock left. The GPU image builds (4.0 GB) but has **not been run on
a GPU yet**. The test machine has no NVIDIA Container Toolkit.

## Install (dev)

```bash
pip install -e ".[local,dev]"     # GPU/CPU-local profile
pip install -e ".[api,dev]"       # API-only profile
pytest
```

## Try it now, with no interview and no API key

The repo ships a synthetic bilingual mock interview: Polish with an English code-switch, two
speakers on two channels, and answers that deliberately contradict the sample CV. Nothing about it
is a real person.

```bash
second-chair analyse examples/sample_session --audio tests/fixtures/mock_interview.wav
```

On an RTX 5060 Ti that transcribes 105 seconds of audio in about 6 seconds and prints a
speaker-tagged transcript plus a proposal for each of the six bank questions it heard being asked.
Every proposal waits for your click — the app never marks a question asked by itself.

## How it will work

1. **Before the call** — load a session bundle (`examples/sample_session/` shows the shape): job
   requirements, question bank in both languages, CV / cover letter / previous-conversation summary,
   primary + secondary language, and the consent record.
2. **During the call** — the browser captures your mic and the call tab's audio and streams it to a
   local server (`127.0.0.1` only). You see the transcript with speaker and language tags, the
   question tracker, and proposals like "you just asked Q3" that you confirm with one click.
   "Save answer" pins the transcript span to the question — from its asked-mark to the next
   question's asked-mark (or to now while it is the latest one); "Save up to…" lets you pick the
   answer's last line in the transcript by mouse or keyboard instead.
   When the *candidate* asks a question — about benefits, salary, remote work — and the bundle
   has an info pack (`docs/job_info.md`, D26), the cockpit shows the matching `##` section of your
   pack word for word, or "Not in your pack". Nothing is generated and nothing changes until you
   speak; text above the first `##` is ignored, and without the file the panel does not appear. The
   pack is never used as evidence: it stays out of the analysis and the report. The match threshold
   (`INFO_PACK_MIN_CONFIDENCE`, default 0.36) was measured with `python scripts/bench_info_pack.py`.
   `examples/sample_session/docs/job_info.md` shows the shape.
3. **After the call** — Stop saves the live transcript to `<session>/transcript.json` (inside the
   bundle, so `purge` removes it) and `second-chair analyse <session>` — no `--audio` —
   produces the report from it: per question a summary, verbatim quotes with
   timestamps, contradictions naming the source document, a score with confidence, or "not
   evidenced". Requirement coverage rolls up into a fit summary. You edit before export.

## Server host (loopback only, D14)

The cockpit binds loopback only. Set the bind host with **`RECRUITER_COPILOT_HOST`** (default
`127.0.0.1`); a non-loopback value there is **refused** at settings load — the deliberate bind is
still blocked. Bare **`HOST`** is consulted only as a fallback, and only when it parses as an
address (an IP literal or `localhost`): any other `HOST` value is **ignored** with one INFO line.
That is why `second-chair serve` starts inside a conda environment whose compiler activation
exports `HOST=x86_64-conda-linux-gnu` — the build triple is ignored, not read as a bind address.
Precedence is unchanged: CLI flag > env > `config/.env` > default — the CLI flag is
`second-chair serve --host <addr>`, which overrides `RECRUITER_COPILOT_HOST` and is refused
the same way for a non-loopback value (D14, no bypass).

The port works the same way: **`PORT`** in the env or `config/.env` (default `8765`), overridden by
`second-chair serve --port <n>` at highest precedence (range-checked 1-65535 like `PORT`) —
e.g. `second-chair serve sessions/demo --port 9001` when 8765 is taken.

One session folder is served by one process at a time: `serve` writes a `.serve.lock` (its PID)
into the session folder and removes it on shutdown (one Ctrl-C or SIGTERM, e.g. `docker stop`,
also with a cockpit tab still open — at most ~5 s for in-flight requests to drain, plus up to 4 s
to decode a live capture's backlog), so a second `serve` on the same folder — on
any port — refuses to start with a message naming the holder; a lock left by a crashed process
(dead PID) is taken over with a warning, and `purge` deletes it with the rest of the folder.

**Stop timeouts for a supervisor (Docker, compose, systemd).** One SIGTERM is a clean stop:
`serve` lets in-flight requests drain for at most `GRACEFUL_SHUTDOWN_S` = **5 s**, removes
`.serve.lock`, and exits 0. If a live capture is streaming (`/audio` open mid-speech), the open
segment is flushed and the STT backlog is decoded for at most `LIVE_DRAIN_TIMEOUT_S` = **4 s**
before `transcript.json` is written, so speech already captured is saved. Segments still not
decoded by then are left out with one WARNING (`shutdown: N live segment(s) not decoded within
4.0 s …`); a decode (or model load) still running then is abandoned, not waited for — the
transcript is already saved and the lock removed before the process exits (one WARNING:
`shutdown: abandoning N in-flight STT call(s) …`). A SIGKILL skips all of that and **leaks the lock**. What the next
`serve` on that folder then does is decided by `session_lock.py`: the leftover lock is taken over
(one WARNING) when it was written on this same hostname and its PID is no longer running, when
its PID is the new `serve`'s own PID but that process never took the lock (a restarted container
where `serve` gets the same PID back, often PID 1), or when the file is unreadable. Otherwise the
new `serve` refuses to start, naming the recorded PID, host and start time, and the path to
delete. The container case that still ends up refused is a new container with a different
hostname (its lock is another host's, so liveness cannot be checked): delete `.serve.lock` by hand. To
avoid the SIGKILL in the first place, give the supervisor a stop timeout above 5 s + 4 s:
`docker stop -t`, compose `stop_grace_period`, systemd `TimeoutStopSec`. Docker's default of
10 s already qualifies, so don't lower it — however slow the STT model, an in-flight decode no
longer adds to that bound.

## Live capture — browser support (G4)

The cockpit captures audio **in the browser** (D13): your microphone via `getUserMedia` and the
call tab via `getDisplayMedia` with audio, resampled to 16 kHz in an `AudioWorklet` and streamed to
the local server over `ws://127.0.0.1/audio` — one socket per role, nothing leaves the machine.
That makes the app installable without PortAudio/PipeWire, at the price of depending on what each
browser lets a page capture:

| Browser (desktop) | Mic (`getUserMedia`) | Call-tab audio (`getDisplayMedia` audio) | Live verdict |
|---|---|---|---|
| **Chrome / Chromium** | yes | **yes** — share a *tab* and tick **"Share tab audio"**. Whole-screen audio only on Windows/ChromeOS | **supported** (the developed-on path; the author's real bilingual call on Linux is the acceptance test) |
| **Edge** (Chromium) | yes | same as Chrome | supported (vendor docs; not verified here) |
| **Firefox** | yes | **no** — tab/system audio is not offered by its screen-share picker | mic only: the candidate side is missing, use post-hoc |
| **Safari** | yes | **no** audio through `getDisplayMedia` | not supported live, use post-hoc |
| Mobile browsers | partial | `getDisplayMedia` unavailable | not supported |

The support facts come from vendor documentation and are unverified in this project until the
real call runs (Chrome on Linux first); the matrix is re-checked as other seats try it. Two more constraints:

- **Wear headphones (G5).** Speakers are attributed by *source*: the mic socket is you, the tab
  socket is the candidate. On loudspeakers the mic also hears the candidate and speaker tags degrade.
- `AudioWorklet` needs Chrome 66+, Firefox 76+, Safari 14.1+; older browsers fall back to a
  `ScriptProcessorNode` resampler automatically (deprecated but functional).

**Post-hoc upload stays the universal path**: record the call with anything you like and run
`second-chair analyse <session> --audio call.wav` — the same segmenter, STT and analyser, on
every OS and browser.

## Privacy & compliance

Audio is discarded after transcription unless you keep it. `second-chair purge <session>`
deletes a candidate's data. Recording cannot start before the consent record is filled. The tool
never ranks or rejects; it is built as human-in-the-loop support with the EU AI Act's treatment of
recruitment AI in mind.
