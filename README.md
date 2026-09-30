# recruiter-copilot

**A disclosed, local-first copilot for the person running the interview.** It transcribes a
bilingual call, keeps the question bank in view (asked / answered / left), pins the candidate's
answers to the questions, and after the call writes an **evidence-cited** analysis: contradictions
against the CV, cover letter and earlier conversations, weighted job-fit coverage, and, only when
the role needs it, an indicative language estimate.

Runs in one of two profiles behind a single setting:

| `PROFILE` | Speech-to-text | Reasoning | Needs |
|---|---|---|---|
| `local` | faster-whisper on your GPU (or CPU) | Ollama | no keys, no data leaves the machine |
| `api` | any OpenAI-compatible transcription endpoint | OpenAI-compatible or Anthropic | an API key; the egress is shown on screen and in the report |

Design principles, in order: **local by default · consent-gated recording · decision support, never
decision making · every finding quotes its evidence.**

> Status: **early preview (v0.0.1).** The post-hoc pipeline, the live cockpit and the post-call
> analysis all work; live browser capture is the newest part and still being validated on real calls.
> Design reasoning: `design/DECISIONS.md`.

## Install

Needs **Python 3.11+** and **Chrome** (live capture uses the browser's mic + tab-audio sharing).

```bash
git clone https://github.com/Drzymek92/recruiter-copilot.git
cd recruiter-copilot
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[local]"            # GPU/CPU-local profile (add ,dev for pytest)
cp config/.env.example config/.env   # then edit if needed
```

**`PROFILE=local` (recommended with an NVIDIA GPU):** install [Ollama](https://ollama.com) and pull
the model once: `ollama pull llama3.1:8b`. No key, nothing leaves the machine. On Windows the GPU
speech-to-text also needs the CUDA 12 cuBLAS + cuDNN 9 runtime DLLs on `PATH` (see the
faster-whisper README); without them set `STT_DEVICE=cpu` and `STT_MODEL=small`.

**`PROFILE=api`:** `pip install -e ".[api]"`, then in `config/.env` set `PROFILE=api` and
`OPENAI_API_KEY` (+ `OPENAI_CHAT_MODEL`, `OPENAI_STT_MODEL`, e.g. `gpt-4o-mini` / `whisper-1`).
Transcription needs an **OpenAI-compatible** key; an Anthropic key alone covers only the analysis.
Prefer exporting the key in your shell over writing it to disk.

Two-person walkthrough with planted contradictions: [`examples/TEST_SCENARIO_two_person.md`](examples/TEST_SCENARIO_two_person.md).

## Try it now, with no interview and no API key

The repo ships a synthetic bilingual mock interview: Polish with an English code-switch, two
speakers on two channels, and answers that deliberately contradict the sample CV. Nothing about it
is a real person.

```bash
recruiter-copilot analyse examples/sample_session --audio tests/fixtures/mock_interview.wav
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
3. **After the call** — Stop saves the live transcript to `<session>/transcript.json` (inside the
   bundle, so `purge` removes it) and `recruiter-copilot analyse <session>` — no `--audio` —
   produces the report from it: per question a summary, verbatim quotes with
   timestamps, contradictions naming the source document, a score with confidence, or "not
   evidenced". Requirement coverage rolls up into a fit summary. You edit before export.

## Server host (loopback only, D14)

The cockpit binds loopback only. Set the bind host with **`RECRUITER_COPILOT_HOST`** (default
`127.0.0.1`); a non-loopback value there is **refused** at settings load — the deliberate bind is
still blocked. Bare **`HOST`** is consulted only as a fallback, and only when it parses as an
address (an IP literal or `localhost`): any other `HOST` value is **ignored** with one INFO line.
That is why `recruiter-copilot serve` starts inside a conda environment whose compiler activation
exports `HOST=x86_64-conda-linux-gnu` — the build triple is ignored, not read as a bind address.
Precedence is unchanged: CLI flag > env > `config/.env` > default — the CLI flag is
`recruiter-copilot serve --host <addr>`, which overrides `RECRUITER_COPILOT_HOST` and is refused
the same way for a non-loopback value (D14, no bypass).

The port works the same way: **`PORT`** in the env or `config/.env` (default `8765`), overridden by
`recruiter-copilot serve --port <n>` at highest precedence (range-checked 1-65535 like `PORT`) —
e.g. `recruiter-copilot serve sessions/demo --port 9001` when 8765 is taken.

One session folder is served by one process at a time: `serve` writes a `.serve.lock` (its PID)
into the session folder and removes it on shutdown, so a second `serve` on the same folder — on
any port — refuses to start with a message naming the holder; a lock left by a crashed process
(dead PID) is taken over with a warning, and `purge` deletes it with the rest of the folder.

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
`recruiter-copilot analyse <session> --audio call.wav` — the same segmenter, STT and analyser, on
every OS and browser.

## Privacy & compliance

Audio is discarded after transcription unless you keep it. `recruiter-copilot purge <session>`
deletes a candidate's data. Recording cannot start before the consent record is filled. The tool
never ranks or rejects; it is built as human-in-the-loop support with the EU AI Act's treatment of
recruitment AI in mind.
