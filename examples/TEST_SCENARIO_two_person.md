# Two-person test scenario — recruiter-copilot

A ~15-minute walkthrough for **two people** to exercise the app's features in one mock interview.
Everything is synthetic (job, questions, "Candidate A", the seeded contradictions) — no real person.

- **Player A — Interviewer (you):** runs the app, drives the cockpit, reads the interviewer lines.
- **Player B — Candidate (your friend):** reads the candidate lines. Deliberately says a few things
  that contradict the CV so the analysis has something to catch.

The mock job is **AI Engineer (LLM applications)**; the question bank is Polish-primary with an
English second language, and the report language is English. If Player B doesn't speak Polish, use
the **English-only** lines below — the app still tags languages and runs every feature.

---

## Part 0 — 2-minute solo smoke test (Player A, before your friend arrives)

Confirms the install works, with no friend and no microphone.

From the repo root:

```bash
pip install -e ".[local,dev]" && pytest -q   # see README > Install
```

```bash
recruiter-copilot serve examples/demo_session
```

Open the printed `http://127.0.0.1:...` URL in **Chrome**, click **Replay demo**, and watch each of
the 6 questions move **pending → asked → answered** as a prepared transcript streams in.

✅ **Check:** the tracker updates itself, the transcript shows **pl** and **en** language tags, and
every state change was a *proposal you could confirm* — the app never marked a question asked on its
own. Then `Ctrl-C` to stop.

---

## Part 1 — Setup for the live call (Player A)

The live path captures **two audio sources**: your **microphone** (interviewer) and a **browser
tab's audio** (candidate). The simplest two-person setup:

1. Start a video call with your friend (Google Meet / Zoom **in a Chrome tab**).
2. **Put on headphones.** Speakers are attributed by source — the mic is *you*, the tab is *the
   candidate*. Without headphones your mic also hears the candidate and the speaker tags degrade.
3. Serve the ready bundle (complete consent, full CV/cover-letter/previous-summary):

   ```bash
   recruiter-copilot serve examples/demo_session
   ```
4. Open the loopback URL in Chrome → **Live capture → Start**. Grant the **microphone**, and for
   the candidate share the **call tab** with **"Share tab audio"** ticked.

> **No headphones or not on Chrome/Linux?** Skip live capture. Record the call any way you like and
> use the **post-hoc** path in Part 3 instead — same segmenter, STT and analysis on every OS.

✅ **Checks at this point:** the **consent gate** let the session open (try `serve
examples/sample_session` once — same bundle, *empty* consent — and confirm the cockpit **refuses**
it). Both **role indicators** (interviewer / candidate) light up once audio is flowing.

---

## Part 2 — The mock interview (~5 min)

Read the questions in order. After Player B answers each one, **Player A** clicks **"mark asked"**
on that question, then **"Save answer"** (or **"Save up to…"** to pick the last answer line by mouse
or keyboard). Watch the transcript fill in with speaker + language tags, and the tracker advance.

The **three planted contradictions** are marked ⚠️ — say those lines roughly as written so the
post-call analysis can catch them.

| # | Interviewer asks (PL / EN) | Candidate answers (say something like…) |
|---|---|---|
| **Q1** | *Opowiedz o ostatnim projekcie produkcyjnym w Pythonie.* / "Tell me about your last production Python project." | "I built a FastAPI backend with Postgres — **document search for the platform team.** Mostly Python, some SQL." *(covers requirement r1: 3+ yrs Python)* |
| **Q2** | *Jaką funkcję opartą na LLM wdrożyłeś i jak wyglądała architektura?* / "Which LLM-backed feature did you ship, and its architecture?" | ⚠️ "A retrieval-augmented assistant — a RAG system. **I built the whole thing myself**, end to end, retrieval and prompting." *(contradicts the previous-conversation summary: it was a **team of three**, candidate owned only the retrieval component)* |
| **Q3** | *Jak mierzyłeś, że zmiana modelu była lepsza?* / "How did you measure that a model change was better?" | "We had an offline eval set of about 120 questions and tracked answer accuracy before and after." *(covers r3: evaluation with a metric)* |
| **Q4** | "Describe explaining a technical trade-off to a non-technical stakeholder." | "I present to **executive stakeholders** regularly — I explained why we chose retrieval over fine-tuning in plain terms." *(English answer — feeds the language estimate)* |
| **Q5** | *Gdzie i jak wdrażałeś modele w chmurze?* / "Where and how did you deploy models in the cloud?" | *(Answer only briefly, or skip — leave this thin so you can check it shows up as weak/uncovered in the fit summary.)* |
| **Q6** | *W CV piszesz o 4 latach w firmie X — jaką rolę pełniłeś w ostatnim roku?* / "Your CV says 4 years at company X — your role this last year?" | ⚠️ "Right — and I've been **building LLM applications in production for about two years now.**" *(contradicts the CV, which shows the first LLM project only in 2025 — about one year, not two; also mismatches the cover letter)* |

Optional third contradiction to reinforce ⚠️: at any point Player B can add "and **I led the whole
evaluation programme**" — the previous summary says they owned only the offline eval set, not led
the programme.

✅ **Checks during the call:**
- Transcript shows **speaker tags** (interviewer vs candidate) and **pl / en** per line; the
  code-switch (Polish sentence with an English phrase) is tagged correctly.
- **Proposals** appear ("you just asked Q3") and only take effect **when you click Confirm**.
- **Save answer** pins the right transcript span to each question; **Save up to…** lets you trim the
  last line by mouse or keyboard (Tab / ↑ ↓ / Home / End / Enter, Esc to cancel).
- The **notes** box saves per question.

When done, click **Stop**. This saves the live transcript into
`examples/demo_session/transcript.json` (inside the bundle).

---

## Part 3 — After the call: the analysis (Player A)

```bash
recruiter-copilot analyse examples/demo_session
```

*(Post-hoc fallback if you skipped live capture — point it at your recording instead:)*

```bash
recruiter-copilot analyse examples/demo_session --audio path/to/your_call.wav
```

✅ **Checks in the report:**
- **Every finding quotes the transcript** verbatim, with a **timestamp**.
- **Contradictions name their source document.** Look for:
  - **Sole-ownership** of the RAG assistant (Q2) vs the *previous conversation summary*.
  - **"two years" of LLM work** (Q6) vs the *CV* (first LLM project in 2025) and the *cover letter*.
- **Weighted fit / requirement coverage** rolls up — Python (r1), LLM feature (r2), eval metric
  (r3) should read as covered; **cloud deployment (Q5)** should read weak or **"not evidenced"** if
  you left it thin.
- A **language estimate** for English appears (the role flags it) **with a disclaimer** — it never
  makes a hiring call.
- Where there's no evidence, the report says **"not evidenced"** rather than inventing an answer.

---

## Part 4 — Privacy / compliance check (Player A)

```bash
recruiter-copilot purge examples/demo_session
```

✅ **Check:** the saved transcript (and any kept audio) inside the bundle is **deleted** — confirm
`transcript.json` is gone. This is the "purgeable candidate data" guarantee.

> Note: `purge` empties this demo bundle's saved outputs. If you want to keep `demo_session`
> pristine for later runs, copy it first (`cp -r examples/demo_session /tmp/rc_test`) and test
> against the copy.

---

## Feature checklist (tick as you go)

- [ ] Runs locally, no API key, no data leaves the machine (`local` profile)
- [ ] Consent gate opens a consented bundle, **refuses** an unconsented one
- [ ] Bilingual transcript with **speaker** + **language (pl/en)** tags, incl. code-switch
- [ ] Question tracker: pending → asked → answered/skipped, driven by **confirmed proposals**
- [ ] Save answer pins the span; **Save up to…** trims by mouse/keyboard; notes persist
- [ ] Analysis: quotes with timestamps; **contradictions cite the source doc**
- [ ] Weighted fit / coverage summary; **"not evidenced"** where there's no evidence
- [ ] Language estimate **with disclaimer**, only because the role flagged it
- [ ] **purge** deletes the candidate's data
