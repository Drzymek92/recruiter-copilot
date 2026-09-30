# Security Invariants — the reconciliation checklist (applies D10)

**Purpose.** This is the project's canonical list of load-bearing **security decision data points**.
Any future decision that touches a security-sensitive surface — **data privacy/locality, transport,
credentials/keys, or a remote/command/agent surface** — **must be reconciled against this list before
implementation** (the gate is installed by **D10**; the routine step lives in
`ROUTINE_add_decision.md`). This doc **applies** the security-bearing decisions in `DECISIONS.md` and
never restates them; the canonical statements live there (D1).

**Starts (almost) empty — it grows with the project.** A brand-new project has few security
invariants. Add an `SI#` row the first time a decision *establishes* one (e.g. "data class X is
local-only", "surface Y is default-deny", "credential Z has a revocation path"), and cite the `D#`
that set it. If the project has **no** security-sensitive surface, leave this a stub and record the
reconciliation step as a one-time `N/A` in `agent/project.md`.

**How to use (at decision time).** For each invariant the new decision could touch, record one line
in the decision's rationale or its `design/NN` note:
**`SI# — Reconciled: <how>`** or **`SI# — N/A: <why>`**.
Mark an invariant **🔒 non-waivable** when it encodes a safety floor (privacy/security): per **D6** a
safety rule cannot be waived by a Policy Override, so a decision that cannot meet a 🔒 invariant does
**not** ship — it redesigns, or explicitly **supersedes** the cited decision in `DECISIONS.md` first.

## Invariants
<!-- Add one row per security invariant as decisions establish them; each cites its source D#.
     Replace the two example rows below with real ones (or delete them for a stub). -->

| SI | 🔒 | Invariant (confirm the new decision honors it) | Source | Confirm question |
|----|----|-----|--------|------------------|
| SI1 | 🔒 | **Candidate data (CV, cover letter, prior summaries), interview audio and transcript are LOCAL-BY-DEFAULT.** They leave the machine only under `PROFILE=api`, which is announced on screen and recorded in the report; a fully-local profile must always exist; no egress may be silent. The server binds loopback only. | D12, D14, D18 | Does the decision add or default-enable an egress of audio/transcript/candidate docs? Is it announced? Does `local` still work? New socket bound beyond 127.0.0.1? |
| SI2 | 🔒 | **No automated hiring decision.** The app never ranks candidates, auto-rejects, or emits a score without cited evidence + confidence, and every output is editable by the interviewer before export. | D11, D17 | Does the decision add a ranking/reject/threshold path, or a score with no evidence attached? If so it does not ship. |
| SI3 | 🔒 | **Recording is consent-gated and data is purgeable.** Capture cannot start before a consent record exists; audio is discarded after transcription unless `KEEP_AUDIO=1`; `purge <session>` removes a candidate's data completely (no hidden copies, caches, or logs holding transcript text). | D18 | Does the decision bypass the consent gate, persist audio silently, or write transcript/candidate text somewhere `purge` does not reach (logs, caches, telemetry)? |

## Non-waivable floor (🔒)
List here the `SI#` that are safety-floor (privacy/security). Per **D6** these cannot be waived by a
Policy Override — a decision that cannot meet one redesigns or supersedes the cited decision first.

- **SI1** — locality of candidate data, audio and transcript (D12, D14, D18).
- **SI2** — no automated hiring decision (D11, D17).
- **SI3** — consent gate + purgeability (D18).

## Provenance
Distilled from the security-bearing rows of `DECISIONS.md` (and any security review the project runs).
This checklist **never** becomes the source of truth — `DECISIONS.md` is (D1). When a decision changes
the security posture, update the cited row here **and** adjust the underlying `D#`.
