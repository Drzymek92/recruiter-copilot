# demo_session (synthetic)

A turnkey bundle for the **M3 cockpit demo**. Identical to `sample_session` (same job, questions and
seeded inconsistencies) except its **consent record is complete**, so the cockpit's consent gate
(D18) opens and the tracker renders. No real person is described; consent is synthetic.

Run the cockpit and drive it without a microphone:

```
recruiter-copilot serve examples/demo_session
```

Open the printed loopback URL and click **Replay demo**: `replay.json` streams a prepared bilingual
transcript into `/events`, and you watch each question move pending → asked → answered as its turn
plays (M3 acceptance — no live audio; that is M4).

`sample_session` keeps its empty consent block, so `serve examples/sample_session` shows the gate
*refusing* the session instead.
