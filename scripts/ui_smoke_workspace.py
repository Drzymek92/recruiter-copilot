"""Browser smoke run of the workspace pages (D28/D29 UI, D14): headless Chrome via Playwright.

    python scripts/ui_smoke_workspace.py [--out DIR] [--real-model] [--timeout-s 900]

Everything is synthetic and lives under ``--out`` (default: a fresh temp folder): a workspace,
a synthetic offer PDF + DOCX (+ a scanned PDF that must be refused) and a synthetic CV. The run:

    home (empty) → new job → drop the PDF + DOCX (a real drop event) + the scan (refused) →
    generate (stub chat, or the local model with --real-model) → review the draft → edit (add a
    question, an inline validator error, move one between sections, reorder by keyboard, delete
    one, change minutes) → accept → save a template → start an interview (synthetic CV + consent)
    → tailor from the CV → edit + accept → open the cockpit (a child serve, new tab) → delete
    refused while served → light/dark × 1280×800/390 px screenshots → ONE SIGINT to the workspace
    app → it stops the cockpit child → checks: exit 0, no child serve, no .serve.lock left.

``--real-model`` drafts with the configured local chat model (Ollama, qwen3:14b) — run it under
the GPU lease board (``python -m commons.coordination.gpu run --vram 10500 --label … -- …``); the
model is unloaded at the end so the GPU is idle again. Default = a stub chat (no model, no GPU).

Writes its own log + ``summary.json`` + screenshots (``NN_<step>.png``) into ``--out``.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

RESERVED_PORTS = {8787}
FORBIDDEN_ROOTS = (
    Path.home() / "SecondChair",
    Path.home() / "Desktop" / "SecondChair",
    PROJECT / "examples",
    PROJECT / "sessions",
)

logger = logging.getLogger("ui_smoke_workspace")

# The flow provokes exactly these refusals: the scanned PDF (422 PUT), the accept with an empty
# wording (422 POST) and the delete while the cockpit serves the interview (409 DELETE).
EXPECTED_HTTP = {(422, "PUT"), (422, "POST"), (409, "DELETE")}

# ── synthetic documents ─────────────────────────────────────────────────────────────────────

OFFER_LINES = [
    "Backend Developer (Python) - Lodz / remote",
    "Requirements:",
    "- at least 3 years of commercial Python experience",
    "- good knowledge of PostgreSQL",
    "- communicative English (daily stand-ups with the team in Dublin)",
    "Nice to have:",
    "- experience with Kafka",
]
BENEFITS = ["What we offer: remote work, private medical care, a training budget."]
BENEFITS_TABLE = [["Stack", "Python, Django, PostgreSQL, Kafka"], ["Team", "8 engineers"]]
CV_LINES = [
    "Synthetic Candidate - Backend Developer",
    "2020-2024: Python developer at ExampleSoft, built a payments API in Django.",
    "Cut PostgreSQL query time by 40% for the reporting module.",
    "Speaks English at work every day.",
]


def write_text_pdf(path: Path, lines: list[str]) -> Path:
    """A tiny PDF with a real text layer (Helvetica, ASCII)."""

    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    ops = ["BT", "/F1 11 Tf", "14 TL", "60 740 Td"]
    ops += [f"({esc(line)}) Tj T*" for line in lines]
    ops.append("ET")
    stream = "\n".join(ops).encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    path.write_bytes(bytes(out))
    return path


def write_docs(folder: Path) -> dict[str, Path]:
    import docx  # noqa: PLC0415
    from pypdf import PdfWriter  # noqa: PLC0415

    folder.mkdir(parents=True, exist_ok=True)
    pdf = write_text_pdf(folder / "Offer Backend Developer.pdf", OFFER_LINES)
    document = docx.Document()
    for p in BENEFITS:
        document.add_paragraph(p)
    table = document.add_table(rows=len(BENEFITS_TABLE), cols=2)
    for r, row in enumerate(BENEFITS_TABLE):
        for c, value in enumerate(row):
            table.cell(r, c).text = value
    docx_path = folder / "benefits.docx"
    document.save(str(docx_path))
    scan = folder / "scanned offer.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with scan.open("wb") as fh:
        writer.write(fh)
    cv = write_text_pdf(folder / "CV Synthetic Candidate.pdf", CV_LINES)
    return {"pdf": pdf, "docx": docx_path, "scan": scan, "cv": cv}


# ── stub chat (default; no model) ───────────────────────────────────────────────────────────


def stub_replies() -> list[dict]:
    q = lambda i, pl, en, rids, intent, assesses=False: {  # noqa: E731
        "id": f"q{i}",
        "text": {"pl": pl, "en": en},
        "intent": intent,
        "expected_signals": ["a concrete example", "own role"],
        "requirement_ids": rids,
        "assesses_language": assesses,
    }
    prep_reply = {
        "requirements": [
            {
                "id": "r1",
                "text": "3+ years of commercial Python",
                "kind": "must",
                "weight": 3,
                "source_quote": "at least 3 years of commercial Python experience",
            },
            {
                "id": "r2",
                "text": "PostgreSQL",
                "kind": "must",
                "weight": 2,
                "source_quote": "good knowledge of PostgreSQL",
            },
            {
                "id": "r3",
                "text": "Communicative English",
                "kind": "must",
                "weight": 1.5,
                "source_quote": "communicative English",
            },
            {
                "id": "r4",
                "text": "Kafka",
                "kind": "nice",
                "weight": 0.5,
                "source_quote": "experience with Kafka",
            },
            {
                "id": "r5",
                "text": "Leads a team",
                "kind": "nice",
                "weight": 1,
                "source_quote": "team leadership experience",
            },  # not in the offer → flagged
        ],
        "questions": [
            q(
                1,
                "Opowiedz o projekcie w Pythonie, z którego jesteś dumny.",
                "Tell me about a Python project you are proud of.",
                ["r1"],
                "Python depth",
            ),
            q(
                2,
                "Jak diagnozujesz wolne zapytanie w PostgreSQL?",
                "How do you diagnose a slow PostgreSQL query?",
                ["r2"],
                "SQL tuning",
            ),
            q(
                3,
                "Opisz swój typowy dzień pracy po angielsku.",
                "Describe a typical working day, in English.",
                ["r3"],
                "hear English",
                True,
            ),
            q(
                4,
                "Gdzie używałeś kolejek wiadomości?",
                "Where have you used message queues?",
                ["r4"],
                "Kafka exposure",
            ),
            q(
                5,
                "Jak dzielisz się wiedzą w zespole?",
                "How do you share knowledge in a team?",
                [],
                "collaboration",
            ),
        ],
        "plan": {
            "target_minutes": 30,
            "sections": [
                {"name": "intro", "minutes": 3, "question_ids": []},
                {"name": "experience", "minutes": 8, "question_ids": ["q1", "q5"]},
                {"name": "technical", "minutes": 8, "question_ids": ["q2", "q4"]},
                {"name": "language check", "minutes": 5, "question_ids": ["q3"]},
                {"name": "candidate questions", "minutes": 4, "question_ids": []},
                {"name": "close", "minutes": 2, "question_ids": []},
            ],
        },
    }
    cv_reply = {
        "questions": [
            {
                "cv_quote": "built a payments API in Django",
                "text": {
                    "pl": "Jak zbudowałeś API płatności w Django?",
                    "en": "How did you build the payments API in Django?",
                },
                "intent": "the payments API claim",
                "expected_signals": ["own role"],
                "requirement_ids": ["r1"],
                "assesses_language": False,
            },
            {
                "cv_quote": "Cut PostgreSQL query time by 40%",
                "text": {
                    "pl": "Jak zmierzyłeś skrócenie czasu zapytań o 40%?",
                    "en": "How did you measure the 40% faster queries?",
                },
                "intent": "the 40% claim",
                "expected_signals": ["measurement"],
                "requirement_ids": ["r2"],
                "assesses_language": False,
            },
            {
                "cv_quote": "Speaks English at work every day",
                "text": {
                    "pl": "W jakich sytuacjach mówisz po angielsku w pracy?",
                    "en": "When do you speak English at work?",
                },
                "intent": "English use",
                "expected_signals": ["examples"],
                "requirement_ids": ["r3"],
                "assesses_language": False,
            },
        ]
    }
    return [prep_reply, cv_reply]


class StubChat:
    model_name = "stub-chat"

    def __init__(self, reply: dict, delay_s: float = 2.5) -> None:
        self.reply = reply
        self.delay_s = delay_s

    def complete(self, prompt: str, system: str | None = None, max_tokens: int | None = None):
        time.sleep(self.delay_s)  # long enough for the progress line to show
        return SimpleNamespace(
            text=json.dumps(self.reply, ensure_ascii=False),
            input_tokens=len(prompt) // 4,
            output_tokens=400,
            truncation_signals=(),
        )


def stub_factory():
    replies = stub_replies()
    calls = {"n": 0}

    def make() -> StubChat:
        reply = replies[min(calls["n"], len(replies) - 1)]
        calls["n"] += 1
        return StubChat(reply)

    return make


# ── the server subprocess ───────────────────────────────────────────────────────────────────


def serve_main(args: argparse.Namespace) -> int:
    """``--_serve``: the workspace app itself (stopped by ONE SIGINT from the parent)."""
    from recruiter_copilot.config import load_settings  # noqa: PLC0415
    from recruiter_copilot.workspace_app import serve_workspace  # noqa: PLC0415

    settings = load_settings(
        overrides={"workspace_dir": args.workspace, "port": args.port, "profile": "local"},
        env_file=PROJECT / "config" / ".env",
    )
    serve_workspace(settings, chat_factory=None if args.real_model else stub_factory())
    return 0


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) != 0


def pick_port(first: int = 8795) -> int:
    for port in range(first, first + 40):
        if port not in RESERVED_PORTS and port_free(port) and port_free(port + 1):
            return port
    raise RuntimeError("no free port")


def check_out_dir(out: Path) -> Path:
    out = out.expanduser().resolve()
    for bad in FORBIDDEN_ROOTS:
        if out == bad.resolve() or out.is_relative_to(bad.resolve()):
            raise SystemExit(f"refusing --out {out}: never under {bad} (synthetic scratch only)")
    out.mkdir(parents=True, exist_ok=True)
    return out


# ── the browser run ─────────────────────────────────────────────────────────────────────────


class Run:
    def __init__(self, out: Path, base: str, real: bool, timeout_s: float) -> None:
        self.out = out
        self.base = base
        self.real = real
        self.deadline = time.monotonic() + timeout_s
        self.shots: list[str] = []
        self.steps: list[dict] = []
        self.console: list[str] = []
        self.http_errors: list[dict] = []
        self.n = 0
        self.t0 = time.monotonic()

    def budget_ms(self, cap_s: float) -> float:
        left = self.deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("smoke run exceeded --timeout-s")
        return min(cap_s, left) * 1000

    def shot(self, page, name: str, full: bool = True) -> None:
        self.n += 1
        path = self.out / f"{self.n:02d}_{name}.png"
        page.screenshot(path=str(path), full_page=full)
        self.shots.append(str(path))
        logger.info("screenshot %s", path)

    def step(self, name: str, **info) -> None:
        info = {"step": name, "t_s": round(time.monotonic() - self.t0, 1), **info}
        self.steps.append(info)
        logger.info("step %s", json.dumps(info, ensure_ascii=False))

    def watch(self, page) -> None:
        def on_console(m) -> None:
            # a 4xx the flow provokes on purpose is logged by Chrome as "Failed to load
            # resource"; those are checked by URL in on_response instead
            if m.type == "error" and "Failed to load resource" not in m.text:
                self.console.append(f"{m.type}: {m.text}")

        def on_response(r) -> None:
            if r.status >= 400:
                self.http_errors.append(
                    {"status": r.status, "url": r.url, "method": r.request.method}
                )

        page.on("console", on_console)
        page.on("pageerror", lambda e: self.console.append(f"pageerror: {e}"))
        page.on("response", on_response)


DROP_JS = """
async ([sel, files]) => {
  const dt = new DataTransfer();
  for (const f of files) {
    const bin = Uint8Array.from(atob(f.b64), (c) => c.charCodeAt(0));
    dt.items.add(new File([bin], f.name, { type: f.type }));
  }
  const el = document.querySelector(sel);
  el.dispatchEvent(new DragEvent("dragover", { dataTransfer: dt, bubbles: true, cancelable: true }));
  el.dispatchEvent(new DragEvent("drop", { dataTransfer: dt, bubbles: true, cancelable: true }));
}
"""


def _file_payload(path: Path, mime: str) -> dict:
    return {"name": path.name, "type": mime, "b64": base64.b64encode(path.read_bytes()).decode()}


def section(page, name: str):
    return page.locator("#ed-body .sec").filter(
        has=page.locator("h3", has_text=re.compile(rf"^\s*{re.escape(name)}\s*$", re.I))
    )


def section_ids(page, name: str) -> list[str]:
    return section(page, name).locator(".q .idtag").all_inner_texts()


def browser_flow(run: Run, docs: dict[str, Path], workspace: Path) -> dict:
    from playwright.sync_api import expect, sync_playwright  # noqa: PLC0415

    result: dict = {}
    gen_cap = 420 if run.real else 60
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)
        ctx = browser.new_context(viewport={"width": 1280, "height": 800}, color_scheme="light")
        page = ctx.new_page()
        run.watch(page)
        page.set_default_timeout(run.budget_ms(30))

        page.goto(run.base + "/")
        expect(page.locator("#jobs")).to_contain_text("No jobs yet")
        run.shot(page, "home_empty")
        run.step("home_empty")

        page.fill("#nj-title", "Backend Developer (Python)")
        page.fill("#nj-primary", "pl")
        page.fill("#nj-secondary", "en")
        page.select_option("#nj-assess", "secondary")
        page.click("#new-job button[type=submit]")
        page.wait_for_url(re.compile(r"/job/backend-developer-python$"))
        expect(page.locator("#job-title")).to_have_text("Backend Developer (Python)")
        run.shot(page, "job_new")
        run.step("job_created", url=page.url)

        page.evaluate(
            DROP_JS,
            [
                "#drop",
                [
                    _file_payload(docs["pdf"], "application/pdf"),
                    _file_payload(
                        docs["docx"],
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    ),
                ],
            ],
        )
        expect(page.locator("#upload-results li.ok")).to_have_count(2)
        page.set_input_files("#offer-files", str(docs["scan"]))
        expect(page.locator("#upload-results li.err")).to_have_count(1)
        expect(page.locator("#upload-results li.err")).to_contain_text("scanned PDF")
        expect(page.locator("#offer-docs li.ok")).to_have_count(2)
        run.shot(page, "offer_dropped")
        run.step("offer_dropped", results=page.locator("#upload-results li").all_inner_texts())

        page.fill("#g-minutes", "30")
        page.fill("#g-questions", "5")
        t_gen = time.monotonic()
        page.click("#generate")
        expect(page.locator("#gen-status")).to_contain_text("Drafting")
        page.wait_for_timeout(1500)
        run.shot(page, "generating", full=False)
        expect(page.locator("#gen-status")).to_contain_text(
            re.compile("Draft ready|No draft"), timeout=run.budget_ms(gen_cap)
        )
        gen_wall = round(time.monotonic() - t_gen, 1)
        status = page.locator("#gen-status").inner_text()
        run.shot(page, "draft_ready")
        if "No draft" in status:
            raise RuntimeError(f"generation failed: {status}")
        draft = page.evaluate(
            "async () => (await fetch('/api/jobs/backend-developer-python/draft')).json()"
        )
        result["generation"] = {
            "browser_wall_s": gen_wall,
            "stats": draft["stats"],
            "questions": len(draft["questions"]),
            "requirements": len(draft["requirements"]),
            "ungrounded": draft["ungrounded"],
        }
        run.step("draft_ready", **result["generation"])

        page.click('#draft-actions [data-act="review-draft"]')
        expect(page.locator("#editor")).to_be_visible()
        run.shot(page, "editor_draft")
        before_ids = page.locator("#ed-body .q .idtag").all_inner_texts()
        run.step("editor_open", questions=before_ids)

        # add a question to "technical", leave its English wording empty → inline error
        tech = section(page, "technical")
        tech.locator('[data-act="q-add"]').click()
        new_q = section(page, "technical").locator(".q").last
        new_id = new_q.locator(".idtag").inner_text()
        new_q.locator('[data-field="wording"][data-lang="pl"]').fill(
            "Jak testujesz kod, który dotyka bazy danych?"
        )
        new_q.locator('[data-field="intent"]').fill("testing practice")
        page.click("#ed-accept")
        expect(page.locator("#ed-summary .status.err")).to_be_visible()
        bad = section(page, "technical").locator(".q").last
        expect(bad.locator('[data-field="wording"][data-lang="en"]')).to_have_attribute(
            "aria-invalid", "true"
        )
        # #2043: the recruiter reads a plain sentence beside the field (machine form in the API)
        expect(bad.locator(".field-err")).to_have_text(
            f"Question {new_id.removeprefix('q')} needs an English wording."
        )
        run.shot(page, "editor_inline_error")
        run.step("inline_error_shown", question=new_id)
        bad.locator('[data-field="wording"][data-lang="en"]').fill(
            "How do you test code that touches the database?"
        )

        # move a question from "experience" into "technical" (select), then reorder by keyboard
        exp_ids = section_ids(page, "experience")
        moved = None
        if exp_ids:
            moved = exp_ids[-1]
            q_card = page.locator("#ed-body .q").filter(
                has=page.locator(".idtag", has_text=re.compile(rf"^\s*{re.escape(moved)}\s*$"))
            )
            q_card.locator('[data-field="section"]').select_option(label="technical")
        tech_before = section_ids(page, "technical")
        last = section(page, "technical").locator(".q").last
        up = last.locator('[data-act="q-up"]')
        up.focus()
        page.keyboard.press("Enter")
        tech_after = section_ids(page, "technical")
        focused_act = page.evaluate("document.activeElement && document.activeElement.dataset.act")
        run.step(
            "moved_and_reordered",
            moved=moved,
            technical_before=tech_before,
            technical_after=tech_after,
            focus_after=focused_act,
        )
        if tech_before == tech_after and len(tech_before) > 1:
            raise RuntimeError("keyboard reorder did not change the order")

        # delete a question that is not in "technical"
        deleted = None
        for name in ("language check", "experience", "intro", "candidate questions", "close"):
            ids = section_ids(page, name)
            if ids and (name != "language check" or len(ids) > 1):
                deleted = ids[0]
                section(page, name).locator(".q").first.locator('[data-act="q-remove"]').click()
                break
        # change minutes: intro +3 → total off target → rescaled in code on accept
        intro = section(page, "intro").locator('[data-field="minutes"]')
        new_intro = str(int(intro.input_value() or "1") + 3)
        intro.fill(new_intro)
        total_text = page.locator("#ed-total").inner_text()
        run.shot(page, "editor_edited")
        run.step("edited", deleted=deleted, intro_minutes=new_intro, total=total_text)

        page.click("#ed-accept")
        expect(page.locator("#gen-status")).to_contain_text("Accepted", timeout=run.budget_ms(20))
        expect(page.locator("#editor")).to_be_hidden()
        spec = page.evaluate(
            "async () => (await fetch('/api/jobs/backend-developer-python')).json()"
        )["spec"]
        bank_ids = [q["id"] for q in spec["questions"]]
        plan = {s["name"]: s for s in spec["plan"]["sections"]}
        result["accepted"] = {
            "bank": bank_ids,
            "technical": plan["technical"]["question_ids"],
            "total_minutes": sum(s["minutes"] for s in spec["plan"]["sections"]),
            "target": spec["plan"]["target_minutes"],
        }
        run.shot(page, "accepted")
        run.step("accepted", **result["accepted"])
        assert new_id in bank_ids and (deleted is None or deleted not in bank_ids)
        assert result["accepted"]["total_minutes"] == result["accepted"]["target"]

        page.fill("#tpl-name", "Backend base")
        page.click("#tpl-save-btn")
        expect(page.locator("#tpl-status")).to_contain_text("Saved template")
        run.shot(page, "template_saved", full=False)
        run.step("template_saved", status=page.locator("#tpl-status").inner_text())

        # start an interview: synthetic CV + consent (box ticked by the recruiter)
        page.fill("#iv-candidate", "Synthetic Candidate")
        page.set_input_files("#iv-cv", str(docs["cv"]))
        page.fill("#iv-informed-by", "Recruiter (synthetic)")
        page.fill("#iv-method", "verbal at call start")
        assert not page.is_checked("#iv-agreed"), "consent box must start unticked"
        page.check("#iv-agreed")
        page.click("#iv-form button[type=submit]")
        expect(page.locator("#iv-status")).to_contain_text("Consent recorded")
        expect(page.locator("#iv-status")).to_contain_text("CV stored")
        row = page.locator("#interviews li.item").first
        expect(row).to_contain_text("consent ok")
        run.shot(page, "interview_created")
        run.step("interview_created", status=page.locator("#iv-status").inner_text())

        row.locator('[data-act="tailor"]').click()
        row_status = page.locator("#interviews [data-iv-status]").first
        expect(row_status).to_contain_text(
            re.compile("Tailored questions ready|Tailoring failed"), timeout=run.budget_ms(gen_cap)
        )
        tailor_status = row_status.inner_text()
        if "failed" in tailor_status:
            raise RuntimeError(f"tailoring failed: {tailor_status}")
        page.locator('#interviews [data-act="cv-review"]').click()
        cv_editor = page.locator("#interviews [data-cv-editor] .q")
        expect(cv_editor.first).to_be_visible()
        n_cv = cv_editor.count()
        run.shot(page, "tailor_review")
        cv_editor.first.locator('[data-lang="en"]').fill(
            "Walk me through the payments API you built: what was your part?"
        )
        if n_cv > 1:
            cv_editor.last.locator('[data-cv-act="remove"]').click()
        page.locator('#interviews [data-cv-act="accept"]').click()
        expect(page.locator("#interviews [data-iv-status]").first).to_contain_text("Added")
        run.shot(page, "tailored_accepted")
        run.step(
            "tailored_accepted",
            tailor_status=tailor_status,
            drafted=n_cv,
            status=page.locator("#interviews [data-iv-status]").first.inner_text(),
        )

        with ctx.expect_page(timeout=run.budget_ms(60)) as info:
            page.locator('#interviews [data-act="cockpit"]').click()
        cockpit = info.value
        run.watch(cockpit)
        cockpit.wait_for_url(re.compile(r"http://127\.0\.0\.1:\d+/"), timeout=run.budget_ms(60))
        expect(cockpit.locator("#questions li").first).to_be_visible(timeout=run.budget_ms(30))
        gate_hidden = cockpit.evaluate("document.getElementById('gate').hidden")
        result["cockpit"] = {
            "url": cockpit.url,
            "gate_hidden": gate_hidden,
            "questions": cockpit.locator("#questions li").count(),
        }
        run.shot(cockpit, "cockpit_child")
        run.step("cockpit_open", **result["cockpit"])
        page.bring_to_front()
        expect(page.locator("#interviews [data-iv-status]").first).to_contain_text(
            "Cockpit running"
        )

        row = page.locator("#interviews li.item").first
        row.locator("details summary").click()
        row.locator('[data-act="purge"]').click()
        expect(page.locator("dialog")).to_be_visible()
        expect(page.locator("dialog .path")).to_contain_text("interviews")
        run.shot(page, "purge_confirm", full=False)
        page.locator('dialog [data-act="yes"]').click()
        expect(page.locator("#interviews [data-iv-status]").first).to_contain_text("being served")
        run.shot(page, "purge_refused_while_served")
        run.step(
            "purge_refused", status=page.locator("#interviews [data-iv-status]").first.inner_text()
        )

        page.goto(run.base + "/")
        expect(page.locator("#jobs")).to_contain_text("Synthetic Candidate")
        run.shot(page, "home_with_job")
        run.step("home_listing")
        ctx.close()

        # light/dark × wide/narrow
        result["layouts"] = []
        for scheme, w, h in (("dark", 1280, 800), ("light", 390, 844), ("dark", 390, 844)):
            c2 = browser.new_context(viewport={"width": w, "height": h}, color_scheme=scheme)
            p2 = c2.new_page()
            run.watch(p2)
            p2.set_default_timeout(run.budget_ms(30))
            p2.goto(run.base + "/")
            expect(p2.locator("#jobs")).to_contain_text("Backend Developer")
            over_home = p2.evaluate("document.documentElement.scrollWidth - window.innerWidth")
            run.shot(p2, f"home_{scheme}_{w}")
            p2.goto(run.base + "/job/backend-developer-python")
            p2.locator('#draft-actions [data-act="edit-bank"]').click()
            expect(p2.locator("#editor")).to_be_visible()
            over_job = p2.evaluate("document.documentElement.scrollWidth - window.innerWidth")
            run.shot(p2, f"job_editor_{scheme}_{w}")
            p2.locator("#ed-body .q").first.scroll_into_view_if_needed()
            run.shot(p2, f"job_editor_view_{scheme}_{w}", full=False)  # readable at real size
            bg = p2.evaluate("getComputedStyle(document.body).backgroundColor")
            result["layouts"].append(
                {
                    "scheme": scheme,
                    "width": w,
                    "overflow_home_px": over_home,
                    "overflow_job_px": over_job,
                    "body_bg": bg,
                }
            )
            c2.close()
        run.step("layouts", layouts=result["layouts"])
        browser.close()
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=None, help="run folder (default: a new temp folder)")
    ap.add_argument("--real-model", action="store_true", help="draft with the local model")
    ap.add_argument("--timeout-s", type=float, default=900.0)
    ap.add_argument("--_serve", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--workspace", help=argparse.SUPPRESS)
    ap.add_argument("--port", type=int, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args._serve:
        return serve_main(args)

    out = check_out_dir(Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="sc_ui_")))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(out / "smoke.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    workspace = out / "workspace"
    docs = write_docs(out / "docs")
    port = pick_port()
    base = f"http://127.0.0.1:{port}"
    server_log = (out / "server.log").open("w", encoding="utf-8")
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_serve",
        "--workspace",
        str(workspace),
        "--port",
        str(port),
    ] + (["--real-model"] if args.real_model else [])
    logger.info("workspace app: %s", " ".join(cmd))
    proc = subprocess.Popen(  # noqa: S603 — our own script
        cmd, cwd=PROJECT, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True
    )
    summary: dict = {"out": str(out), "real_model": args.real_model, "port": port, "ok": False}
    run = Run(out, base, args.real_model, args.timeout_s)
    try:
        deadline = time.monotonic() + 30
        while port_free(port):
            if proc.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f"workspace app did not start (see {out / 'server.log'})")
            time.sleep(0.2)
        summary.update(browser_flow(run, docs, workspace))
        summary["ok"] = True
    except Exception as e:  # noqa: BLE001 — record, then always stop the server
        logger.exception("smoke run failed")
        summary["error"] = f"{type(e).__name__}: {e}"
    finally:
        child_ports = [p for p in range(port + 1, port + 41) if not port_free(p)]
        t_stop = time.monotonic()
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)  # ONE SIGINT: the app stops its cockpit child itself
        try:
            code = proc.wait(timeout=40)
        except subprocess.TimeoutExpired:
            logger.error("workspace app ignored SIGINT for 40 s; SIGTERM")
            proc.terminate()
            code = proc.wait(timeout=20)
        server_log.close()
        locks = [str(p) for p in workspace.rglob(".serve.lock")]
        summary["stop"] = {
            "exit_code": code,
            "seconds": round(time.monotonic() - t_stop, 1),
            "child_ports_before": child_ports,
            "listening_after": [p for p in [port, *child_ports] if not port_free(p)],
            "serve_locks_left": locks,
        }
        if args.real_model:
            summary["model_unloaded"] = unload_model()
    unexpected = [
        e
        for e in run.http_errors
        if (e["status"], e["method"]) not in EXPECTED_HTTP
        and not e["url"].endswith("/favicon.ico")  # the cockpit page has no icon link
    ]
    summary.update(
        steps=run.steps,
        screenshots=run.shots,
        console_errors=run.console,
        http_errors=run.http_errors,
        unexpected_http_errors=unexpected,
        wall_s=round(time.monotonic() - run.t0, 1),
    )
    stop = summary["stop"]
    clean = stop["exit_code"] == 0 and not stop["listening_after"] and not stop["serve_locks_left"]
    summary["ok"] = bool(summary["ok"] and clean and not run.console and not unexpected)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), "utf-8")
    logger.info("summary → %s (ok=%s)", out / "summary.json", summary["ok"])
    return 0 if summary["ok"] else 1


def unload_model() -> bool:
    """Real-model runs: unload the chat model (Ollama keep_alive 0) so the GPU is idle again."""
    from recruiter_copilot.config import load_settings  # noqa: PLC0415
    from recruiter_copilot.llm import OllamaProvider  # noqa: PLC0415

    settings = load_settings(env_file=PROJECT / "config" / ".env")
    ok = OllamaProvider(settings).reset_prompt_cache()
    logger.info("model %s unloaded: %s", settings.ollama_model, ok)
    return ok


if __name__ == "__main__":
    os.chdir(PROJECT)
    sys.exit(main())
