"""Workspace app API (D28/D29 UI, D14) through FastAPI's TestClient with a stub chat.

No model, no GPU, no network, no real browser; every workspace is a ``tmp_path``. All documents
are synthetic. The browser run lives in ``scripts/ui_smoke_workspace.py``.
"""

from __future__ import annotations

import json
import signal
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import write_docx, write_scanned_pdf, write_text_pdf
from fastapi.testclient import TestClient
from test_prep import CV, OFFER, StubChat, _cv_reply, good_reply

from recruiter_copilot import prep, session_lock
from recruiter_copilot.cli import build_parser
from recruiter_copilot.config import Settings
from recruiter_copilot.models import JobSpec, Profile, Session, load_json
from recruiter_copilot.workspace_app import (
    CockpitChildren,
    Refused,
    consent_from,
    create_workspace_app,
    safe_upload_name,
)

BASE = "http://127.0.0.1:8765"


class Chats:
    """A chat factory handing out queued StubChats (one per generation)."""

    def __init__(self, *chats: object) -> None:
        self.chats = list(chats)

    def __call__(self) -> object:
        assert self.chats, "no stub chat left"
        return self.chats.pop(0)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "ws"


def make_client(root: Path, chats: Chats | None = None, **kw) -> TestClient:
    app = create_workspace_app(Settings(), root, chat_factory=chats or Chats(), **kw)
    return TestClient(app, base_url=BASE)


@pytest.fixture
def client(root: Path) -> TestClient:
    return make_client(root, Chats(StubChat(good_reply()), StubChat(_cv_reply())))


def new_job(client: TestClient, offer: bool = True) -> str:
    r = client.post(
        "/api/jobs",
        json={
            "title": "Backend Developer",
            "primary": "pl",
            "secondary": "en",
            "assess_language": "en",
        },
    )
    assert r.status_code == 200, r.text
    slug = r.json()["slug"]
    if offer:
        assert client.post(f"/api/jobs/{slug}/paste", json={"text": OFFER}).status_code == 200
    return slug


def put_file(client: TestClient, url: str, path: Path, name: str | None = None):
    return client.put(url, content=path.read_bytes(), headers={"X-Filename": name or path.name})


def wait_task(client: TestClient, task: dict, timeout_s: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        t = client.get(f"/api/tasks/{task['id']}").json()
        if t["state"] != "running":
            return t
        time.sleep(0.02)
    raise AssertionError(f"task {task['id']} still running after {timeout_s}s")


def generate(client: TestClient, slug: str) -> dict:
    r = client.post(f"/api/jobs/{slug}/generate", json={"target_minutes": 45, "n_questions": 3})
    assert r.status_code == 200, r.text
    return wait_task(client, r.json())


def editor_body(draft: dict) -> dict:
    return {
        "target_minutes": draft["target_minutes"],
        "requirements": draft["requirements"],
        "questions": draft["questions"],
        "plan": draft["plan"],
    }


# ── pages, assets, guards ───────────────────────────────────────────────────────────────────


def test_pages_and_assets_are_served_with_a_csp(client: TestClient) -> None:
    home = client.get("/")
    assert home.status_code == 200 and "<title>Second Chair workspace</title>" in home.text
    assert "default-src 'self'" in home.headers["content-security-policy"]
    job = client.get("/job/anything")
    assert job.status_code == 200 and 'id="editor"' in job.text
    assert client.get("/static/workspace.js").headers["content-type"].startswith("text/javascript")
    assert client.get("/static/workspace.css").headers["content-type"].startswith("text/css")
    assert client.get("/static/cockpit.html").status_code == 404  # whitelist only
    assert client.get("/static/..%2Fworkspace_app.py").status_code == 404


def test_non_loopback_host_and_cross_origin_writes_are_refused(client: TestClient) -> None:
    assert client.get("/api/workspace", headers={"host": "evil.example:8765"}).status_code == 403
    r = client.post("/api/jobs", json={"title": "x"}, headers={"origin": "http://evil.example"})
    assert r.status_code == 403
    r = client.post("/api/jobs", json={"title": "x"}, headers={"origin": "http://127.0.0.1:9999"})
    assert r.status_code == 403  # another loopback port is another origin
    r = client.post("/api/jobs", json={"title": "Same origin"}, headers={"origin": BASE})
    assert r.status_code == 200


def test_empty_workspace_then_a_job(client: TestClient, root: Path) -> None:
    view = client.get("/api/workspace").json()
    assert view["root"] == str(root) and view["jobs"] == [] and view["templates"] == []
    assert view["disclosure"]["profile"] == "local" and view["disclosure"]["line"] == ""
    slug = new_job(client)
    view = client.get("/api/workspace").json()
    assert [j["slug"] for j in view["jobs"]] == [slug]
    assert view["jobs"][0]["languages"] == {
        "primary": "pl",
        "secondary": "en",
        "assess_language": "en",
        "report_language": None,
    }
    bad = client.post("/api/jobs", json={"title": "X", "primary": "pl", "assess_language": "de"})
    assert bad.status_code == 422 and "assessed language" in bad.json()["error"]


def test_api_profile_payloads_carry_the_si1_line(root: Path) -> None:
    app = create_workspace_app(Settings(profile=Profile.API), root, chat_factory=Chats())
    c = TestClient(app, base_url=BASE)
    line = c.get("/api/workspace").json()["disclosure"]["line"]
    assert line.startswith("!! api profile") and "SI1" in line


# ── uploads ─────────────────────────────────────────────────────────────────────────────────


def test_upload_each_format_and_a_refused_scan(client: TestClient, root: Path, tmp_path) -> None:
    slug = new_job(client, offer=False)
    src = tmp_path / "src"
    src.mkdir()
    pdf = write_text_pdf(src / "Offer Backend.pdf", ["Backend Developer", "Python 3+ years"])
    docx = write_docx(src / "benefits.docx", ["Benefits: remote work"], [["Stack", "Python"]])
    (src / "notes.md").write_text("# Notes\nKafka nice to have", encoding="utf-8")
    (src / "extra.txt").write_bytes("Zespół w Łodzi".encode("cp1250"))
    url = f"/api/jobs/{slug}/offer"
    for path in (pdf, docx, src / "notes.md", src / "extra.txt"):
        r = put_file(client, url, path)
        assert r.status_code == 200, r.text
        assert r.json()["docs"][0]["chars"] > 0
    scan = put_file(client, url, write_scanned_pdf(src / "scan.pdf"))
    assert scan.status_code == 422 and "scanned PDF" in scan.json()["error"]
    exe = client.put(url, content=b"MZ", headers={"X-Filename": "tool.exe"})
    assert exe.status_code == 422 and "unsupported type" in exe.json()["error"]
    spec = load_json(JobSpec, root / "jobs" / slug / "job.json")
    assert [d.kind for d in spec.offer_docs] == ["pdf", "docx", "md", "txt"]
    offer = root / "jobs" / slug / "offer"
    assert "Zespół w Łodzi" in (offer / "extra.txt").read_text(encoding="utf-8")
    assert not list(offer.glob("scan*")) and not list(offer.glob("tool*"))
    assert not list((root / "jobs" / slug).glob(".upload-*"))  # stages always removed


@pytest.mark.parametrize(
    "name", ["../evil.txt", "..%2Fevil.txt", "a/b.txt", "a%5Cb.txt", ".hidden.txt", "", "x..txt"]
)
def test_upload_names_with_paths_are_refused(client: TestClient, root: Path, name: str) -> None:
    slug = new_job(client, offer=False)
    r = client.put(f"/api/jobs/{slug}/offer", content=b"text", headers={"X-Filename": name})
    assert r.status_code == 422
    assert not any(root.parent.glob("evil*")) and not any(root.glob("**/evil*"))


def test_oversized_upload_is_refused_while_reading(root: Path) -> None:
    c = make_client(root, max_upload_bytes=1024)
    slug = new_job(c, offer=False)
    r = c.put(f"/api/jobs/{slug}/offer", content=b"x" * 5000, headers={"X-Filename": "big.txt"})
    assert r.status_code == 422 and "limit" in r.json()["error"]
    assert load_json(JobSpec, root / "jobs" / slug / "job.json").offer_docs == []


def test_safe_upload_name_rules() -> None:
    assert safe_upload_name("Oferta%20pracy.PDF") == "Oferta pracy.PDF"
    for bad in ("..", "a/../b.md", "/etc/passwd.txt", "C:\\x.txt", "a\x00.txt", "x.exe", None):
        with pytest.raises(ValueError):
            safe_upload_name(bad)


def test_paste_box_validates_like_a_file(client: TestClient) -> None:
    slug = new_job(client, offer=False)
    assert client.post(f"/api/jobs/{slug}/paste", json={"text": "   "}).status_code == 422
    assert client.post(f"/api/jobs/{slug}/paste", json={"text": 3}).status_code == 422


# ── path guard ──────────────────────────────────────────────────────────────────────────────


def test_path_traversal_is_refused(client: TestClient, root: Path, tmp_path: Path) -> None:
    slug = new_job(client)
    for bad in ("..", "...", "x..y", ".hidden"):
        assert client.get(f"/api/jobs/{bad}").status_code in (400, 404)
    assert client.get("/api/jobs/%2E%2E").status_code in (400, 404)
    assert client.get("/api/jobs/no-such-job").status_code == 404
    assert client.get(f"/api/interviews/{slug}/..").status_code in (400, 404, 405)
    # a symlinked job / interview pointing outside the workspace is refused
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "job.json").write_text(json.dumps({"title": "x"}), encoding="utf-8")
    (outside / "session.json").write_text("{}", encoding="utf-8")
    (root / "jobs" / "linked").symlink_to(outside)
    assert client.get("/api/jobs/linked").status_code == 403
    (root / "jobs" / slug / "interviews" / "2026-01-01_x").symlink_to(outside)
    r = client.delete(f"/api/interviews/{slug}/2026-01-01_x")
    assert r.status_code == 403 and outside.is_dir() and (outside / "session.json").is_file()
    assert client.get(f"/api/jobs/{slug}").status_code == 200  # the link is not listed


def test_reports_are_served_read_only_from_inside_the_workspace(
    client: TestClient, root: Path, tmp_path: Path
) -> None:
    slug = new_job(client)
    name = client.post(f"/api/jobs/{slug}/interviews", json={"candidate": "Ann Example"}).json()[
        "name"
    ]
    folder = root / "jobs" / slug / "interviews" / name
    (folder / "report_20261124_101010.html").write_text("<html><body>R</body></html>", "utf-8")
    (folder / "report_20261124_101010.md").write_text("# R", "utf-8")
    secret = tmp_path / "secret.html"
    secret.write_text("SECRET", encoding="utf-8")
    (folder / "report_link.html").symlink_to(secret)
    ok = client.get(f"/report/{slug}/{name}/report_20261124_101010.html")
    assert ok.status_code == 200 and "R" in ok.text
    assert "default-src 'none'" in ok.headers["content-security-policy"]
    md = client.get(f"/report/{slug}/{name}/report_20261124_101010.md")
    assert md.headers["content-type"].startswith("text/plain")
    assert client.get(f"/report/{slug}/{name}/report_link.html").status_code == 403
    assert client.get(f"/report/{slug}/{name}/session.json").status_code == 404
    assert client.get(f"/report/{slug}/{name}/..%2F..%2Fjob.json").status_code == 404
    assert client.post(f"/report/{slug}/{name}/report_20261124_101010.html").status_code == 405
    listed = client.get(f"/api/jobs/{slug}").json()["interviews"][0]["reports"]
    assert {r["name"] for r in listed} >= {"report_20261124_101010.html"}


# ── generate → poll → draft → edit → accept ─────────────────────────────────────────────────


def test_generate_runs_in_the_background_and_writes_a_pending_draft(
    client: TestClient, root: Path
) -> None:
    slug = new_job(client)
    before = (root / "jobs" / slug / "job.json").read_text(encoding="utf-8")
    task = generate(client, slug)
    assert task["state"] == "done" and task["calls"] == 1, task
    assert (root / "jobs" / slug / "job.json").read_text(encoding="utf-8") == before  # D29
    draft = client.get(f"/api/jobs/{slug}/draft").json()
    assert [q["id"] for q in draft["questions"]] == ["q1", "q2", "q3"]
    assert draft["status"] == "pending" and draft["ungrounded"] == []
    assert client.get(f"/api/jobs/{slug}").json()["has_draft"] is True


def test_generate_needs_offer_docs_and_one_task_at_a_time(root: Path) -> None:
    gate = threading.Event()

    class SlowChat(StubChat):
        def complete(self, prompt, system=None, max_tokens=None):
            gate.wait(5)
            return super().complete(prompt, system, max_tokens)

    c = make_client(root, Chats(SlowChat(good_reply())))
    empty = new_job(c, offer=False)
    r = c.post(f"/api/jobs/{empty}/generate", json={})
    assert r.status_code == 422 and "offer documents" in r.json()["error"]
    slug = new_job(c)
    first = c.post(f"/api/jobs/{slug}/generate", json={}).json()
    busy = c.post(f"/api/jobs/{slug}/generate", json={})
    assert busy.status_code == 409 and "busy" in busy.json()["error"]
    assert c.get(f"/api/jobs/{slug}").json()["task"]["id"] == first["id"]  # resumable on reload
    gate.set()
    assert wait_task(c, first)["state"] == "done"
    bad = c.post(f"/api/jobs/{slug}/generate", json={"target_minutes": 3})
    assert bad.status_code == 422


def test_generation_failure_is_reported_with_reasons(root: Path) -> None:
    broken = good_reply()
    broken["questions"][0]["requirement_ids"] = ["r99"]
    c = make_client(root, Chats(StubChat("not json", broken)))
    slug = new_job(c)
    task = generate(c, slug)
    assert task["state"] == "error" and task["calls"] == 2
    assert any("'r99' does not resolve" in r for r in task["reasons"])
    assert c.get(f"/api/jobs/{slug}/draft").status_code == 422  # no pending draft written


def test_edited_draft_accept_ok(client: TestClient, root: Path) -> None:
    slug = new_job(client)
    generate(client, slug)
    body = editor_body(client.get(f"/api/jobs/{slug}/draft").json())
    # edit: add a question, delete one, reorder, change minutes
    body["questions"] = [q for q in body["questions"] if q["id"] != "q2"]
    body["questions"].append(
        {
            "id": "q4",
            "text": {"pl": "Jak testujesz kod?", "en": "How do you test code?"},
            "intent": "testing",
            "expected_signals": ["pytest"],
            "requirement_ids": ["r1"],
            "assesses_language": False,
        }
    )
    secs = {s["name"]: s for s in body["plan"]["sections"]}
    secs["technical"]["question_ids"] = ["q4"]
    secs["experience"]["question_ids"] = ["q1"]
    secs["intro"]["minutes"] = 7
    secs["close"]["minutes"] = 2  # still sums to 45
    r = client.post(f"/api/jobs/{slug}/draft/accept", json=body)
    assert r.status_code == 200, r.text
    assert r.json()["accepted"] == ["q1", "q3", "q4"] and r.json()["rescaled"] is False
    spec = load_json(JobSpec, root / "jobs" / slug / "job.json")
    assert [q.id for q in spec.questions] == ["q1", "q3", "q4"]
    by = {s.name: s for s in spec.plan.sections}
    assert by["technical"].question_ids == ["q4"] and by["intro"].minutes == 7
    assert spec.plan.total_minutes == 45
    assert not (root / "jobs" / slug / prep.DRAFT_FILE).exists()  # archived
    assert list((root / "jobs" / slug).glob("draft.accepted-*.json"))


def test_edited_draft_with_errors_saves_nothing(client: TestClient, root: Path) -> None:
    slug = new_job(client)
    generate(client, slug)
    job_before = (root / "jobs" / slug / "job.json").read_text(encoding="utf-8")
    draft_before = (root / "jobs" / slug / prep.DRAFT_FILE).read_text(encoding="utf-8")
    body = editor_body(client.get(f"/api/jobs/{slug}/draft").json())
    body["questions"][1]["text"]["en"] = "  "
    body["requirements"][0]["weight"] = 9
    body["plan"]["sections"][0]["minutes"] = 0
    body["questions"].append({"id": "q9", "text": {"pl": "a", "en": "b"}})  # in no section
    r = client.post(f"/api/jobs/{slug}/draft/accept", json=body)
    assert r.status_code == 422
    errors = r.json()["errors"]
    assert any(
        e.startswith("questions[1] (q2): no wording in session language 'en'") for e in errors
    )
    assert any(e.startswith("requirements[0] (r1): weight") for e in errors)
    assert any(e.startswith("plan.sections[0]") and "minutes" in e for e in errors)
    assert any("'q9' must appear in exactly one section" in e for e in errors)
    assert (root / "jobs" / slug / "job.json").read_text(encoding="utf-8") == job_before
    assert (root / "jobs" / slug / prep.DRAFT_FILE).read_text(encoding="utf-8") == draft_before
    garbage = client.post(f"/api/jobs/{slug}/draft/accept", json={"requirements": "x"})
    assert garbage.status_code == 422  # never a 500 on a malformed body


def test_accepted_bank_is_edited_in_place_and_revalidated(client: TestClient, root: Path) -> None:
    slug = new_job(client)
    generate(client, slug)
    client.post(
        f"/api/jobs/{slug}/draft/accept",
        json=editor_body(client.get(f"/api/jobs/{slug}/draft").json()),
    )
    spec = client.get(f"/api/jobs/{slug}").json()["spec"]
    body = {
        "target_minutes": spec["plan"]["target_minutes"],
        "requirements": spec["requirements"],
        "questions": spec["questions"][:2],
        "plan": {
            "sections": [
                {"name": "intro", "minutes": 10, "question_ids": []},
                {"name": "technical", "minutes": 30, "question_ids": ["q2", "q1"]},
                {"name": "close", "minutes": 10, "question_ids": []},  # 50 ≠ 45 → rescaled
            ]
        },
    }
    r = client.post(f"/api/jobs/{slug}/bank", json=body)
    assert r.status_code == 200 and r.json()["rescaled"] is True, r.text
    saved = load_json(JobSpec, root / "jobs" / slug / "job.json")
    assert [q.id for q in saved.questions] == ["q1", "q2"]  # replaced, not appended
    assert saved.plan.sections[1].question_ids == ["q2", "q1"] and saved.plan.total_minutes == 45
    before = (root / "jobs" / slug / "job.json").read_text(encoding="utf-8")
    body["plan"]["sections"][1]["question_ids"] = ["q2"]  # q1 now in no section
    bad = client.post(f"/api/jobs/{slug}/bank", json=body)
    assert bad.status_code == 422 and any("'q1'" in e for e in bad.json()["errors"])
    assert (root / "jobs" / slug / "job.json").read_text(encoding="utf-8") == before


def test_discard_draft(client: TestClient, root: Path) -> None:
    slug = new_job(client)
    generate(client, slug)
    assert client.delete(f"/api/jobs/{slug}/draft").status_code == 200
    assert client.delete(f"/api/jobs/{slug}/draft").status_code == 404


# ── interviews, consent, CV, tailoring, templates ───────────────────────────────────────────


def _accepted_job(client: TestClient) -> str:
    slug = new_job(client)
    generate(client, slug)
    r = client.post(
        f"/api/jobs/{slug}/draft/accept",
        json=editor_body(client.get(f"/api/jobs/{slug}/draft").json()),
    )
    assert r.status_code == 200
    return slug


def test_start_interview_with_consent(client: TestClient, root: Path) -> None:
    slug = _accepted_job(client)
    consent = {
        "informed_by": "Recruiter A",
        "method": "verbal at call start",
        "candidate_agreed": True,
    }
    r = client.post(
        f"/api/jobs/{slug}/interviews",
        json={"candidate": "Ann Example", "date": "2026-12-02", "consent": consent},
    )
    assert r.status_code == 200 and r.json()["consent_recorded"] is True, r.text
    folder = root / "jobs" / slug / "interviews" / "2026-12-02_ann-example"
    session = load_json(Session, folder / "session.json")
    assert session.consent.is_complete and session.consent.informed_at
    assert [q.id for q in session.questions] == ["q1", "q2", "q3"]
    row = r.json()["interview"]
    assert row["consent_ok"] and row["status"] == "no transcript" and not row["has_cv"]


def test_consent_is_never_assumed(client: TestClient, root: Path) -> None:
    slug = _accepted_job(client)
    # who/how filled but the box NOT ticked → created, consent not recorded
    r = client.post(
        f"/api/jobs/{slug}/interviews",
        json={
            "candidate": "Bo Example",
            "date": "2026-12-03",
            "consent": {"informed_by": "A", "method": "email", "candidate_agreed": False},
        },
    )
    assert r.status_code == 200 and r.json()["consent_recorded"] is False
    name = r.json()["name"]
    assert not load_json(
        Session, root / "jobs" / slug / "interviews" / name / "session.json"
    ).consent.is_complete
    # ticked but half a record → refused before any folder exists
    r = client.post(
        f"/api/jobs/{slug}/interviews",
        json={
            "candidate": "Cy Example",
            "date": "2026-12-04",
            "consent": {"informed_by": "", "method": "email", "candidate_agreed": True},
        },
    )
    assert r.status_code == 422
    assert not (root / "jobs" / slug / "interviews" / "2026-12-04_cy-example").exists()
    # completed later on the row
    url = f"/api/interviews/{slug}/{name}/consent"
    assert client.post(url, json={"informed_by": "A", "method": "email"}).status_code == 422
    ok = client.post(url, json={"informed_by": "A", "method": "email", "candidate_agreed": True})
    assert ok.status_code == 200
    assert client.get(f"/api/jobs/{slug}").json()["interviews"][0]["consent_ok"] is True


def test_consent_from_rules() -> None:
    assert consent_from(None) is None
    assert consent_from({"informed_by": "A", "method": "m"}) is None
    with pytest.raises(Refused):
        consent_from({"candidate_agreed": "yes", "informed_by": "A", "method": "m"})
    c = consent_from({"candidate_agreed": True, "informed_by": " A ", "method": "m"})
    assert c is not None and c.is_complete and c.informed_by == "A"


def test_cv_upload_tailor_accept_into_the_interview_only(
    client: TestClient, root: Path, tmp_path: Path
) -> None:
    slug = _accepted_job(client)
    name = client.post(
        f"/api/jobs/{slug}/interviews",
        json={"candidate": "Synthetic Candidate", "date": "2026-12-02"},
    ).json()["name"]
    folder = root / "jobs" / slug / "interviews" / name
    ascii_cv = CV.replace("—", "-").replace("–", "-")  # the tiny test PDF writer is latin-1
    cv_pdf = write_text_pdf(tmp_path / "CV Synthetic.pdf", ascii_cv.splitlines())
    r = put_file(client, f"/api/interviews/{slug}/{name}/cv", cv_pdf)
    assert r.status_code == 200 and r.json()["doc"]["name"] == "cv.pdf", r.text
    assert (folder / "docs" / "cv.txt").is_file() and not list(folder.glob(".upload-*"))
    again = put_file(client, f"/api/interviews/{slug}/{name}/cv", cv_pdf)
    assert again.status_code == 409
    task = client.post(f"/api/interviews/{slug}/{name}/tailor", json={"n": 3}).json()
    assert wait_task(client, task)["state"] == "done"
    pending = client.get(f"/api/interviews/{slug}/{name}/cv-draft").json()
    assert [q["id"] for q in pending["draft"]["questions"]] == ["cv1", "cv2", "cv3"]
    assert pending["draft"]["ungrounded"] == ["cv3"] and pending["languages"] == ["pl", "en"]
    edited = pending["draft"]["questions"][:2]
    edited[0]["text"]["en"] = "Walk me through the payments API you built."
    bad = [dict(edited[0], text={"pl": "x", "en": ""})]
    r = client.post(f"/api/interviews/{slug}/{name}/cv-draft/accept", json={"questions": bad})
    assert r.status_code == 422 and any("'en'" in e for e in r.json()["errors"])
    r = client.post(f"/api/interviews/{slug}/{name}/cv-draft/accept", json={"questions": edited})
    assert r.status_code == 200 and r.json()["added"] == ["cv1", "cv2"], r.text
    session = load_json(Session, folder / "session.json")
    tailored = [q for q in session.questions if q.cv_tailored]
    assert [q.id for q in tailored] == ["cv1", "cv2"]
    assert tailored[0].text["en"].startswith("Walk me through")
    assert "cv1" in next(s for s in session.plan.sections if s.name == "experience").question_ids
    job = load_json(JobSpec, root / "jobs" / slug / "job.json")
    assert not any(q.cv_tailored for q in job.questions)  # D28: never into job.json
    stored = (folder / "session.json").read_text(encoding="utf-8")
    assert ascii_cv.splitlines()[2] not in stored  # the CV text is never folded into session.json


def test_templates_exclude_cv_tailored_and_apply(client: TestClient, root: Path) -> None:
    slug = _accepted_job(client)
    name = client.post(
        f"/api/jobs/{slug}/interviews",
        json={"candidate": "Synthetic Candidate", "date": "2026-12-02"},
    ).json()["name"]
    folder = root / "jobs" / slug / "interviews" / name
    (folder / "docs" / "cv.md").write_text(CV, encoding="utf-8")
    task = client.post(f"/api/interviews/{slug}/{name}/tailor", json={}).json()
    wait_task(client, task)
    draft = client.get(f"/api/interviews/{slug}/{name}/cv-draft").json()["draft"]
    client.post(
        f"/api/interviews/{slug}/{name}/cv-draft/accept", json={"questions": draft["questions"]}
    )
    r = client.post(f"/api/interviews/{slug}/{name}/template", json={"name": "From interview"})
    assert r.status_code == 200, r.text
    assert [x["id"] for x in r.json()["left_out"]] == ["cv1", "cv2", "cv3"]
    assert r.json()["left_out"][0]["text"]  # the wording is shown, so the recruiter knows which
    saved = json.loads((root / "templates" / "from-interview.json").read_text(encoding="utf-8"))
    assert [q["id"] for q in saved["questions"]] == ["q1", "q2", "q3"]
    r = client.post(f"/api/jobs/{slug}/template", json={"name": "Backend base"})
    assert r.status_code == 200 and r.json()["left_out"] == []
    dup = client.post(f"/api/jobs/{slug}/template", json={"name": "Backend base"})
    assert dup.status_code == 422 and "exists" in dup.json()["error"]
    assert (
        client.post(
            f"/api/jobs/{slug}/template", json={"name": "Backend base", "overwrite": True}
        ).status_code
        == 200
    )
    other = new_job(client)
    r = client.post(f"/api/jobs/{other}/apply-template", json={"name": "backend-base"})
    assert r.status_code == 200 and r.json()["added"] == ["q1", "q2", "q3"]
    view = client.get("/api/workspace").json()
    assert {t["slug"] for t in view["templates"]} == {"backend-base", "from-interview"}
    assert client.get("/api/templates/backend-base").json()["name"] == "Backend base"
    assert client.delete("/api/templates/backend-base").status_code == 200
    assert client.delete("/api/templates/backend-base").status_code == 404
    assert client.delete("/api/templates/..").status_code in (404, 422)


def test_purge_refused_while_served_then_deletes_only_the_bundle(
    client: TestClient, root: Path
) -> None:
    slug = _accepted_job(client)
    name = client.post(f"/api/jobs/{slug}/interviews", json={"candidate": "Dee Example"}).json()[
        "name"
    ]
    folder = root / "jobs" / slug / "interviews" / name
    session_lock.acquire(folder)  # this process now "serves" it (a live lock)
    try:
        r = client.delete(f"/api/interviews/{slug}/{name}")
        assert r.status_code == 409 and "being served" in r.json()["error"]
        consent = client.post(
            f"/api/interviews/{slug}/{name}/consent",
            json={"informed_by": "A", "method": "m", "candidate_agreed": True},
        )
        assert consent.status_code == 409  # session.json is the cockpit's while it runs
        assert client.get(f"/api/jobs/{slug}").json()["interviews"][0]["served"] is True
    finally:
        session_lock.release(folder)
    assert folder.is_dir()
    r = client.delete(f"/api/interviews/{slug}/{name}")
    assert r.status_code == 200 and r.json()["deleted_files"] >= 1
    assert not folder.exists() and (root / "jobs" / slug / "job.json").is_file()


# ── cockpit children (fake popen — no real serve here) ──────────────────────────────────────


class FakeProc:
    def __init__(self, cmd: list[str]) -> None:
        self.cmd = cmd
        self.pid = 424242
        self.signals: list[int] = []
        self.code: int | None = None

    def poll(self):
        return self.code

    def send_signal(self, sig: int) -> None:
        self.signals.append(sig)
        self.code = 0

    def wait(self, timeout=None):
        return self.code

    def terminate(self) -> None:  # pragma: no cover — must not be needed
        raise AssertionError("terminate after one SIGINT")


def test_open_cockpit_spawns_one_child_and_stops_it_with_one_sigint(root: Path) -> None:
    procs: list[FakeProc] = []
    listening: set[int] = set()

    def popen(cmd):
        procs.append(FakeProc(cmd))
        listening.add(int(cmd[cmd.index("--port") + 1]))
        return procs[-1]

    kids = CockpitChildren(
        Settings(),
        popen=popen,
        port_free=lambda p: p not in listening and p != 8766,
        own_port=8765,
    )
    c = make_client(root, Chats(StubChat(good_reply())), children=kids)
    slug = _accepted_job(c)
    name = c.post(f"/api/jobs/{slug}/interviews", json={"candidate": "Eve Example"}).json()["name"]
    folder = (root / "jobs" / slug / "interviews" / name).resolve()
    r = c.post(f"/api/interviews/{slug}/{name}/cockpit")
    assert r.status_code == 200, r.text
    assert r.json()["url"] == "http://127.0.0.1:8767/"  # 8766 busy, 8787 never, own port skipped
    cmd = procs[0].cmd
    assert cmd[1:5] == ["-m", "recruiter_copilot.cli", "serve", str(folder)]
    assert (
        cmd[cmd.index("--host") + 1] == "127.0.0.1" and cmd[cmd.index("--profile") + 1] == "local"
    )
    again = c.post(f"/api/interviews/{slug}/{name}/cockpit")
    assert again.json()["port"] == 8767 and len(procs) == 1  # reused, not a second serve
    assert (
        c.get(f"/api/jobs/{slug}").json()["interviews"][0]["cockpit_url"]
        == "http://127.0.0.1:8767/"
    )
    assert c.post(f"/api/interviews/{slug}/{name}/cockpit/stop").status_code == 200
    assert procs[0].signals == [signal.SIGINT]
    assert c.post(f"/api/interviews/{slug}/{name}/cockpit/stop").status_code == 404
    c.post(f"/api/interviews/{slug}/{name}/cockpit")
    with c:  # lifespan: leaving the app stops every child with one SIGINT
        pass
    assert procs[1].signals == [signal.SIGINT]


def test_children_skip_reserved_ports() -> None:
    kids = CockpitChildren(
        Settings(port=8786), popen=lambda cmd: None, port_free=lambda p: True, own_port=8786
    )
    assert kids._pick_port() == 8788  # 8787 is never used


def test_children_use_the_configured_port_range() -> None:
    """D28 + compose: the child range is a setting; the default (workspace port + 1) is unchanged."""
    s = Settings(cockpit_port_first=9100, cockpit_port_count=3)
    busy = {9100}
    kids = CockpitChildren(s, popen=lambda cmd: None, port_free=lambda p: p not in busy)
    assert kids._pick_port() == 9101
    busy |= {9101, 9102}
    with pytest.raises(Refused, match="9100..9102"):
        kids._pick_port()
    default = CockpitChildren(Settings(), popen=lambda cmd: None, port_free=lambda p: True)
    assert default.port_range() == range(8766, 8806) and default._pick_port() == 8766


def test_children_bind_loopback_unless_the_container_flag_is_set(tmp_path: Path) -> None:
    """D14 amendment: children inherit the parent's wildcard host only under the container flag."""
    plain = CockpitChildren(Settings(), popen=lambda cmd: None)
    cmd = plain.command(tmp_path, 8766)
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
    boxed = CockpitChildren(Settings(host="0.0.0.0", in_container=True), popen=lambda cmd: None)
    cmd = boxed.command(tmp_path, 8766)
    assert cmd[cmd.index("--host") + 1] == "0.0.0.0"
    loop = CockpitChildren(Settings(host="localhost", in_container=True), popen=lambda cmd: None)
    cmd = loop.command(tmp_path, 8766)
    assert cmd[cmd.index("--host") + 1] == "localhost"


# ── CLI + launcher ──────────────────────────────────────────────────────────────────────────


def test_launcher_defaults_to_the_workspace_home(tmp_path: Path, monkeypatch) -> None:
    import importlib.util

    script = Path(__file__).resolve().parent.parent / "scripts" / "launch_cockpit.py"
    if not script.exists():
        pytest.skip("desktop launcher is not shipped in the public repository")
    spec = importlib.util.spec_from_file_location("launch_cockpit_under_test", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    home = mod.serve_command(None, 8765)
    assert home[1:] == ["-m", "recruiter_copilot.cli", "serve", "--port", "8765"]
    assert mod.serve_command(None, 8770, tmp_path)[3:] == [
        "serve",
        "--workspace",
        str(tmp_path),
        "--port",
        "8770",
    ]
    assert mod.serve_command(tmp_path / "s", 8771)[3:] == [
        "serve",
        str(tmp_path / "s"),
        "--port",
        "8771",
    ]
    seen = {}
    monkeypatch.setattr(
        mod,
        "run",
        lambda folder, open_browser=True, workspace=None: seen.update(
            folder=folder, workspace=workspace
        )
        or 0,
    )
    monkeypatch.setattr(mod.signal, "signal", lambda *a: None)
    assert mod.main(["--no-browser"]) == 0 and seen == {"folder": None, "workspace": None}
    mod.main(["--no-browser", "--session", str(tmp_path)])
    assert seen["folder"] == tmp_path.resolve()


# ── CLI ─────────────────────────────────────────────────────────────────────────────────────


def test_serve_without_a_bundle_means_the_workspace_home(tmp_path: Path, monkeypatch) -> None:
    from recruiter_copilot import cli, workspace_app

    args = build_parser().parse_args(["serve", "--workspace", str(tmp_path / "w")])
    assert args.session_dir is None and args.workspace_dir == str(tmp_path / "w")
    seen = {}
    monkeypatch.setattr(workspace_app, "serve_workspace", lambda s, root=None: seen.update(s=s))
    out = SimpleNamespace(lines=[], write=lambda s: out.lines.append(s), flush=lambda: None)
    rc = cli.main(
        ["serve", "--workspace", str(tmp_path / "w"), "--env-file", str(tmp_path / "none")], out=out
    )
    assert rc == 0 and seen["s"].workspace_dir == tmp_path / "w"
    assert "workspace home for" in "".join(out.lines)
