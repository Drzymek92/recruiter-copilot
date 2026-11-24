"""D28/D29 CLI round trip in tmp_path: workspace → job → template → interview → purge."""

from __future__ import annotations

import io
from pathlib import Path

from conftest import write_docx, write_scanned_pdf, write_text_pdf

from recruiter_copilot.cli import main
from recruiter_copilot.store import load_session
from recruiter_copilot.workspace import load_job

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run(*argv: str, root: Path, env_file: Path) -> tuple[int, str]:
    out = io.StringIO()
    rc = main([*argv, "--workspace", str(root), "--env-file", str(env_file)], out=out)
    return rc, out.getvalue()


def test_cli_round_trip(tmp_path: Path, capsys) -> None:
    root, env = tmp_path / "ws", tmp_path / "no.env"
    pdf = write_text_pdf(tmp_path / "offer.pdf", ["Backend Developer", "Must: Python"])
    docx = write_docx(tmp_path / "extra.docx", ["Nice: Kafka"])
    paste = tmp_path / "paste.txt"
    paste.write_text("Zespół w Łodzi", encoding="utf-8")
    cv = write_docx(tmp_path / "cv.docx", ["Synthetic CV — 5 lat Python"])

    assert _run("workspace", "init", root=root, env_file=env)[0] == 0
    rc, text = _run(
        "job",
        "new",
        "Backend Developer",
        "--offer",
        str(pdf),
        "--offer",
        str(docx),
        "--paste-file",
        str(paste),
        root=root,
        env_file=env,
    )
    assert rc == 0 and "created job backend-developer" in text and "offer/offer.pdf" in text
    assert "backend-developer\tBackend Developer" in _run("job", "list", root=root, env_file=env)[1]

    demo = str(PROJECT_ROOT / "examples" / "demo_session")  # read-only source
    assert _run("template", "save", "demo", "--from-session", demo, root=root, env_file=env)[0] == 0
    rc, text = _run("template", "apply", "demo", "backend-developer", root=root, env_file=env)
    assert rc == 0 and "6 question(s) added" in text
    assert len(load_job(root / "jobs" / "backend-developer").questions) == 6
    rc, text = _run("template", "save", "bd", "--job", "backend-developer", root=root, env_file=env)
    assert rc == 0
    assert "demo\t" in _run("template", "list", root=root, env_file=env)[1]
    assert "q1:" in _run("template", "show", "demo", root=root, env_file=env)[1]
    assert _run("template", "delete", "bd", root=root, env_file=env)[0] == 0

    rc, text = _run(
        "interview",
        "new",
        "backend-developer",
        "--candidate",
        "Łucja Żak",
        "--cv",
        str(cv),
        "--date",
        "2026-11-24",
        root=root,
        env_file=env,
    )
    assert rc == 0 and "6 question(s)" in text
    bundle = root / "jobs" / "backend-developer" / "interviews" / "2026-11-24_lucja-zak"
    session = load_session(bundle)
    assert "5 lat Python" in session.candidate.cv and session.template_ref == "demo"
    assert "Must: Python" in session.job.raw_text

    rc, text = _run("workspace", "list", root=root, env_file=env)
    assert rc == 0 and "2026-11-24_lucja-zak  [no transcript]" in text and "templates (1)" in text

    rc, text = _run("purge", str(bundle), "--yes", root=root, env_file=env)
    assert rc == 0 and "file(s) deleted" in text and not bundle.exists()
    assert (root / "jobs" / "backend-developer" / "job.json").is_file()
    assert (root / "jobs" / "backend-developer" / "offer" / "offer.pdf").is_file()
    assert (root / "templates" / "demo.json").is_file()


def test_cli_errors_are_clean(tmp_path: Path, capsys) -> None:
    root, env = tmp_path / "ws", tmp_path / "no.env"
    scan = write_scanned_pdf(tmp_path / "scan.pdf")
    rc, _ = _run("job", "new", "QA", "--offer", str(scan), root=root, env_file=env)
    assert rc == 1 and "paste the text instead" in capsys.readouterr().err
    assert not (root / "jobs" / "qa").exists()  # a refused doc leaves no job behind
    assert _run("interview", "new", "nope", "--candidate", "A", root=root, env_file=env)[0] == 1
    assert "no job" in capsys.readouterr().err
    assert _run("template", "show", "missing", root=root, env_file=env)[0] == 1
    rc, _ = _run("purge", str(tmp_path), "--yes", root=root, env_file=env)
    assert rc == 1 and "not a session bundle" in capsys.readouterr().err
    assert tmp_path.is_dir()


def test_env_workspace_used_without_flag(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "env-ws"
    monkeypatch.setenv("SECOND_CHAIR_WORKSPACE", str(root))
    out = io.StringIO()
    assert main(["workspace", "init", "--env-file", str(tmp_path / "none")], out=out) == 0
    assert (root / "jobs").is_dir() and str(root) in out.getvalue()
