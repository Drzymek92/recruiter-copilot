"""Render a :class:`~recruiter_copilot.models.Report` to markdown and self-contained HTML (D17, D18).

The report is the interviewer's deliverable and the project's honesty surface:

- **D17 — evidence-cited or "not evidenced".** Every finding renders its verbatim quote(s) with
  timestamps, its score and confidence; every contradiction renders BOTH sides and names the source
  document; a finding the verifier stripped bare is rendered *as* ``not evidenced``, never hidden.
- **D18 / SI1 — disclosure + consent.** The report states which profile ran (``local``/``api``),
  whether any data left the machine, and the consent record (who informed whom, when, how).
- **D14 / SI1 — self-contained HTML.** The HTML makes NO third-party request: inline CSS only, no
  CDN, no webfont, no external ``src``/``href``. ``tests/test_report.py`` asserts this.

Rendering is **pure** (MOD): ``Report`` in, strings out — no file I/O, no clock, no settings read.
The CLI writes the strings to disk (``output_paths`` names the two files with one shared timestamp),
into the session folder so ``recruiter-copilot purge`` removes them (D18). ``build_report`` is the
thin, still-pure assembly seam that turns an ``AnalysisResult`` + ``Settings`` + ``Session`` into a
``Report``; only it reads ``Settings`` (to fill the disclosure), and it takes no I/O either.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from .models import (
    Profile,
    ProviderDisclosure,
    Report,
    Session,
)

if TYPE_CHECKING:  # avoid a runtime import cycle; analysis imports nothing from here
    from .analysis import AnalysisResult
    from .config import Settings


# ── assembly (pure; the only place Settings is read) ─────────────────────────────────────────


def disclosure_from_settings(settings: Settings) -> ProviderDisclosure:
    """SI1: capture exactly what processed the data, so the report can state the egress honestly."""
    if settings.profile is Profile.API:
        chat_model = (
            settings.anthropic_model if settings.anthropic_api_key else settings.openai_chat_model
        )
        stt_model = settings.openai_stt_model
    else:
        chat_model = settings.ollama_model
        stt_model = settings.stt_model
    return ProviderDisclosure(
        profile=settings.profile,
        stt_provider=settings.stt_provider_name(),
        chat_provider=settings.chat_provider_name(),
        stt_model=stt_model,
        chat_model=chat_model,
        data_left_machine=settings.data_leaves_machine(),
    )


def build_report(
    session: Session,
    result: AnalysisResult,
    settings: Settings,
    *,
    generated_at: str | None = None,
) -> Report:
    """Assemble a ``Report`` from an ``AnalysisResult`` (disclosure from settings, consent from the
    session). Pure: no file I/O. ``generated_at`` defaults to now (UTC, ISO-8601 to the second)."""
    when = generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    return Report(
        session_id=session.id,
        generated_at=when,
        disclosure=disclosure_from_settings(settings),
        consent=session.consent,
        analyses=result.analyses,
        fit=result.fit,
        language=result.language,
    )


def output_paths(folder: Path, when: datetime | None = None) -> dict[str, Path]:
    """The two report paths for one run, sharing a single ``<YYYYMMDD>_<HHMMSS>`` timestamp
    (house output-naming). Pure — builds paths, touches no disk."""
    when = when or datetime.now()
    stamp = when.strftime("%Y%m%d_%H%M%S")
    base = f"report_{stamp}"
    return {"md": folder / f"{base}.md", "html": folder / f"{base}.html"}


# ── shared helpers ───────────────────────────────────────────────────────────────────────────


def _yn(value: bool) -> str:
    return "yes" if value else "no"


def _span(ev) -> str:
    return f"[{ev.t_start:.2f}–{ev.t_end:.2f}s]"


def _egress_line(d: ProviderDisclosure) -> str:
    if d.data_left_machine:
        return (
            f"yes — the transcript and candidate documents were sent to "
            f"{d.chat_provider} (chat) / {d.stt_provider} (transcription)"
        )
    return "no — all processing stayed on this machine"


# ── markdown ─────────────────────────────────────────────────────────────────────────────────


def render_markdown(report: Report) -> str:
    d = report.disclosure
    c = report.consent
    lines: list[str] = [
        f"# Interview analysis — {report.session_id}",
        "",
        f"_Generated {report.generated_at}. Decision support only — not an automated "
        "hiring decision (SI2); every finding is editable before use._",
        "",
        "## Disclosure (D18 / SI1)",
        "",
        f"- Runtime profile: **{d.profile.value}**",
        f"- Data left this machine: **{_egress_line(d)}**",
        f"- Transcription: {d.stt_provider}" + (f" ({d.stt_model})" if d.stt_model else ""),
        f"- Analysis model: {d.chat_provider}" + (f" ({d.chat_model})" if d.chat_model else ""),
        "",
        "## Consent (SI3)",
        "",
        f"- Candidate agreed to recording: **{_yn(c.candidate_agreed)}**",
        f"- Informed by: {c.informed_by or '—'}",
        f"- Informed at: {c.informed_at or '—'}",
        f"- Method: {c.method or '—'}",
    ]
    if not c.is_complete:
        lines.append(
            "- ⚠ Consent record is INCOMPLETE — recording must not begin until it is "
            "filled (D18)."
        )
    lines += ["", "## Fit assessment", ""]
    lines += _fit_md(report)
    lines += ["", "## Per-question findings", ""]
    for qa in report.analyses:
        lines += _question_md(qa)
    if report.language is not None:
        lines += ["", "## Language assessment (indicative)", ""]
        lines += _language_md(report.language)
    if report.interviewer_summary:
        lines += ["", "## Interviewer summary", "", report.interviewer_summary]
    return "\n".join(lines).rstrip() + "\n"


def _fit_md(report: Report) -> list[str]:
    fit = report.fit
    score = "not computed" if fit.weighted_score is None else f"{fit.weighted_score:.2f} / 1.00"
    out = [f"- Weighted requirement score: **{score}**"]
    if fit.summary:
        out.append(f"- {fit.summary}")
    if fit.coverage:
        out += ["", "| Requirement | Coverage |", "| --- | --- |"]
        out += [f"| {rc.requirement_id} | {rc.coverage.value} |" for rc in fit.coverage]
    return out


def _question_md(qa) -> list[str]:
    if qa.not_evidenced:
        head = f"### {qa.question_id} — NOT EVIDENCED"
    else:
        score = "n/a" if qa.score is None else f"{qa.score}/5"
        head = f"### {qa.question_id} — score {score} (confidence {qa.confidence:.2f})"
    out = [head, ""]
    if qa.interviewer_edit:
        out += [f"> Interviewer edit: {qa.interviewer_edit}", ""]
    if qa.summary:
        out += [qa.summary, ""]
    if qa.not_evidenced:
        out += [
            "_No transcript evidence survived grounding verification for this question " "(D17)._",
            "",
        ]
    if qa.evidence:
        out.append("**Evidence:**")
        out += [f'- "{ev.quote}" {_span(ev)}' for ev in qa.evidence]
        out.append("")
    for con in qa.contradictions:
        out += [
            f"**Contradiction ({con.severity.value}):** {con.claim}",
            f'- Transcript: "{con.transcript_quote.quote}" {_span(con.transcript_quote)}',
            f'- {con.source_doc}: "{con.source_quote}"',
        ]
        if con.explanation:
            out.append(f"- {con.explanation}")
        out.append("")
    return out


def _language_md(language) -> list[str]:
    cefr = language.cefr_estimate or "—"
    out = [
        f"- Language: {language.language}",
        f"- CEFR estimate: **{cefr}**",
    ]
    for obs in language.observations:
        out.append(f"- {obs}")
    if language.evidence:
        out += ["", "**Evidence:**"]
        out += [f'- "{ev.quote}" {_span(ev)}' for ev in language.evidence]
    out += ["", f"_{language.disclaimer}_"]
    return out


# ── HTML (self-contained: inline CSS, no external request — D14 / SI1) ──────────────────────

# CSS uses only /* */ comments and carries no url() / import, so the page pulls nothing.
_STYLE = """
:root { color-scheme: light dark; }
body { font: 15px/1.5 system-ui, sans-serif; max-width: 52rem; margin: 2rem auto;
       padding: 0 1rem; color: #1a1a1a; background: #fff; }
h1 { font-size: 1.6rem; } h2 { font-size: 1.2rem; margin-top: 2rem;
     border-bottom: 1px solid #ddd; padding-bottom: .2rem; }
h3 { font-size: 1.02rem; margin-bottom: .2rem; }
.meta { color: #555; font-size: .9rem; }
.egress-yes { color: #a0410a; font-weight: 600; }
.egress-no { color: #1a6f2b; font-weight: 600; }
.warn { color: #a0410a; font-weight: 600; }
blockquote { margin: .3rem 0 .3rem .5rem; padding-left: .7rem;
             border-left: 3px solid #bbb; color: #333; }
.ts { color: #777; font-size: .82rem; }
.notev { color: #777; font-style: italic; }
.sev-high { color: #a0410a; font-weight: 600; }
.sev-medium { color: #9a6a00; font-weight: 600; }
.sev-low { color: #555; font-weight: 600; }
table { border-collapse: collapse; margin: .5rem 0; }
th, td { border: 1px solid #ccc; padding: .25rem .6rem; text-align: left; }
.disclaimer { color: #555; font-size: .88rem; font-style: italic; }
section { margin-bottom: .5rem; }
"""


def _esc(text: str) -> str:
    return html.escape(text or "")


def _span_html(ev) -> str:
    return f'<span class="ts">[{ev.t_start:.2f}–{ev.t_end:.2f}s]</span>'


def render_html(report: Report) -> str:
    d = report.disclosure
    c = report.consent
    body: list[str] = [
        f"<h1>Interview analysis — {_esc(report.session_id)}</h1>",
        f'<p class="meta">Generated {_esc(report.generated_at)}. Decision support only '
        "— not an automated hiring decision (SI2); every finding is editable before use.</p>",
        "<h2>Disclosure (D18 / SI1)</h2>",
        "<ul>",
        f"<li>Runtime profile: <strong>{_esc(d.profile.value)}</strong></li>",
        f'<li>Data left this machine: <span class="{"egress-yes" if d.data_left_machine else "egress-no"}">'
        f"{_esc(_egress_line(d))}</span></li>",
        f"<li>Transcription: {_esc(d.stt_provider)}"
        + (f" ({_esc(d.stt_model)})" if d.stt_model else "")
        + "</li>",
        f"<li>Analysis model: {_esc(d.chat_provider)}"
        + (f" ({_esc(d.chat_model)})" if d.chat_model else "")
        + "</li>",
        "</ul>",
        "<h2>Consent (SI3)</h2>",
        "<ul>",
        f"<li>Candidate agreed to recording: <strong>{_yn(c.candidate_agreed)}</strong></li>",
        f"<li>Informed by: {_esc(c.informed_by) or '—'}</li>",
        f"<li>Informed at: {_esc(c.informed_at) or '—'}</li>",
        f"<li>Method: {_esc(c.method) or '—'}</li>",
        "</ul>",
    ]
    if not c.is_complete:
        body.append(
            '<p class="warn">⚠ Consent record is INCOMPLETE — recording must not begin '
            "until it is filled (D18).</p>"
        )
    body += ["<h2>Fit assessment</h2>", *_fit_html(report)]
    body.append("<h2>Per-question findings</h2>")
    for qa in report.analyses:
        body += _question_html(qa)
    if report.language is not None:
        body += ["<h2>Language assessment (indicative)</h2>", *_language_html(report.language)]
    if report.interviewer_summary:
        body += [
            "<h2>Interviewer summary</h2>",
            f"<p>{_esc(report.interviewer_summary)}</p>",
        ]
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>Interview analysis — {_esc(report.session_id)}</title>\n"
        f"<style>{_STYLE}</style>\n</head>\n<body>\n" + "\n".join(body) + "\n</body>\n</html>\n"
    )


def _fit_html(report: Report) -> list[str]:
    fit = report.fit
    score = "not computed" if fit.weighted_score is None else f"{fit.weighted_score:.2f} / 1.00"
    out = [f"<p>Weighted requirement score: <strong>{_esc(score)}</strong></p>"]
    if fit.summary:
        out.append(f"<p>{_esc(fit.summary)}</p>")
    if fit.coverage:
        out.append("<table><thead><tr><th>Requirement</th><th>Coverage</th></tr></thead><tbody>")
        out += [
            f"<tr><td>{_esc(rc.requirement_id)}</td><td>{_esc(rc.coverage.value)}</td></tr>"
            for rc in fit.coverage
        ]
        out.append("</tbody></table>")
    return out


def _question_html(qa) -> list[str]:
    if qa.not_evidenced:
        head = f"<h3>{_esc(qa.question_id)} — NOT EVIDENCED</h3>"
    else:
        score = "n/a" if qa.score is None else f"{qa.score}/5"
        head = (
            f"<h3>{_esc(qa.question_id)} — score {_esc(score)} "
            f"(confidence {qa.confidence:.2f})</h3>"
        )
    out = ["<section>", head]
    if qa.interviewer_edit:
        out.append(f"<p><em>Interviewer edit:</em> {_esc(qa.interviewer_edit)}</p>")
    if qa.summary:
        out.append(f"<p>{_esc(qa.summary)}</p>")
    if qa.not_evidenced:
        out.append(
            '<p class="notev">No transcript evidence survived grounding verification for this '
            "question (D17).</p>"
        )
    if qa.evidence:
        out.append("<p><strong>Evidence:</strong></p>")
        out += [
            f"<blockquote>“{_esc(ev.quote)}” {_span_html(ev)}</blockquote>" for ev in qa.evidence
        ]
    for con in qa.contradictions:
        out.append(
            f'<p><strong>Contradiction (<span class="sev-{con.severity.value}">'
            f"{_esc(con.severity.value)}</span>):</strong> {_esc(con.claim)}</p>"
        )
        out.append(
            f"<blockquote>Transcript: “{_esc(con.transcript_quote.quote)}” "
            f"{_span_html(con.transcript_quote)}</blockquote>"
        )
        out.append(f"<blockquote>{_esc(con.source_doc)}: “{_esc(con.source_quote)}”</blockquote>")
        if con.explanation:
            out.append(f"<p>{_esc(con.explanation)}</p>")
    out.append("</section>")
    return out


def _language_html(language) -> list[str]:
    cefr = language.cefr_estimate or "—"
    out = [
        "<ul>",
        f"<li>Language: {_esc(language.language)}</li>",
        f"<li>CEFR estimate: <strong>{_esc(cefr)}</strong></li>",
    ]
    out += [f"<li>{_esc(obs)}</li>" for obs in language.observations]
    out.append("</ul>")
    if language.evidence:
        out.append("<p><strong>Evidence:</strong></p>")
        out += [
            f"<blockquote>“{_esc(ev.quote)}” {_span_html(ev)}</blockquote>"
            for ev in language.evidence
        ]
    out.append(f'<p class="disclaimer">{_esc(language.disclaimer)}</p>')
    return out
