"""Question ids are escaped wherever the cockpit page splices them into markup or selectors (#2024).

Ids come from the user's question-bank file, so an id holding ``"`` or ``<`` must not break the
markup the page builds by string concatenation. No JS runtime on this machine, so these are static
checks on the page source:

- an id-ish expression concatenated into an attribute value (``'…="' + q.id + '"'``) or into text
  between tags (``'>' + q.id + '<'``) must be wrapped in ``escapeHtml(…)``;
- an attribute selector built from an id (``'[data-qid="' + qid + '"]'``) must use ``CSS.escape(…)``
  (``escapeHtml`` is the wrong escaping there; ``getElementById`` would need neither).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PAGE = (
    Path(__file__).resolve().parent.parent / "src" / "recruiter_copilot" / "static" / "cockpit.html"
)

# An expression that carries a question id: q.id, p.question_id, qid, saved.qid, btn.dataset.qid.
_ID_EXPR = re.compile(r"(\.id\b|\bquestion_id\b|\bqid\b)")

# '<lead><attr>="<prefix>' + EXPR + '"…   (lead "[" = selector, whitespace = markup attribute)
_ATTR_CONCAT = re.compile(
    r"(?P<lead>[\[\s])(?P<attr>[\w-]+)=\"(?P<prefix>[^\"'\n]*)'\s*\+\s*(?P<expr>[^+;]+?)\s*\+\s*(?='\")"
)

# '…>' + EXPR + '<…   (an expression spliced in as element text)
_TEXT_CONCAT = re.compile(r">'\s*\+\s*(?P<expr>[^+;]+?)\s*\+\s*(?='<)")


@pytest.fixture(scope="module")
def script() -> str:
    html = PAGE.read_text(encoding="utf-8")
    blocks = re.findall(r"<script\b[^>]*>(.*?)</script>", html, flags=re.S)
    assert blocks, "cockpit.html has no inline <script>"
    return "\n".join(blocks)


def _id_sites(script: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """(markup sites, selector sites) as (attr, expr) pairs whose expr carries an id."""
    markup: list[tuple[str, str]] = []
    selectors: list[tuple[str, str]] = []
    for m in _ATTR_CONCAT.finditer(script):
        expr = m["expr"].strip()
        if not _ID_EXPR.search(expr):
            continue
        (selectors if m["lead"] == "[" else markup).append((m["attr"], expr))
    for m in _TEXT_CONCAT.finditer(script):
        expr = m["expr"].strip()
        if _ID_EXPR.search(expr):
            markup.append(("#text", expr))
    return markup, selectors


def test_the_scanner_sees_every_known_site(script: str) -> None:
    # Guard against a regex that silently matches nothing: data-qid x5 (state, save, pick, confirm,
    # dismiss), aria-label, label for, textarea id + data-notes-qid, the .qid badge text; and the
    # four selector consumers (pick-up-to refocus, two button-refocus branches, notes refocus).
    markup, selectors = _id_sites(script)
    assert len(markup) >= 10, markup
    assert len(selectors) >= 4, selectors


def test_ids_in_markup_go_through_escape_html(script: str) -> None:
    markup, _ = _id_sites(script)
    raw = [(attr, expr) for attr, expr in markup if not expr.startswith("escapeHtml(")]
    assert not raw, f"id spliced into markup without escapeHtml: {raw}"


def test_selectors_built_from_ids_use_css_escape(script: str) -> None:
    _, selectors = _id_sites(script)
    raw = [(attr, expr) for attr, expr in selectors if not expr.startswith("CSS.escape(")]
    assert not raw, f"attribute selector built from an id without CSS.escape: {raw}"


def _selector_sites(script: str) -> list[tuple[str, str]]:
    """Every attribute selector built by concatenation (``'[attr="' + EXPR + '"]'``), id or not."""
    return [
        (m["attr"], m["expr"].strip()) for m in _ATTR_CONCAT.finditer(script) if m["lead"] == "["
    ]


def test_every_concatenated_attribute_selector_uses_css_escape(script: str) -> None:
    # #2030: not only ids — any value spliced into a selector (restoreNotesFocus's saved.action /
    # saved.to come from the DOM's own data-* attributes) must be CSS-escaped, or a quote in it
    # throws from querySelector instead of matching.
    sites = _selector_sites(script)
    attrs = {attr for attr, _ in sites}
    assert {"data-action", "data-to", "data-qid", "data-notes-qid"} <= attrs, sites  # scanner live
    raw = [(attr, expr) for attr, expr in sites if not expr.startswith("CSS.escape(")]
    assert not raw, f"attribute selector built by concatenation without CSS.escape: {raw}"


def test_requirement_ids_are_escaped_in_the_meta_line(script: str) -> None:
    # requirement_ids also come from the bank file and land as element text.
    assert re.search(r"escapeHtml\(\s*q\.requirement_ids\.join\(", script)
    assert not re.search(r"\+\s*q\.requirement_ids\.join\(", script)
