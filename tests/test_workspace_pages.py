"""Static contract of the workspace pages (D14 amended 11-24, D28/D29 UI) — sibling of
``test_cockpit_id_escaping.py`` / ``test_cockpit_ui.py`` for ``home.html``, ``job.html`` and the
shared ``workspace.js`` / ``workspace.css``.

No JS runtime here, so these pin the page source; behaviour is checked in a real browser
(``scripts/ui_smoke_workspace.py``). Escaping rule, stricter than the cockpit's id-only scan:
every expression concatenated into an attribute value (``'…="' + EXPR``) or into element text
(``'…>' + EXPR``) must be ``escapeHtml(…)``, a selector value ``CSS.escape(…)``, ready markup from a
helper/variable named ``…Html``, a local assigned from ``escapeHtml(…)``, or a ternary of literals.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parent.parent / "src" / "recruiter_copilot" / "static"
PAGES = ("home.html", "job.html")
JS = STATIC / "workspace.js"
CSS = STATIC / "workspace.css"
COCKPIT = STATIC / "cockpit.html"
PREFS_KEY = "recruiter-copilot.cockpit.prefs"

_LITERAL_THEN_PLUS = re.compile(r"'((?:[^'\\\n]|\\.)*)'\s*\+\s*")
_ESCAPED_LOCAL = re.compile(r"\bconst\s+(\w+)\s*=\s*escapeHtml\(")
_LITERAL_TERNARY = re.compile(
    r"""^\([^'"]*\?\s*(?:'[^'\n]*'|"[^"\n]*")\s*:\s*(?:'[^'\n]*'|"[^"\n]*")\s*\)$"""
)


def _scripts(name: str) -> str:
    html = (STATIC / name).read_text(encoding="utf-8")
    return "\n".join(re.findall(r"<script\b[^>]*>(.*?)</script>", html, flags=re.S))


@pytest.fixture(scope="module")
def sources() -> dict[str, str]:
    out = {name: _scripts(name) for name in PAGES}
    out["workspace.js"] = JS.read_text(encoding="utf-8")
    assert all(out.values()), "a workspace page lost its inline script"
    return out


def _expr_after(src: str, start: int) -> str:
    """The concatenated operand starting at ``start``: up to the next top-level ``+``/``,``/``;``."""
    depth = 0
    i = start
    quote = ""
    while i < len(src):
        c = src[i]
        if quote:
            if c == "\\":
                i += 2
                continue
            if c == quote:
                quote = ""
        elif c in "'\"`":
            quote = c
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            if depth == 0:
                break
            depth -= 1
        elif depth == 0 and c in "+,;\n":
            break
        i += 1
    return src[start:i].strip()


def splice_sites(src: str) -> list[tuple[str, str]]:
    """(context, expr) for every operand concatenated right after a literal that opens an
    attribute value (``…="…``) or sits in element text (``…>text``)."""
    sites: list[tuple[str, str]] = []
    for m in _LITERAL_THEN_PLUS.finditer(src):
        lit = m.group(1)
        if re.search(r'="[^"]*$', lit):
            ctx = "attr"
        elif ">" in lit and "<" not in lit[lit.rfind(">") :]:
            ctx = "text"
        else:
            continue
        sites.append((ctx, _expr_after(src, m.end())))
    return sites


def _top_level_ternary(expr: str) -> tuple[str, str] | None:
    """``(cond ? a : b)`` → (a, b) when the whole operand is one parenthesised ternary."""
    if not (expr.startswith("(") and expr.endswith(")")):
        return None
    body = expr[1:-1]
    depth, quote, q_at, c_at = 0, "", -1, -1
    i = 0
    while i < len(body):
        c = body[i]
        if quote:
            if c == "\\":
                i += 2
                continue
            if c == quote:
                quote = ""
        elif c in "'\"`":
            quote = c
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth < 0:
                return None  # "(a) + (b)" — not one parenthesised operand
        elif depth == 0 and c == "?" and q_at < 0:
            q_at = i
        elif depth == 0 and c == ":" and q_at >= 0:
            c_at = i
            break
        i += 1
    if q_at < 0 or c_at < 0:
        return None
    return body[q_at + 1 : c_at].strip(), body[c_at + 1 :].strip()


# ``list.map((x) => '<li>…' …).join("")`` — the literals inside are scanned on their own; the
# callback must build markup (start with a tag literal) or call an ``…Html`` helper.
_MARKUP_MAP = re.compile(
    r"""^[\w.$\[\]()]*?\.map\(\s*(?:\w+Html\)|\(?[\w, ]*\)?\s*=>\s*(?:'<|\w+Html\())"""
)


def _is_safe(expr: str, escaped_locals: set[str]) -> bool:
    if expr.startswith(("escapeHtml(", "CSS.escape(", "'", '"')):  # a literal is its own site
        return True
    if re.fullmatch(r"\w+Html(\(.*\))?", expr, flags=re.S):
        return True
    if expr in escaped_locals:
        return True
    if _LITERAL_TERNARY.match(expr) or _MARKUP_MAP.match(expr):
        return True
    branches = _top_level_ternary(expr)
    return branches is not None and all(_is_safe(b, escaped_locals) for b in branches)


def test_the_scanner_sees_the_splice_sites(sources) -> None:
    # Guard against a scanner that silently matches nothing.
    counts = {name: len(splice_sites(src)) for name, src in sources.items()}
    assert counts["job.html"] >= 80, counts
    assert counts["home.html"] >= 15, counts
    assert counts["workspace.js"] >= 4, counts


def test_the_scanner_flags_raw_values() -> None:
    src = (
        "x = '<b>' + name + '</b>' + '<a href=\"' + url + '\">' + '<i>' + escapeHtml(ok) + '</i>';"
        " y = '<p>' + (c ? '<u>' + escapeHtml(a) + '</u>' : bad) + '</p>';"
        " z = '<ul>' + xs.map((x) => x.name).join('') + '</ul>';"
    )
    flagged = [e for _ctx, e in splice_sites(src) if not _is_safe(e, set())]
    assert flagged == [
        "name",
        "url",
        "(c ? '<u>' + escapeHtml(a) + '</u>' : bad)",
        "xs.map((x) => x.name).join('')",
    ], flagged


def test_every_spliced_value_is_escaped(sources) -> None:
    bad: list[tuple[str, str, str]] = []
    for name, src in sources.items():
        escaped_locals = set(_ESCAPED_LOCAL.findall(src))
        for ctx, expr in splice_sites(src):
            if not _is_safe(expr, escaped_locals):
                bad.append((name, ctx, expr[:90]))
    assert not bad, f"unescaped values spliced into markup: {bad}"


def test_attribute_selectors_use_css_escape(sources) -> None:
    sel = re.compile(r"\[[\w-]+=\"'\s*\+\s*(?P<expr>[^+;]+?)\s*\+\s*'\"\]")
    found = [(name, m["expr"].strip()) for name, src in sources.items() for m in sel.finditer(src)]
    assert len(found) >= 8, found  # scanner live: editor refocus, row status, CV editor …
    raw = [f for f in found if not f[1].startswith("CSS.escape(")]
    assert not raw, f"attribute selector built without CSS.escape: {raw}"


def test_escape_helper_is_the_cockpits() -> None:
    js = JS.read_text(encoding="utf-8")
    cockpit = COCKPIT.read_text(encoding="utf-8")
    fn = re.compile(r"function escapeHtml\(s\) \{.*?\n\}", flags=re.S)
    assert fn.search(js) and fn.search(cockpit)
    assert fn.search(js).group(0) == fn.search(cockpit).group(0)


def _root_tokens(css: str, selector: str) -> dict[str, str]:
    block = re.search(re.escape(selector) + r"\s*\{(.*?)\}", css, flags=re.S)
    assert block, selector
    return {
        k: " ".join(v.split()) for k, v in re.findall(r"(--[\w-]+):\s*([^;]+);", block.group(1))
    }


def test_tokens_and_themes_are_the_cockpits() -> None:
    cockpit_css = "\n".join(
        re.findall(r"<style>(.*?)</style>", COCKPIT.read_text(encoding="utf-8"), flags=re.S)
    )
    ours = CSS.read_text(encoding="utf-8")
    for selector in (":root", ':root[data-theme="dark"]', ':root[data-theme="contrast"]'):
        theirs = _root_tokens(cockpit_css, selector)
        mine = _root_tokens(ours, selector)
        differ = {
            k: (v, mine.get(k)) for k, v in theirs.items() if k != "--gutter" and mine.get(k) != v
        }
        assert not differ, f"{selector}: tokens differ from the cockpit: {differ}"
    assert re.search(
        r'@media \(prefers-color-scheme: dark\) \{\s*:root:not\(\[data-theme="light"\]\)', ours
    )
    assert "prefers-reduced-motion: reduce" in ours
    assert ":focus-visible" in ours
    assert re.findall(r"font(?:-size)?:\s*[\d.]+px", ours) == []  # text scales from the root


@pytest.mark.parametrize("name", [*PAGES, "workspace.js", "workspace.css"])
def test_no_third_party_requests(name: str) -> None:
    text = (STATIC / name).read_text(encoding="utf-8")
    assert "http://" not in text and "https://" not in text  # D14: no CDN, no webfont
    assert "@import" not in text and "@font-face" not in text
    for src in re.findall(r'<(?:script|link)\b[^>]*(?:src|href)="([^"]+)"', text):
        assert src.startswith("/static/") or src == "data:,", src  # data:, = "no favicon"


def test_pages_load_the_shared_assets_in_head() -> None:
    for name in PAGES:
        head = (STATIC / name).read_text(encoding="utf-8").split("</head>", 1)[0]
        # in <head>, not deferred: the display prefs apply before first paint (no theme flash)
        assert '<script src="/static/workspace.js"></script>' in head
        assert '<link rel="stylesheet" href="/static/workspace.css">' in head
        assert "<title>Second Chair" in head  # D27


def test_storage_is_display_prefs_only_under_the_cockpits_key(sources) -> None:
    js = sources["workspace.js"]
    uses = [m.start() for m in re.finditer(r"localStorage", js)]
    assert uses
    for pos in uses:
        before = js[:pos]
        assert before.rfind("try {") > before.rfind("catch"), js[pos - 80 : pos + 40]
    assert set(re.findall(r'"(recruiter-copilot\.[\w.-]+)"', js)) == {PREFS_KEY}
    for name, src in sources.items():
        assert "sessionStorage" not in src and "indexedDB" not in src, name
        if name != "workspace.js":
            assert "localStorage" not in src, f"{name}: storage only through scPrefs"


def test_consent_box_is_never_pre_ticked() -> None:
    job = (STATIC / "job.html").read_text(encoding="utf-8")
    box = re.search(r'<input type="checkbox" id="iv-agreed"[^>]*>', job)
    assert box and "checked" not in box.group(0)
    # the per-row consent form's box is built without "checked" too
    assert re.search(r"<input type=\"checkbox\" data-c=\"agreed\">", job)


def test_reordering_is_keyboard_reachable_not_drag_only() -> None:
    job = (STATIC / "job.html").read_text(encoding="utf-8")
    assert "draggable" not in job
    for act in ("q-up", "q-down", "s-up", "s-down", "q-add", "q-remove", "req-add", "req-remove"):
        assert f'data-act="{act}"' in job, act
    assert 'data-field="section"' in job  # move a question between sections with a <select>
    drop = re.search(r'<div class="drop" id="drop"[^>]*>', job)
    assert drop and 'role="button"' in drop.group(0) and 'tabindex="0"' in drop.group(0)


def test_si1_line_has_a_place_before_generate_and_tailor() -> None:
    job = (STATIC / "job.html").read_text(encoding="utf-8")
    assert job.index('id="si1-gen"') < job.index('id="generate"')
    assert "JOB.disclosure.line" in job  # the interview row shows it next to "Tailor"
