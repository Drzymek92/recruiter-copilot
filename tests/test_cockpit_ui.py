"""Cockpit page UI contract (#2022): resizable split, Settings dialog, guarded UI-prefs storage.

Static checks on the one hand-written page (D14: inline CSS/JS, no CDN, no webfont, no build step).
Behaviour (drag, keys, persistence across a reload) is verified in a browser; these tests pin the
markup and the invariants a browser pass cannot be relied on to re-check every run.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

PAGE = (
    Path(__file__).resolve().parent.parent / "src" / "recruiter_copilot" / "static" / "cockpit.html"
)


class _Collector(HTMLParser):
    """Every start tag as (tag, attrs-dict), in document order."""

    def __init__(self) -> None:
        super().__init__()
        self.tags: list[tuple[str, dict[str, str | None]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append((tag, dict(attrs)))


@pytest.fixture(scope="module")
def html() -> str:
    return PAGE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tags(html: str) -> list[tuple[str, dict[str, str | None]]]:
    parser = _Collector()
    parser.feed(html)
    return parser.tags


def _by_id(tags: list[tuple[str, dict[str, str | None]]], el_id: str) -> tuple[str, dict]:
    hits = [(t, a) for t, a in tags if a.get("id") == el_id]
    assert len(hits) == 1, f"expected exactly one #{el_id}, found {len(hits)}"
    return hits[0]


def _radio_values(tags: list[tuple[str, dict[str, str | None]]], name: str) -> list[str]:
    return [
        str(a.get("value"))
        for t, a in tags
        if t == "input" and a.get("type") == "radio" and a.get("name") == name
    ]


def _css(html: str) -> str:
    return "\n".join(re.findall(r"<style>(.*?)</style>", html, flags=re.S))


def test_split_separator_has_aria_and_keyboard_focus(tags) -> None:
    tag, a = _by_id(tags, "split")
    assert a.get("role") == "separator"
    assert a.get("aria-orientation") == "vertical"
    assert a.get("tabindex") == "0"  # keyboard-operable: ←/→, Home/End
    assert a.get("aria-label") or a.get("aria-labelledby")
    lo, hi, now = (
        float(a[k] or "nan") for k in ("aria-valuemin", "aria-valuemax", "aria-valuenow")
    )
    assert (lo, hi) == (25, 75) and lo <= now <= hi
    # it resizes a real panel, and sits between the two columns of <main>
    controlled = a.get("aria-controls")
    assert controlled and _by_id(tags, controlled)
    ids = [x.get("id") for _, x in tags]
    assert ids.index(controlled) < ids.index("split") < ids.index("live-panel")


def test_split_is_a_css_custom_property_hidden_on_narrow_screens(html: str) -> None:
    css = _css(html)
    assert "var(--split" in css  # one property drives grid-template-columns
    narrow = re.search(r"@media \(max-width: 800px\) \{(.*?)\n  \}", css, flags=re.S)
    assert narrow, "the one-column breakpoint at 800px must stay"
    assert "grid-template-columns: 1fr" in narrow.group(1)
    assert re.search(r"#split\s*\{[^}]*display:\s*none", narrow.group(1))


def test_settings_dialog_offers_fonts_themes_and_sizes(tags, html: str) -> None:
    tag, btn = _by_id(tags, "settings-btn")
    assert tag == "button" and btn.get("aria-controls") == "settings"
    dlg_tag, dlg = _by_id(tags, "settings")
    assert dlg_tag == "dialog" and dlg.get("aria-labelledby")
    assert _radio_values(tags, "font") == ["sans", "serif", "mono"]
    assert _radio_values(tags, "theme") == ["system", "light", "dark", "contrast"]
    assert _radio_values(tags, "size") == ["s", "m", "l", "xl"]
    assert _by_id(tags, "settings-reset")[0] == "button"
    # the gear is inline SVG with no namespace URL — the page must carry no http(s):// at all (D14)
    assert re.search(r'<button id="settings-btn"[^>]*>\s*<svg[^>]*aria-hidden="true"', html)
    assert "http://" not in html and "https://" not in html


def test_prefs_apply_before_first_paint_and_override_system_theme(html: str) -> None:
    head = html.split("</head>", 1)[0]
    assert "<script>" in head and "data-theme" in head  # no flash: applied in <head>
    css = _css(html)
    assert ':root[data-theme="dark"]' in css
    assert re.search(
        r'@media \(prefers-color-scheme: dark\) \{\s*:root:not\(\[data-theme="light"\]\)', css
    )
    for size in ("s", "l", "xl"):
        assert f':root[data-size="{size}"]' in css
    for font in ("serif", "mono"):
        assert f':root[data-font="{font}"]' in css


def test_font_sizes_scale_from_the_root(html: str) -> None:
    # text size works by changing one root value — so no font size may be pinned in px
    css = _css(html)
    pinned = re.findall(r"font(?:-size)?:\s*[\d.]+px", css)
    assert pinned == []
    assert "zoom:" not in css


def test_storage_is_ui_prefs_only_and_every_access_is_guarded(html: str) -> None:
    script = "\n".join(re.findall(r"<script>(.*?)</script>", html, flags=re.S))
    uses = [m.start() for m in re.finditer(r"localStorage", script)]
    assert uses, "UI prefs must persist (localStorage)"
    for pos in uses:
        before = script[:pos]
        opened = before.rfind("try {")
        assert opened != -1 and before.rfind("catch") < opened, (
            "localStorage outside a try block: " + script[pos - 60 : pos + 40]
        )
        assert script.find("catch", pos) != -1
    assert "sessionStorage" not in script and "indexedDB" not in script
    assert len(set(re.findall(r'"(recruiter-copilot\.[\w.-]+)"', script))) == 1  # one prefs key


def test_cockpit_title_and_header_say_second_chair_but_prefs_key_is_kept(html: str) -> None:
    """D27: user-visible name is Second Chair; the storage key keeps its old name (D14 prefs)."""
    assert re.search(r"<title>Second Chair cockpit</title>", html)
    assert '<span class="brand" id="brand">Second Chair</span>' in html
    assert '"recruiter-copilot.cockpit.prefs"' in html
    assert "recruiter-copilot analyse" not in html  # commands shown use the new name
