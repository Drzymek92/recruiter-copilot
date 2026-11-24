"""D14 (amended 11-23, #776 / D27 packaging): the shipped Docker files keep the loopback boundary
at the HOST edge — every published port is ``127.0.0.1:…``, the bundled Ollama publishes none,
the cockpit-children range is published with the same numbers inside and out, and the build
context is an allow-list (no candidate data, no secrets can enter the image)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml  # dev extra only (pyproject [dev]); the app never needs it

PROJECT = Path(__file__).resolve().parent.parent
COMPOSE_FILES = ("compose.yaml", "compose.gpu.yaml")


def _load(name: str) -> dict:
    return yaml.safe_load((PROJECT / name).read_text(encoding="utf-8"))


def _published(port: object) -> str:
    """Normalise one ``ports:`` entry to ``host_ip:published:target`` text."""
    if isinstance(port, dict):  # long syntax
        return f"{port.get('host_ip', '')}:{port.get('published', '')}:{port.get('target', '')}"
    return str(port)


def _range(text: str) -> range:
    first, _, last = text.partition("-")
    return range(int(first), int(last or first) + 1)


@pytest.mark.parametrize("name", COMPOSE_FILES)
def test_every_published_port_is_host_loopback(name: str) -> None:
    services = _load(name)["services"]
    for svc, spec in services.items():
        assert spec.get("network_mode") != "host", f"{name}:{svc} uses the host network"
        for port in spec.get("ports", []) or []:
            assert _published(port).startswith("127.0.0.1:"), f"{name}:{svc} publishes {port!r}"


@pytest.mark.parametrize("name", COMPOSE_FILES)
def test_bundled_ollama_publishes_nothing(name: str) -> None:
    ollama = _load(name)["services"].get("ollama", {})
    assert not ollama.get("ports"), f"{name}: the bundled Ollama must not be published (D14)"
    assert ollama.get("network_mode") is None


def test_cockpit_range_and_workspace_port_published_with_the_same_numbers() -> None:
    app = _load("compose.yaml")["services"]["app"]
    env = app["environment"]
    first, count = int(env["COCKPIT_PORT_FIRST"]), int(env["COCKPIT_PORT_COUNT"])
    published: dict[str, range] = {}
    for port in app["ports"]:
        _ip, host, container = _published(port).split(":")
        assert _range(host) == _range(container), f"{port!r}: host and container ports differ"
        published[host] = _range(host)
    ranges = list(published.values())
    assert range(int(env["PORT"]), int(env["PORT"]) + 1) in ranges
    assert range(first, first + count) in ranges, "the cockpit range is not published exactly"
    assert int(env["PORT"]) not in range(first, first + count)
    assert 8787 not in range(first, first + count)  # RESERVED_PORTS in workspace_app
    assert env["OLLAMA_BASE_URL"] == "http://ollama:11434/v1"


def test_gpu_override_reserves_the_gpu_for_both_services_and_adds_no_port() -> None:
    gpu = _load("compose.gpu.yaml")["services"]
    for svc in ("app", "ollama"):
        devices = gpu[svc]["deploy"]["resources"]["reservations"]["devices"]
        assert any(d["driver"] == "nvidia" and "gpu" in d["capabilities"] for d in devices)
        assert not gpu[svc].get("ports")
    assert gpu["app"]["build"]["target"] == "gpu"


def test_stop_grace_covers_serves_clean_stop_and_sessions_live_on_a_host_mount() -> None:
    app = _load("compose.yaml")["services"]["app"]
    m = re.fullmatch(r"(\d+)s", str(app["stop_grace_period"]))
    assert m and int(m.group(1)) >= 15  # serve: 5 s drain + 4 s backlog, children after that
    assert app.get("init") is True
    binds = [v for v in app["volumes"] if isinstance(v, dict) and v.get("type") == "bind"]
    assert [b["target"] for b in binds] == ["/workspace"]
    assert "sessions" not in str(binds[0]["source"])  # never the project's real sessions/


def test_image_sets_the_container_flag_and_compose_does_not() -> None:
    dockerfile = (PROJECT / "Dockerfile").read_text(encoding="utf-8")
    assert re.search(r"SECOND_CHAIR_IN_CONTAINER=1\b", dockerfile)
    assert re.search(r"RECRUITER_COPILOT_HOST=0\.0\.0\.0\b", dockerfile)
    assert re.search(r"^USER app$", dockerfile, re.M)  # non-root
    for name in COMPOSE_FILES:
        for svc, spec in _load(name)["services"].items():
            env = spec.get("environment") or {}
            assert "RECRUITER_COPILOT_HOST" not in env, f"{name}:{svc} overrides the bind host"


def test_build_context_is_an_allow_list() -> None:
    lines = [
        ln.strip()
        for ln in (PROJECT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    assert lines[0] == "*"
    assert {ln for ln in lines if ln.startswith("!")} == {"!pyproject.toml", "!README.md", "!src/"}
    for denied in ("**/.env", "**/sessions", "**/workspace"):
        assert denied in lines
