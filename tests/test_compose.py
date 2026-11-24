"""D14 (amended 11-23, #776 / D27 packaging): the shipped Docker files keep the loopback boundary
at the HOST edge — every published port is ``127.0.0.1:…``, the bundled Ollama publishes none,
the cockpit-children range is published with the same numbers inside and out, and the build
context is an allow-list (no candidate data, no secrets can enter the image). #2045: every image
input is pinned — base and frontend by digest, the CPU Ollama by a sha256-checked release archive,
the GPU Ollama by digest — and both Ollama paths run the same version."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml  # dev extra only (pyproject [dev]); the app never needs it

PROJECT = Path(__file__).resolve().parent.parent
COMPOSE_FILES = ("compose.yaml", "compose.gpu.yaml")
OLLAMA_CPU_DOCKERFILE = PROJECT / "docker" / "ollama-cpu" / "Dockerfile"


class _ComposeLoader(yaml.SafeLoader):
    """``yaml.safe_load`` plus Compose's ``!reset`` merge tag (an override drops a key)."""


_ComposeLoader.add_constructor("!reset", lambda _loader, _node: None)


def _load(name: str) -> dict:
    text = (PROJECT / name).read_text(encoding="utf-8")
    return yaml.load(text, Loader=_ComposeLoader)  # a SafeLoader subclass


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


DIGEST_PINNED = re.compile(r"^[\w.\-/]+:[\w.\-]+@sha256:[0-9a-f]{64}$")


def _syntax_pinned(dockerfile: str) -> None:
    syntax = re.match(r"# syntax=(\S+)\n", dockerfile)  # the BuildKit frontend runs at build time
    assert syntax and DIGEST_PINNED.match(syntax.group(1)), "the # syntax= frontend is not pinned"


def test_app_base_image_and_frontend_are_pinned_by_digest() -> None:
    """#2045: a re-pushed tag must not change what gets built (tag kept for readability)."""
    dockerfile = (PROJECT / "Dockerfile").read_text(encoding="utf-8")
    _syntax_pinned(dockerfile)
    m = re.search(r"^ARG PYTHON_IMAGE=(\S+)$", dockerfile, re.M)
    assert m, "the Dockerfile lost its PYTHON_IMAGE default"
    assert DIGEST_PINNED.match(m.group(1)), f"base image not digest-pinned: {m.group(1)}"
    assert m.group(1).startswith("python:3.12-slim-bookworm@")  # the gpu LD_LIBRARY_PATH needs 3.12
    assert re.search(r"^FROM \$\{PYTHON_IMAGE\} AS base$", dockerfile, re.M)
    for source in re.findall(r"^FROM\s+(\S+)", dockerfile, re.M):
        assert source in {"${PYTHON_IMAGE}", "base"}, f"FROM {source} bypasses the pinned base"


def test_every_pulled_image_is_pinned_by_digest() -> None:
    for name in COMPOSE_FILES:
        for svc, spec in _load(name)["services"].items():
            if spec.get("build") is not None or "image" not in spec:
                continue  # built here (second-chair*), FROM pinned inputs — see the tests around
            image = spec["image"]
            assert DIGEST_PINNED.match(image), f"{name}:{svc} image not digest-pinned: {image}"


def test_cpu_ollama_is_a_checksum_pinned_release_and_gpu_runs_the_same_version() -> None:
    """#2045: the CPU Ollama is built from the official release archive, sha256-checked by BuildKit
    (a mismatch fails the build); the GPU file resets that build to the official image."""
    cpu = _load("compose.yaml")["services"]["ollama"]
    assert cpu["build"]["context"] == "docker/ollama-cpu"
    gpu = _load("compose.gpu.yaml")["services"]["ollama"]
    assert "build" in gpu and gpu["build"] is None, "compose.gpu.yaml must !reset the CPU build"
    gpu_version = re.fullmatch(r"ollama/ollama:([\d.]+)@sha256:[0-9a-f]{64}", gpu["image"])
    assert gpu_version, gpu["image"]

    dockerfile = OLLAMA_CPU_DOCKERFILE.read_text(encoding="utf-8")
    _syntax_pinned(dockerfile)
    args = dict(re.findall(r"^ARG (\w+)=(\S+)$", dockerfile, re.M))
    assert DIGEST_PINNED.match(args["BASE"]), "the Ollama base image is not digest-pinned"
    assert re.fullmatch(r"[0-9a-f]{64}", args["OLLAMA_SHA256"])
    assert args["OLLAMA_VERSION"] == gpu_version.group(1), "CPU and GPU Ollama versions differ"
    assert f":{args['OLLAMA_VERSION']}-cpu" in cpu["image"]
    assert re.search(
        r"^ADD --checksum=sha256:\$\{OLLAMA_SHA256\} \\\n\s+https://github\.com/ollama/ollama/"
        r"releases/download/v\$\{OLLAMA_VERSION\}/ollama-linux-amd64\.tar\.zst",
        dockerfile,
        re.M,
    ), "the archive is not fetched from the official release with ADD --checksum"
    assert "sha256sum -c" in dockerfile
    assert not re.search(r"curl[^\n]*\|\s*(ba)?sh", dockerfile), "never curl | sh"
    for source in re.findall(r"^FROM\s+(\S+)", dockerfile, re.M):
        assert source == "${BASE}", f"FROM {source} bypasses the pinned base"


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
