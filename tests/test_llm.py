"""The chat seam (D12): profile switching, key gates, and the Ollama truncation guard."""

from __future__ import annotations

import pytest

from recruiter_copilot.config import Settings
from recruiter_copilot.llm import (
    AnthropicChatProvider,
    ChatError,
    ChatReply,
    OllamaProvider,
    OpenAIChatProvider,
    build_chat_provider,
)
from recruiter_copilot.models import Profile


def test_truncation_is_detected_from_the_reported_token_count() -> None:
    """The measured trap: Ollama truncates silently and reports the TRUNCATED prompt_tokens."""
    honest = ChatReply("ok", "m", "p", input_tokens=9000, sent_chars=36000)
    assert not honest.truncated  # 4 chars/token — plausible
    silent_loss = ChatReply("ok", "m", "p", input_tokens=2050, sent_chars=37800)
    assert silent_loss.truncated  # 18 chars/token — the backend cannot have seen it all


def test_truncation_flag_is_off_without_usage_data() -> None:
    assert not ChatReply("ok", "m", "p").truncated
    assert not ChatReply("ok", "m", "p", input_tokens=100).truncated


# ── #2014: context-based truncation signal (real prompts run ~3.4 chars/token, not > 7) ────────


def test_prompt_counted_at_the_context_window_is_flagged_by_the_context_signal() -> None:
    """A silently-cut prompt reports prompt_tokens at the loaded window; at a realistic 3.4
    chars/token the chars/token signal alone (> 7) never fires, so this is the case it missed."""
    cut = ChatReply(
        "ok", "m", "p", input_tokens=4096, sent_chars=int(4096 * 3.4), context_window=4096
    )
    assert cut.truncated
    assert cut.truncation_signals == ("context",)
    just_under = ChatReply("ok", "m", "p", input_tokens=4000, sent_chars=13600, context_window=4096)
    assert just_under.truncation_signals == ("context",)


def test_normal_prompt_well_inside_the_window_is_not_flagged() -> None:
    """The measured real-call peak (#2006): 2714 prompt tokens at ~3.4 chars/token in a 4096 window."""
    normal = ChatReply(
        "ok", "m", "p", input_tokens=2714, sent_chars=int(2714 * 3.4), context_window=4096
    )
    assert not normal.truncated
    assert normal.truncation_signals == ()


def test_chars_per_token_signal_still_fires_and_both_can_fire() -> None:
    far_below = ChatReply("ok", "m", "p", input_tokens=2050, sent_chars=37800, context_window=4096)
    assert far_below.truncation_signals == ("chars_per_token",)
    both = ChatReply("ok", "m", "p", input_tokens=4096, sent_chars=40000, context_window=4096)
    assert both.truncation_signals == ("context", "chars_per_token")


def test_without_a_context_window_only_the_chars_per_token_signal_applies() -> None:
    """context_window=0 (the default; the api providers) keeps today's chars/token-only check."""
    at_limit = ChatReply("ok", "m", "p", input_tokens=4096, sent_chars=int(4096 * 3.4))
    assert at_limit.context_window == 0
    assert not at_limit.truncated


def _fake_chat_openai(monkeypatch, input_tokens: int) -> None:
    """Replace langchain's ChatOpenAI with an offline fake reporting ``input_tokens``."""
    import langchain_openai

    class _Msg:
        content = "{}"
        usage_metadata = {"input_tokens": input_tokens, "output_tokens": 1}

    class _Fake:
        def __init__(self, **_kw: object) -> None:
            pass

        def invoke(self, _messages: object) -> _Msg:
            return _Msg()

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", _Fake)


def test_local_provider_flags_a_cut_prompt_and_names_the_context_signal(
    monkeypatch, caplog
) -> None:
    _fake_chat_openai(monkeypatch, input_tokens=4096)
    provider = OllamaProvider(Settings())  # OLLAMA_CONTEXT_TOKENS default: 4096 (measured load)
    with caplog.at_level("WARNING", logger="recruiter_copilot.llm"):
        reply = provider.complete("x" * int(4096 * 3.4))
    assert reply.context_window == 4096
    assert reply.truncated and reply.truncation_signals == ("context",)
    (msg,) = [r.getMessage() for r in caplog.records if "PROMPT TRUNCATED" in r.getMessage()]
    assert "context signal" in msg and "4096-token context window" in msg
    assert "OLLAMA_CONTEXT_TOKENS" in msg


def test_local_provider_context_window_follows_the_setting(monkeypatch) -> None:
    _fake_chat_openai(monkeypatch, input_tokens=4096)
    reply = OllamaProvider(Settings(ollama_context_tokens=8192)).complete("x" * int(4096 * 3.4))
    assert reply.context_window == 8192 and not reply.truncated
    off = OllamaProvider(Settings(ollama_context_tokens=0)).complete("x" * int(4096 * 3.4))
    assert off.context_window == 0 and not off.truncated  # 0 = context signal disabled


def test_api_provider_keeps_the_chars_per_token_check_only(monkeypatch, caplog) -> None:
    """Hosted endpoints error on overflow instead of cutting silently: no context signal there."""
    _fake_chat_openai(monkeypatch, input_tokens=4096)
    provider = OpenAIChatProvider(Settings(profile=Profile.API, openai_api_key="k"))
    with caplog.at_level("WARNING", logger="recruiter_copilot.llm"):
        reply = provider.complete("x" * int(4096 * 3.4))
    assert reply.context_window == 0 and not reply.truncated
    assert not [r for r in caplog.records if "PROMPT TRUNCATED" in r.getMessage()]
    _fake_chat_openai(monkeypatch, input_tokens=2050)
    with caplog.at_level("WARNING", logger="recruiter_copilot.llm"):
        lossy = provider.complete("x" * 37800)
    assert lossy.truncation_signals == ("chars_per_token",)
    (msg,) = [r.getMessage() for r in caplog.records if "PROMPT TRUNCATED" in r.getMessage()]
    assert "chars/token signal" in msg


def test_total_tokens() -> None:
    assert ChatReply("x", "m", "p", input_tokens=10, output_tokens=5).total_tokens == 15


def test_local_profile_selects_ollama_and_needs_no_key() -> None:
    provider = build_chat_provider(Settings())
    assert isinstance(provider, OllamaProvider)
    assert provider.name.startswith("ollama:")


def test_api_profile_prefers_anthropic_when_its_key_is_set() -> None:
    s = Settings(profile=Profile.API, anthropic_api_key="k", openai_api_key="k2")
    assert isinstance(build_chat_provider(s), AnthropicChatProvider)


def test_api_profile_falls_back_to_the_openai_compatible_endpoint() -> None:
    s = Settings(profile=Profile.API, openai_api_key="k")
    provider = build_chat_provider(s)
    assert isinstance(provider, OpenAIChatProvider)
    assert provider.base_url == s.openai_base_url


def test_api_providers_refuse_to_construct_without_a_key() -> None:
    with pytest.raises(ChatError, match="OPENAI_API_KEY"):
        OpenAIChatProvider(Settings(profile=Profile.API))
    with pytest.raises(ChatError, match="ANTHROPIC_API_KEY"):
        AnthropicChatProvider(Settings(profile=Profile.API))


def test_ollama_uses_the_configured_placeholder_token_not_a_real_credential() -> None:
    provider = OllamaProvider(Settings())
    assert provider.token == "local-no-auth"
    assert provider.base_url.startswith("http://localhost")


def test_an_unreachable_backend_raises_one_typed_error() -> None:
    s = Settings(ollama_base_url="http://127.0.0.1:1/v1", ollama_model="nope")
    with pytest.raises(ChatError, match="unreachable"):
        OllamaProvider(s).complete("hello")


# ── #2006: a fresh prompt cache per analysis batch ──────────────────────────────────────────


class _FakeOllama:
    """Stands in for Ollama's native API: /api/ps lists loaded models, keep_alive=0 unloads."""

    def __init__(self, loaded: list[str]) -> None:
        self.loaded = list(loaded)
        self.calls: list[tuple[str, dict | None]] = []

    def __call__(self, url: str, body: dict | None = None, timeout: float = 10.0) -> dict:
        del timeout
        self.calls.append((url, body))
        if url.endswith("/api/ps"):
            return {"models": [{"name": m, "model": m} for m in self.loaded]}
        if url.endswith("/api/generate") and body and body.get("keep_alive") == 0:
            self.loaded = [m for m in self.loaded if m != body["model"]]
            return {"done": True, "done_reason": "unload"}
        raise AssertionError(f"unexpected call {url} {body}")


def test_reset_prompt_cache_unloads_the_loaded_model_through_the_native_api(monkeypatch) -> None:
    """A warm model re-uses its prompt cache, which measurably changes temp-0 output (#2006)."""
    from recruiter_copilot import llm

    fake = _FakeOllama(["qwen3:14b", "other:8b"])
    monkeypatch.setattr(llm, "_ollama_json", fake)
    provider = OllamaProvider(Settings(ollama_model="qwen3:14b"))
    assert provider.reset_prompt_cache() is True
    unloads = [b for u, b in fake.calls if u.endswith("/api/generate")]
    assert unloads == [{"model": "qwen3:14b", "keep_alive": 0}]
    assert all(u.startswith("http://localhost:11434/api/") for u, _ in fake.calls)  # /v1 dropped
    assert fake.loaded == ["other:8b"]  # only our model is touched


def test_reset_prompt_cache_is_a_no_op_when_the_model_is_not_loaded(monkeypatch) -> None:
    from recruiter_copilot import llm

    fake = _FakeOllama(["other:8b"])
    monkeypatch.setattr(llm, "_ollama_json", fake)
    assert OllamaProvider(Settings(ollama_model="qwen3")).reset_prompt_cache() is True
    assert not [u for u, _ in fake.calls if u.endswith("/api/generate")]


def test_reset_prompt_cache_matches_an_untagged_model_as_latest(monkeypatch) -> None:
    from recruiter_copilot import llm

    fake = _FakeOllama(["qwen3:latest"])
    monkeypatch.setattr(llm, "_ollama_json", fake)
    assert OllamaProvider(Settings(ollama_model="qwen3")).reset_prompt_cache() is True
    assert fake.loaded == []


# ── #2012: opt-in thinking control (reasoning_effort) ──────────────────────────────────────────


def _capture_wire_body(monkeypatch, settings_kw: dict, api: bool = False, send=None) -> dict:
    """Send one real ``complete`` through langchain to a loopback fake OpenAI server; return the
    JSON body it received. Offline: no Ollama, no GPU — this is the exact request serialisation.
    ``api=True`` sends it through ``OpenAIChatProvider`` (the hosted path) instead of Ollama.
    ``send(provider)`` (#2035) replaces the default ``complete`` call, e.g. with a matcher call."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    for var in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.delenv(var, raising=False)
    captured: list[bytes] = []
    reply = json.dumps(
        {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "qwen3:14b",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "{}"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        }
    ).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            captured.append(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args: object) -> None:
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    try:
        if api:
            settings = Settings(
                openai_base_url=url, openai_api_key="test-key", openai_chat_model="qwen3:14b"
            )
            provider: object = OpenAIChatProvider(settings)
        else:
            settings = Settings(ollama_base_url=url, ollama_model="qwen3:14b", **settings_kw)
            provider = OllamaProvider(settings)
        if send is not None:
            send(provider)
        else:
            provider.complete("PROMPT", system="SYSTEM", max_tokens=900)  # type: ignore[attr-defined]
    finally:
        srv.shutdown()
    return json.loads(captured[0])


# The default request as sent BEFORE #2012 (captured from the pre-change code, langchain_openai 1.3.5).
_PRE_2012_BODY = {
    "messages": [{"content": "SYSTEM", "role": "system"}, {"content": "PROMPT", "role": "user"}],
    "model": "qwen3:14b",
    "max_completion_tokens": 900,
    "stream": False,
    "temperature": 0.0,
}


# #2019: the local default body is the pre-#2012 body MINUS the token cap. langchain sends
# `max_tokens` as `max_completion_tokens`, which Ollama 0.40.2 ignores (outputs reached 1957
# tokens under a "900" cap), so the local request no longer claims a cap it does not have. That key
# is the ONLY difference — nothing else in the local request changed.
_LOCAL_DEFAULT_BODY = {k: v for k, v in _PRE_2012_BODY.items() if k != "max_completion_tokens"}


def test_default_local_request_drops_only_the_ignored_cap_and_has_no_reasoning_effort(
    monkeypatch,
) -> None:
    body = _capture_wire_body(monkeypatch, {})
    assert body == _LOCAL_DEFAULT_BODY
    assert set(_PRE_2012_BODY) - set(body) == {"max_completion_tokens"}
    assert "max_tokens" not in body  # no total cap either: it would cut off the JSON (#2012)


def test_reasoning_effort_reaches_the_wire_only_when_set(monkeypatch) -> None:
    body = _capture_wire_body(monkeypatch, {"ollama_reasoning_effort": "none"})
    assert body == {**_LOCAL_DEFAULT_BODY, "reasoning_effort": "none"}


def test_client_kwargs_add_reasoning_effort_only_when_configured() -> None:
    assert OllamaProvider(Settings()).client_kwargs(900) == {}  # #2019: cap not sent locally
    assert OllamaProvider(Settings()).client_kwargs() == {}
    off = OllamaProvider(Settings(ollama_reasoning_effort="none"))
    assert off.client_kwargs(900) == {"reasoning_effort": "none"}


# ── #2019: the api profile keeps sending its cap exactly as before ─────────────────────────────


def test_openai_api_provider_still_sends_the_cap(monkeypatch) -> None:
    s = Settings(profile=Profile.API, openai_api_key="k")
    assert OpenAIChatProvider(s).client_kwargs(900) == {"max_tokens": 900}
    assert OpenAIChatProvider(s).client_kwargs() == {}
    body = _capture_wire_body(monkeypatch, {}, api=True)
    assert body == _PRE_2012_BODY  # hosted OpenAI honours max_completion_tokens: unchanged


# ── #2035: the matcher asks Ollama with thinking off; the api matcher request is unchanged ──────


def _ask_matcher(provider: object) -> None:
    from recruiter_copilot.matcher import QuestionIndex, propose_for_line
    from recruiter_copilot.models import Question, Speaker, TranscriptLine

    index = QuestionIndex.build([Question("q1", {"en": "Tell me about your last project."})])
    line = TranscriptLine(0.0, 3.0, "Tell me about your last project?", Speaker.INTERVIEWER, "en")
    propose_for_line(line, 0, index, "llm", 0.45, chat=provider)


def test_without_thinking_is_a_think_off_copy_and_leaves_the_provider_alone() -> None:
    base = OllamaProvider(Settings(ollama_reasoning_effort="high"))
    quick = base.without_thinking()
    assert quick.client_kwargs(16) == {"reasoning_effort": "none"}  # still no cap locally
    assert base.client_kwargs(16) == {"reasoning_effort": "high"}  # analysis calls untouched
    assert not hasattr(OpenAIChatProvider, "without_thinking")
    assert not hasattr(AnthropicChatProvider, "without_thinking")


def test_local_matcher_request_turns_thinking_off(monkeypatch) -> None:
    body = _capture_wire_body(monkeypatch, {}, send=_ask_matcher)
    assert body["reasoning_effort"] == "none"
    assert "max_completion_tokens" not in body and "max_tokens" not in body  # #2019 unchanged
    assert {k for k in body if k != "reasoning_effort"} == set(_LOCAL_DEFAULT_BODY)


def test_api_matcher_request_is_unchanged(monkeypatch) -> None:
    body = _capture_wire_body(monkeypatch, {}, api=True, send=_ask_matcher)
    assert body["max_completion_tokens"] == 16  # the cap still binds on the hosted endpoint
    assert "reasoning_effort" not in body
    assert set(body) == set(_PRE_2012_BODY)


def test_anthropic_provider_still_sends_max_tokens(monkeypatch) -> None:
    """``max_tokens`` is REQUIRED by Claude's API: the passed cap, or 4096 when none is given."""
    import sys
    import types

    calls: list[dict] = []

    class _Messages:
        def create(self, **kw: object) -> object:
            calls.append(kw)
            return types.SimpleNamespace(
                content=[types.SimpleNamespace(type="text", text="{}")],
                usage=types.SimpleNamespace(input_tokens=1, output_tokens=1),
            )

    fake = types.ModuleType("anthropic")
    fake.NOT_GIVEN = object()  # type: ignore[attr-defined]
    fake.Anthropic = lambda api_key: types.SimpleNamespace(messages=_Messages())  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", fake)

    provider = AnthropicChatProvider(Settings(profile=Profile.API, anthropic_api_key="k"))
    provider.complete("PROMPT", system="SYSTEM", max_tokens=900)
    provider.complete("PROMPT")
    assert [c["max_tokens"] for c in calls] == [900, 4096]
    assert calls[0]["system"] == "SYSTEM" and calls[1]["system"] is fake.NOT_GIVEN


def test_reset_prompt_cache_never_raises_when_ollama_is_unreachable(monkeypatch) -> None:
    from recruiter_copilot import llm

    def boom(url: str, body: dict | None = None, timeout: float = 10.0) -> dict:  # noqa: ARG001
        raise OSError("connection refused")

    monkeypatch.setattr(llm, "_ollama_json", boom)
    assert OllamaProvider(Settings()).reset_prompt_cache() is False
