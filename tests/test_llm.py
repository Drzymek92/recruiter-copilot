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
