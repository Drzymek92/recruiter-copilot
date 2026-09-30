"""Chat providers (D12: one interface, two profiles).

``PROFILE=local`` talks to Ollama through ``langchain_openai`` (an OpenAI-compatible endpoint,
so no vendor SDK is imported — the house rule holds here unmodified). ``PROFILE=api`` uses the
same client against a hosted OpenAI-compatible endpoint, or the ``anthropic`` SDK when an
Anthropic key is set, which is the one narrow Policy Override this project carries: Claude has
no OpenAI-compatible endpoint, so ``langchain_openai`` cannot reach it.

Unlike the sibling project this seam does **not** stream. Analysis (M2) is a post-call batch
job, so the 3x client overhead that forced a raw-HTTP streaming path there buys nothing here,
and the house rule is kept instead.

**The truncation trap is guarded, not assumed.** Ollama silently truncates an over-long prompt,
returns no error, and reports the *truncated* count as ``prompt_tokens`` — a 37.8k-character
prompt came back as 2050 tokens with most of the context gone. ``ChatReply.truncated`` flags it
so a caller can refuse to build a report from a half-read bundle.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Protocol

from .config import Settings
from .models import Profile

logger = logging.getLogger("recruiter_copilot.llm")

# Above this chars-per-token ratio the backend cannot have seen everything that was sent.
TRUNCATION_CHARS_PER_TOKEN = 7.0


class ChatError(RuntimeError):
    """The provider was unreachable or refused the request."""


@dataclass
class ChatReply:
    text: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    sent_chars: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def truncated(self) -> bool:
        """True when the reported prompt tokens cannot account for what was sent."""
        if not self.input_tokens or not self.sent_chars:
            return False
        return (self.sent_chars / self.input_tokens) > TRUNCATION_CHARS_PER_TOKEN

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ChatProvider(Protocol):
    name: str
    model_name: str

    def complete(
        self, prompt: str, system: str | None = None, max_tokens: int | None = None
    ) -> ChatReply: ...


class _OpenAICompatibleProvider:
    """Shared implementation for any OpenAI-compatible endpoint (Ollama or hosted)."""

    def __init__(
        self, settings: Settings, base_url: str, token: str, model: str, label: str
    ) -> None:
        self.settings = settings
        self.base_url = base_url
        self.token = token
        self.model_name = model
        self.name = f"{label}:{model}"
        self._label = label

    def complete(
        self, prompt: str, system: str | None = None, max_tokens: int | None = None
    ) -> ChatReply:
        from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415
        from langchain_openai import ChatOpenAI  # noqa: PLC0415

        kwargs: dict[str, object] = {}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        client = ChatOpenAI(
            model=self.model_name,
            base_url=self.base_url,
            api_key=self.token,
            temperature=0.0,  # analysis must be reproducible; this is not creative writing
            timeout=300,
            max_retries=1,
            **kwargs,
        )
        messages = ([SystemMessage(content=system)] if system else []) + [
            HumanMessage(content=prompt)
        ]
        sent_chars = len(prompt) + len(system or "")
        t0 = time.perf_counter()
        try:
            reply = client.invoke(messages)
        except Exception as e:  # noqa: BLE001 — surfaced as one typed error to the caller
            raise ChatError(f"{self.name} unreachable: {e}") from e
        latency = time.perf_counter() - t0

        usage = getattr(reply, "usage_metadata", None) or {}
        result = ChatReply(
            text=str(reply.content).strip(),
            model=self.model_name,
            provider=self._label,
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            latency_seconds=latency,
            sent_chars=sent_chars,
        )
        if result.truncated:
            logger.warning(
                "PROMPT TRUNCATED: sent %d chars but the backend counted only %d prompt tokens "
                "(%.1f chars/token). Raise the model's context window — Ollama truncates "
                "silently and its OpenAI endpoint IGNORES options.num_ctx, so the window must be "
                "baked into the model with a Modelfile.",
                sent_chars,
                result.input_tokens,
                sent_chars / max(1, result.input_tokens),
            )
        return result


class OllamaProvider(_OpenAICompatibleProvider):
    """Local Ollama (``PROFILE=local``). Nothing leaves the machine (SI1)."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(
            settings,
            base_url=settings.ollama_base_url,
            token=settings.ollama_token,
            model=settings.ollama_model,
            label="ollama",
        )


class OpenAIChatProvider(_OpenAICompatibleProvider):
    """Hosted OpenAI-compatible endpoint (``PROFILE=api``). **This is an egress (SI1).**"""

    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ChatError("PROFILE=api with the OpenAI-compatible provider needs OPENAI_API_KEY")
        super().__init__(
            settings,
            base_url=settings.openai_base_url,
            token=settings.openai_api_key,
            model=settings.openai_chat_model or "gpt-4o-mini",
            label="openai-compatible",
        )


class AnthropicChatProvider:
    """Claude via the official SDK (``PROFILE=api``). **This is an egress (SI1).**

    POLICY OVERRIDE (D6, recorded in agent/project.md): the house rule says never import a
    vendor SDK. It cannot be honoured here — Claude's API is ``/v1/messages`` with ``x-api-key``,
    which ``langchain_openai`` cannot speak. The import is lazy, so a local-profile run never
    loads it. Two Claude-specific traps this path handles: ``max_tokens`` is required, and
    ``temperature`` is rejected by some models, so it is not sent.
    """

    def __init__(self, settings: Settings) -> None:
        if not settings.anthropic_api_key:
            raise ChatError("the Anthropic provider needs ANTHROPIC_API_KEY")
        self.settings = settings
        self.model_name = settings.anthropic_model or "claude-sonnet-5"
        self.name = f"anthropic:{self.model_name}"

    def complete(
        self, prompt: str, system: str | None = None, max_tokens: int | None = None
    ) -> ChatReply:
        import anthropic  # noqa: PLC0415 — Policy Override; lazy so local never imports it

        client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
        sent_chars = len(prompt) + len(system or "")
        t0 = time.perf_counter()
        try:
            message = client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens or 4096,  # required by the API, not optional
                system=system or anthropic.NOT_GIVEN,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:  # noqa: BLE001
            raise ChatError(f"{self.name} call failed: {e}") from e
        latency = time.perf_counter() - t0
        text = "".join(b.text for b in message.content if b.type == "text").strip()
        return ChatReply(
            text=text,
            model=self.model_name,
            provider="anthropic",
            input_tokens=int(message.usage.input_tokens),
            output_tokens=int(message.usage.output_tokens),
            latency_seconds=latency,
            sent_chars=sent_chars,
        )


def build_chat_provider(settings: Settings) -> ChatProvider:
    """The D12 switch: one setting picks the provider, nothing downstream changes."""
    if settings.profile is Profile.API:
        if settings.anthropic_api_key:
            return AnthropicChatProvider(settings)
        return OpenAIChatProvider(settings)
    return OllamaProvider(settings)
