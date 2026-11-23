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
so a caller can refuse to build a report from a half-read bundle. Two signals (#2014): prompt
tokens at the loaded context window (``OLLAMA_CONTEXT_TOKENS``, local profile only — hosted APIs
error on overflow instead), and a chars/token ratio no real prompt reaches.

**Reasoning models (#2012).** Ollama 0.40.2's OpenAI endpoint returns qwen3's thinking in a
separate ``message.reasoning`` field, which ``langchain_openai`` drops, so ``ChatReply.text`` is
the answer alone (no ``<think>`` block) while ``output_tokens`` still counts the thinking. The
opt-in ``OLLAMA_REASONING_EFFORT=none`` turns thinking off; unset, nothing extra is sent.
"""

from __future__ import annotations

import copy
import json
import logging
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol

from .config import Settings
from .models import Profile

logger = logging.getLogger("recruiter_copilot.llm")

# Signal 1 (chars/token): above this ratio the backend cannot have seen everything that was sent.
# Coarse on its own — real analysis prompts run ~3.4 chars/token (#2006), so a moderate cut stays
# under 7 and passes; signal 2 below catches those.
TRUNCATION_CHARS_PER_TOKEN = 7.0
# Signal 2 (context, #2014; local profile only): Ollama cuts an over-long prompt to the loaded
# window and reports the CUT count as prompt_tokens, so a prompt counted within this many tokens
# of ``ChatReply.context_window`` was cut — or left no room for the answer. The measured real-call
# peak (2714 of 4096) is far below the line.
TRUNCATION_CONTEXT_MARGIN_TOKENS = 128


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
    # The loaded context window the prompt was counted against (#2014). Set by the local profile
    # only; 0 (the api providers, which error on overflow instead of cutting) = no context check.
    context_window: int = 0

    @property
    def truncation_signals(self) -> tuple[str, ...]:
        """Which truncation signals fired: ``"context"`` (prompt tokens at the loaded window)
        and/or ``"chars_per_token"`` (too few tokens for what was sent). Empty = not truncated."""
        if not self.input_tokens:
            return ()
        signals: list[str] = []
        if (
            self.context_window
            and self.input_tokens >= self.context_window - TRUNCATION_CONTEXT_MARGIN_TOKENS
        ):
            signals.append("context")
        if self.sent_chars and (self.sent_chars / self.input_tokens) > TRUNCATION_CHARS_PER_TOKEN:
            signals.append("chars_per_token")
        return tuple(signals)

    @property
    def truncated(self) -> bool:
        """True when the backend cannot have read the whole prompt (any signal fired)."""
        return bool(self.truncation_signals)

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
        self,
        settings: Settings,
        base_url: str,
        token: str,
        model: str,
        label: str,
        reasoning_effort: str = "",
        context_window: int = 0,
    ) -> None:
        self.settings = settings
        # #2014: >0 enables the context truncation signal (local profile only; see ChatReply).
        self.context_window = context_window
        self.base_url = base_url
        self.token = token
        self.model_name = model
        self.name = f"{label}:{model}"
        self._label = label
        # Opt-in (#2012): sent only when set, so the default request carries no reasoning_effort.
        self.reasoning_effort = reasoning_effort

    def client_kwargs(self, max_tokens: int | None = None) -> dict[str, object]:
        """The optional ``ChatOpenAI`` kwargs; empty by default (the request is unchanged)."""
        kwargs: dict[str, object] = {}
        if max_tokens is not None:
            # langchain_openai 1.x sends this as `max_completion_tokens`; hosted OpenAI honours it.
            # OllamaProvider overrides this method to drop it (#2019).
            kwargs["max_tokens"] = max_tokens
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        return kwargs

    def complete(
        self, prompt: str, system: str | None = None, max_tokens: int | None = None
    ) -> ChatReply:
        from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415
        from langchain_openai import ChatOpenAI  # noqa: PLC0415

        kwargs = self.client_kwargs(max_tokens)
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
            context_window=self.context_window,
        )
        if result.truncated:
            logger.warning(_truncation_warning(result))
        return result


def _truncation_warning(reply: ChatReply) -> str:
    """The PROMPT TRUNCATED warning, naming which signal(s) fired (#2014)."""
    ratio = reply.sent_chars / max(1, reply.input_tokens)
    reasons: list[str] = []
    if "context" in reply.truncation_signals:
        reasons.append(
            f"context signal: the backend counted {reply.input_tokens} prompt tokens, within "
            f"{TRUNCATION_CONTEXT_MARGIN_TOKENS} of the {reply.context_window}-token context "
            "window (Ollama cuts an over-long prompt to the window and reports the cut count)"
        )
    if "chars_per_token" in reply.truncation_signals:
        reasons.append(
            f"chars/token signal: sent {reply.sent_chars} chars but the backend counted only "
            f"{reply.input_tokens} prompt tokens ({ratio:.1f} chars/token > "
            f"{TRUNCATION_CHARS_PER_TOKEN:.1f})"
        )
    text = (
        f"PROMPT TRUNCATED ({'; '.join(reasons)}). Raise the model's context window — Ollama "
        "truncates silently and its OpenAI endpoint IGNORES options.num_ctx, so the window must "
        "be baked into the model with a Modelfile."
    )
    if reply.context_window:
        text += " Then set OLLAMA_CONTEXT_TOKENS to the window it loads with."
    return text


def _ollama_json(url: str, body: dict | None = None, timeout: float = 10.0) -> dict:
    """One call to Ollama's native JSON API (GET without ``body``, POST with). Stdlib only."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — local Ollama URL
        return json.loads(resp.read().decode("utf-8") or "{}")


def _with_tag(model: str) -> str:
    return model if ":" in model else f"{model}:latest"


class OllamaProvider(_OpenAICompatibleProvider):
    """Local Ollama (``PROFILE=local``). Nothing leaves the machine (SI1)."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(
            settings,
            base_url=settings.ollama_base_url,
            token=settings.ollama_token,
            model=settings.ollama_model,
            label="ollama",
            reasoning_effort=settings.ollama_reasoning_effort,
            context_window=settings.ollama_context_tokens,
        )

    def client_kwargs(self, max_tokens: int | None = None) -> dict[str, object]:
        """As the base, but the token cap is NEVER sent on the local profile (#2019).

        langchain_openai 1.x serialises ``max_tokens`` as ``max_completion_tokens``, which Ollama
        0.40.2's OpenAI endpoint ignores (measured in #2012: replies reached 1957 tokens under a
        "900" cap), so sending it only made the request claim a cap it did not have. The raw
        ``max_tokens`` key is not a fix either: Ollama would apply it as a TOTAL cap that counts
        the qwen3 thinking (on by default, D23), cutting the JSON off mid-answer (#2012). So the
        local request carries no cap; ``max_tokens`` is accepted and deliberately dropped.
        """
        return super().client_kwargs(None)

    def without_thinking(self) -> OllamaProvider:
        """A copy of this provider whose requests send ``reasoning_effort: "none"`` (#2035).

        For one-token answers (the matcher's question id), whatever ``OLLAMA_REASONING_EFFORT``
        says. Measured on the mock interview, qwen3:14b: same P/R/F1 (1.0) and NONE on 12/12
        off-bank questions with thinking on or off, but ~5 s → ~0.1 s per call (~210 → 3 output
        tokens) — and on local the matcher's ``max_tokens`` cap is never sent (#2019), so thinking
        was its only, unbounded, cost. A non-thinking model (llama3.1:8b) accepts it unchanged.
        Only Ollama has this method: the api profile's matcher request is untouched.
        """
        clone = copy.copy(self)
        clone.reasoning_effort = "none"
        return clone

    def reset_prompt_cache(self, wait_seconds: float = 15.0) -> bool:
        """Unload this model so the next call starts from an EMPTY prompt cache (#2006).

        Measured: at temperature 0 Ollama's output depends on how much of each prompt it serves
        from its KV prompt cache (a warm model re-reads the previous run's prompts), so identical
        runs diverge unless each starts cold. The OpenAI endpoint cannot switch the cache off, so
        this uses the native ``keep_alive: 0`` unload. Never raises: on any failure it logs and
        returns False, and the analysis runs anyway (just without the reproducibility guarantee).
        """
        root = self.base_url.rstrip("/").removesuffix("/v1")
        want = _with_tag(self.model_name)

        def loaded() -> bool:
            models = _ollama_json(f"{root}/api/ps").get("models", [])
            return any(want in {m.get("name"), m.get("model")} for m in models)

        try:
            if not loaded():
                return True
            _ollama_json(f"{root}/api/generate", {"model": want, "keep_alive": 0})
            deadline = time.monotonic() + wait_seconds
            while loaded():
                if time.monotonic() > deadline:
                    logger.warning(
                        "%s still loaded after %.0fs; cache not reset", want, wait_seconds
                    )
                    return False
                time.sleep(0.25)
        except (OSError, ValueError) as e:
            logger.warning("could not reset the Ollama prompt cache for %s: %s", want, e)
            return False
        logger.info("unloaded %s so this analysis starts from an empty prompt cache (#2006)", want)
        return True


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
