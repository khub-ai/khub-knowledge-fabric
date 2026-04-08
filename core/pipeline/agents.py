"""
core/pipeline/agents.py — Multi-backend LLM infrastructure.

Domain-agnostic: no references to grids, tasks, or prompt files.
Each use case imports these primitives and builds its own agent
runner functions on top.

Supports two backends:
  - "anthropic"  — Claude models via Anthropic SDK (default)
  - "together"   — Open-source models via Together.ai (OpenAI-compatible)

The backend is auto-detected from the model string: models starting with
"claude-" use Anthropic; everything else uses Together.ai.

Provides:
  - get_client()          lazy AsyncAnthropic singleton
  - call_agent()          raw LLM call with retry logic and cost tracking
  - CostTracker           accumulates token usage / USD cost per task run
  - reset_cost_tracker()  called by harness between tasks
  - get_cost_tracker()    returns the module-level singleton
  - DEFAULT_MODEL         current recommended model string
  - SHOW_PROMPTS          module flag; set True by harness to debug prompts
"""

from __future__ import annotations
import asyncio
import os
import time
from typing import Optional, Union

import anthropic


# ---------------------------------------------------------------------------
# Model defaults
# ---------------------------------------------------------------------------

# Switch DEFAULT_MODEL to change the model used across the entire pipeline.
# Anthropic:  "claude-sonnet-4-6"
# Together:   "Qwen/Qwen3.5-9B", "meta-llama/Llama-3.3-70B-Instruct-Turbo",
#             "deepseek-ai/DeepSeek-V3.1", "Qwen/Qwen3.5-397B-A17B"
DEFAULT_MODEL      = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 4096

# Set to True by harness/ensemble to print prompts before each call
SHOW_PROMPTS: bool = False


def _is_anthropic_model(model: str) -> bool:
    """Return True if the model string refers to an Anthropic model."""
    return model.startswith("claude-")


# ---------------------------------------------------------------------------
# Anthropic client (lazy singleton)
# ---------------------------------------------------------------------------

_client: Optional[anthropic.AsyncAnthropic] = None


def get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable not set")
        _client = anthropic.AsyncAnthropic(api_key=api_key)
    return _client


# ---------------------------------------------------------------------------
# Together.ai client (lazy singleton, OpenAI-compatible)
# ---------------------------------------------------------------------------

_together_client = None  # openai.AsyncOpenAI


def _get_together_client():
    global _together_client
    if _together_client is None:
        from openai import AsyncOpenAI
        api_key = os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            raise RuntimeError("TOGETHER_API_KEY environment variable not set")
        _together_client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.together.xyz/v1",
        )
    return _together_client


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------

# Pricing tables (USD per 1M tokens).
# Anthropic — Claude Sonnet 4.6
_ANTHROPIC_PRICING = {
    "input":          3.00,
    "cache_creation": 3.75,
    "cache_read":     0.30,
    "output":        15.00,
}

# Together.ai — per-model pricing (input/output per 1M tokens).
# Source: https://www.together.ai/pricing (April 2026)
_TOGETHER_PRICING = {
    "Qwen/Qwen3.5-9B":                                  (0.10,  0.15),
    "Qwen/Qwen3.5-397B-A17B":                           (0.60,  3.60),
    "Qwen/Qwen2.5-7B-Instruct-Turbo":                   (0.30,  0.30),
    "Qwen/Qwen3-235B-A22B-Instruct-2507-tput":          (0.60,  3.60),
    "meta-llama/Llama-3.3-70B-Instruct-Turbo":           (0.88,  0.88),
    "deepseek-ai/DeepSeek-V3.1":                         (0.60,  1.70),
    "deepseek-ai/DeepSeek-R1-0528":                      (3.00,  7.00),
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503":     (0.10,  0.30),
    "Qwen/Qwen3-Coder-Next-FP8":                        (0.50,  1.20),
}
# Fallback for unlisted Together models
_TOGETHER_DEFAULT_PRICING = (0.50, 1.00)


def _together_token_price(model: str, direction: str) -> float:
    """Return USD per token for a Together.ai model."""
    pair = _TOGETHER_PRICING.get(model, _TOGETHER_DEFAULT_PRICING)
    idx = 0 if direction == "input" else 1
    return pair[idx] / 1_000_000


class CostTracker:
    """Accumulates token usage and computes USD cost for one task run."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.input_tokens:          int = 0
        self.cache_creation_tokens: int = 0
        self.cache_read_tokens:     int = 0
        self.output_tokens:         int = 0
        self.api_calls:             int = 0
        # Together.ai tracks cost directly (model-specific pricing)
        self._together_cost:      float = 0.0

    def add(self, input_tokens: int, output_tokens: int,
            cache_creation: int = 0, cache_read: int = 0) -> None:
        self.input_tokens          += input_tokens
        self.cache_creation_tokens += cache_creation
        self.cache_read_tokens     += cache_read
        self.output_tokens         += output_tokens
        self.api_calls             += 1

    def add_together(self, input_tokens: int, output_tokens: int,
                     model: str) -> None:
        """Track a Together.ai call with model-specific pricing."""
        self.input_tokens  += input_tokens
        self.output_tokens += output_tokens
        self.api_calls     += 1
        self._together_cost += (
            input_tokens  * _together_token_price(model, "input") +
            output_tokens * _together_token_price(model, "output")
        )

    def cost_usd(self) -> float:
        anthropic_cost = (
            self.input_tokens          * _ANTHROPIC_PRICING["input"]  / 1_000_000 +
            self.cache_creation_tokens * _ANTHROPIC_PRICING["cache_creation"] / 1_000_000 +
            self.cache_read_tokens     * _ANTHROPIC_PRICING["cache_read"]     / 1_000_000 +
            self.output_tokens         * _ANTHROPIC_PRICING["output"] / 1_000_000
        )
        if self._together_cost > 0:
            # When Together calls are tracked, the input/output tokens include
            # both backends. Anthropic cost is only meaningful for Anthropic
            # calls, but since we can't separate after the fact, use Together
            # cost directly when it was recorded.
            return self._together_cost
        return anthropic_cost

    def to_dict(self) -> dict:
        return {
            "input_tokens":          self.input_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "cache_read_tokens":     self.cache_read_tokens,
            "output_tokens":         self.output_tokens,
            "api_calls":             self.api_calls,
            "cost_usd":              round(self.cost_usd(), 6),
        }


# Module-level singleton — reset between tasks by the harness/ensemble
_cost_tracker = CostTracker()


def reset_cost_tracker() -> None:
    _cost_tracker.reset()


def get_cost_tracker() -> CostTracker:
    return _cost_tracker


# ---------------------------------------------------------------------------
# Core LLM call — routes to Anthropic or Together.ai based on model string
# ---------------------------------------------------------------------------

async def call_agent(
    agent_id: str,
    user_message: Union[str, list],
    system_prompt: str = "",
    model: str = "",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_retries: int = 5,
) -> tuple[str, int]:
    """
    Make a single LLM call and return (response_text, duration_ms).

    Routes automatically:
      - "claude-*" models → Anthropic SDK
      - Everything else   → Together.ai (OpenAI-compatible)

    Args:
        agent_id:      Label used for logging/display (e.g. "MEDIATOR").
        user_message:  The user-turn message.  Pass a string for text-only
                       calls.  Pass a list of Anthropic content blocks for
                       multimodal calls (e.g. image + text for vision tasks):
                           [
                             {"type": "image", "source": {
                               "type": "base64",
                               "media_type": "image/jpeg",
                               "data": "<base64-string>",
                             }},
                             {"type": "text", "text": "What species is this?"},
                           ]
        system_prompt: System prompt string.
        model:         Model string (Anthropic or Together.ai). Empty = DEFAULT_MODEL.
        max_tokens:    Maximum output tokens.
        max_retries:   Retry attempts on transient errors.
    """
    # Resolve at call time so runtime overrides of DEFAULT_MODEL take effect.
    if not model:
        model = DEFAULT_MODEL

    if SHOW_PROMPTS:
        _print_prompt(agent_id, system_prompt, user_message, model)

    # Optional: route string-only Claude calls through the Claude Code CLI
    # subprocess (KF_USE_CLAUDE_CLI=1). This lets testing/knowledge-accumulation
    # runs piggyback on the user's Max plan instead of API billing.
    if _is_anthropic_model(model) and isinstance(user_message, str):
        try:
            from core.pipeline import claude_cli as _cli
        except Exception:
            _cli = None
        if _cli is not None and _cli.is_enabled():
            return await _cli.call_via_cli(
                agent_id, user_message, system_prompt, model
            )

    if _is_anthropic_model(model):
        return await _call_anthropic(agent_id, user_message, system_prompt,
                                     model, max_tokens, max_retries)
    else:
        return await _call_together(agent_id, user_message, system_prompt,
                                    model, max_tokens, max_retries)


# ---------------------------------------------------------------------------
# Anthropic backend
# ---------------------------------------------------------------------------

async def _call_anthropic(
    agent_id: str,
    user_message: Union[str, list],
    system_prompt: str,
    model: str,
    max_tokens: int,
    max_retries: int,
) -> tuple[str, int]:
    client = get_client()
    t0 = time.time()

    for attempt in range(max_retries):
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            duration_ms = int((time.time() - t0) * 1000)
            text = response.content[0].text if response.content else ""
            if response.usage:
                u = response.usage
                _cost_tracker.add(
                    u.input_tokens,
                    u.output_tokens,
                    cache_creation=getattr(u, "cache_creation_input_tokens", 0) or 0,
                    cache_read=getattr(u, "cache_read_input_tokens", 0) or 0,
                )
            return text, duration_ms
        except anthropic.RateLimitError:
            if attempt < max_retries - 1:
                wait = 60 * (attempt + 1)
                print(f"  [rate-limit] {agent_id} retry {attempt+1}/{max_retries-1} in {wait}s...")
                await asyncio.sleep(wait)
            else:
                raise
        except anthropic.APIStatusError as e:
            if e.status_code == 529 and attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  [overloaded] {agent_id} retry {attempt+1}/{max_retries-1} in {wait}s...")
                await asyncio.sleep(wait)
            else:
                raise


# ---------------------------------------------------------------------------
# Together.ai backend (OpenAI-compatible)
# ---------------------------------------------------------------------------

async def _call_together(
    agent_id: str,
    user_message: Union[str, list],
    system_prompt: str,
    model: str,
    max_tokens: int,
    max_retries: int,
) -> tuple[str, int]:
    client = _get_together_client()
    t0 = time.time()

    # Build messages in OpenAI format
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Convert user_message: string or Anthropic-style content blocks
    if isinstance(user_message, str):
        messages.append({"role": "user", "content": user_message})
    elif isinstance(user_message, list):
        # Convert Anthropic content blocks to OpenAI format
        oai_content = []
        for block in user_message:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    oai_content.append({"type": "text", "text": block["text"]})
                elif block.get("type") == "image":
                    src = block.get("source", {})
                    media = src.get("media_type", "image/png")
                    data = src.get("data", "")
                    oai_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{media};base64,{data}"},
                    })
            else:
                oai_content.append({"type": "text", "text": str(block)})
        messages.append({"role": "user", "content": oai_content})
    else:
        messages.append({"role": "user", "content": str(user_message)})

    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=messages,
            )
            duration_ms = int((time.time() - t0) * 1000)
            text = response.choices[0].message.content if response.choices else ""
            # Track tokens + cost
            if response.usage:
                _cost_tracker.add_together(
                    response.usage.prompt_tokens or 0,
                    response.usage.completion_tokens or 0,
                    model,
                )
            return text, duration_ms
        except Exception as e:
            err_str = str(e)
            is_rate_limit = "rate" in err_str.lower() or "429" in err_str
            is_overloaded = "overloaded" in err_str.lower() or "529" in err_str
            if (is_rate_limit or is_overloaded) and attempt < max_retries - 1:
                wait = 2 ** attempt if is_overloaded else 30 * (attempt + 1)
                print(f"  [{'rate-limit' if is_rate_limit else 'overloaded'}] "
                      f"{agent_id} retry {attempt+1}/{max_retries-1} in {wait}s...")
                await asyncio.sleep(wait)
            else:
                raise


def _print_prompt(agent_id: str, system: str, user: Union[str, list], model: str) -> None:
    """Print agent prompt to terminal (used when SHOW_PROMPTS is True)."""
    if isinstance(user, list):
        # Summarize multimodal content blocks for display
        parts = []
        for block in user:
            if isinstance(block, dict):
                if block.get("type") == "image":
                    parts.append("[image]")
                elif block.get("type") == "text":
                    parts.append(block.get("text", "")[:400])
        usr_preview = " ".join(parts)
    else:
        usr_preview = user[:1200] + ("..." if len(user) > 1200 else "")
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text
        c = Console()
        sys_preview = system[:600] + ("..." if len(system) > 600 else "")
        c.print(Panel(
            Text(sys_preview, style="dim"),
            title=f"[bold magenta]{agent_id} -- system prompt[/bold magenta]  [dim]{model}[/dim]",
            border_style="magenta",
        ))
        c.print(Panel(
            Text(usr_preview),
            title=f"[bold magenta]{agent_id} -- user message[/bold magenta]",
            border_style="magenta",
        ))
    except ImportError:
        print(f"\n=== {agent_id} ({model}) ===")
        print(f"SYSTEM: {system[:400]}")
        print(f"USER:   {usr_preview[:800]}")
