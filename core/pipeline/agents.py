"""
core/pipeline/agents.py — Generic Anthropic API infrastructure.

Domain-agnostic: no references to grids, tasks, or prompt files.
Each use case imports these primitives and builds its own agent
runner functions on top.

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

DEFAULT_MODEL      = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 4096

# Set to True by harness/ensemble to print prompts before each call
SHOW_PROMPTS: bool = False


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
# Cost tracking
# ---------------------------------------------------------------------------

# Claude Sonnet 4.6 pricing (USD per token).
# Verify against https://www.anthropic.com/pricing if estimates diverge.
# Anthropic auto-caches repeated system prompts:
#   cache_creation costs 25% more, cache_read costs 90% less.
_PRICE_INPUT_PER_TOKEN          = 3.00  / 1_000_000
_PRICE_CACHE_CREATION_PER_TOKEN = 3.75  / 1_000_000
_PRICE_CACHE_READ_PER_TOKEN     = 0.30  / 1_000_000
_PRICE_OUTPUT_PER_TOKEN         = 15.00 / 1_000_000


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

    def add(self, input_tokens: int, output_tokens: int,
            cache_creation: int = 0, cache_read: int = 0) -> None:
        self.input_tokens          += input_tokens
        self.cache_creation_tokens += cache_creation
        self.cache_read_tokens     += cache_read
        self.output_tokens         += output_tokens
        self.api_calls             += 1

    def cost_usd(self) -> float:
        return (
            self.input_tokens          * _PRICE_INPUT_PER_TOKEN +
            self.cache_creation_tokens * _PRICE_CACHE_CREATION_PER_TOKEN +
            self.cache_read_tokens     * _PRICE_CACHE_READ_PER_TOKEN +
            self.output_tokens         * _PRICE_OUTPUT_PER_TOKEN
        )

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
# Core LLM call
# ---------------------------------------------------------------------------

async def call_agent(
    agent_id: str,
    user_message: Union[str, list],
    system_prompt: str = "",
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_retries: int = 5,
) -> tuple[str, int]:
    """
    Make a single LLM call and return (response_text, duration_ms).

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
        system_prompt: System prompt string.  Each use case loads its own
                       prompt files and passes the content here.
        model:         Anthropic model string.
        max_tokens:    Maximum output tokens.
        max_retries:   Retry attempts on rate-limit / overloaded errors.

    Retries on 529 overloaded errors (exponential backoff) and rate-limit
    errors (60s × attempt).  Raises on any other error or exhausted retries.
    """
    if SHOW_PROMPTS:
        _print_prompt(agent_id, system_prompt, user_message, model)

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
