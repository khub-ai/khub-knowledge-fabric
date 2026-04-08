"""
Claude Code CLI subprocess backend.

Provides a drop-in alternative to the Anthropic SDK path in ``core.pipeline.agents``
that invokes the ``claude`` CLI in headless print mode (``claude -p``).

Why this exists:
  - The Anthropic SDK path bills against an API key / API account.
  - The ``claude`` CLI, when NOT run with --bare, uses the user's Claude Code
    authentication (OAuth / Max plan subscription), which can avoid per-token
    API charges for testing and knowledge accumulation workflows.

Toggle via the ``KF_USE_CLAUDE_CLI`` environment variable. When set to a truthy
value, ``core.pipeline.agents.call_agent`` will route string-only Claude calls
through this module instead of the SDK.

Limitations:
  - String user messages only. Multimodal content (images) falls back to SDK.
  - No cost/usage accounting — the CLI does not expose per-call token counts
    in ``--output-format text``. Reported cost is 0.
  - Subprocess startup latency is higher than SDK (~5-15s extra per call).
  - Hooks, auto-memory, and CLAUDE.md auto-discovery run unless the caller
    takes care to isolate ``cwd``.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time
from typing import Optional


CLAUDE_BIN = os.environ.get("KF_CLAUDE_BIN", "claude")
DEFAULT_TIMEOUT_S = int(os.environ.get("KF_CLAUDE_CLI_TIMEOUT", "300"))
# Isolate CLI runs in a scratch dir so repo CLAUDE.md / hooks don't fire.
DEFAULT_CWD = os.environ.get("KF_CLAUDE_CLI_CWD") or os.path.join(
    os.path.expanduser("~"), ".cache", "kf_claude_cli"
)


def is_enabled() -> bool:
    """Return True if KF_USE_CLAUDE_CLI is set to a truthy value."""
    v = os.environ.get("KF_USE_CLAUDE_CLI", "")
    return v.strip().lower() in ("1", "true", "yes", "on")


def _resolve_model_alias(model: str) -> str:
    """Map SDK model strings to CLI --model aliases.

    The CLI accepts either aliases (``sonnet``, ``haiku``, ``opus``) or full
    model names. Aliases are preferred for forward compatibility.
    """
    m = (model or "").lower()
    if "haiku" in m:
        return "haiku"
    if "opus" in m:
        return "opus"
    return "sonnet"


async def call_via_cli(
    agent_id: str,
    user_message: str,
    system_prompt: str = "",
    model: str = "",
    timeout_s: Optional[int] = None,
    cwd: Optional[str] = None,
) -> tuple[str, int]:
    """Single-shot Claude Code CLI call. Returns (text, duration_ms).

    Raises RuntimeError on timeout or non-zero exit.
    """
    if not isinstance(user_message, str):
        raise TypeError(
            "claude_cli.call_via_cli only accepts string user_message; "
            "multimodal content must go through the SDK path."
        )
    if shutil.which(CLAUDE_BIN) is None:
        raise RuntimeError(
            f"claude CLI not found on PATH (looked for {CLAUDE_BIN!r}). "
            "Install Claude Code or set KF_CLAUDE_BIN."
        )

    run_cwd = cwd or DEFAULT_CWD
    os.makedirs(run_cwd, exist_ok=True)

    # Pipe user message via stdin to avoid Windows ARG_MAX (~32KB) limits.
    # Pass the system prompt via a temp file if it's large enough to risk
    # the same limit, otherwise inline via --append-system-prompt.
    args: list[str] = [
        CLAUDE_BIN,
        "-p",
        "--output-format", "text",
        "--model", _resolve_model_alias(model),
        "--no-session-persistence",
        "--dangerously-skip-permissions",
    ]
    import tempfile
    _sys_tmp_path: Optional[str] = None
    if system_prompt:
        if len(system_prompt) > 4000:
            fd, _sys_tmp_path = tempfile.mkstemp(
                prefix="kf_sysprompt_", suffix=".txt", text=True
            )
            with os.fdopen(fd, "w", encoding="utf-8") as _f:
                _f.write(system_prompt)
            args.extend(["--append-system-prompt-file", _sys_tmp_path])
        else:
            args.extend(["--append-system-prompt", system_prompt])

    t0 = time.time()
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=run_cwd,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(input=user_message.encode("utf-8")),
                timeout=timeout_s or DEFAULT_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError(
                f"[{agent_id}] claude CLI timed out after "
                f"{timeout_s or DEFAULT_TIMEOUT_S}s"
            )
    except FileNotFoundError as e:
        raise RuntimeError(f"claude CLI spawn failed: {e}")
    finally:
        if _sys_tmp_path:
            try:
                os.unlink(_sys_tmp_path)
            except Exception:
                pass

    duration_ms = int((time.time() - t0) * 1000)
    text = (stdout_b or b"").decode("utf-8", errors="replace").strip()
    if proc.returncode != 0:
        err = (stderr_b or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"[{agent_id}] claude CLI exit {proc.returncode}: {err[:500]}"
        )
    return text, duration_ms
