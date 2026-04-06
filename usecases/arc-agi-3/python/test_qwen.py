"""
test_qwen.py — Compare Sonnet 4.6 vs Qwen2.5-7B-Instruct on LS20 OBSERVER prompt.

Loads real API keys from P:/_access/Security/api_keys.env, runs one OBSERVER
call against both models using the actual LS20 initial game state, and prints
a side-by-side comparison.

Usage (arc conda env required):
  python test_qwen.py
"""

from __future__ import annotations
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE    = Path(__file__).parent
_KF_ROOT = _HERE.resolve().parents[3]
for _p in (str(_KF_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Load API keys
# ---------------------------------------------------------------------------
_key_file = Path("P:/_access/Security/api_keys.env")
if _key_file.exists():
    for _line in _key_file.read_text().splitlines():
        for _prefix in ("ANTHROPIC_API_KEY=", "arc_api_key=", "TOGETHER_API_KEY="):
            if _line.startswith(_prefix):
                _var = _prefix.rstrip("=").upper()
                if not os.environ.get(_var):
                    os.environ[_var] = _line.split("=", 1)[1].strip()

_TOGETHER_KEY = os.environ.get("TOGETHER_API_KEY", "")
_ARC_KEY      = os.environ.get("ARC_API_KEY", "")
_ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

if not _ANTHROPIC_KEY:
    print("ERROR: ANTHROPIC_API_KEY not set")
    sys.exit(1)
if not _TOGETHER_KEY:
    print("ERROR: TOGETHER_API_KEY not set")
    sys.exit(1)

print(f"Anthropic key: ...{_ANTHROPIC_KEY[-8:]}")
print(f"Together key:  ...{_TOGETHER_KEY[-8:]}")
print(f"ARC API key:   ...{_ARC_KEY[-8:] if _ARC_KEY else '(none)'}")

# ---------------------------------------------------------------------------
# Build real OBSERVER user message from initial LS20 state
# ---------------------------------------------------------------------------
import arc_agi
from ensemble import _KNOWN_SUBPLANS
from agents import (
    load_prompt,
    obs_frame, obs_levels_completed, obs_state_name,
    frame_to_str, frame_shape, format_action_space, format_action_history,
    _format_action_effects, summarize_current_objects, format_structural_context,
    compute_trend_predictions,
)

print("\nCreating LS20 environment (initial state)...")
arc_obj = arc_agi.Arcade(arc_api_key=_ARC_KEY)
env     = arc_obj.make("ls20", render_mode=None)
obs     = env.reset()
AS      = {a.name: a for a in env.action_space}

# Take one action to get a real observation with some action history
obs = env.step(AS["ACTION1"])

# Build prompt inputs
frame = obs_frame(obs)
grid_str    = frame_to_str(frame)
shape       = frame_shape(frame)
levels      = obs_levels_completed(obs)
state       = obs_state_name(obs)
actions_str = format_action_space(list(env.action_space))

action_history = [{"action": "ACTION1", "data": None, "levels": levels, "state": state}]
history_str    = format_action_history(action_history)
effects_str    = "  (no action effects recorded yet -- all actions are unexplored)"
objects_str    = summarize_current_objects(frame, None)
structural_str = format_structural_context(frame)
predictions_str = "  (none yet — not enough data)"
concepts_str    = "  (none identified yet)"

SYSTEM_PROMPT = load_prompt("OBSERVER")

USER_MESSAGE = (
    f"## Current game state\n\n"
    f"State: {state}\n"
    f"Levels completed: {levels}\n"
    f"Steps remaining this episode: 299\n\n"
    f"## Current frame ({shape})\n\n"
    f"{grid_str}\n\n"
    f"## Current objects (non-background)\n\n"
    f"{objects_str}\n\n"
    f"## Structural context (containment & spatial alignment — zero-cost)\n\n"
    f"{structural_str}\n\n"
    f"## Known concept bindings\n\n"
    f"{concepts_str}\n\n"
    f"## Trend predictions (zero-cost projections)\n\n"
    f"{predictions_str}\n\n"
    f"## Observed action effects (accumulated this episode)\n\n"
    f"{effects_str}\n\n"
    f"## Available actions\n\n"
    f"{actions_str}\n\n"
    f"## Recent action history\n\n"
    f"{history_str}\n"
)

print(f"User message: {len(USER_MESSAGE)} chars  |  System prompt: {len(SYSTEM_PROMPT)} chars")

# ---------------------------------------------------------------------------
# Call helpers
# ---------------------------------------------------------------------------

async def call_anthropic(model: str, system: str, user: str, max_tokens: int = 1024) -> tuple[str, int]:
    """Call via Anthropic SDK."""
    import anthropic as _ant
    client = _ant.AsyncAnthropic(api_key=_ANTHROPIC_KEY)
    t0 = time.time()
    resp = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    ms   = int((time.time() - t0) * 1000)
    text = resp.content[0].text if resp.content else ""
    return text, ms


async def call_together(model: str, system: str, user: str, max_tokens: int = 1024) -> tuple[str, int]:
    """Call Qwen via Together.ai OpenAI-compatible endpoint."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        api_key=_TOGETHER_KEY,
        base_url="https://api.together.xyz/v1",
    )
    t0 = time.time()
    resp = await client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    ms   = int((time.time() - t0) * 1000)
    text = resp.choices[0].message.content if resp.choices else ""
    return text, ms


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def extract_json_blocks(text: str) -> list[dict]:
    """Extract all ```json ... ``` blocks and parse them."""
    blocks = []
    for raw in re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text or ""):
        try:
            blocks.append(json.loads(raw))
        except (json.JSONDecodeError, ValueError):
            pass
    return blocks


def evaluate_response(label: str, text: str, ms: int) -> None:
    """Print quality assessment for one model response."""
    print(f"\n{'='*60}")
    print(f"  {label}  ({ms}ms)")
    print(f"{'='*60}")

    if not text:
        print("  ERROR: empty response")
        return

    print(f"  Response length: {len(text)} chars")

    # JSON validity
    blocks = extract_json_blocks(text)
    if blocks:
        print(f"  JSON blocks found: {len(blocks)} — VALID")
        for i, blk in enumerate(blocks):
            keys = list(blk.keys())
            print(f"    Block {i+1} keys: {keys}")
            # Check for expected OBSERVER output fields
            expected = {"level_description", "hypothesized_goal", "concept_bindings",
                        "reasoning", "visual_observations"}
            present = expected & set(keys)
            missing = expected - set(keys)
            print(f"    Expected keys present: {sorted(present)}")
            if missing:
                print(f"    Missing keys: {sorted(missing)}")
    else:
        print("  JSON blocks found: 0 — INVALID (no parseable JSON)")

    # GUESS / CONFIRMED discipline
    guess_count     = text.upper().count("[GUESS]")
    confirmed_count = text.upper().count("[CONFIRMED]")
    print(f"  [GUESS] labels: {guess_count}  |  [CONFIRMED] labels: {confirmed_count}")

    # Concept bindings
    if blocks:
        for blk in blocks:
            cb = blk.get("concept_bindings", {})
            if cb:
                print(f"  Concept bindings proposed: {dict(list(cb.items())[:5])}")

    # First 400 chars of text
    print(f"\n  --- Response preview ---")
    print(f"  {text[:400].replace(chr(10), chr(10)+'  ')}")
    print(f"  {'...' if len(text) > 400 else '(end)'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MODELS = [
    ("Sonnet 4.6 (Anthropic)",              "claude-sonnet-4-6",                          call_anthropic),
    ("Qwen2.5-7B-Instruct (Together)",      "Qwen/Qwen2.5-7B-Instruct-Turbo",            call_together),
    ("Qwen3.5-9B (Together)",               "Qwen/Qwen3.5-9B",                            call_together),
    ("Qwen3-235B-A22B tput MoE (Together)", "Qwen/Qwen3-235B-A22B-Instruct-2507-tput",   call_together),
]


async def main() -> None:
    results = []
    for label, model_id, caller in MODELS:
        print(f"\nCalling {label}...")
        try:
            text, ms = await caller(model_id, SYSTEM_PROMPT, USER_MESSAGE, max_tokens=3000)
            results.append((label, text, ms))
            print(f"  Done in {ms}ms  ({len(text)} chars)")
        except Exception as exc:
            print(f"  ERROR: {exc}")
            results.append((label, "", 0))

    print("\n\n" + "#"*60)
    print("# EVALUATION RESULTS")
    print("#"*60)

    for label, text, ms in results:
        evaluate_response(label, text, ms)

    # JSON validity summary
    print("\n\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for label, text, ms in results:
        blocks = extract_json_blocks(text)
        valid  = "YES" if blocks else "NO"
        guess  = text.upper().count("[GUESS]") if text else 0
        conf   = text.upper().count("[CONFIRMED]") if text else 0
        print(f"  {label:<40}  JSON:{valid:<4}  [GUESS]:{guess:<3}  [CONFIRMED]:{conf:<3}  {ms}ms")


if __name__ == "__main__":
    asyncio.run(main())
