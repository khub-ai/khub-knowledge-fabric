"""
test_qwen_extended.py — Extended evaluation of Llama-3.3-70B vs Sonnet 4.6.

Tests:
  1. OBSERVER consistency — 3 repeated calls on the same game state
  2. MEDIATOR quality     — plan quality using real OBSERVER output
  3. Live episode         — 10-step mini-episode driven by Llama-3.3-70B vs Sonnet 4.6

Usage (arc conda env required):
  python test_qwen_extended.py
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
# Path setup + API keys
# ---------------------------------------------------------------------------
_HERE    = Path(__file__).parent
_KF_ROOT = _HERE.resolve().parents[3]
for _p in (str(_KF_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

for _line in Path("P:/_access/Security/api_keys.env").read_text().splitlines():
    for _pfx in ("ANTHROPIC_API_KEY=", "arc_api_key=", "TOGETHER_API_KEY="):
        if _line.startswith(_pfx):
            _var = _pfx.rstrip("=").upper()
            if not os.environ.get(_var):
                os.environ[_var] = _line.split("=", 1)[1].strip()

_TOGETHER_KEY  = os.environ["TOGETHER_API_KEY"]
_ARC_KEY       = os.environ.get("ARC_API_KEY", "")
_ANTHROPIC_KEY = os.environ["ANTHROPIC_API_KEY"]

print(f"Anthropic: ...{_ANTHROPIC_KEY[-8:]}  Together: ...{_TOGETHER_KEY[-8:]}")

# ---------------------------------------------------------------------------
# Game imports
# ---------------------------------------------------------------------------
import arc_agi
from ensemble import _KNOWN_SUBPLANS
from agents import (
    load_prompt, parse_action_plan,
    obs_frame, obs_levels_completed, obs_state_name,
    frame_to_str, frame_shape, format_action_space, format_action_history,
    _format_action_effects, summarize_current_objects, format_structural_context,
    compute_trend_predictions,
)

# ---------------------------------------------------------------------------
# LLM call helpers
# ---------------------------------------------------------------------------

async def call_anthropic(model: str, system: str, user: str, max_tokens: int = 3000) -> tuple[str, int]:
    import anthropic as _ant
    client = _ant.AsyncAnthropic(api_key=_ANTHROPIC_KEY)
    t0 = time.time()
    resp = await client.messages.create(
        model=model, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": user}],
    )
    return (resp.content[0].text if resp.content else ""), int((time.time() - t0) * 1000)


async def call_together(model: str, system: str, user: str, max_tokens: int = 3000) -> tuple[str, int]:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=_TOGETHER_KEY, base_url="https://api.together.xyz/v1")
    t0 = time.time()
    resp = await client.chat.completions.create(
        model=model, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return (resp.choices[0].message.content if resp.choices else ""), int((time.time() - t0) * 1000)


SONNET   = ("claude-sonnet-4-6",                        call_anthropic)
LLAMA70B = ("meta-llama/Llama-3.3-70B-Instruct-Turbo", call_together)

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_observer_message(obs, available_actions, action_history=None, action_effects=None):
    frame       = obs_frame(obs)
    grid_str    = frame_to_str(frame)
    shape       = frame_shape(frame)
    levels      = obs_levels_completed(obs)
    state       = obs_state_name(obs)
    actions_str = format_action_space(list(available_actions))
    history_str = format_action_history(action_history or [])
    effects_str = _format_action_effects(action_effects or {})
    objects_str = summarize_current_objects(frame, None)
    struct_str  = format_structural_context(frame)
    preds_str   = "  (none yet — not enough data)"
    concepts_str = "  (none identified yet)"
    return (
        f"## Current game state\n\nState: {state}\nLevels completed: {levels}\n"
        f"Steps remaining this episode: 299\n\n"
        f"## Current frame ({shape})\n\n{grid_str}\n\n"
        f"## Current objects (non-background)\n\n{objects_str}\n\n"
        f"## Structural context\n\n{struct_str}\n\n"
        f"## Known concept bindings\n\n{concepts_str}\n\n"
        f"## Trend predictions\n\n{preds_str}\n\n"
        f"## Observed action effects\n\n{effects_str}\n\n"
        f"## Available actions\n\n{actions_str}\n\n"
        f"## Recent action history\n\n{history_str}\n"
    )


def build_mediator_message(observer_text, available_actions, action_history=None):
    msg  = f"## OBSERVER analysis\n\n{observer_text}\n"
    msg += f"\n## Available actions\n\n{format_action_space(list(available_actions))}\n"
    if action_history:
        msg += f"\n## Full action history\n\n{format_action_history(action_history, last_n=15)}\n"
    return msg


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def extract_json(text: str) -> list[dict]:
    blocks = []
    for raw in re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text or ""):
        try:
            blocks.append(json.loads(raw))
        except (json.JSONDecodeError, ValueError):
            pass
    return blocks


def score_observer(text: str) -> dict:
    blocks    = extract_json(text)
    valid     = bool(blocks)
    guess_ct  = text.upper().count("[GUESS]")
    conf_ct   = text.upper().count("[CONFIRMED]")
    expected  = {"level_description", "visual_observations", "concept_bindings",
                 "hypothesized_goal", "reasoning"}
    keys_present = set()
    bindings  = {}
    if blocks:
        keys_present = expected & set(blocks[0].keys())
        bindings     = blocks[0].get("concept_bindings", {})
    return {
        "valid_json":     valid,
        "keys_present":   len(keys_present),
        "missing_keys":   sorted(expected - keys_present),
        "guess":          guess_ct,
        "confirmed":      conf_ct,
        "bindings":       bindings,
        "response_len":   len(text),
    }


def score_mediator(text: str) -> dict:
    plan      = parse_action_plan(text)
    blocks    = extract_json(text)
    valid     = bool(plan)
    actions   = [s["action"] for s in plan]
    return {
        "valid_plan":   valid,
        "action_count": len(plan),
        "actions":      actions[:10],
        "has_json":     bool(blocks),
        "response_len": len(text),
    }


def print_score(label: str, s: dict, ms: int) -> None:
    print(f"  {label:<42}  {ms:5}ms  ", end="")
    if "valid_json" in s:
        status = "JSON:OK" if s["valid_json"] else "JSON:FAIL"
        print(f"{status}  keys:{s['keys_present']}/5  "
              f"[G]:{s['guess']:2d}  [C]:{s['confirmed']:2d}  "
              f"bindings:{len(s['bindings'])}  {s['response_len']}ch")
    else:
        status = "PLAN:OK" if s["valid_plan"] else "PLAN:FAIL"
        print(f"{status}  actions:{s['action_count']}  {s['actions']}")


# ===========================================================================
# TEST 1: OBSERVER CONSISTENCY (3 runs each model)
# ===========================================================================

async def test_observer_consistency():
    print("\n" + "="*70)
    print("TEST 1: OBSERVER CONSISTENCY (3 runs, same initial state)")
    print("="*70)

    arc_obj = arc_agi.Arcade(arc_api_key=_ARC_KEY)
    env     = arc_obj.make("ls20", render_mode=None)
    obs     = env.reset()
    obs     = env.step(list(env.action_space)[0])  # one step for some history
    AS      = {a.name: a for a in env.action_space}

    obs_sp = load_prompt("OBSERVER")
    um     = build_observer_message(obs, env.action_space,
                                    action_history=[{"action": "ACTION1", "data": None,
                                                     "levels": 0, "state": "NOT_FINISHED"}])

    results = {name: [] for name in ("Sonnet 4.6", "Llama-3.3-70B")}
    for run in range(3):
        print(f"\n  --- Run {run+1}/3 ---")
        for label, (model_id, caller) in [("Sonnet 4.6", SONNET), ("Llama-3.3-70B", LLAMA70B)]:
            text, ms = await caller(model_id, obs_sp, um)
            s = score_observer(text)
            results[label].append(s)
            print_score(f"  {label} run{run+1}", s, ms)

    # Consistency summary: how often do bindings agree across 3 runs?
    print("\n  --- Consistency summary ---")
    for label, runs in results.items():
        valid_runs = sum(1 for r in runs if r["valid_json"])
        all_binding_keys = [set(r["bindings"].keys()) for r in runs if r["valid_json"]]
        common = set.intersection(*all_binding_keys) if all_binding_keys else set()
        guess_var  = max(r["guess"] for r in runs) - min(r["guess"] for r in runs)
        print(f"  {label:<15}  valid:{valid_runs}/3  "
              f"common_bindings:{sorted(common)}  [GUESS] variance:{guess_var}")


# ===========================================================================
# TEST 2: MEDIATOR QUALITY
# ===========================================================================

async def test_mediator_quality():
    print("\n" + "="*70)
    print("TEST 2: MEDIATOR QUALITY (using Sonnet OBSERVER output as shared input)")
    print("="*70)

    arc_obj = arc_agi.Arcade(arc_api_key=_ARC_KEY)
    env     = arc_obj.make("ls20", render_mode=None)
    obs     = env.reset()
    obs     = env.step(list(env.action_space)[0])

    obs_sp = load_prompt("OBSERVER")
    med_sp = load_prompt("MEDIATOR")
    um_obs = build_observer_message(obs, env.action_space,
                                    action_history=[{"action": "ACTION1", "data": None,
                                                     "levels": 0, "state": "NOT_FINISHED"}])

    # Use Sonnet for the OBSERVER output so both models see the same input
    print("\n  Getting shared OBSERVER output (Sonnet)...")
    shared_obs_text, obs_ms = await SONNET[1](SONNET[0], obs_sp, um_obs)
    print(f"  OBSERVER done in {obs_ms}ms  ({len(shared_obs_text)} chars)")

    um_med = build_mediator_message(shared_obs_text, env.action_space,
                                    action_history=[{"action": "ACTION1", "data": None,
                                                     "levels": 0, "state": "NOT_FINISHED"}])

    print("\n  Running MEDIATOR on same input...")
    for label, (model_id, caller) in [("Sonnet 4.6", SONNET), ("Llama-3.3-70B", LLAMA70B)]:
        text, ms = await caller(model_id, med_sp, um_med)
        s = score_mediator(text)
        print_score(f"  {label}", s, ms)
        if s["valid_plan"]:
            print(f"    Full plan: {s['actions']}")
        else:
            print(f"    No parseable plan. Preview: {text[:300]}")


# ===========================================================================
# TEST 3: LIVE MINI-EPISODE (10 steps driven by each model)
# ===========================================================================

async def live_episode(label: str, model_id: str, caller):
    print(f"\n  --- {label} ---")
    arc_obj = arc_agi.Arcade(arc_api_key=_ARC_KEY)
    env     = arc_obj.make("ls20", render_mode=None)
    obs     = env.reset()
    AS      = {a.name: a for a in env.action_space}
    obs_sp  = load_prompt("OBSERVER")
    med_sp  = load_prompt("MEDIATOR")

    action_history: list[dict] = []
    step = 0
    levels_reached = 0
    total_obs_ms = total_med_ms = 0
    valid_plans = invalid_plans = 0

    while step < 10:
        # OBSERVER
        um_obs = build_observer_message(obs, env.action_space, action_history)
        obs_text, obs_ms = await caller(model_id, obs_sp, um_obs)
        total_obs_ms += obs_ms

        # MEDIATOR
        um_med = build_mediator_message(obs_text, env.action_space, action_history)
        med_text, med_ms = await caller(model_id, med_sp, um_med)
        total_med_ms += med_ms

        plan = parse_action_plan(med_text)
        if not plan:
            invalid_plans += 1
            print(f"    step {step+1}: no valid plan — skipping cycle")
            step += 1
            continue

        valid_plans += 1
        for action_dict in plan:
            if step >= 10:
                break
            aname = action_dict["action"]
            if aname not in AS:
                print(f"    step {step+1}: unknown action {aname!r}")
                step += 1
                continue
            obs = env.step(AS[aname])
            lvl = obs_levels_completed(obs)
            levels_reached = max(levels_reached, lvl)
            action_history.append({
                "action": aname, "data": None,
                "levels": lvl, "state": obs_state_name(obs),
            })
            step += 1

    print(f"    steps:{step}  levels_reached:{levels_reached}  "
          f"valid_plans:{valid_plans}  invalid_plans:{invalid_plans}")
    print(f"    avg OBSERVER:{total_obs_ms // max(valid_plans,1)}ms  "
          f"avg MEDIATOR:{total_med_ms // max(valid_plans,1)}ms")
    print(f"    actions taken: {[h['action'] for h in action_history]}")
    return levels_reached


async def test_live_episode():
    print("\n" + "="*70)
    print("TEST 3: LIVE MINI-EPISODE (10 game steps each model)")
    print("="*70)

    for label, (model_id, caller) in [("Sonnet 4.6", SONNET), ("Llama-3.3-70B", LLAMA70B)]:
        lvl = await live_episode(label, model_id, caller)
        print(f"    => {label} reached level {lvl+1}")


# ===========================================================================
# Main
# ===========================================================================

async def main():
    await test_observer_consistency()
    await test_mediator_quality()
    await test_live_episode()
    print("\n\nDone.")

if __name__ == "__main__":
    asyncio.run(main())
