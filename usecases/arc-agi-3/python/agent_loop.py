"""
agent_loop.py — Bounded tool-loop OBSERVER for ARC-AGI-3 competition mode.

This module replaces the single-shot OBSERVER call with a hard-bounded
Anthropic tool-use loop. The agent gets a tiny custom toolset (no Bash,
no Read, no Write, no network) and a strict turn cap. It can probe the
current frame, query the object tracker, recall learned concepts from
the persistent ConceptRegistry, and record new concepts before emitting
its final observation JSON.

Hard caps (enforced in this module, not just prompted):
  - max_turns: default 4 tool-call iterations
  - max_tokens: default 2000 per model call
  - per-episode invocation cap (tracked by caller via ConceptRegistry usage)

The bespoke per-game solvers (LS20 BFS, TR87 slot-strip detector, etc.)
remain available in training mode. This module is the COMPETITION_MODE
replacement: game-agnostic primitives + cross-episode learned concepts.

ARC-AGI-3 concept kind vocabulary (caller-defined, opaque to core registry):
  - "object_role"          : a color or sprite plays a structural role
                             (player, wall, target, counter, container, ...)
  - "action_effect"        : pressing an action has a characterized effect
                             (ACTION1=move-up, ACTION4=peek, ...)
  - "mechanic"             : an environmental rule (push pads slide, counter
                             ticks per move, contact triggers, ...)
  - "compositional"        : multi-step pattern (visit-target-then-cross,
                             rotation-changer-pair, ...)

A concept's `signature` dict for ARC-AGI-3 should include:
  - trigger:    short text describing how to detect this concept
  - evidence:   what was observed when first/last confirmed (frame indices,
                action sequence, object ids, etc.)
A concept's `abstraction` dict should include:
  - summary:    one-line human-readable description
  - hint:       actionable guidance for the OBSERVER/MEDIATOR when matched
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

# Make core importable when this file is invoked from the use-case dir.
_KF_ROOT = Path(__file__).resolve().parents[3]
if str(_KF_ROOT) not in sys.path:
    sys.path.insert(0, str(_KF_ROOT))

from core.knowledge.concept_registry import Concept, ConceptRegistry  # noqa: E402

DOMAIN = "arc-agi-3"

ALLOWED_KINDS = {"object_role", "action_effect", "mechanic", "compositional"}

DEFAULT_MAX_TURNS = 4
DEFAULT_MAX_TOKENS = 2000
# Competition-mode OBSERVER uses Opus 4.6 by default — strongest at the
# multi-step spatial reasoning ls20/tr87 require. Override per-call via
# the `model` arg to run_observer_agent.
DEFAULT_AGENT_MODEL = "claude-sonnet-4-6"
DEFAULT_REGISTRY_PATH = _KF_ROOT / "core" / "knowledge" / "store" / "concepts.json"


# ----------------------------------------------------------------------
# Tool schemas (Anthropic tool-use format)
# ----------------------------------------------------------------------

def _tool_specs() -> list[dict]:
    return [
        {
            "name": "inspect_region",
            "description": (
                "Return a sub-rectangle of the current frame as a compact "
                "row-major grid of integer color codes. Use this to examine "
                "specific areas closely. Coordinates are inclusive."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "r0": {"type": "integer"},
                    "c0": {"type": "integer"},
                    "r1": {"type": "integer"},
                    "c1": {"type": "integer"},
                },
                "required": ["r0", "c0", "r1", "c1"],
            },
        },
        {
            "name": "list_objects",
            "description": (
                "List non-background objects detected in the current frame "
                "with their color, bounding box, pixel count, and any role "
                "the OBSERVER has previously bound to this color."
            ),
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "diff_last_frames",
            "description": (
                "Return a summary of what changed between the previous and "
                "current frames: pixel delta count and per-action observed "
                "effects accumulated this episode. Use this to test "
                "hypotheses about action mechanics."
            ),
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "recall_concepts",
            "description": (
                "Query the persistent ConceptRegistry for concepts that may "
                "apply to the current game. Concepts learned from past games "
                "(possibly in other domains) will be returned, ranked by "
                "domain match and confidence."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Free-text substring on name/summary; empty for all.",
                    },
                    "kind": {
                        "type": "string",
                        "enum": sorted(ALLOWED_KINDS),
                    },
                    "limit": {"type": "integer", "default": 5},
                },
            },
        },
        {
            "name": "record_concept",
            "description": (
                "Persist a NEW concept you have discovered or strongly "
                "hypothesized in the current episode. Be conservative: false "
                "concepts pollute the cross-episode knowledge base. Provide "
                "concrete evidence in the signature."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "kind": {"type": "string", "enum": sorted(ALLOWED_KINDS)},
                    "trigger": {
                        "type": "string",
                        "description": "How to detect this concept in any game.",
                    },
                    "summary": {"type": "string"},
                    "hint": {
                        "type": "string",
                        "description": "Actionable guidance when this concept matches.",
                    },
                    "evidence": {
                        "type": "string",
                        "description": "Concrete observation supporting this concept.",
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 0.5,
                    },
                },
                "required": ["name", "kind", "trigger", "summary", "evidence"],
            },
        },
        {
            "name": "confirm_concept",
            "description": (
                "Mark an existing concept (returned by recall_concepts) as "
                "confirmed in the current episode. Bumps its confidence and "
                "appends evidence to its provenance."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "concept_id": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["concept_id", "evidence"],
            },
        },
        {
            "name": "submit_observation",
            "description": (
                "Terminal tool. Emit your final OBSERVER observation as a "
                "markdown text block and exit the loop. Call this exactly "
                "once when you have enough information."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "observation_markdown": {"type": "string"},
                },
                "required": ["observation_markdown"],
            },
        },
    ]


# ----------------------------------------------------------------------
# Tool execution context
# ----------------------------------------------------------------------

class _LoopContext:
    """Holds the data the tools read from. Built fresh per OBSERVER call."""

    def __init__(
        self,
        *,
        current_frame: list[list[int]],
        previous_frame: Optional[list[list[int]]],
        action_effects: dict,
        objects_summary: str,
        concept_bindings: dict,
        registry: ConceptRegistry,
        episode_meta: dict,
    ):
        self.current_frame = current_frame
        self.previous_frame = previous_frame
        self.action_effects = action_effects or {}
        self.objects_summary = objects_summary
        self.concept_bindings = concept_bindings or {}
        self.registry = registry
        self.episode_meta = episode_meta
        self.recorded_ids: list[str] = []
        self.confirmed_ids: list[str] = []

    # -- tool handlers ------------------------------------------------

    def tool_inspect_region(self, args: dict) -> str:
        r0 = max(0, int(args["r0"]))
        c0 = max(0, int(args["c0"]))
        r1 = min(len(self.current_frame) - 1, int(args["r1"]))
        if not self.current_frame:
            return "(empty frame)"
        c1 = min(len(self.current_frame[0]) - 1, int(args["c1"]))
        if r1 < r0 or c1 < c0:
            return "(empty region)"
        rows = []
        for r in range(r0, r1 + 1):
            row = self.current_frame[r][c0 : c1 + 1]
            rows.append(" ".join(f"{v:2d}" for v in row))
        return f"region rows {r0}..{r1} cols {c0}..{c1}:\n" + "\n".join(rows)

    def tool_list_objects(self, args: dict) -> str:
        return self.objects_summary or "(no objects)"

    def tool_diff_last_frames(self, args: dict) -> str:
        diff = 0
        if self.previous_frame:
            for r in range(min(len(self.current_frame), len(self.previous_frame))):
                cur_row = self.current_frame[r]
                prev_row = self.previous_frame[r]
                for c in range(min(len(cur_row), len(prev_row))):
                    if cur_row[c] != prev_row[c]:
                        diff += 1
        lines = [f"pixel delta vs previous frame: {diff}"]
        if self.action_effects:
            # Use the formatted summary which includes *** ATTRIBUTE CHANGE *** markers
            # to surface orientation/color/shape changes that indicate CHANGER cells.
            try:
                from object_tracker import summarize_action_effects
                lines.append("accumulated action effects this episode:")
                lines.append(summarize_action_effects(self.action_effects))
            except Exception:
                lines.append("accumulated action effects this episode:")
                for act, eff in sorted(self.action_effects.items()):
                    lines.append(f"  {act}: {eff}")
        else:
            lines.append("no action effects recorded yet")
        return "\n".join(lines)

    def tool_recall_concepts(self, args: dict) -> str:
        query = args.get("query") or None
        kind = args.get("kind") or None
        limit = int(args.get("limit", 5))
        hits = self.registry.recall(
            domain=DOMAIN,
            kind=kind,
            name_query=query,
            limit=limit,
            include_cross_domain=True,
            cross_domain_kinds=["compositional", "mechanic"],
        )
        if not hits:
            return "(no concepts found)"
        lines = []
        for c in hits:
            tag = "" if c.domain == DOMAIN else f" [from {c.domain}]"
            lines.append(
                f"- id={c.id} name={c.name} kind={c.kind} "
                f"conf={c.confidence:.2f}{tag}\n"
                f"    trigger: {c.signature.get('trigger', '')}\n"
                f"    summary: {c.abstraction.get('summary', '')}\n"
                f"    hint:    {c.abstraction.get('hint', '')}"
            )
        return "\n".join(lines)

    def tool_record_concept(self, args: dict) -> str:
        kind = args.get("kind", "")
        if kind not in ALLOWED_KINDS:
            return f"error: kind must be one of {sorted(ALLOWED_KINDS)}"
        name = (args.get("name") or "").strip()
        trigger = (args.get("trigger") or "").strip()
        summary = (args.get("summary") or "").strip()
        evidence = (args.get("evidence") or "").strip()
        if not (name and trigger and summary and evidence):
            return "error: name, trigger, summary, evidence are all required"
        try:
            cid = self.registry.record(
                name=name,
                domain=DOMAIN,
                kind=kind,
                signature={"trigger": trigger, "evidence": evidence},
                abstraction={
                    "summary": summary,
                    "hint": (args.get("hint") or "").strip(),
                },
                provenance={
                    "first_seen": dict(self.episode_meta),
                    "evidence_history": [{"evidence": evidence}],
                },
                confidence=float(args.get("confidence", 0.5)),
            )
            self.recorded_ids.append(cid)
            return f"recorded concept id={cid}"
        except (ValueError, TypeError) as e:
            return f"error: {e}"

    def tool_confirm_concept(self, args: dict) -> str:
        cid = args.get("concept_id", "")
        ev = args.get("evidence", "")
        if not cid:
            return "error: concept_id required"
        before = self.registry.get(cid)
        if before is None:
            return f"error: no concept with id {cid}"
        self.registry.confirm(cid, evidence={"evidence": ev}, confidence_delta=0.05)
        self.confirmed_ids.append(cid)
        after = self.registry.get(cid)
        return f"confirmed {cid}; confidence {before.confidence:.2f} -> {after.confidence:.2f}"

    def dispatch(self, name: str, args: dict) -> str:
        handler = {
            "inspect_region":   self.tool_inspect_region,
            "list_objects":     self.tool_list_objects,
            "diff_last_frames": self.tool_diff_last_frames,
            "recall_concepts":  self.tool_recall_concepts,
            "record_concept":   self.tool_record_concept,
            "confirm_concept":  self.tool_confirm_concept,
        }.get(name)
        if handler is None:
            return f"error: unknown tool {name}"
        try:
            return handler(args)
        except Exception as e:  # noqa: BLE001
            return f"error: {type(e).__name__}: {e}"


# ----------------------------------------------------------------------
# System prompt
# ----------------------------------------------------------------------

_SYSTEM_PROMPT = """You are the OBSERVER for an ARC-AGI-3 game-playing system,
running in COMPETITION MODE. You see only the current frame and a small set of
zero-cost tools — there is no game source code, no per-game solver, no labels.

Your job is to produce a structured markdown observation that the MEDIATOR
will use to refine its hypothesis and choose the next action plan. You have
a HARD CAP of 6 tool calls; be efficient.

You will receive a `## Prior hypothesis` section in the user message. Treat
it as the working theory: your job is to REFINE it (confirm, sharpen, or
correct), not throw it away. You will also see `## Recent action outcomes`
showing the last few steps. Lines marked WALL/no-movement mean that action
did nothing — DO NOT plan to repeat it from the same position.

Strategy (follow this order; at most 6 tool calls total):
  TURN 1 (mandatory): recall_concepts with no filters. Cross-domain hits
     marked [from <other>] only apply if their trigger plainly matches.
  TURN 2-3: list_objects and diff_last_frames to characterize the scene
     and the most recent action's effect. Use inspect_region only if you
     need a closer look at a specific area.
  TURN 4 (constant learning — do this most cycles): If you discovered a
     NEW reusable concept (object role, action effect, or mechanic that
     would help solve a DIFFERENT game), call record_concept. If a recalled
     concept clearly matched what just happened, call confirm_concept.
     Skip ONLY if registry already covers the situation and nothing new
     was learned this turn.
  FINAL TURN: call submit_observation exactly once. Your observation_markdown
     MUST contain these sections, in this order:

       ## Current hypothesis
       <one or two sentences refining the prior hypothesis with what you
        now know. State it as a concrete theory of the game, not a list
        of observations.>

       ## Plan (next 1-3 actions)
       <the specific actions you recommend, with reasoning. NEVER include
        an action that just hit a wall from the current position. If the
        plan needs spatial movement, name the direction and approximate
        target.>

       ## Concept bindings
       <one line per binding, format: `color<N>: <role>` — e.g.
        `color9: player`. Omit if nothing confident.>

       ## Evidence
       <bullet list of the concrete observations from this cycle that
        support the hypothesis and plan.>

Constraints:
  - Do NOT speculate about game internals you cannot see.
  - Do NOT record concepts that are tautologies or restate the obvious.
  - Do NOT plan an action you just saw fail (diff <= 4) from the same spot.
  - Be terse. The downstream MEDIATOR is the planner; you advise it.
"""


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------

_REGISTRY_SINGLETON: Optional[ConceptRegistry] = None


def get_registry() -> ConceptRegistry:
    global _REGISTRY_SINGLETON
    if _REGISTRY_SINGLETON is None:
        _REGISTRY_SINGLETON = ConceptRegistry(DEFAULT_REGISTRY_PATH)
    return _REGISTRY_SINGLETON


async def run_observer_agent(
    *,
    current_frame: list[list[int]],
    previous_frame: Optional[list[list[int]]],
    action_effects: dict,
    objects_summary: str,
    concept_bindings: dict,
    episode_meta: dict,
    user_context: str,
    registry: Optional[ConceptRegistry] = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    model: str = "",
    verbose: bool = True,
) -> tuple[str, int, dict]:
    """
    Run the bounded OBSERVER tool loop.

    Returns (observation_markdown, duration_ms, stats).
    `stats` includes turn count, tool-call count, recorded/confirmed concept ids.

    Caller is responsible for setting `episode_meta` to something like
        {"env_id": "ls20", "episode": 3, "level": 1, "step": 42}
    so any newly-recorded concepts have proper provenance.
    """
    import anthropic  # local import: keep core dependency optional at import time
    from core.pipeline.agents import DEFAULT_MODEL, get_client, _cost_tracker  # type: ignore

    reg = registry or get_registry()

    # ------------------------------------------------------------------
    # Optional CLI backend path (KF_USE_CLAUDE_CLI=1)
    # ------------------------------------------------------------------
    # When the Claude Code CLI is used as the backend, we cannot give the
    # model access to our Python tool functions (inspect_region, etc.).
    # Instead we inline the observable data into a single prompt and ask
    # for the same markdown contract as the tool loop. Any concept
    # record/confirm/recall is done programmatically here from the
    # caller-supplied context (no agent-side registry writes in this path).
    try:
        from core.pipeline import claude_cli as _cli  # type: ignore
    except Exception:
        _cli = None  # type: ignore
    if _cli is not None and _cli.is_enabled():
        t0 = time.time()
        # Pre-recall concepts so the CLI OBSERVER sees what's been learned.
        _hits = reg.recall(
            domain=DOMAIN, limit=8, include_cross_domain=True,
            cross_domain_kinds=list(ALLOWED_KINDS),
        )
        _recall_lines = []
        if _hits:
            _recall_lines.append("## Recalled concepts")
            for _c in _hits:
                _tag = "" if _c.domain == DOMAIN else f" [from {_c.domain}]"
                _recall_lines.append(
                    f"- **{_c.name}** ({_c.kind}, conf={_c.confidence:.2f}){_tag}: "
                    f"{_c.abstraction.get('summary','')}"
                )
        # Compact scene description.
        _h = len(current_frame)
        _w = len(current_frame[0]) if current_frame else 0
        _ae_lines = []
        for _a, _info in (action_effects or {}).items():
            _ae_lines.append(f"- {_a}: {_info}")
        _scene = [
            f"## Frame",
            f"{_h}x{_w} grid.",
            f"",
            f"## Objects summary",
            objects_summary or "(none)",
            f"",
            f"## Action effects observed so far",
            ("\n".join(_ae_lines) or "(none)"),
            f"",
            f"## Current concept bindings",
            (str(concept_bindings) if concept_bindings else "(none)"),
            f"",
        ]
        if _recall_lines:
            _scene.append("\n".join(_recall_lines))
            _scene.append("")
        _scene.append(user_context)
        _cli_user = "\n".join(_scene)
        try:
            _text, _dur = await _cli.call_via_cli(
                "OBSERVER", _cli_user, _SYSTEM_PROMPT,
                model=(model or DEFAULT_AGENT_MODEL or DEFAULT_MODEL),
            )
        except Exception as _e:
            if verbose:
                print(f"  [agent_loop:cli] failed: {_e} — falling back to SDK")
            _text = ""
        if _text:
            dur_ms = int((time.time() - t0) * 1000)
            stats = {
                "turns": 1, "tool_calls": 0,
                "recorded": [], "confirmed": [],
                "backend": "claude_cli",
            }
            return _text, dur_ms, stats
        # else: fall through to SDK path
    ctx = _LoopContext(
        current_frame=current_frame,
        previous_frame=previous_frame,
        action_effects=action_effects,
        objects_summary=objects_summary,
        concept_bindings=concept_bindings,
        registry=reg,
        episode_meta=episode_meta,
    )

    client = get_client()
    use_model = model or DEFAULT_AGENT_MODEL or DEFAULT_MODEL
    tools = _tool_specs()
    messages: list[dict] = [{"role": "user", "content": user_context}]

    final_text = ""
    turns = 0
    tool_calls = 0
    t0 = time.time()
    truncated = False

    while turns < max_turns:
        turns += 1
        try:
            resp = await client.messages.create(
                model=use_model,
                max_tokens=max_tokens,
                system=_SYSTEM_PROMPT,
                tools=tools,
                messages=messages,
            )
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  [agent_loop] model call failed on turn {turns}: {e}")
            break

        if resp.usage:
            u = resp.usage
            try:
                _cost_tracker.add(
                    u.input_tokens, u.output_tokens,
                    cache_creation=getattr(u, "cache_creation_input_tokens", 0) or 0,
                    cache_read=getattr(u, "cache_read_input_tokens", 0) or 0,
                )
            except Exception:
                pass

        # Append assistant message verbatim (model expects round-trip).
        assistant_blocks = [b.model_dump() if hasattr(b, "model_dump") else dict(b)
                            for b in resp.content]
        messages.append({"role": "assistant", "content": assistant_blocks})

        stop_reason = resp.stop_reason
        if stop_reason == "tool_use":
            tool_results: list[dict] = []
            submitted = False
            for block in resp.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                tool_calls += 1
                name = block.name
                args = block.input or {}
                if name == "submit_observation":
                    final_text = str(args.get("observation_markdown") or "").strip()
                    submitted = True
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "ok",
                    })
                    if verbose:
                        print(f"    [tool t{turns}] submit_observation ({len(final_text)} chars)")
                else:
                    result = ctx.dispatch(name, args)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
                    if verbose:
                        snippet = str(result).replace("\n", " ")[:90]
                        arg_summary = ""
                        if name == "recall_concepts":
                            arg_summary = f" q={args.get('query','')!r} kind={args.get('kind','')!r}"
                        elif name == "record_concept":
                            arg_summary = f" name={args.get('name','')!r} kind={args.get('kind','')!r}"
                        elif name == "confirm_concept":
                            arg_summary = f" id={args.get('concept_id','')!r}"
                        elif name == "inspect_region":
                            arg_summary = f" r={args.get('r0','?')}..{args.get('r1','?')} c={args.get('c0','?')}..{args.get('c1','?')}"
                        print(f"    [tool t{turns}] {name}{arg_summary} -> {snippet}")
            messages.append({"role": "user", "content": tool_results})
            if submitted:
                break
        else:
            # end_turn or max_tokens — pull any plain text and stop.
            for block in resp.content:
                if getattr(block, "type", None) == "text":
                    final_text = (final_text + "\n" + block.text).strip()
            break
    else:
        truncated = True

    if not final_text:
        # Fallback: synthesize a minimal observation so the MEDIATOR isn't blank.
        final_text = (
            "## Current state\n\n(agent loop produced no observation; "
            "MEDIATOR should rely on structural context)\n"
        )

    duration_ms = int((time.time() - t0) * 1000)
    stats = {
        "turns": turns,
        "tool_calls": tool_calls,
        "recorded": list(ctx.recorded_ids),
        "confirmed": list(ctx.confirmed_ids),
        "truncated": truncated,
    }
    if verbose:
        print(
            f"  [agent_loop] OBSERVER done in {duration_ms}ms; "
            f"turns={turns} tool_calls={tool_calls} "
            f"recorded={len(ctx.recorded_ids)} confirmed={len(ctx.confirmed_ids)}"
        )
    return final_text, duration_ms, stats
