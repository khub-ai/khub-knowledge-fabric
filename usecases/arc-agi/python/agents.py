"""
agents.py — Async Anthropic API calls for each ensemble agent.

Loads system prompts from the prompts/ directory next to this package,
injects prior knowledge and task context, and returns structured responses.
"""

from __future__ import annotations
import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import anthropic

from grid_tools import Grid, grid_to_str, summarize
from typing import TYPE_CHECKING
from metadata import SolverEntry, MediatorDecision, extract_json_grid

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

PROMPT_FILES = {
    "SOLVER":             PROMPTS_DIR / "solver.md",
    "SOLVER-SPATIAL":     PROMPTS_DIR / "solver-spatial.md",
    "SOLVER-PROCEDURAL":  PROMPTS_DIR / "solver-procedural.md",
    "SOLVER-ANALOGICAL":  PROMPTS_DIR / "solver-analogical.md",
    "MEDIATOR":           PROMPTS_DIR / "mediator.md",
}

# Default solver set — single solver for efficiency.
# Swap in multiple specialist solvers for hard puzzles:
#   DEFAULT_SOLVERS = ["SOLVER-SPATIAL", "SOLVER-PROCEDURAL", "SOLVER-ANALOGICAL"]
DEFAULT_SOLVERS: list[str] = ["SOLVER"]

_prompt_cache: dict[str, str] = {}

def load_prompt(agent_id: str) -> str:
    if agent_id not in _prompt_cache:
        path = PROMPT_FILES[agent_id]
        _prompt_cache[agent_id] = path.read_text(encoding="utf-8")
    return _prompt_cache[agent_id]


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
# Task formatting
# ---------------------------------------------------------------------------

def format_task_for_prompt(task: dict) -> str:
    """Render a task's train/test pairs as readable text for injection into prompts."""
    lines = ["## Task\n"]
    for i, pair in enumerate(task.get("train", []), 1):
        lines.append(f"### Demo pair {i}")
        lines.append("**Input:**")
        lines.append(grid_to_str(pair["input"]))
        lines.append("**Output:**")
        lines.append(grid_to_str(pair["output"]))
        lines.append(f"*Shape: {summarize(pair['input'])} -> {summarize(pair['output'])}*\n")
    for i, t in enumerate(task.get("test", []), 1):
        lines.append(f"### Test input {i}")
        lines.append(grid_to_str(t["input"]))
        inp = t["input"]
        lines.append(f"*Shape: {summarize(inp)}*\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core call
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 4096

# Set to True by harness/ensemble to print prompts before each call
SHOW_PROMPTS: bool = False

# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------

# Sonnet 4.6 pricing (USD per token)
# Verify against https://www.anthropic.com/pricing if estimates diverge from Anthropic invoices.
# Anthropic auto-caches repeated system prompts; cache_creation costs 25% more, cache_read 90% less.
_PRICE_INPUT_PER_TOKEN          = 3.00  / 1_000_000   # standard input
_PRICE_CACHE_CREATION_PER_TOKEN = 3.75  / 1_000_000   # cache write (25% surcharge)
_PRICE_CACHE_READ_PER_TOKEN     = 0.30  / 1_000_000   # cache hit  (90% discount)
_PRICE_OUTPUT_PER_TOKEN         = 15.00 / 1_000_000   # output


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


async def call_agent(
    agent_id: str,
    user_message: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_retries: int = 5,
) -> tuple[str, int]:
    """
    Call an agent with its system prompt + a user message.
    Returns (response_text, duration_ms).
    Retries on 529 overloaded errors with exponential backoff.
    """
    system_prompt = load_prompt(agent_id)

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
                wait = 60 * (attempt + 1)  # 60s, 120s, 180s, 240s
                print(f"  [rate-limit] {agent_id} retry {attempt+1}/{max_retries-1} in {wait}s...")
                await asyncio.sleep(wait)
            else:
                raise
        except anthropic.APIStatusError as e:
            if e.status_code == 529 and attempt < max_retries - 1:
                wait = 2 ** attempt  # 1s, 2s, 4s, 8s, 16s
                print(f"  [overloaded] {agent_id} retry {attempt+1}/{max_retries-1} in {wait}s...")
                await asyncio.sleep(wait)
            else:
                raise


def _print_prompt(agent_id: str, system: str, user: str, model: str) -> None:
    """Print agent prompt to terminal (used when SHOW_PROMPTS is True)."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text
        c = Console()
        sys_preview = system[:600] + ("..." if len(system) > 600 else "")
        usr_preview = user[:1200] + ("..." if len(user) > 1200 else "")
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
        print(f"USER:   {user[:800]}")


# ---------------------------------------------------------------------------
# Solver response parsing (text-only — no grid extraction)
# ---------------------------------------------------------------------------

def extract_solver_hypothesis(text: str) -> dict:
    """
    Extract hypothesis fields from a solver's text-only response.
    Returns dict with: rule, confidence, reasoning, suggested_tools, suggested_steps, category.
    """
    block_re = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
    result = {
        "rule": "",
        "confidence": "medium",
        "reasoning": "",
        "suggested_tools": [],
        "suggested_steps": [],
        "category": "",
    }
    for raw in block_re.findall(text):
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and "rule" in obj:
                result["rule"] = obj.get("rule", "")
                result["confidence"] = obj.get("confidence", "medium")
                result["reasoning"] = obj.get("reasoning", "")
                result["suggested_tools"] = obj.get("suggested_tools", [])
                result["suggested_steps"] = obj.get("suggested_steps", [])
                result["category"] = obj.get("category", "")
                break
        except (json.JSONDecodeError, Exception):
            continue
    # Fallback: use entire response as rule if no JSON found
    if not result["rule"]:
        result["rule"] = text[:500]
    return result


# ---------------------------------------------------------------------------
# Round 1 — Solver initial hypotheses (parallel, text-only)
# ---------------------------------------------------------------------------

async def run_solvers_round1(
    task: dict,
    prior_knowledge: str = "",
    human_hypothesis: str = "",
    solver_ids: list[str] | None = None,
    preference_priors: str = "",
) -> list[SolverEntry]:
    """Run solvers in parallel for Round 1 (text-only hypotheses).

    solver_ids defaults to DEFAULT_SOLVERS. Pass a different list to run
    multiple specialist solvers (e.g. for hard puzzles).

    preference_priors: optional soft-prior section from preference rules.
    These are injected separately from task-specific prior_knowledge so the
    solver can distinguish learned preferences from matched task rules.
    """
    if solver_ids is None:
        solver_ids = DEFAULT_SOLVERS

    task_text = format_task_for_prompt(task)
    knowledge_section = (
        f"\n## Prior Knowledge\n{prior_knowledge}\n" if prior_knowledge.strip() else ""
    )
    preference_section = (
        f"\n{preference_priors}\n" if preference_priors.strip() else ""
    )
    human_section = (
        f"\n## Human Hypothesis\n{human_hypothesis}\n"
        "(A human member of the ensemble has offered this observation -- "
        "consider it, but form your own independent analysis.)\n"
        if human_hypothesis.strip() else ""
    )
    user_msg = f"{knowledge_section}{preference_section}{human_section}\n{task_text}\n\nPlease analyze this task and propose your transformation rule."

    async def run_one(agent_id: str) -> SolverEntry:
        text, ms = await call_agent(agent_id, user_msg)
        hyp = extract_solver_hypothesis(text)
        return SolverEntry(
            agent=agent_id,
            round=1,
            rule=hyp["rule"],
            confidence=hyp["confidence"],
            grid=None,  # text-only — no grid
            raw_response=text,
            duration_ms=ms,
        )

    results = await asyncio.gather(*[run_one(sid) for sid in solver_ids])
    return list(results)


# ---------------------------------------------------------------------------
# Round 2 — MEDIATOR synthesizes pseudo-code
# ---------------------------------------------------------------------------

async def run_mediator_synthesize(
    task: dict,
    solver_entries: list[SolverEntry],
    prior_knowledge: str = "",
    human_insight: str = "",
    rule_section: str = "",
    tool_section: str = "",
) -> tuple[str, list[dict], int]:
    """
    Ask MEDIATOR to synthesize solver hypotheses into pseudo-code.
    Returns (raw_response, pseudocode_steps, duration_ms).
    """
    task_text = format_task_for_prompt(task)

    proposals = []
    for e in solver_entries:
        proposals.append(
            f"### {e.agent} (confidence: {e.confidence})\n"
            f"Rule: {e.rule}\n"
            f"Full reasoning:\n{e.raw_response[:1500]}"
        )

    knowledge_section = (
        f"\n## Applicable Rules\n{prior_knowledge}\n" if prior_knowledge.strip() else ""
    )
    human_section = (
        f"\n## Human Insight\n{human_insight}\n" if human_insight.strip() else ""
    )
    rule_mgmt_section = f"\n{rule_section}\n" if rule_section.strip() else ""
    tool_avail_section = f"\n{tool_section}\n" if tool_section.strip() else ""

    user_msg = (
        f"{knowledge_section}{human_section}"
        f"{task_text}\n\n"
        "## Solver Hypotheses\n\n"
        + "\n\n".join(proposals)
        + "\n\nPlease synthesize these hypotheses into a pseudo-code sequence of tool calls "
        "that the EXECUTOR can run against the demo pairs."
        + tool_avail_section
        + rule_mgmt_section
    )

    text, ms = await call_agent("MEDIATOR", user_msg)

    # Parse pseudo-code from response
    from executor import parse_pseudocode
    steps = parse_pseudocode(text)

    return text, steps, ms


# ---------------------------------------------------------------------------
# Round 3+ — MEDIATOR revises pseudo-code after execution failure
# ---------------------------------------------------------------------------

async def run_mediator_revise(
    task: dict,
    solver_entries: list[SolverEntry],
    previous_pseudocode: list[dict],
    execution_trace: str,
    human_insight: str = "",
    failed_tools: list[str] | None = None,
) -> tuple[str, list[dict], int]:
    """
    Ask MEDIATOR to revise pseudo-code based on execution failure.
    Returns (raw_response, revised_steps, duration_ms).
    failed_tools: tool names that have already failed across previous revisions.
    """
    task_text = format_task_for_prompt(task)

    proposals = []
    for e in solver_entries:
        proposals.append(f"### {e.agent}: {e.rule[:200]}")

    prev_code = json.dumps(previous_pseudocode, indent=2)

    user_msg = (
        f"{task_text}\n\n"
        "## Solver Hypotheses (for reference)\n"
        + "\n".join(proposals)
        + f"\n\n## Previous pseudo-code (FAILED)\n```json\n{prev_code}\n```\n\n"
        f"## Execution Trace\n{execution_trace}\n\n"
    )

    if failed_tools:
        user_msg += (
            "## Tools that have already failed across previous revisions\n"
            + "\n".join(f"- `{t}`" for t in failed_tools)
            + "\n\nDo NOT reuse any of these tools. Try a different decomposition "
            "or request new tools with more precise behavior descriptions.\n\n"
        )

    if human_insight:
        user_msg += f"## Human Insight\n{human_insight}\n\n"

    user_msg += (
        "The pseudo-code failed on one or more demo pairs. "
        "Please analyze the execution trace, identify what went wrong, "
        "and produce a REVISED pseudo-code sequence.\n\n"
        "**Before revising, consider:**\n"
        "- Does the current approach assume all sources/groups are processed "
        "simultaneously? If so, test whether a *sequential* interpretation "
        "(one group processed first, its filled cells acting as barriers for "
        "subsequent groups) produces different results on the failing cells. "
        "Look for contested cells — output cells reachable from multiple sources "
        "— and check which source's color appears there.\n"
        "- If the same tool has failed twice, do NOT reuse it. Try a fundamentally "
        "different decomposition.\n"
        "- If the approach uses a directional rule (e.g., 'shoot toward nearest edge', "
        "'align with top'), consider whether the direction is measured in the *grid* "
        "reference frame (absolute row/col) or the *shape-local* reference frame "
        "(where the marker sits within its enclosing shape — top/bottom/left/right "
        "extremum of the shape). These produce different results when the shape is not "
        "near the edge the marker points toward. Verify which interpretation matches "
        "all demo pairs.\n"
        "- **Honor the transition census.** Only cells with a specific input value change. "
        "If the solver says only 5→2 transitions occur (for example), then your tool must "
        "target cells with value 5, NOT cells with value 0 (background). Passing "
        "`background=0` to a fill tool will change 0-cells, not 5-cells. Use the correct "
        "target value in tool parameters."
    )

    text, ms = await call_agent("MEDIATOR", user_msg)

    from executor import parse_pseudocode
    steps = parse_pseudocode(text)

    return text, steps, ms


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Rule generalization pass — called after a successful run with new task rules
# ---------------------------------------------------------------------------

async def run_generalization_pass(
    task_id: str,
    new_rule_ids: list[str],
    new_rules: list[dict],
    existing_rules_summary: str,
) -> str:
    """
    After a successful run, ask MEDIATOR to propose generalized variants of the
    newly created task rules.

    Generalized rules start as 'candidate' status — they are included in Round 0
    matching but labeled as unconfirmed. A candidate is promoted to 'active' on
    its first independent success on a DIFFERENT task, and deprecated after 1
    failure on a different task.

    This allows the system to speculatively extend knowledge without corrupting the
    rule base with unconfirmed generalizations.

    Args:
        task_id:              The task that just succeeded.
        new_rule_ids:         IDs of rules created/updated during this run.
        new_rules:            Full rule dicts for context.
        existing_rules_summary: Formatted existing rules (for dedup checking).

    Returns:
        Raw MEDIATOR response text (caller parses rule_updates from it).
    """
    if not new_rules:
        return ""

    rules_section = "\n".join(
        f"- [{r['id']}]\n  CONDITION: {r['condition']}\n  ACTION: {r['action']}"
        for r in new_rules
    )

    user_msg = (
        f"## Generalization request for task: {task_id}\n\n"
        f"The following rule(s) were just created after a successful solve:\n\n"
        f"{rules_section}\n\n"
        f"## Existing rules (for dedup — do NOT recreate these)\n\n"
        f"{existing_rules_summary}\n\n"
        "## Your task\n\n"
        "For each new rule above, propose **one generalized variant** with a broader "
        "condition that would match a wider family of puzzles with the same underlying "
        "transformation logic.\n\n"
        "A good generalization:\n"
        "- Removes constraints that are specific to this one task (specific colors, "
        "  exact counts, specific grid sizes, specific positions)\n"
        "- Preserves the essential structural property that determines WHEN the "
        "  transformation applies\n"
        "- Keeps the same action (same tool/approach), but makes the condition trigger "
        "  on more puzzle variations\n"
        "- Identifies what DIMENSION of variation is being generalized "
        "  (e.g. 'works for any number of sequences, not just 2')\n\n"
        "Consider these generalization dimensions:\n"
        "  1. **Count**: relax 'exactly N' to 'at least N' or 'N or more'\n"
        "  2. **Role assignment**: relax 'longest = spine' to 'some criterion = spine'\n"
        "  3. **Orientation**: relax 'horizontal or vertical' to 'any linear arrangement'\n"
        "  4. **Position**: relax 'at edge/corner' to 'anywhere in grid'\n"
        "  5. **Color**: relax specific color values to 'any non-background colors'\n\n"
        "Only propose a generalization if you are confident it describes a real broader "
        "pattern class. If the rule is already maximally general, say so and omit it.\n\n"
        "Generalized rules MUST use `action: \"generalize\"` with the parent rule's ID.\n"
        "They will be stored as CANDIDATE rules (unconfirmed) and only promoted to active "
        "after succeeding independently on a different task.\n\n"
        "```json\n"
        '{"rule_updates": [\n'
        '  {\n'
        '    "action": "generalize",\n'
        '    "parent_id": "r_NNN",\n'
        '    "condition": "[category] Broader condition...",\n'
        '    "rule_action": "Same action as parent...",\n'
        '    "reason": "What dimension was generalized and why",\n'
        '    "tags": ["...relevant-tags..."]\n'
        '  }\n'
        "]}\n"
        "```\n"
        "Omit the block entirely if no useful generalizations can be made."
    )

    text, _ms = await call_agent("MEDIATOR", user_msg, max_tokens=1024)
    return text


# ---------------------------------------------------------------------------
# Preference rule extraction — called when insight + success
# ---------------------------------------------------------------------------

async def run_mediator_extract_preference(
    task_id: str,
    wrong_hypotheses: list[str],
    human_insight: str,
    correct_approach: str,
    existing_preference_rules: str = "",
) -> str:
    """
    After a correction event (wrong hypothesis → human insight → success), ask
    MEDIATOR to extract a preference rule capturing *what to prefer in future*.

    A preference rule is NOT about how to solve a specific puzzle type.
    It encodes a *reasoning bias*: given ambiguous evidence, prefer hypothesis
    properties that correlate with correct human-natural interpretations.

    Args:
        task_id:           The task that was corrected.
        wrong_hypotheses:  Solver hypotheses that were initially wrong.
        human_insight:     The corrective hint provided by the human.
        correct_approach:  The transformation approach that ultimately worked.
        existing_preference_rules: Current preference rules (to avoid duplication).

    Returns:
        Raw MEDIATOR response text (caller parses rule_updates from it).
    """
    existing_section = (
        f"\n## Existing preference rules (avoid duplicating these)\n{existing_preference_rules}\n"
        if existing_preference_rules.strip() else ""
    )

    wrong_section = (
        "### What the solver initially proposed (from a prior FAILED run — this is what was wrong):\n"
        + "\n".join(f"- {h}" for h in wrong_hypotheses)
        if wrong_hypotheses else
        "### Prior failed hypotheses: not available for this task."
    )

    user_msg = (
        f"## Correction event on task: {task_id}\n\n"
        f"{wrong_section}\n\n"
        f"### Human insight that corrected it:\n{human_insight}\n\n"
        f"### Approach that ultimately succeeded:\n{correct_approach}\n"
        f"{existing_section}\n"
        "## Your task\n\n"
        "Extract a **preference rule** from this correction event.\n\n"
        "**Primary focus — reason backwards from the insight:**\n"
        "The human insight tells you what the solver SHOULD have used. "
        "Ask: what property was the solver likely preferring INSTEAD? "
        "For example, if the insight says 'topological hole count', the solver was "
        "probably choosing a simpler metric like pixel count or bounding-box area "
        "because those are computationally easier but less human-natural. "
        "The preference rule should name the RIGHT property (from the insight) and the "
        "WRONG property (what the solver defaults to) and explain why the right one "
        "is more human-natural.\n\n"
        "If prior failed hypotheses ARE provided above, use them as direct evidence "
        "of what was wrong. If they are not available, infer the wrong property from "
        "the insight and from common solver failure modes.\n\n"
        "Key properties of a good preference rule:\n"
        "- It names the property to prefer (from the insight) vs the property to "
        "  de-prioritize (what the solver defaults to, e.g. pixel count, bounding box area)\n"
        "- It explains *why* the preferred property is more human-natural\n"
        "- It is general enough to transfer to other puzzles with the same ambiguity\n"
        "- It is falsifiable: future puzzles could provide counter-evidence\n"
        "- The `condition` field: describe the situation where this bias applies\n"
        "- The `rule_action` field: 'prefer X over Y because humans perceive X first/more reliably'\n\n"
        "Emit a rule_updates JSON block with `rule_type: \"preference\"`:\n"
        "```json\n"
        '{"rule_updates": [\n'
        '  {\n'
        '    "action": "new",\n'
        '    "rule_type": "preference",\n'
        '    "condition": "[preference] When ...",\n'
        '    "rule_action": "Prefer ... over ... because ...",\n'
        '    "tags": ["preference", "...relevant-category..."]\n'
        '  }\n'
        "]}\n"
        "```\n"
        "Omit the block if no generalizable preference can be extracted "
        "(e.g. the correction was purely task-specific and unlikely to transfer)."
    )

    text, _ms = await call_agent("MEDIATOR", user_msg, max_tokens=1024)
    return text


# ---------------------------------------------------------------------------
# Tool generator — Claude writes Python code for new tools on demand
# ---------------------------------------------------------------------------

_TOOL_GENERATOR_SYSTEM = """You are a Python code generator for ARC-AGI grid transformation tools.

Write a single Python function that implements the requested grid transformation.

Requirements:
- Signature: def {name}(grid, **kwargs) -> list
  where `grid` is list[list[int]] (0 = background color)
- Return a NEW 2D list of ints — never modify the input in-place
- Must be deterministic and handle edge cases (empty grid, single cell) gracefully
- No docstring, no imports at module level — just the function body
- CRITICAL: NEVER hardcode color values (like 6, 8, 3) or grid positions you observed in a specific example. Colors and key positions vary per input — always compute them dynamically from the grid.
- CRITICAL: When the behavior mentions blocks separated by a separator value: find separator row/col POSITIONS first by scanning (e.g., rows where all values == separator), then compute band_starts as the rows/cols between separators. Do NOT assume uniform spacing like br*block_size — separators create irregular offsets (e.g., seps at rows 3,7,11,15,19 → band_starts=[0,4,8,12,16,20], NOT [0,3,6,9,12,15]).

Available in scope (no need to import):
- `np` / `numpy`: NumPy
- `to_np(grid)` → numpy 2D int array;  `to_list(arr)` → list[list[int]]
- `flood_fill(grid, start_row, start_col, fill_color)` → new grid
- `replace_color(grid, from_color, to_color)` → new grid
- `unique_colors(grid)` → set of ints present in grid
- `color_count(grid, color)` → int count of cells with that color
- `bounding_box(grid, color)` → (r_min, c_min, r_max, c_max) or None
- `count_connected_components(grid, color)` → int
- `grids_equal(a, b)` → bool

Return ONLY the function code inside a ```python block. No explanation outside the block."""


def _format_demo_examples(task: dict, max_demos: int = 3) -> str:
    """Format demo pairs as concrete input/output examples for the tool generator.
    Also includes the test input (no expected output) so the generator can see
    any structural variations not present in the demos."""
    lines = ["## Concrete examples your function MUST pass (verified by the executor):"]
    for i, pair in enumerate(task.get("train", [])[:max_demos], 1):
        lines.append(f"\n### Example {i}")
        lines.append("Input grid:")
        lines.append(grid_to_str(pair["input"]))
        lines.append("Expected output grid:")
        lines.append(grid_to_str(pair["output"]))
    for i, t in enumerate(task.get("test", []), 1):
        lines.append(f"\n### Test input {i} (expected output unknown — your function MUST handle this input correctly)")
        lines.append("Input grid:")
        lines.append(grid_to_str(t["input"]))
        lines.append("*(No expected output shown — but note any structural differences from the demo examples above, such as opposite direction, mirrored orientation, or objects on the other side of a divider. Ensure your implementation handles all such variants.)*")
    return "\n".join(lines)


async def run_tool_generator(tool_spec: dict, task: dict | None = None) -> tuple[str, int]:
    """
    Ask Claude to generate Python code for a new grid transformation tool.
    Returns (python_code_str, duration_ms).
    If task is provided, demo input/output pairs are included so the generator
    has concrete examples to write against.
    """
    name = tool_spec.get("name", "unnamed_tool")
    system = _TOOL_GENERATOR_SYSTEM.replace("{name}", name)

    args_desc = json.dumps(tool_spec.get("args", {}), indent=2)
    behavior = tool_spec.get("behavior", "")
    description = tool_spec.get("description", "")

    examples_section = f"\n\n{_format_demo_examples(task)}" if task else ""

    user_msg = (
        f"Tool name: `{name}`\n"
        f"Description: {description}\n"
        f"Arguments:\n{args_desc}\n\n"
        f"Behavior:\n{behavior}"
        f"{examples_section}\n\n"
        f"Write `def {name}(grid, **kwargs)` implementing the above."
    )

    client = get_client()
    t0 = time.time()
    for attempt in range(5):
        try:
            response = await client.messages.create(
                model=DEFAULT_MODEL,
                max_tokens=2048,
                system=system,
                messages=[{"role": "user", "content": user_msg}],
            )
            break
        except anthropic.RateLimitError:
            if attempt < 4:
                wait = 60 * (attempt + 1)
                print(f"  [rate-limit] tool-gen retry {attempt+1}/4 in {wait}s...")
                await asyncio.sleep(wait)
            else:
                raise
        except anthropic.APIStatusError as e:
            if e.status_code == 529 and attempt < 4:
                await asyncio.sleep(2 ** attempt)
            else:
                raise

    duration_ms = int((time.time() - t0) * 1000)
    raw = response.content[0].text if response.content else ""
    if response.usage:
        u = response.usage
        _cost_tracker.add(
            u.input_tokens, u.output_tokens,
            cache_creation=getattr(u, "cache_creation_input_tokens", 0) or 0,
            cache_read=getattr(u, "cache_read_input_tokens", 0) or 0,
        )

    # Extract code from ```python block
    match = re.search(r"```python\s*(.*?)\s*```", raw, re.DOTALL)
    code = match.group(1).strip() if match else raw.strip()

    return code, duration_ms


_TOOL_FIX_SYSTEM = """You are fixing a Python grid transformation function that is producing wrong output.

You will be given:
1. The original behavior specification
2. The current (buggy) function code
3. One or more execution traces showing exactly which cells are wrong (multiple traces = multiple failed attempts)

Your job: return a corrected version of the function.

Requirements (same as before):
- Signature: def {name}(grid, **kwargs) -> list
- Return a NEW 2D list of ints — never modify in-place
- Deterministic, handles edge cases gracefully

Available in scope (no need to import):
- `np` / `numpy`: NumPy
- `to_np(grid)` → numpy 2D int array;  `to_list(arr)` → list[list[int]]
- `flood_fill(grid, start_row, start_col, fill_color)` → new grid
- `replace_color(grid, from_color, to_color)` → new grid
- `unique_colors(grid)` → set of ints present in grid
- `color_count(grid, color)` → int count of cells with that color
- `bounding_box(grid, color)` → (r_min, c_min, r_max, c_max) or None
- `count_connected_components(grid, color)` → int

Study ALL traces before fixing — the same bug may manifest differently across demos.
Return ONLY the corrected function code inside a ```python block. No explanation outside the block."""


async def run_tool_generator_fix(
    tool_spec: dict,
    buggy_code: str,
    trace: str,
    task: dict | None = None,
) -> tuple[str, int]:
    """
    Ask Claude to fix a previously generated tool that failed verification.
    Returns (corrected_python_code, duration_ms).
    """
    name = tool_spec.get("name", "unnamed_tool")
    system = _TOOL_FIX_SYSTEM.replace("{name}", name)

    behavior = tool_spec.get("behavior", "")
    description = tool_spec.get("description", "")
    examples_section = f"\n\n{_format_demo_examples(task)}" if task else ""

    rationale = tool_spec.get("rationale", "")
    rationale_section = f"\n\n## MEDIATOR rationale (why this tool was requested)\n{rationale}" if rationale else ""

    user_msg = (
        f"Tool name: `{name}`\n"
        f"Description: {description}\n\n"
        f"Original behavior spec:\n{behavior}"
        f"{rationale_section}"
        f"{examples_section}\n\n"
        f"## Buggy code\n```python\n{buggy_code}\n```\n\n"
        f"## Execution trace (showing what went wrong)\n{trace}\n\n"
        f"Fix `def {name}(grid, **kwargs)` so it passes all examples above."
    )

    client = get_client()
    t0 = time.time()
    for attempt in range(5):
        try:
            response = await client.messages.create(
                model=DEFAULT_MODEL,
                max_tokens=2048,
                system=system,
                messages=[{"role": "user", "content": user_msg}],
            )
            break
        except anthropic.RateLimitError:
            if attempt < 4:
                wait = 60 * (attempt + 1)
                print(f"  [rate-limit] tool-fix retry {attempt+1}/4 in {wait}s...")
                await asyncio.sleep(wait)
            else:
                raise
        except anthropic.APIStatusError as e:
            if e.status_code == 529 and attempt < 4:
                await asyncio.sleep(2 ** attempt)
            else:
                raise

    duration_ms = int((time.time() - t0) * 1000)
    raw = response.content[0].text if response.content else ""
    if response.usage:
        u = response.usage
        _cost_tracker.add(
            u.input_tokens, u.output_tokens,
            cache_creation=getattr(u, "cache_creation_input_tokens", 0) or 0,
            cache_read=getattr(u, "cache_read_input_tokens", 0) or 0,
        )

    match = re.search(r"```python\s*(.*?)\s*```", raw, re.DOTALL)
    code = match.group(1).strip() if match else raw.strip()

    return code, duration_ms
