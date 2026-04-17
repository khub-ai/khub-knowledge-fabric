"""
harness.py — CLI test runner for the ARC-AGI-3 ensemble.

Usage:
  python harness.py                            # 3 episodes of ls20, no render
  python harness.py --env ls20 --episodes 5
  python harness.py --env ls20 --episodes 1 --render terminal-fast
  python harness.py --env ls20 --max-steps 300 --max-cycles 50
  python harness.py --env ls20 --prompts      # print full LLM prompts

Runs the OBSERVER/MEDIATOR/ACTOR ensemble on an ARC-AGI-3 environment.
Requires the 'arc' conda environment:
  conda run -n arc python harness.py --env ls20
"""

from __future__ import annotations
import argparse
import asyncio
import datetime
import io
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Force UTF-8 stdout/stderr on Windows
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ---------------------------------------------------------------------------
# Resolve paths before local imports
# ---------------------------------------------------------------------------
_HERE    = Path(__file__).parent
_KF_ROOT = _HERE.resolve().parents[3]
for _p in (str(_KF_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Enable persistent LLM disk cache before importing agents (which reads the env
# var at module load time).  Cached responses survive between runs, cutting
# re-run costs to near zero for identical prompts.
_LLM_CACHE_DIR = _HERE.parent / ".tmp" / "llm_cache"
_LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("LLM_CACHE_DIR", str(_LLM_CACHE_DIR))

import arc_agi
import ensemble as ens
from ensemble import run_episode, EpisodeMetadata, MAX_STEPS, MAX_CYCLES
from rules import RuleEngine
from tools import ToolRegistry
import agents
from agents import DEFAULT_MODEL

console = Console()

_TMP = Path(__file__).parent.parent / ".tmp"
DEFAULT_OUTPUT = str(_TMP / "results.json")


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _save_results(
    output_path: Path,
    episodes: list[EpisodeMetadata],
    env_id: str,
    model: str,
    rules: RuleEngine,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total       = len(episodes)
    best_levels = max((e.levels_completed for e in episodes), default=0)
    won_count   = sum(1 for e in episodes if e.won)
    total_cost  = sum(e.cost_usd for e in episodes)
    total_steps = sum(e.steps_taken for e in episodes)

    payload = {
        "summary": {
            "env_id":          env_id,
            "model":           model,
            "episodes":        total,
            "won":             won_count,
            "best_levels":     best_levels,
            "total_steps":     total_steps,
            "total_cost_usd":  round(total_cost, 6),
            "avg_cost_usd":    round(total_cost / max(total, 1), 6),
            "timestamp":       datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "rules":           rules.stats_summary(),
        },
        "episodes": [asdict(e) for e in episodes],
    }
    tmp = output_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, output_path)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ARC-AGI-3 Ensemble Test Harness")
    p.add_argument("--env",      default="ls20",
                   help="ARC-AGI-3 environment ID (default: ls20)")
    p.add_argument("--episodes", type=int, default=3,
                   help="Number of episodes to run (default: 3)")
    p.add_argument("--max-steps",  dest="max_steps",  type=int, default=5,
                   help="Max env.step() calls per episode (default: 5)")
    p.add_argument("--max-cycles", dest="max_cycles", type=int, default=MAX_CYCLES,
                   help=f"Max OBSERVER-MEDIATOR cycles per episode (default: {MAX_CYCLES})")
    p.add_argument("--output",   default=DEFAULT_OUTPUT,
                   help="Output JSON file path (default: results.json)")
    p.add_argument("--render",   default="none",
                   choices=["terminal", "terminal-fast", "human", "none"],
                   help="Render mode (default: none)")
    p.add_argument("--prompts",  action="store_true",
                   help="Print full prompts sent to each agent")
    p.add_argument("--quiet",    action="store_true",
                   help="Minimal output (suppress per-step logging)")
    p.add_argument("--rules",    default="",
                   help="Path to rules.json (default: auto-resolve to python/rules.json)")
    p.add_argument("--dataset-tag", dest="dataset_tag", default="arc-agi-3",
                   help="Namespace tag for rule/tool filtering (default: arc-agi-3)")
    p.add_argument("--playlog", dest="playlog", default=str(_TMP / "playlogs"),
                   help="Directory for per-step playlog JSON files (default: .tmp/playlogs). "
                        "Pass empty string to disable.")
    p.add_argument("--start-level", dest="start_level", type=int, default=1,
                   help="Level to start playing from (default: 1). "
                        "If the scorecard already has prior levels completed, skips pre-solve. "
                        "If not, executes the known subplan for this env to advance levels.")
    p.add_argument("--competition", action="store_true",
                   help="Run in competition mode: iterate the curated list of "
                        "competition-ready games with per-game budgets, one scorecard, "
                        "and write a submission summary.")
    p.add_argument("--model", default="",
                   help="Override LLM model. Examples: claude-sonnet-4-6, "
                        "Qwen/Qwen3.5-9B, meta-llama/Llama-3.3-70B-Instruct-Turbo, "
                        "deepseek-ai/DeepSeek-V3.1, Qwen/Qwen3.5-397B-A17B")
    p.add_argument("--quick", action="store_true",
                   help="Quick-test mode: max-cycles=15 (overrides --max-cycles).")
    p.add_argument("--stop-on-level", dest="stop_on_level", type=int, default=0,
                   help="Stop all episodes as soon as any episode reaches this level "
                        "(default: 0 = disabled). Useful for fast targeted testing.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Competition mode
# ---------------------------------------------------------------------------

# Curated list of games we currently consider competition-ready, with budgets
# tuned to known-good runs. Add games here as they reach competition quality.
COMPETITION_GAMES: list[dict] = [
    # Budget covers level 1 (~15 steps) + level 2 (BFS-optimal ~45 steps).
    {"env_id": "ls20", "max_steps": 80, "max_cycles": 40, "episodes": 1},
]


async def _run_competition(arc, args, render_mode) -> None:
    """Run the curated competition game list under a single scorecard."""
    rules    = RuleEngine(args.rules or None, dataset_tag=args.dataset_tag)
    tool_reg = ToolRegistry(read_only=False, dataset_tag=args.dataset_tag)
    playlog_root = Path(args.playlog) if args.playlog else None

    console.print(Panel(
        f"[bold]ARC-AGI-3 COMPETITION MODE[/bold]\n"
        f"Model:    [cyan]{DEFAULT_MODEL}[/cyan]\n"
        f"Games:    {', '.join(g['env_id'] for g in COMPETITION_GAMES)}\n"
        f"Playlogs: {playlog_root or '(disabled)'}",
        title="Harness — Competition"
    ))

    # Competition mode: exclude ALL per-game artifacts (LS20 BFS solver,
    # TR87 slot-strip detector, hardcoded subplans). KF must solve every
    # game using only game-agnostic primitives + concepts learned from
    # training. Bespoke solvers remain available in training mode as
    # ground-truth references.
    ens.COMPETITION_MODE = True

    submission: list[dict] = []
    output_path = Path(args.output).with_name("competition_results.json")
    started = time.time()

    for game in COMPETITION_GAMES:
        env_id   = game["env_id"]
        max_st   = game["max_steps"]
        max_cy   = game["max_cycles"]
        n_eps    = game["episodes"]

        console.rule(f"[bold cyan]{env_id}[/bold cyan]")
        env = arc.make(env_id, render_mode=render_mode)
        if env is None:
            console.print(f"[red]Failed to create env: {env_id}[/red]")
            continue

        ens.MAX_STEPS  = max_st
        ens.MAX_CYCLES = max_cy

        episodes: list[EpisodeMetadata] = []
        shared_dirs: dict = {}
        best_levels = 0

        for ep in range(1, n_eps + 1):
            meta = await run_episode(
                env                     = env,
                episode_num             = ep,
                env_id                  = env_id,
                rule_engine             = rules,
                tool_registry           = tool_reg,
                max_steps               = max_st,
                max_cycles              = max_cy,
                verbose                 = not args.quiet,
                playlog_root            = playlog_root,
                known_action_directions = shared_dirs,
                start_level             = 1,
            )
            shared_dirs.update(meta.action_directions)
            episodes.append(meta)
            best_levels = max(best_levels, meta.levels_completed)
            if meta.won:
                console.print(f"[green bold]WIN {env_id} ep{ep}[/green bold]")
                break

        submission.append({
            "env_id":          env_id,
            "episodes_run":    len(episodes),
            "won":             any(e.won for e in episodes),
            "best_levels":     best_levels,
            "total_steps":     sum(e.steps_taken for e in episodes),
            "total_cost_usd":  round(sum(e.cost_usd for e in episodes), 6),
            "episodes":        [asdict(e) for e in episodes],
        })

        # Incremental write after every game.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({
            "model":     DEFAULT_MODEL,
            "started":   datetime.datetime.utcfromtimestamp(started).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "elapsed_s": round(time.time() - started, 1),
            "games":     submission,
        }, indent=2), encoding="utf-8")

    # Final summary table
    table = Table(title="Competition Summary")
    table.add_column("Game"); table.add_column("Won"); table.add_column("Levels")
    table.add_column("Steps"); table.add_column("Cost USD")
    for g in submission:
        table.add_row(g["env_id"], "yes" if g["won"] else "no",
                      str(g["best_levels"]), str(g["total_steps"]),
                      f"${g['total_cost_usd']:.4f}")
    console.print(table)
    console.print(f"Submission written to [bold]{output_path}[/bold]")

    # Close the shared scorecard
    try:
        scorecard = arc.close_scorecard()
        if scorecard is not None:
            sc_path = output_path.with_name("competition_scorecard.json")
            if hasattr(scorecard, "model_dump_json"):
                sc_path.write_text(scorecard.model_dump_json(indent=2), encoding="utf-8")
            else:
                import dataclasses
                sc_dict = (dataclasses.asdict(scorecard)
                           if dataclasses.is_dataclass(scorecard)
                           else {k: v for k, v in vars(scorecard).items()
                                 if not k.startswith("_")})
                sc_path.write_text(json.dumps(sc_dict, indent=2), encoding="utf-8")
            console.print(f"Scorecard written to [bold]{sc_path}[/bold]")
    except Exception as exc:
        console.print(f"[yellow]Could not close scorecard: {exc}[/yellow]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    args = parse_args()
    verbose = not args.quiet

    if args.prompts:
        import core.pipeline.agents as _ca
        _ca.SHOW_PROMPTS = True

    # Override model across all modules
    if args.model:
        import core.pipeline.agents as _ca2
        _ca2.DEFAULT_MODEL = args.model
        agents.DEFAULT_MODEL = args.model
        ens.DEFAULT_MODEL = args.model
        global DEFAULT_MODEL
        DEFAULT_MODEL = args.model

    # Load API keys
    _key_file = Path("P:/_access/Security/api_keys.env")
    if _key_file.exists():
        for _line in _key_file.read_text().splitlines():
            for _prefix in ("ANTHROPIC_API_KEY=", "arc_api_key=", "TOGETHER_API_KEY="):
                if _line.startswith(_prefix):
                    _var = _prefix.rstrip("=").upper()
                    if not os.environ.get(_var):
                        os.environ[_var] = _line.split("=", 1)[1].strip()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]ANTHROPIC_API_KEY not set[/red]")
        sys.exit(1)

    # Build environment
    render_mode = None if args.render == "none" else args.render
    arc = arc_agi.Arcade(arc_api_key=os.environ.get("ARC_API_KEY", ""))

    # Competition mode: run the curated game list and exit.
    if args.competition:
        await _run_competition(arc, args, render_mode)
        return

    env = arc.make(args.env, render_mode=render_mode)
    if env is None:
        console.print(f"[red]Failed to create environment: {args.env}[/red]")
        sys.exit(1)

    # Knowledge engines
    rules_path = args.rules or None
    rules    = RuleEngine(rules_path, dataset_tag=args.dataset_tag)
    tool_reg = ToolRegistry(read_only=False, dataset_tag=args.dataset_tag)
    # Note: arc-agi-3 tools are action sequences stored as text, not Python functions.
    # We don't call load_into_executor() here — there is no Python executor in this pipeline.

    # Allow ensemble limits to be overridden
    ens.MAX_STEPS  = args.max_steps
    ens.MAX_CYCLES = 15 if args.quick else args.max_cycles

    playlog_root = Path(args.playlog) if args.playlog else None

    console.print(Panel(
        f"[bold]ARC-AGI-3 Ensemble[/bold]\n"
        f"Model:       [cyan]{DEFAULT_MODEL}[/cyan]\n"
        f"Environment: [cyan]{args.env}[/cyan]\n"
        f"Episodes:    {args.episodes}   Start level: {args.start_level}\n"
        f"Max steps/ep: {args.max_steps}   Max cycles/ep: {args.max_cycles}\n"
        f"Render:      {args.render}\n"
        f"NS tag:      [cyan]{args.dataset_tag}[/cyan]\n"
        f"Rules:       {rules.path}  {rules.stats_summary()}\n"
        f"Tools:       {tool_reg.path}  {tool_reg.stats_summary()}\n"
        f"Playlogs:    {playlog_root or '(disabled)'}",
        title="Harness"
    ))

    all_episodes: list[EpisodeMetadata] = []
    output_path = Path(args.output)
    best_levels = 0
    shared_action_directions: dict = {}   # persists across episodes

    for ep in range(1, args.episodes + 1):
        console.rule(f"[{ep}/{args.episodes}] Episode {ep}")

        meta = await run_episode(
            env                     = env,
            episode_num             = ep,
            env_id                  = args.env,
            rule_engine             = rules,
            tool_registry           = tool_reg,
            max_steps               = args.max_steps,
            max_cycles              = args.max_cycles,
            verbose                 = verbose,
            playlog_root            = playlog_root,
            known_action_directions = shared_action_directions,
            start_level             = args.start_level,
        )
        # Carry confirmed directions into the next episode
        shared_action_directions.update(meta.action_directions)

        all_episodes.append(meta)
        best_levels = max(best_levels, meta.levels_completed)

        # Incremental save after every episode
        _save_results(output_path, all_episodes, args.env, DEFAULT_MODEL, rules)

        if meta.won:
            console.print(f"[green bold]WIN in episode {ep}![/green bold]")
            break

        if args.stop_on_level and best_levels >= args.stop_on_level:
            console.print(f"[yellow]Reached level {best_levels} — stopping early "
                          f"(--stop-on-level {args.stop_on_level})[/yellow]")
            break

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    total       = len(all_episodes)
    won_count   = sum(1 for e in all_episodes if e.won)
    total_cost  = sum(e.cost_usd for e in all_episodes)
    total_steps = sum(e.steps_taken for e in all_episodes)

    table = Table(title="Run Summary")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Episodes run",      str(total))
    table.add_row("Won",               f"{won_count}/{total}")
    table.add_row("Best levels",       str(best_levels))
    table.add_row("Total steps",       str(total_steps))
    table.add_row("Total cost (USD)",  f"${total_cost:.4f}")
    table.add_row("Avg cost/episode",  f"${total_cost / max(total, 1):.4f}")
    table.add_row("Model",             DEFAULT_MODEL)
    rs = rules.stats_summary()
    table.add_row("Rules (active)",    str(rs["active"]))
    table.add_row("Rules (total)",     str(rs["total"]))
    console.print(table)
    console.print(f"Results written to [bold]{output_path}[/bold]")

    # Try to close the ARC scorecard
    try:
        scorecard = arc.close_scorecard()
        if scorecard is not None:
            scorecard_path = output_path.with_name("scorecard.json")
            if hasattr(scorecard, "model_dump_json"):
                scorecard_path.write_text(scorecard.model_dump_json(indent=2), encoding="utf-8")
            else:
                import dataclasses
                if dataclasses.is_dataclass(scorecard):
                    sc_dict = dataclasses.asdict(scorecard)
                else:
                    sc_dict = {k: v for k, v in vars(scorecard).items()
                               if not k.startswith("_")}
                scorecard_path.write_text(json.dumps(sc_dict, indent=2), encoding="utf-8")
            console.print(f"Scorecard written to [bold]{scorecard_path}[/bold]")
    except Exception as exc:
        console.print(f"[yellow]Could not close scorecard: {exc}[/yellow]")


if __name__ == "__main__":
    asyncio.run(main())
