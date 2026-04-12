"""
probe_rscd.py — PUPIL Domain Readiness Probe for road surface conditions.

Runs the five-step probe against a fixed benchmark manifest and produces a
structured report with go / partial / no-go verdict and per-dimension scores.

TUTOR and VALIDATOR outputs are cached between runs — testing a second PUPIL
model costs only PUPIL API calls (typically < $0.01 for Qwen3-VL-8B).

Usage:
  # First run — all models called, TUTOR/VALIDATOR results cached
  python probe_rscd.py

  # Different PUPIL model — TUTOR/VALIDATOR served from cache
  python probe_rscd.py --pupil-model llava-hf/llava-1.5-7b-hf

  # Specific pair and custom models
  python probe_rscd.py --pair dry_vs_wet \\
      --pupil-model     qwen/qwen3-vl-8b-instruct \\
      --tutor-model     claude-opus-4-6 \\
      --validator-model claude-sonnet-4-6

  # List available probe manifests and exit
  python probe_rscd.py --list-manifests

  # Clear TUTOR/VALIDATOR cache (re-run full probe from scratch)
  python probe_rscd.py --clear-cache

Output:
  benchmarks/reports/probe_{pair}_{model_tag}.json
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_KF_ROOT = _HERE.parents[4]
if str(_KF_ROOT) not in sys.path:
    sys.path.insert(0, str(_KF_ROOT))

import agents
from dataset import load as load_rscd, CONFUSABLE_PAIRS, DEFAULT_DATA_DIR
from domain_config import ROAD_SURFACE_CONFIG
from benchmark import to_probe_images, list_benchmarks

from core.benchmark import registry
from core.dialogic_distillation.probe import (
    probe,
    save_report,
    get_probe_costs,
    reset_probe_costs,
    clear_probe_cache,
    VERDICT_GO,
    VERDICT_PARTIAL,
    VERDICT_NO_GO,
)

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_PUPIL     = "qwen/qwen3-vl-8b-instruct"
DEFAULT_TUTOR     = "claude-opus-4-6"
DEFAULT_VALIDATOR = "claude-sonnet-4-6"
DEFAULT_PAIR      = "dry_vs_wet"

_TMP_DIR = (_HERE / ".." / ".." / ".." / ".." / ".tmp" / "rscd_session").resolve()
_REPORTS_DIR = _HERE.parent / "benchmarks" / "reports"


# ---------------------------------------------------------------------------
# API key loader
# ---------------------------------------------------------------------------

def _load_api_keys():
    kf = Path("P:/_access/Security/api_keys.env")
    if kf.exists():
        for line in kf.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip()
                if k in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY") and not os.environ.get(k):
                    os.environ[k] = v


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="PUPIL Domain Readiness Probe — road surface conditions")
    p.add_argument("--pair",           default=DEFAULT_PAIR,
                   choices=[cp["pair_id"] for cp in CONFUSABLE_PAIRS])
    p.add_argument("--pupil-model",    default=DEFAULT_PUPIL)
    p.add_argument("--tutor-model",    default=DEFAULT_TUTOR)
    p.add_argument("--validator-model", default=DEFAULT_VALIDATOR)
    p.add_argument("--probe-manifest", default="",
                   help="Explicit benchmark_id or path. Default: auto from --pair.")
    p.add_argument("--seed-rule",      default="",
                   help="Path to a JSON file containing a seed rule dict.")
    p.add_argument("--n-feature-queries", type=int, default=12)
    p.add_argument("--data-dir",       default=str(DEFAULT_DATA_DIR))
    p.add_argument("--output-dir",     default=str(_REPORTS_DIR))
    p.add_argument("--list-manifests", action="store_true",
                   help="List available probe manifests and exit")
    p.add_argument("--clear-cache",    action="store_true",
                   help="Clear TUTOR/VALIDATOR probe cache (disk + memory)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    args = parse_args()
    _load_api_keys()

    # Handle --list-manifests
    if args.list_manifests:
        registry.refresh()
        all_bm = registry.list_all()
        probe_bm = [b for b in all_bm if b["benchmark_type"] == "probe"]
        console.print("\n[bold]Available probe manifests:[/bold]")
        tbl = Table(show_header=True, header_style="bold")
        tbl.add_column("benchmark_id")
        tbl.add_column("domain")
        tbl.add_column("pair_id")
        tbl.add_column("images")
        tbl.add_column("path")
        for b in probe_bm:
            tbl.add_row(b["benchmark_id"], b["domain"],
                        b["pair_id"], str(b["n_images"]), b["path"])
        console.print(tbl)
        return

    # Handle --clear-cache
    if args.clear_cache:
        clear_probe_cache(disk=True)
        console.print("[green]Probe cache cleared (memory + disk).[/green]")
        return

    console.rule("[bold]PUPIL Domain Readiness Probe — Road Surface[/bold]")
    console.print(f"  PUPIL:      [cyan]{args.pupil_model}[/cyan]")
    console.print(f"  TUTOR:      [cyan]{args.tutor_model}[/cyan]")
    console.print(f"  VALIDATOR:  [cyan]{args.validator_model}[/cyan]")
    console.print(f"  Pair:       [cyan]{args.pair}[/cyan]")

    # Load seed rule if provided
    seed_rule = None
    if args.seed_rule:
        import json
        with open(args.seed_rule, encoding="utf-8") as f:
            seed_rule = json.load(f)
        console.print(f"  Seed rule:  [dim]{args.seed_rule}[/dim]")

    # Resolve pair info
    pair_info = next(cp for cp in CONFUSABLE_PAIRS if cp["pair_id"] == args.pair)

    # Load dataset
    console.print(f"\n[dim]Loading RSCD from {args.data_dir}...[/dim]")
    _TMP_DIR.mkdir(parents=True, exist_ok=True)
    try:
        ds = load_rscd(args.data_dir)
    except FileNotFoundError as e:
        console.print(f"\n[red]Dataset not found:[/red] {e}")
        console.print(
            "\n[yellow]Download RSCD:[/yellow]\n"
            "  kaggle datasets download cristvollerei/rscd-dataset-1million\n"
            "  Place zip at C:\\\\backup\\\\ml\\\\data\\\\rscd-dataset-1million.zip"
        )
        return

    total = sum(sum(s.values()) for s in ds.class_stats().values())
    console.print(f"  Loaded: {total:,} images")

    # Resolve probe manifest
    if args.probe_manifest:
        manifest_ref = args.probe_manifest
    else:
        # Auto-resolve from pair name
        manifest_ref = f"road_surface_{args.pair}_probe_v1"

    console.print(f"\n[dim]Loading probe manifest: {manifest_ref}[/dim]")
    try:
        # Try registry first (by benchmark_id)
        manifest_path = registry.find(manifest_ref)
    except KeyError:
        # Fall back to treating it as a direct file path
        manifest_path = Path(manifest_ref)
        if not manifest_path.exists():
            console.print(f"[red]Manifest not found:[/red] '{manifest_ref}'")
            console.print(
                "Generate it first:\n"
                f"  python create_benchmark.py --pair {args.pair} --types probe"
            )
            return

    probe_images = to_probe_images(manifest_path, ds, _TMP_DIR)
    console.print(f"  {len(probe_images)} probe images loaded")

    # Run probe
    agents.ACTIVE_MODEL = args.tutor_model
    reset_probe_costs()

    report = await probe(
        pupil_model     = args.pupil_model,
        tutor_model     = args.tutor_model,
        validator_model = args.validator_model,
        domain_config   = ROAD_SURFACE_CONFIG,
        probe_images    = probe_images,
        pair_info       = pair_info,
        seed_rule       = seed_rule,
        call_agent_fn   = agents.call_agent,
        n_feature_queries = args.n_feature_queries,
        console         = console,
    )

    # ---------------------------------------------------------------------------
    # Cost summary
    # ---------------------------------------------------------------------------
    console.print()
    console.rule("[bold]Cost breakdown[/bold]")
    costs = get_probe_costs()
    tbl = Table(show_header=True, header_style="bold")
    tbl.add_column("Role")
    tbl.add_column("API calls", justify="right")
    tbl.add_column("Input tok", justify="right")
    tbl.add_column("Output tok", justify="right")
    tbl.add_column("Cost (USD)", justify="right")
    total_cost = 0.0
    for role, c in costs.items():
        tbl.add_row(
            role,
            str(c["api_calls"]),
            str(c["input_tokens"]),
            str(c["output_tokens"]),
            f"${c['cost_usd']:.4f}",
        )
        total_cost += c["cost_usd"]
    tbl.add_section()
    tbl.add_row("[bold]TOTAL[/bold]", "", "", "", f"[bold]${total_cost:.4f}[/bold]")
    console.print(tbl)

    # ---------------------------------------------------------------------------
    # Save report
    # ---------------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_tag = args.pupil_model.replace("/", "_").replace("-", "_").replace(".", "_")
    out_path  = output_dir / f"probe_{args.pair}_{model_tag}.json"
    save_report(report, out_path)
    console.print(f"\n  Report saved: [cyan]{out_path}[/cyan]")
    console.print(
        f"  To commit: [dim]git add {out_path.relative_to(_KF_ROOT)}[/dim]"
    )

    # ---------------------------------------------------------------------------
    # Final verdict banner
    # ---------------------------------------------------------------------------
    verdict = report["verdict"]
    colour  = {VERDICT_GO: "green", VERDICT_PARTIAL: "yellow", VERDICT_NO_GO: "red"}[verdict]
    console.print(Panel(
        f"[{colour}][bold]{verdict.upper()}[/bold][/{colour}]\n\n"
        f"Perception:      {report['perception_score']:.2f}  "
        f"(feature detection accuracy)\n"
        f"Rule delta:      {report['rule_comprehension_delta']:+.2f}  "
        f"(zero-shot → rule-aided accuracy)\n"
        f"Consistency:     {report['consistency_score']:.2f}  "
        f"(same answer on repeated runs)\n\n"
        + ("\n".join(f"  ⚠ {wp}" for wp in report["weak_points"])
           if report["weak_points"] else "  No critical weak points."),
        title=f"Verdict — {args.pupil_model}",
        border_style=colour,
    ))


if __name__ == "__main__":
    asyncio.run(main())
