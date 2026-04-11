"""
distill_dialogic.py — Three-party dialogic knowledge distillation

Demonstrates that multi-round dialog between PUPIL, TUTOR, and KF
produces grounded rules that pass validation, where single-shot
elicitation fails (0/4 in elicit_from_failures.py).

Three parties:
  PUPIL  — the cheap VLM that failed (provides wrong prediction + reasoning)
  TUTOR  — Claude Opus (expert with dermoscopic subtype knowledge)
  KF     — orchestrator that steers the dialog:
           1. Surfaces PUPIL's failure + reasoning to TUTOR
           2. TUTOR authors a corrective rule (Round 1)
           3. KF immediately tests the rule on the trigger image (grounding check)
           4. If preconditions don't fire → KF feeds validator observations
              back to TUTOR with specific guidance on what failed
           5. TUTOR refines the rule (Round 2+)
           6. Once grounded → KF runs the standard pool gate
           7. If pool gate fails → contrastive tightening round

Evidence structure: each failure produces a dialog transcript showing
every KF steering move and the rule's evolution across rounds.

Usage:
  python distill_dialogic.py
  python distill_dialogic.py --max-rounds 4 --tutor-model claude-opus-4-6
  python distill_dialogic.py --failure-ids ISIC_0024410,ISIC_0024647
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_KF_ROOT = _HERE.parents[4]
if str(_KF_ROOT) not in sys.path:
    sys.path.insert(0, str(_KF_ROOT))

import agents
from dataset import load as load_ham10000
from harness import CONFUSABLE_PAIRS
from domain_config import DERM_CONFIG

from core.dialogic_distillation import run_dialogic_distillation

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# ── defaults ────────────────────────────────────────────────────────────────
DEFAULT_DATA_DIR   = r"C:\_backup\ml\data\DermaMNIST_HAM10000"
DEFAULT_PAIR       = "melanoma_vs_melanocytic_nevus"
DEFAULT_TUTOR      = "claude-opus-4-6"
DEFAULT_VALIDATOR  = "claude-sonnet-4-6"
DEFAULT_MAX_ROUNDS = 4
DEFAULT_VAL_PER_CLASS = 10
DEFAULT_FAILURE_IDS = [
    "ISIC_0024410",
    "ISIC_0024647",
    "ISIC_0024911",
    "ISIC_0025128",
]


# ── API keys ────────────────────────────────────────────────────────────────
def _load_api_keys():
    kf = Path("P:/_access/Security/api_keys.env")
    if kf.exists():
        for line in kf.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                k = k.strip(); v = v.strip()
                if k in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY") \
                        and not os.environ.get(k):
                    os.environ[k] = v


# ── CLI + main ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Three-party dialogic knowledge distillation")
    p.add_argument("--failure-ids", default="",
                   help="Comma-separated ISIC IDs (default: 4 canonical failures)")
    p.add_argument("--tutor-model", default=DEFAULT_TUTOR)
    p.add_argument("--validator-model", default=DEFAULT_VALIDATOR)
    p.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    p.add_argument("--val-per-class", type=int, default=DEFAULT_VAL_PER_CLASS)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--pair", default=DEFAULT_PAIR)
    p.add_argument("--output", default="distill_dialogic_session.json")
    return p.parse_args()


async def main():
    args = parse_args()
    _load_api_keys()

    failure_ids = (args.failure_ids.split(",") if args.failure_ids
                   else DEFAULT_FAILURE_IDS)

    pair_info = next(p for p in CONFUSABLE_PAIRS if p["pair_id"] == args.pair)
    label_mel = pair_info["class_a"]   # "Melanoma"
    label_nv  = pair_info["class_b"]   # "Melanocytic Nevus"

    console.rule("[bold]Three-Party Dialogic Distillation[/bold]")
    console.print(f"  TUTOR:     [cyan]{args.tutor_model}[/cyan]")
    console.print(f"  Validator: [cyan]{args.validator_model}[/cyan]")
    console.print(f"  Max rounds: {args.max_rounds}")
    console.print(f"  Pool size/class: {args.val_per_class}")
    console.print(f"  Failures: {failure_ids}")

    # Load dataset
    console.print(f"\n[dim]Loading HAM10000...[/dim]")
    ds = load_ham10000(args.data_dir)

    # Build image_id → path map
    all_mel = (ds.sample_images("mel", 500, split="test", seed=0)
               + ds.sample_images("mel", 500, split="train", seed=0))
    all_nv = (ds.sample_images("nv", 500, split="test", seed=0)
              + ds.sample_images("nv", 500, split="train", seed=0))
    img_map = {img.image_id: str(img.file_path)
               for img in all_mel + all_nv}

    # Sample validation pool (seed=42, balanced)
    pool_mel = [(str(img.file_path), label_mel)
                for img in ds.sample_images(
                    pair_info["dx_a"], args.val_per_class,
                    split="train", seed=42)]
    pool_nv = [(str(img.file_path), label_nv)
               for img in ds.sample_images(
                   pair_info["dx_b"], args.val_per_class,
                   split="train", seed=42)]
    pool_images = pool_mel + pool_nv
    console.print(f"  Pool: {len(pool_mel)} {label_mel} + "
                  f"{len(pool_nv)} {label_nv} = {len(pool_images)}")

    # Process each failure
    all_transcripts = []
    for fid in failure_ids:
        path = img_map.get(fid)
        if not path:
            console.print(f"\n[red]Image not found: {fid}[/red]")
            continue

        console.rule(f"[bold]{fid}[/bold]")

        transcript = await run_dialogic_distillation(
            image_path=path,
            image_id=fid,
            correct_label=label_mel,
            wrong_prediction=label_nv,
            pupil_reasoning="(cheap model predicted Melanocytic Nevus)",
            pair_info=pair_info,
            config=DERM_CONFIG,
            tutor_model=args.tutor_model,
            validator_model=args.validator_model,
            max_rounds=args.max_rounds,
            pool_images=pool_images,
            call_agent_fn=agents.call_agent,
            console=console,
        )
        all_transcripts.append(transcript)

    # ── Summary ──────────────────────────────────────────────────────────
    console.print()
    console.rule("[bold]Summary[/bold]")

    tbl = Table(show_header=True, header_style="bold")
    tbl.add_column("Image ID")
    tbl.add_column("Grounded?")
    tbl.add_column("Round")
    tbl.add_column("Pool")
    tbl.add_column("Outcome")

    n_grounded = 0
    n_accepted = 0
    for t in all_transcripts:
        grounded = t["grounded_at_round"] is not None
        n_grounded += grounded
        pool = t.get("pool_result") or {}
        pool_t = t.get("pool_result_after_tighten")
        # Use tightened result if it exists and passed
        if pool_t and pool_t.get("accepted"):
            pool = pool_t
        accepted = pool.get("accepted", False)
        n_accepted += accepted

        tbl.add_row(
            t["image_id"],
            "[green]Yes[/green]" if grounded else "[red]No[/red]",
            str(t["grounded_at_round"] or "—"),
            (f"TP={pool.get('tp',0)} FP={pool.get('fp',0)} "
             f"prec={pool.get('precision',0):.2f}")
            if pool else "—",
            ("[green]ACCEPTED[/green]" if accepted
             else "[yellow]grounded[/yellow]" if grounded
             else "[red]not grounded[/red]"),
        )

    console.print(tbl)

    # Comparison with single-shot
    console.print(
        f"\n  Single-shot baseline (elicit_from_failures.py): "
        f"[red]0/{len(all_transcripts)} grounded, 0 accepted[/red]"
    )
    console.print(
        f"  Dialogic (this run): "
        f"[cyan]{n_grounded}/{len(all_transcripts)} grounded, "
        f"{n_accepted} accepted[/cyan]"
    )

    # Save session
    session = {
        "tutor_model": args.tutor_model,
        "validator_model": args.validator_model,
        "max_rounds": args.max_rounds,
        "pool_size_per_class": args.val_per_class,
        "pair": args.pair,
        "failure_ids": failure_ids,
        "transcripts": all_transcripts,
        "summary": {
            "total_failures": len(all_transcripts),
            "grounded": n_grounded,
            "accepted": n_accepted,
            "single_shot_grounded": 0,
            "single_shot_accepted": 0,
        },
    }

    out_path = _HERE / args.output
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(session, f, indent=2, ensure_ascii=False)
    console.print(f"\n  Session saved to [cyan]{args.output}[/cyan]")


if __name__ == "__main__":
    asyncio.run(main())
