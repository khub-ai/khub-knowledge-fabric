"""
distill_dialogic.py — Three-party dialogic knowledge distillation for road surface conditions.

Demonstrates that multi-round dialogue between PUPIL, TUTOR, and KF
produces grounded rules that pass validation on the safety-critical
road surface condition classification task.

Three parties:
  PUPIL  — the cheap VLM that failed (provides wrong prediction + reasoning)
  TUTOR  — Claude (expert with pavement engineering knowledge)
  KF     — orchestrator that steers the dialog:
           1. Surfaces PUPIL's failure + reasoning to TUTOR
           2. TUTOR authors a corrective rule (Round 1)
           3. KF immediately tests the rule on the trigger image (grounding check)
           4. If preconditions don't fire → KF feeds validator observations
              back to TUTOR with specific guidance on what failed
           5. TUTOR refines the rule (Round 2+)
           6. Once grounded → KF runs the standard pool gate
           7. If pool gate fails → contrastive tightening rounds

Key domain challenge:
  Road surface confusable pairs (e.g. wet vs ice) appear visually similar.
  Expert rules must describe OBSERVABLE visual differences — e.g. "ice
  completely obscures surface texture, while wet preserves faint aggregate
  visibility through the film" — that a validator model can confirm by
  looking at the image without temperature or environmental context.

Usage:
  python distill_dialogic.py
  python distill_dialogic.py --pair wet_vs_ice --tutor-model claude-opus-4-6
  python distill_dialogic.py --failure-ids 00012345,00023456,00034567
  python distill_dialogic.py --val-per-class 15 --max-rounds 4
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
from dataset import load as load_rscd, CONFUSABLE_PAIRS, DEFAULT_DATA_DIR
from domain_config import ROAD_SURFACE_CONFIG

from core.dialogic_distillation import run_dialogic_distillation

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_PAIR          = "dry_vs_wet"
DEFAULT_TUTOR         = "claude-opus-4-6"
DEFAULT_VALIDATOR     = "claude-sonnet-4-6"
DEFAULT_MAX_ROUNDS    = 4
DEFAULT_VAL_PER_CLASS = 10
DEFAULT_OUTPUT        = "distill_dialogic_session.json"

# Temp dir for extracting zip-backed images during session
_TMP_DIR = _HERE / ".." / ".." / ".." / ".." / ".tmp" / "rscd_session"
_TMP_DIR = _TMP_DIR.resolve()

# Canonical failure cases for wet_vs_ice pair.
# Replace with real RSCD image IDs once dataset is downloaded.
# These should be images where the cheap VLM predicted "Wet Road" but
# the ground truth is "Icy Road".
DEFAULT_FAILURE_IDS = [
    # Placeholder IDs — replace with actual RSCD filenames (without extension)
    # e.g. "00045231", "00089012", "00123456", "00234567"
]


# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------

def _load_api_keys():
    """Load API keys from local env file if present."""
    kf = Path("P:/_access/Security/api_keys.env")
    if kf.exists():
        for line in kf.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                k = k.strip(); v = v.strip()
                if k in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY") \
                        and not os.environ.get(k):
                    os.environ[k] = v


# ---------------------------------------------------------------------------
# Failure discovery (when no explicit IDs provided)
# ---------------------------------------------------------------------------

async def discover_failures(
    ds,
    pair_info: dict,
    pupil_model: str,
    n_per_class: int,
    split: str = "test",
) -> list[dict]:
    """Run the PUPIL model on test images and collect failure cases.

    Returns list of failure dicts:
      {image_id, image_path, correct_label, wrong_prediction, pupil_reasoning}
    """
    from core.dialogic_distillation.agents import image_block, parse_json_block

    friction_a     = pair_info["friction_a"]
    friction_b     = pair_info["friction_b"]
    class_a        = pair_info["class_a"]
    class_b        = pair_info["class_b"]
    mat_filter     = pair_info.get("material_filter")

    images_a = ds.sample_images(friction_a, n_per_class, split=split,
                                material_filter=mat_filter)
    images_b = ds.sample_images(friction_b, n_per_class, split=split,
                                material_filter=mat_filter)

    console.print(f"  Discovering failures on {len(images_a)} {class_a} + "
                  f"{len(images_b)} {class_b} images...")

    failures = []
    total = images_a + images_b
    for img in total:
        true_label   = class_a if img.friction == friction_a else class_b
        other_label  = class_b if true_label == class_a else class_a

        content = [
            image_block(str(img.resolve_path(_TMP_DIR))),
            {
                "type": "text",
                "text": (
                    f"Look at this road surface image carefully.\n\n"
                    f"Your task: classify this road surface as one of exactly two options:\n"
                    f"  A) {class_a}\n"
                    f"  B) {class_b}\n\n"
                    f"Describe what you observe, then give your classification.\n\n"
                    f"Respond with JSON:\n"
                    f'{{"prediction": "{class_a}" or "{class_b}", '
                    f'"reasoning": "what you observed"}}'
                ),
            },
        ]

        raw, _ = await agents.call_agent(
            "PUPIL",
            content,
            system_prompt=(
                f"You are a vision model classifying road surface conditions. "
                f"Answer only with the JSON format requested."
            ),
            model=pupil_model,
            max_tokens=256,
        )

        result = parse_json_block(raw)
        prediction = (result or {}).get("prediction", "").strip()
        reasoning  = (result or {}).get("reasoning", raw[:300])

        # Normalize prediction to exact class label
        if class_a.lower() in prediction.lower():
            prediction = class_a
        elif class_b.lower() in prediction.lower():
            prediction = class_b
        else:
            prediction = class_a if "A" in prediction else class_b

        if prediction != true_label:
            failures.append({
                "image_id":        img.image_id,
                "image_path":      str(img.resolve_path(_TMP_DIR)),
                "correct_label":   true_label,
                "wrong_prediction": prediction,
                "pupil_reasoning": reasoning,
            })

    console.print(f"  Found [yellow]{len(failures)}/{len(total)}[/yellow] failures "
                  f"({len(failures)/len(total)*100:.0f}% error rate)")
    return failures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Three-party dialogic distillation — road surface conditions")
    p.add_argument("--pair", default=DEFAULT_PAIR,
                   choices=[cp["pair_id"] for cp in CONFUSABLE_PAIRS],
                   help="Confusable pair to work on")
    p.add_argument("--failure-ids", default="",
                   help="Comma-separated RSCD image IDs (stem only, no .jpg). "
                        "If not provided, auto-discovers failures using --pupil-model.")
    p.add_argument("--pupil-model",  default="qwen/qwen3-vl-8b-instruct",
                   help="Cheap model to test as PUPIL (used when --failure-ids not given). "
                        "Defaults to Qwen3-VL-8B via OpenRouter — same as birds/dermatology "
                        "experiments for cross-domain comparability.")
    p.add_argument("--tutor-model",  default=DEFAULT_TUTOR)
    p.add_argument("--validator-model", default=DEFAULT_VALIDATOR)
    p.add_argument("--max-rounds",   type=int, default=DEFAULT_MAX_ROUNDS)
    p.add_argument("--val-per-class", type=int, default=DEFAULT_VAL_PER_CLASS,
                   help="Number of pool images per class for validation")
    p.add_argument("--n-failures",   type=int, default=4,
                   help="Max failures to process (when auto-discovering)")
    p.add_argument("--data-dir",     default=str(DEFAULT_DATA_DIR))
    p.add_argument("--output",       default=DEFAULT_OUTPUT)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    args = parse_args()
    _load_api_keys()

    pair_info = next(cp for cp in CONFUSABLE_PAIRS if cp["pair_id"] == args.pair)
    class_a   = pair_info["class_a"]
    class_b   = pair_info["class_b"]
    mat_filter = pair_info.get("material_filter")

    console.rule("[bold]Three-Party Dialogic Distillation — Road Surface Conditions[/bold]")
    console.print(f"  Pair:      [cyan]{args.pair}[/cyan]  ({class_a} vs {class_b})")
    console.print(f"  TUTOR:     [cyan]{args.tutor_model}[/cyan]")
    console.print(f"  Validator: [cyan]{args.validator_model}[/cyan]")
    console.print(f"  Max rounds: {args.max_rounds}")
    console.print(f"  Pool size/class: {args.val_per_class}")
    if mat_filter:
        console.print(f"  Material filter: [dim]{mat_filter}[/dim]")

    # Load dataset
    console.print(f"\n[dim]Loading RSCD from {args.data_dir}...[/dim]")
    _TMP_DIR.mkdir(parents=True, exist_ok=True)
    try:
        ds = load_rscd(args.data_dir)
    except FileNotFoundError as e:
        console.print(f"\n[red]Dataset not found:[/red] {e}")
        console.print(
            "\n[yellow]To download RSCD:[/yellow]\n"
            "  kaggle datasets download cristvollerei/rscd-dataset-1million\n"
            "  Place zip at C:\\\\backup\\\\ml\\\\data\\\\rscd-dataset-1million.zip"
        )
        return

    stats = ds.class_stats()
    console.print(f"  Loaded: {sum(sum(s.values()) for s in stats.values()):,} images")

    # Build validation pool from training split
    pool_a = [
        (str(img.resolve_path(_TMP_DIR)), class_a)
        for img in ds.sample_images(
            pair_info["friction_a"], args.val_per_class,
            split="train", seed=42, material_filter=mat_filter)
    ]
    pool_b = [
        (str(img.resolve_path(_TMP_DIR)), class_b)
        for img in ds.sample_images(
            pair_info["friction_b"], args.val_per_class,
            split="train", seed=42, material_filter=mat_filter)
    ]
    pool_images = pool_a + pool_b
    console.print(f"  Pool: {len(pool_a)} {class_a} + {len(pool_b)} {class_b} "
                  f"= {len(pool_images)} images")

    # Gather failure cases
    if args.failure_ids:
        # Build image_id → resolved path map from full dataset
        img_map = {img.image_id: str(img.resolve_path(_TMP_DIR)) for img in ds._images}
        failure_ids = [fid.strip() for fid in args.failure_ids.split(",") if fid.strip()]
        failures = []
        for fid in failure_ids:
            path = img_map.get(fid)
            if not path:
                console.print(f"  [red]Image not found: {fid}[/red]")
                continue
            # Determine the ground truth from the dataset
            img_rec = next((img for img in ds._images if img.image_id == fid), None)
            if img_rec is None:
                continue
            true_label = class_a if img_rec.friction == pair_info["friction_a"] else class_b
            # Assume opposite label was predicted (user provides known failures)
            wrong = class_b if true_label == class_a else class_a
            failures.append({
                "image_id":        fid,
                "image_path":      path,
                "correct_label":   true_label,
                "wrong_prediction": wrong,
                "pupil_reasoning": "(provided by user as known failure case)",
            })
    else:
        console.print(f"\n  No --failure-ids provided; "
                      f"running PUPIL ({args.pupil_model}) to discover failures...")
        agents.ACTIVE_MODEL = args.pupil_model
        all_failures = await discover_failures(
            ds, pair_info,
            pupil_model=args.pupil_model,
            n_per_class=30,
            split="test",
        )
        failures = all_failures[:args.n_failures]
        if not failures:
            console.print("[yellow]No failures found — PUPIL may already be accurate on "
                          "this pair, or the dataset slice is too small.[/yellow]")
            return
        console.print(f"  Using first {len(failures)} failures for distillation.")

    console.print(f"\n  Processing [bold]{len(failures)}[/bold] failure(s)...\n")

    # Run dialogic distillation on each failure
    agents.ACTIVE_MODEL = args.tutor_model
    agents.reset_cost_tracker()

    all_transcripts = []
    for failure in failures:
        fid     = failure["image_id"]
        path    = failure["image_path"]
        correct = failure["correct_label"]
        wrong   = failure["wrong_prediction"]
        reason  = failure["pupil_reasoning"]

        console.rule(f"[bold]{fid}[/bold]  correct={correct}  predicted={wrong}")

        transcript = await run_dialogic_distillation(
            image_path=path,
            image_id=fid,
            correct_label=correct,
            wrong_prediction=wrong,
            pupil_reasoning=reason,
            pair_info=pair_info,
            config=ROAD_SURFACE_CONFIG,
            tutor_model=args.tutor_model,
            validator_model=args.validator_model,
            max_rounds=args.max_rounds,
            pool_images=pool_images,
            call_agent_fn=agents.call_agent,
            console=console,
        )
        all_transcripts.append(transcript)

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    tracker = agents.get_cost_tracker()
    console.print()
    console.rule("[bold]Summary[/bold]")

    tbl = Table(show_header=True, header_style="bold")
    tbl.add_column("Image ID")
    tbl.add_column("Grounded?")
    tbl.add_column("Round")
    tbl.add_column("Pool result")
    tbl.add_column("Outcome")

    n_grounded = 0
    n_accepted = 0
    for t in all_transcripts:
        grounded  = t["grounded_at_round"] is not None
        n_grounded += grounded
        pool       = t.get("pool_result") or {}
        pool_tight = t.get("pool_result_after_tighten")
        if pool_tight and pool_tight.get("accepted"):
            pool = pool_tight
        accepted = pool.get("accepted", False)
        n_accepted += accepted

        tbl.add_row(
            t["image_id"],
            "[green]Yes[/green]" if grounded else "[red]No[/red]",
            str(t["grounded_at_round"] or "—"),
            (f"TP={pool.get('tp', 0)} FP={pool.get('fp', 0)} "
             f"prec={pool.get('precision', 0):.2f}")
            if pool else "—",
            ("[green]ACCEPTED[/green]" if accepted
             else "[yellow]grounded[/yellow]" if grounded
             else "[red]not grounded[/red]"),
        )

    console.print(tbl)
    console.print(
        f"\n  {n_grounded}/{len(all_transcripts)} grounded  |  "
        f"{n_accepted} accepted"
    )
    try:
        console.print(f"  Cost: [cyan]${tracker.cost_usd():.4f}[/cyan]")
    except Exception:
        pass  # cost display is best-effort

    # Save session
    session = {
        "pair":                args.pair,
        "tutor_model":         args.tutor_model,
        "validator_model":     args.validator_model,
        "max_rounds":          args.max_rounds,
        "pool_size_per_class": args.val_per_class,
        "transcripts":         all_transcripts,
        "summary": {
            "total_failures":  len(all_transcripts),
            "grounded":        n_grounded,
            "accepted":        n_accepted,
        },
    }

    out_path = _HERE / args.output
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(session, f, indent=2, ensure_ascii=False)
    console.print(f"  Session saved to [cyan]{out_path.name}[/cyan]")


if __name__ == "__main__":
    asyncio.run(main())
