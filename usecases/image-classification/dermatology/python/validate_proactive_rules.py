"""
validate_proactive_rules.py

Validates proactively-elicited rules (no failure trigger needed) against
a held-out image pool. Reports TP, FP, precision for each rule.

Usage:
  python validate_proactive_rules.py \
    --rules patch_rules_proactive_nodular_raw.json \
    --pair melanoma_vs_melanocytic_nevus \
    --max-val-per-class 10 \
    --validator-model claude-sonnet-4-6
"""
from __future__ import annotations
import argparse, asyncio, json, os, sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_KF_ROOT = _HERE.parents[4]
if str(_KF_ROOT) not in sys.path:
    sys.path.insert(0, str(_KF_ROOT))

import agents
from dataset import load as load_ham10000
from harness import CONFUSABLE_PAIRS
from rich.console import Console
from rich.table import Table

console = Console()


def _load_api_keys():
    key_file = Path("P:/_access/Security/api_keys.env")
    if key_file.exists():
        for line in key_file.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                k = k.strip(); v = v.strip()
                if k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY") \
                        and not os.environ.get(k):
                    os.environ[k] = v


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules", default="patch_rules_proactive_nodular_raw.json")
    parser.add_argument("--pair", default="melanoma_vs_melanocytic_nevus")
    parser.add_argument("--max-val-per-class", type=int, default=10)
    parser.add_argument("--validator-model", default="claude-sonnet-4-6")
    parser.add_argument("--data-dir",
                        default=r"C:\_backup\ml\data\DermaMNIST_HAM10000")
    parser.add_argument("--output", default="proactive_validation_results.json")
    args = parser.parse_args()

    _load_api_keys()

    # Load rules
    rules_path = _HERE / args.rules
    with open(rules_path, encoding="utf-8") as f:
        raw_rules = json.load(f)

    # Convert to KF rule format
    kf_rules = []
    for r in raw_rules:
        kf_rules.append({
            "id": r.get("rule_id", r.get("id", "unknown")),
            "condition": r["condition"],
            "action": r["action"],
            "status": "active",
            "source": r.get("source", "proactive_elicitation"),
            "subtype": r.get("subtype", ""),
            "rationale": r.get("rationale", ""),
            "description": r.get("description", ""),
        })

    console.print(f"Loaded [bold]{len(kf_rules)}[/bold] rules from {args.rules}")

    # Load dataset
    pair_info = next(p for p in CONFUSABLE_PAIRS if p["pair_id"] == args.pair)
    dx_a = pair_info["dx_a"]
    dx_b = pair_info["dx_b"]
    label_a = pair_info["class_a"]   # "Melanoma"
    label_b = pair_info["class_b"]   # "Melanocytic Nevus"

    ds = load_ham10000(args.data_dir)

    # Sample validation pool (seed=42)
    pool_a = [(str(img.file_path), label_a) for img in ds.sample_images(
        dx_a, args.max_val_per_class, split="train", seed=42)]
    pool_b = [(str(img.file_path), label_b) for img in ds.sample_images(
        dx_b, args.max_val_per_class, split="train", seed=42)]
    pool = pool_a + pool_b

    # Use first melanoma image as dummy trigger (required by validate_candidate_rule API)
    # Proactive rules have no natural trigger — any melanoma image works as placeholder
    dummy_trigger_path, dummy_trigger_label = pool_a[0]

    console.print(f"Pool: {len(pool_a)} {label_a} + {len(pool_b)} {label_b} "
                  f"= {len(pool)} images (seed=42)")
    console.print(f"Validator: [cyan]{args.validator_model}[/cyan]")
    console.print(f"Dummy trigger: [dim]{Path(dummy_trigger_path).name}[/dim] "
                  f"(proactive rule — no real trigger image)\n")

    # Validate each rule
    all_results = {}
    for rule in kf_rules:
        rule_id = rule["id"]
        console.print(f"[bold]Validating {rule_id}[/bold]: "
                      f"{rule.get('description','')[:70]}")
        result = await agents.validate_candidate_rule(
            candidate_rule=rule,
            validation_images=pool,
            trigger_image_path=dummy_trigger_path,
            trigger_correct_label=dummy_trigger_label,
            model=args.validator_model,
            early_exit_fp=5,   # don't exit early — we want full pool results
        )
        all_results[rule_id] = result
        prec = result.get("precision")
        prec_str = f"{prec:.2f}" if prec is not None else "N/A"
        fires = result.get("fires_on_trigger", False)
        tp = result.get("tp", 0)
        fp = result.get("fp", 0)
        passes = result.get("accept", False)
        status = "[green]PASS[/green]" if passes else "[red]FAIL[/red]"
        console.print(
            f"  fires_on_trigger={fires}  TP={tp}  FP={fp}  "
            f"precision={prec_str}  {status}"
        )

    # Summary table
    console.print()
    table = Table(title="Proactive Rule Validation Summary")
    table.add_column("Rule ID")
    table.add_column("Description", max_width=45)
    table.add_column("Subtype", max_width=20)
    table.add_column("TP")
    table.add_column("FP")
    table.add_column("Precision")
    table.add_column("Gate")

    passes_count = 0
    for rule in kf_rules:
        rid = rule["id"]
        r = all_results[rid]
        prec = r.get("precision")
        prec_str = f"{prec:.2f}" if prec is not None else "—"
        gate = "PASS" if r.get("accept", False) else "FAIL"
        if r.get("accept", False):
            passes_count += 1
        table.add_row(
            rid,
            rule.get("description", "")[:45],
            rule.get("subtype", ""),
            str(r.get("tp", 0)),
            str(r.get("fp", 0)),
            prec_str,
            gate,
        )

    console.print(table)
    console.print(
        f"\n[bold]{passes_count}/{len(kf_rules)} rules passed the precision gate[/bold] "
        f"(fires_on_trigger, FP<=1, precision>=0.75)"
    )

    # Save
    out = {
        "rules_file": args.rules,
        "pair": args.pair,
        "pool_size_per_class": args.max_val_per_class,
        "validator_model": args.validator_model,
        "results": all_results,
        "rules": kf_rules,
        "passes": passes_count,
        "total": len(kf_rules),
    }
    out_path = _HERE / args.output
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    console.print(f"Results saved to [cyan]{args.output}[/cyan]")


if __name__ == "__main__":
    asyncio.run(main())
