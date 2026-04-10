"""
Check whether each elicited rule fires on the image it was authored from,
and measure precision on the validation pool.
This separates the trigger-firing check from pool precision.
"""
import asyncio, base64, json, os, re, sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_KF_ROOT = _HERE.parents[4]
if str(_KF_ROOT) not in sys.path:
    sys.path.insert(0, str(_KF_ROOT))

import agents
from dataset import load as load_ham10000
from harness import CONFUSABLE_PAIRS
from rich.console import Console
console = Console()

DATA_DIR = r"C:\_backup\ml\data\DermaMNIST_HAM10000"
VALIDATOR = "claude-sonnet-4-6"
VAL_PER_CLASS = 10


def _to_validator_format(rule: dict) -> dict:
    """Convert patch-rule condition string to the preconditions/favors format
    expected by run_rule_validator_on_image.

    Input rule may have:
      "condition": "[Patch rule — pair_id] precond1 AND precond2 AND precond3"
      "action":    "classify as Melanoma"

    OR already have:
      "preconditions": [...]
      "favors": "..."
    """
    if "preconditions" in rule and "favors" in rule:
        return rule  # already in validator format

    condition = rule.get("condition", "")
    action    = rule.get("action", "")

    # Extract precondition text after the "] " header
    bracket_end = condition.find("]")
    if bracket_end != -1:
        raw_preds = condition[bracket_end + 1:].strip()
    else:
        raw_preds = condition.strip()

    # Split on " AND " (Opus uses this as separator in candidate pool rules)
    preconditions = [p.strip() for p in raw_preds.split(" AND ") if p.strip()]

    # Derive favors from action field ("classify as Melanoma" → "Melanoma")
    favors = action.replace("classify as ", "").strip() if action else ""

    rule_text = f"When {raw_preds}, {action}."

    return {
        **rule,
        "rule": rule_text,
        "preconditions": preconditions,
        "favors": favors,
    }


def _load_keys():
    kf = Path("P:/_access/Security/api_keys.env")
    for line in kf.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            k = k.strip(); v = v.strip()
            if k in ("ANTHROPIC_API_KEY",) and not os.environ.get(k):
                os.environ[k] = v


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules", default="patch_rules_elicited_from_failures.json")
    args = parser.parse_args()

    _load_keys()

    with open(_HERE / args.rules, encoding="utf-8") as f:
        rules = json.load(f)

    ds = load_ham10000(DATA_DIR)
    pair_info = next(p for p in CONFUSABLE_PAIRS
                     if p["pair_id"] == "melanoma_vs_melanocytic_nevus")
    label_mel = pair_info["class_a"]   # "Melanoma"
    label_nv  = pair_info["class_b"]   # "Melanocytic Nevus"

    # Pool (seed=42, excluding source images)
    pool_mel = [(str(img.file_path), label_mel)
                for img in ds.sample_images("mel", VAL_PER_CLASS, split="train", seed=42)]
    pool_nv  = [(str(img.file_path), label_nv)
                for img in ds.sample_images("nv",  VAL_PER_CLASS, split="train", seed=42)]
    pool = pool_mel + pool_nv

    # Build image_id -> path map for source images (both splits)
    all_mel = ds.sample_images("mel", 500, split="test", seed=0) + \
              ds.sample_images("mel", 500, split="train", seed=0)
    img_map = {img.image_id: str(img.file_path) for img in all_mel}

    console.print(f"Pool: {len(pool_mel)} mel + {len(pool_nv)} nv = {len(pool)}\n")

    for rule in rules:
        rid = rule["rule_id"]
        src_id = rule.get("source_image", "")
        src_path = img_map.get(src_id, "")

        vrule = _to_validator_format(rule)   # convert to preconditions/favors format

        console.print(f"[bold]{rid}[/bold]  source={src_id}")
        console.print(f"  preconditions: {vrule.get('preconditions','')}")

        # 1. Does rule fire on its own source image?
        if src_path:
            src_result, _ = await agents.run_rule_validator_on_image(
                image_path=src_path,
                ground_truth=label_mel,
                candidate_rule=vrule,
                model=VALIDATOR,
            )
            fires_src = src_result.get("precondition_met", False)
            console.print(f"  fires on source image: [{'green' if fires_src else 'red'}]{fires_src}[/]")
            console.print(f"  obs: {src_result.get('observations','')[:120]}")
        else:
            fires_src = False
            console.print("  [red]source image not found[/red]")

        # 2. Precision on pool
        tp = fp = 0
        for img_path, gt in pool:
            res, _ = await agents.run_rule_validator_on_image(
                image_path=img_path,
                ground_truth=gt,
                candidate_rule=vrule,
                model=VALIDATOR,
            )
            if res.get("precondition_met", False):
                if gt == label_mel:
                    tp += 1
                else:
                    fp += 1

        fires_pool = tp + fp
        prec = tp / fires_pool if fires_pool > 0 else None
        prec_str = f"{prec:.2f}" if prec is not None else "—"
        passes = fires_src and fires_pool > 0 and fp <= 1 and (prec or 0) >= 0.75
        status = "[green]PASS[/green]" if passes else "[red]FAIL[/red]"
        console.print(f"  pool: fires={fires_pool} TP={tp} FP={fp} "
                      f"precision={prec_str}  {status}\n")


if __name__ == "__main__":
    asyncio.run(main())
