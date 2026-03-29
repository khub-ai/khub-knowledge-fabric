"""
Experiment evaluator for UC-02.

Runs all four conditions across all 15 confusable pairs and writes results
to results/. Produces per-pair and aggregate accuracy tables.

Usage:
  python evaluator.py --condition zero_shot
  python evaluator.py --condition few_shot
  python evaluator.py --condition kf_taught
  python evaluator.py --condition all        # runs all three
"""

from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).parent))

from config import MAX_TEST_IMAGES_PER_CLASS, RESULTS_DIR, SEED
from dataset import load, CUBDataset
from confusable_pairs import CONFUSABLE_PAIRS, ConfusablePair
from baseline import build_client, classify_zero_shot, classify_few_shot
from kf_teacher import load_rules, rules_to_prompt

Condition = Literal["zero_shot", "few_shot", "kf_taught"]


# ---------------------------------------------------------------------------

def normalise(name: str) -> str:
    """Lowercase and strip for comparison."""
    return name.strip().lower().replace("-", " ").replace("_", " ")


def is_correct(prediction: str, true_class_name: str) -> bool:
    """Flexible match: true_class_name is like '029.American_Crow'."""
    species = true_class_name.split(".", 1)[1].replace("_", " ")
    return normalise(prediction) == normalise(species)


# ---------------------------------------------------------------------------

def evaluate_pair(
    client,
    pair: ConfusablePair,
    ds: CUBDataset,
    condition: Condition,
    n_per_class: int = MAX_TEST_IMAGES_PER_CLASS,
) -> list[dict]:

    results = []

    # Collect test images for both classes
    test_a = ds.sample_test_images(pair.class_id_a, n=n_per_class, seed=SEED)
    test_b = ds.sample_test_images(pair.class_id_b, n=n_per_class, seed=SEED)
    test_images = test_a + test_b

    # Few-shot examples (from training split)
    if condition == "few_shot":
        examples_a = ds.sample_test_images.__func__(ds, pair.class_id_a, n=3, seed=SEED + 1)  # type: ignore
        examples_b = ds.sample_test_images.__func__(ds, pair.class_id_b, n=3, seed=SEED + 1)  # type: ignore
        # Actually use train split for few-shot examples
        examples_a = ds.images_for_class(pair.class_id_a, split="train")[:3]
        examples_b = ds.images_for_class(pair.class_id_b, split="train")[:3]

    # KF rules
    rules_text = None
    if condition == "kf_taught":
        rules = load_rules(pair)
        if not rules:
            print(f"  [WARN] No KB rules found for {pair.class_name_a} vs {pair.class_name_b} — skipping kf_taught")
            return []
        rules_text = rules_to_prompt(rules)

    for img in test_images:
        print(f"  [{condition}] image {img.image_id} ({ds.class_name_clean(img.class_id)}) ...", end=" ", flush=True)
        try:
            if condition == "zero_shot":
                result = classify_zero_shot(client, pair, img)
            elif condition == "few_shot":
                result = classify_few_shot(client, pair, img, examples_a, examples_b)
            elif condition == "kf_taught":
                result = classify_zero_shot(client, pair, img, rules_text=rules_text)
            else:
                raise ValueError(f"Unknown condition: {condition}")

            result["correct"] = is_correct(result.get("prediction", ""), img.class_name)
            print("✓" if result["correct"] else "✗")
        except Exception as e:
            print(f"ERROR: {e}")
            result = {
                "image_id": img.image_id,
                "true_class": img.class_name,
                "condition": condition,
                "prediction": None,
                "correct": None,
                "error": str(e),
            }
        results.append(result)

    return results


# ---------------------------------------------------------------------------

def summarise(all_results: list[dict]) -> dict:
    from collections import defaultdict
    by_condition: dict[str, list] = defaultdict(list)
    for r in all_results:
        cond = r.get("condition", "unknown")
        if r.get("correct") is not None:
            by_condition[cond].append(r["correct"])

    summary = {}
    for cond, corrects in by_condition.items():
        n = len(corrects)
        acc = sum(corrects) / n if n else 0.0
        summary[cond] = {"n": n, "correct": sum(corrects), "accuracy": round(acc, 4)}
    return summary


def print_summary(summary: dict):
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    for cond, stats in sorted(summary.items()):
        bar = "█" * int(stats["accuracy"] * 30)
        print(f"  {cond:<15} {stats['accuracy']*100:5.1f}%  {bar}  ({stats['correct']}/{stats['n']})")
    print()


# ---------------------------------------------------------------------------

def run(conditions: list[Condition]):
    ds = load()
    client = build_client()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    all_results = []

    for pair in CONFUSABLE_PAIRS:
        print(f"\n{'─'*60}")
        print(f"Pair: {pair.class_name_a} vs {pair.class_name_b}  (sim={pair.cosine_sim:.4f})")
        for cond in conditions:
            results = evaluate_pair(client, pair, ds, cond)
            all_results.extend(results)

    # Save raw results
    out_file = RESULTS_DIR / f"results_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRaw results saved → {out_file}")

    # Summary
    summary = summarise(all_results)
    print_summary(summary)

    summary_file = RESULTS_DIR / f"summary_{timestamp}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved → {summary_file}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition",
        choices=["zero_shot", "few_shot", "kf_taught", "all"],
        default="all",
    )
    args = parser.parse_args()

    if args.condition == "all":
        conditions: list[Condition] = ["zero_shot", "few_shot", "kf_taught"]
    else:
        conditions = [args.condition]

    run(conditions)
