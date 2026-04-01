"""
migrate_rules.py -- Import knowledge_base/*.json rules into KF RuleEngine.

Reads the 15 per-pair JSON files from knowledge_base/ and registers each rule
into rules.json with:
  - dataset_tag = "bird-uc200"
  - observability_filter=True  (rejects vocalizations, habitat, range, season)
  - condition  = rule["rule"]  (the natural-language rule text)
  - rule_action = "Classify as {favors}" (+ confidence note)
  - tags        = ["bird-uc200", pair_id]

Run standalone:
  python migrate_rules.py           # import all rules
  python migrate_rules.py --dry-run # preview without saving

Run from harness:
  python harness.py --migrate
  python harness.py --migrate --dry-run

After migration, run python harness.py to start classifying.
"""

from __future__ import annotations
import io
import json
import sys
from pathlib import Path

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_HERE    = Path(__file__).parent
_KF_ROOT = _HERE.resolve().parents[3]
for _p in (str(_KF_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rules import RuleEngine, is_visually_observable

_KB_DIR = _HERE.parent / "knowledge_base"
DATASET_TAG = "bird-uc200"


def _pair_id_from_filename(path: Path) -> str:
    """Derive a normalized pair_id from the knowledge_base filename."""
    stem = path.stem  # e.g. "american_crow_vs_fish_crow"
    # Normalize hyphens to underscores
    return stem.replace("-", "_")


def run(rule_engine: RuleEngine | None = None, dry_run: bool = False) -> None:
    """Import all knowledge_base rules into rule_engine.

    Args:
        rule_engine: RuleEngine instance. Created fresh if None.
        dry_run: If True, print what would be imported without saving.
    """
    if rule_engine is None:
        rule_engine = RuleEngine(dataset_tag=DATASET_TAG)

    kb_files = sorted(_KB_DIR.glob("*.json"))
    if not kb_files:
        print(f"No JSON files found in {_KB_DIR}")
        return

    total_seen  = 0
    total_added = 0
    total_skipped_obs = 0  # rejected by observability filter
    total_skipped_dup = 0  # already exists in rules.json

    print(f"{'DRY RUN — ' if dry_run else ''}Migrating {len(kb_files)} knowledge_base file(s) "
          f"into {rule_engine.path}\n")

    for kb_path in kb_files:
        pair_id = _pair_id_from_filename(kb_path)
        try:
            data = json.loads(kb_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  [{kb_path.name}] ERROR reading file: {e}")
            continue

        rules_raw = data.get("rules", [])
        pair_added = 0
        pair_skipped_obs = 0

        for raw in rules_raw:
            total_seen += 1
            rule_text = raw.get("rule", "").strip()
            favors    = raw.get("favors", "").strip()
            confidence = raw.get("confidence", "medium").strip()

            if not rule_text or not favors:
                continue

            condition   = rule_text
            rule_action = f"Classify as {favors} (visual evidence confidence: {confidence})"
            tags        = [DATASET_TAG, pair_id]

            # Check observability filter (would be applied by add_rule)
            combined = f"{condition} {rule_action}"
            if not is_visually_observable(combined):
                if dry_run:
                    print(f"  [SKIP-nonvisual] {rule_text[:80]}")
                pair_skipped_obs += 1
                total_skipped_obs += 1
                continue

            if dry_run:
                print(f"  [ADD] {condition[:80]}")
                print(f"        -> {rule_action}")
                pair_added += 1
                total_added += 1
                continue

            # Add with observability_filter=True (belt-and-suspenders)
            result = rule_engine.add_rule(
                condition=condition,
                action=rule_action,
                source="knowledge_base",
                source_task=pair_id,
                tags=tags,
                observability_filter=True,
            )
            if result is not None:
                pair_added += 1
                total_added += 1
            else:
                total_skipped_dup += 1

        print(f"  {kb_path.name:55s}  "
              f"added={pair_added}  skipped_nonvisual={pair_skipped_obs}")

    print(f"\nTotal: seen={total_seen}  added={total_added}  "
          f"skipped_nonvisual={total_skipped_obs}  "
          f"skipped_duplicate={total_skipped_dup}")

    if not dry_run:
        rule_engine.save()
        print(f"Rules saved to {rule_engine.path}")
        stats = rule_engine.stats_summary()
        print(f"Rule stats: {stats}")
    else:
        print("\n[dry-run] No changes written.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Migrate knowledge_base rules into KF RuleEngine")
    p.add_argument("--dry-run", action="store_true", help="Preview without saving")
    p.add_argument("--rules",   default="", help="Path to rules.json (default: auto)")
    args = p.parse_args()

    engine = RuleEngine(args.rules or None, dataset_tag=DATASET_TAG)
    run(engine, dry_run=args.dry_run)
