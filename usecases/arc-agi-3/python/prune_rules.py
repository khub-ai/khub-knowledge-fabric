"""
prune_rules.py — One-time rule-base cleanup script.

Removes three categories of dead weight from rules.json:
  1. Deprecated/archived rules     — already soft-deleted, reclaim space
  2. Stale-orphan candidates       — tasks_seen==0, never evaluated
  3. Stale co-occurrence candidates — seen 20+ times without ever firing

Run from the arc-agi-3/python directory:
    python prune_rules.py [--dry-run] [--co-occ-threshold N]

Going forward, hard_prune() is called automatically at the end of every
episode in ensemble.py, so this script is only needed for the initial
cleanup of the pre-existing bloat.
"""

import argparse
import sys
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
_HERE    = Path(__file__).resolve().parent
_KF_ROOT = _HERE.parents[2]
if str(_KF_ROOT) not in sys.path:
    sys.path.insert(0, str(_KF_ROOT))

from rules import RuleEngine, DEFAULT_PATH  # noqa: E402 (arc-agi-3 shim)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prune stale rules from rules.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be removed without writing changes")
    ap.add_argument("--co-occ-threshold", type=int, default=20, metavar="N",
                    help="Hard-delete co-occurrence candidates with tasks_seen >= N "
                         "and 0 fires (default: 20)")
    ap.add_argument("--rules-path", type=Path, default=DEFAULT_PATH,
                    help=f"Path to rules.json (default: {DEFAULT_PATH})")
    args = ap.parse_args()

    engine = RuleEngine(path=args.rules_path, dataset_tag="arc-agi-3")
    before = len(engine.rules)

    print(f"Rules before pruning: {before}")
    print(f"  active:     {sum(1 for r in engine.rules if r['status']=='active')}")
    print(f"  candidate:  {sum(1 for r in engine.rules if r['status']=='candidate')}")
    print(f"  deprecated: {sum(1 for r in engine.rules if r['status']=='deprecated')}")
    print(f"  flagged:    {sum(1 for r in engine.rules if r['status']=='flagged')}")
    print()

    if args.dry_run:
        # Simulate without writing
        def _total_fires(r):
            return sum(ns.get("fires", 0) for ns in r.get("stats_by_ns", {}).values())

        deprecated = [r for r in engine.rules if r["status"] in ("deprecated", "archived")]
        orphans    = [r for r in engine.rules
                      if r["status"] == "candidate" and r.get("tasks_seen", 0) == 0]
        co_stale   = [r for r in engine.rules
                      if "co-occurrence" in r.get("tags", [])
                      and r["status"] == "candidate"
                      and _total_fires(r) == 0
                      and r.get("tasks_seen", 0) >= args.co_occ_threshold]

        print("[DRY RUN] Would remove:")
        print(f"  {len(deprecated):4d} deprecated/archived")
        print(f"  {len(orphans):4d} stale orphans (tasks_seen=0)")
        print(f"  {len(co_stale):4d} stale co-occurrence (tasks_seen >= {args.co_occ_threshold})")
        total = len({r["id"] for r in deprecated + orphans + co_stale})
        print(f"  {total:4d} total  ->  {before - total} would remain")
    else:
        counts = engine.hard_prune(
            remove_deprecated=True,
            remove_stale_orphans=True,
            co_occ_stale_tasks=args.co_occ_threshold,
        )
        after = len(engine.rules)
        print(f"Pruned {counts['total']} rules:")
        print(f"  {counts['deprecated']:4d} deprecated/archived")
        print(f"  {counts['orphans']:4d} stale orphans (tasks_seen=0)")
        print(f"  {counts['co_occ_stale']:4d} stale co-occurrence (tasks_seen >= {args.co_occ_threshold})")
        print()
        print(f"Rules after pruning: {after}")
        print(f"  active:     {sum(1 for r in engine.rules if r['status']=='active')}")
        print(f"  candidate:  {sum(1 for r in engine.rules if r['status']=='candidate')}")
        print(f"  flagged:    {sum(1 for r in engine.rules if r['status']=='flagged')}")


if __name__ == "__main__":
    main()
