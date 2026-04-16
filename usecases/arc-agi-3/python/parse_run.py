#!/usr/bin/env python
"""parse_run.py — quick summary extractor for harness playlogs.

Usage:
  python parse_run.py playlogs/run60.log
  python parse_run.py playlogs/run60.log --full   # also show EXPLORE lines
"""
import re
import sys
from pathlib import Path

SIGNALS = [
    (r"rc_visits_done=(\d+)",          "rc_visits"),
    (r"RING.*done.*target\s+(\S+)",    "ring_tgt"),
    (r"Detour to ring\s+(\S+)",        "ring_det"),
    (r"diff=(\d+).*\*\*\* LARGE",      "large_diff"),
    (r"levels=(\d+)",                   "levels"),
    (r"Episode\s+\d+ complete",         "ep_end"),
    (r"levels=2",                       "LEVEL2!"),
    (r"CONFIRMED ring",                 "ring_cfm"),
    (r"DYN-EXPLORE\] .+ -> (.+):",     "explore"),
    (r"\[WIN\]|\blevels=2\b",           "WIN"),
]

def parse(path: str, full: bool = False):
    log = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = log.splitlines()

    prev_rc = None
    ep = 0
    for i, line in enumerate(lines):
        if "Episode" in line and "complete" in line:
            ep += 1
            print(f"\n=== {line.strip()} ===")
            continue

        # rc_visits changes
        m = re.search(r"rc_visits_done=(\d+)", line)
        if m and m.group(1) != prev_rc:
            prev_rc = m.group(1)
            print(f"  [rc_visits={prev_rc}] {line.strip()[:120]}")
            continue

        # Large diffs (resets, RC triggers)
        if "LARGE DIFF" in line:
            print(f"  {line.strip()[:120]}")
            continue

        # Ring events
        if "[RING]" in line and ("Detour" in line or "CONFIRMED" in line
                                 or "RC done" in line or "Cannot safely" in line):
            print(f"  {line.strip()[:120]}")
            continue

        # Level 2 completion
        if "levels=2" in line:
            print(f"\n  *** LEVEL 2 REACHED: {line.strip()} ***\n")
            continue

        # Explore targets (verbose)
        if full and "DYN-EXPLORE" in line:
            print(f"  {line.strip()[:120]}")

    # Final summary
    ep_lines = [l for l in lines if "Episode" in l and "complete" in l]
    level2_lines = [l for l in lines if "levels=2" in l]
    print(f"\n--- Summary ---")
    print(f"Episodes completed: {len(ep_lines)}")
    print(f"Level 2 events:     {len(level2_lines)}")
    for l in ep_lines:
        print(f"  {l.strip()}")

if __name__ == "__main__":
    full = "--full" in sys.argv
    path = next((a for a in sys.argv[1:] if not a.startswith("--")), None)
    if not path:
        print("Usage: python parse_run.py <logfile> [--full]")
        sys.exit(1)
    parse(path, full)
