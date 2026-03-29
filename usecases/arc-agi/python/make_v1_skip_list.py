"""
make_v1_skip_list.py — fetch the 400 ARC-AGI-v1 training task IDs from GitHub
and save them as v1_ids.json in this directory.

Run once before using --skip-ids:
  python make_v1_skip_list.py
  python harness.py --all --skip-ids v1_ids.json --output results_v2only.json

The script hits the GitHub contents API (no auth required for public repos).
"""

import json
import sys
import urllib.request
from pathlib import Path

GITHUB_API = (
    "https://api.github.com/repos/fchollet/ARC-AGI/contents/data/training"
)
OUT_FILE = Path(__file__).parent / "v1_ids.json"


def fetch_v1_ids():
    req = urllib.request.Request(GITHUB_API, headers={"User-Agent": "arc-ensemble"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        entries = json.loads(resp.read().decode())
    ids = [e["name"][:-5] for e in entries if e["name"].endswith(".json")]
    return sorted(ids)


def main() -> None:
    print(f"Fetching v1 training IDs from {GITHUB_API} ...")
    try:
        ids = fetch_v1_ids()
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print(f"Found {len(ids)} v1 task IDs.")
    OUT_FILE.write_text(json.dumps(ids, indent=2), encoding="utf-8")
    print(f"Saved to {OUT_FILE}")
    print()
    print("Usage:")
    print(f"  python harness.py --all --skip-ids {OUT_FILE.name} "
          f"--output results_v2only.json --dataset v2-training")


if __name__ == "__main__":
    main()
