"""Knowledge-base manipulation tools for the provenance-tagged KB format.

Provides `strip_authored(kb_text)` which returns a version of the KB with
all `provenance: authored-by-claude-code` and `provenance:
authored-interpretation` items removed (plus corroborated items that
don't have enough corroborating sessions yet).  This is used to evaluate
TUTOR on a "rediscovery" loop where the scaffolding Claude Code added
has been peeled back.

The KB format is markdown with a fenced ```yaml ... ``` block
containing `beliefs:`, `hypotheses:`, `traps:`, and `open_experiments:`
lists.  Each item has `provenance: <value>` plus content fields.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


# Provenance values considered "authored" and removed by strip_authored.
_AUTHORED = {"authored-by-claude-code", "authored-interpretation"}

# A corroborated item is kept only if it has at least this many entries
# in `corroborating_sessions` (as evidence that TUTOR has verified it).
_MIN_CORROBORATING_SESSIONS = 2


def _split_yaml_block(kb_text: str) -> tuple[str, str, str] | None:
    """Split KB text into (prose_before, yaml_block_body, prose_after).

    Returns None if no yaml block is found.
    """
    start_marker = "```yaml"
    end_marker = "```"
    start = kb_text.find(start_marker)
    if start < 0:
        return None
    yaml_start = start + len(start_marker)
    # Find the NEXT closing ``` after the start marker
    end = kb_text.find(end_marker, yaml_start)
    if end < 0:
        return None
    prose_before = kb_text[:start]
    yaml_body    = kb_text[yaml_start:end]
    prose_after  = kb_text[end + len(end_marker):]
    return prose_before, yaml_body, prose_after


def _filter_items(
    items:           list[dict[str, Any]],
    keep_authored:   bool,
) -> list[dict[str, Any]]:
    """Filter items by provenance according to strip-authored rules."""
    kept: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            kept.append(item)
            continue
        prov = item.get("provenance", "discovered")
        if prov in _AUTHORED and not keep_authored:
            continue
        if prov == "corroborated":
            sessions = item.get("corroborating_sessions") or []
            if len(sessions) < _MIN_CORROBORATING_SESSIONS and not keep_authored:
                continue
        kept.append(item)
    return kept


def strip_authored(kb_text: str) -> str:
    """Return a copy of `kb_text` with authored items removed.

    Prose sections outside the YAML block are kept unchanged.  If there
    is no YAML block, the text is returned unchanged.
    """
    split = _split_yaml_block(kb_text)
    if split is None:
        return kb_text
    prose_before, yaml_body, prose_after = split

    try:
        data = yaml.safe_load(yaml_body) or {}
    except yaml.YAMLError:
        # Malformed — better to return original than to silently corrupt
        return kb_text

    for key in ("beliefs", "hypotheses", "traps", "open_experiments"):
        if key in data and isinstance(data[key], list):
            data[key] = _filter_items(data[key], keep_authored=False)

    new_body = yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
    return f"{prose_before}```yaml\n{new_body}```{prose_after}"


def provenance_summary(kb_text: str) -> dict[str, dict[str, int]]:
    """Count items by (section, provenance) for reporting purposes."""
    split = _split_yaml_block(kb_text)
    if split is None:
        return {}
    _, yaml_body, _ = split
    try:
        data = yaml.safe_load(yaml_body) or {}
    except yaml.YAMLError:
        return {}
    out: dict[str, dict[str, int]] = {}
    for key in ("beliefs", "hypotheses", "traps", "open_experiments"):
        items = data.get(key) or []
        sec: dict[str, int] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            prov = item.get("provenance", "unknown")
            sec[prov] = sec.get(prov, 0) + 1
        if sec:
            out[key] = sec
    return out


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(
        description="Strip authored items from a KB for rediscovery evaluation."
    )
    ap.add_argument("kb_path", type=Path,
                    help="path to the per-game KB markdown file")
    ap.add_argument("--summary", action="store_true",
                    help="print provenance counts instead of stripped text")
    ap.add_argument("--out", type=Path, default=None,
                    help="write stripped KB to this path (default: stdout)")
    a = ap.parse_args()

    kb = a.kb_path.read_text(encoding="utf-8")
    if a.summary:
        import json
        print(json.dumps(provenance_summary(kb), indent=2))
        sys.exit(0)
    stripped = strip_authored(kb)
    if a.out:
        a.out.write_text(stripped, encoding="utf-8")
        print(f"wrote {a.out} ({len(stripped)} chars, original {len(kb)})")
    else:
        sys.stdout.write(stripped)
