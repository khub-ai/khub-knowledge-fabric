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


def apply_patch(kb_text: str, patch: dict) -> tuple[str, list[str]]:
    """Apply a JSON patch dict to the KB, returning (updated_kb_text, warnings).

    Supported patch keys:
      belief_updates, hypothesis_updates, trap_updates, open_experiment_updates

    Per-key ops:
      {"op": "add",  "id": "<new_id>", ...rest of item fields...}
        — appends a new item if <id> not already present in the section.

      {"op": "corroborate", "id": "<existing_id>", "session_id": "<sid>"}
        — adds session_id to corroborating_sessions; promotes provenance
          from "discovered" to "corroborated" if not already.

      {"op": "update_status", "id": "<existing_id>",
       "status": "supported|refuted|proposed",
       "evidence_session": "<sid>",  (optional)
       "note": "one sentence"}       (optional)
        — updates hypothesis status and appends to evidence.supporting or
          evidence.contradicting depending on new status.

      {"op": "close", "id": "<existing_id>", "session_id": "<sid>",
       "result": "..."}
        — marks an open_experiment resolved by appending a result note.
    """
    split = _split_yaml_block(kb_text)
    if split is None:
        return kb_text, ["no YAML block found — patch not applied"]
    prose_before, yaml_body, prose_after = split

    try:
        data = yaml.safe_load(yaml_body) or {}
    except yaml.YAMLError as exc:
        return kb_text, [f"YAML parse error: {exc}"]

    section_map = {
        "belief_updates":           "beliefs",
        "hypothesis_updates":       "hypotheses",
        "trap_updates":             "traps",
        "open_experiment_updates":  "open_experiments",
    }

    warnings: list[str] = []

    for patch_key, section_key in section_map.items():
        updates = patch.get(patch_key) or []
        if not updates:
            continue
        items: list = data.setdefault(section_key, [])
        existing_ids = {
            item.get("id")
            for item in items
            if isinstance(item, dict) and item.get("id")
        }

        for upd in updates:
            if not isinstance(upd, dict):
                continue
            op  = upd.get("op")
            uid = upd.get("id")

            if op == "add":
                if uid in existing_ids:
                    warnings.append(
                        f"{section_key}: id {uid!r} already exists — skipped add"
                    )
                    continue
                new_item = {k: v for k, v in upd.items() if k != "op"}
                items.append(new_item)
                existing_ids.add(uid)

            elif op == "corroborate":
                found = False
                for item in items:
                    if isinstance(item, dict) and item.get("id") == uid:
                        sess = upd.get("session_id") or ""
                        if sess:
                            cs = item.setdefault("corroborating_sessions", [])
                            if sess not in cs:
                                cs.append(sess)
                        if item.get("provenance") == "discovered":
                            item["provenance"] = "corroborated"
                        found = True
                        break
                if not found:
                    warnings.append(
                        f"{section_key}: id {uid!r} not found for corroborate"
                    )

            elif op == "update_status":
                found = False
                for item in items:
                    if isinstance(item, dict) and item.get("id") == uid:
                        new_status = upd.get("status")
                        if new_status:
                            item["status"] = new_status
                        evidence_session = upd.get("evidence_session") or ""
                        note = upd.get("note") or ""
                        if evidence_session:
                            ev = item.setdefault("evidence", {})
                            if new_status in ("supported", "corroborated"):
                                supp = ev.setdefault("supporting", [])
                                if evidence_session not in supp:
                                    supp.append(evidence_session)
                            elif new_status == "refuted":
                                contra = ev.setdefault("contradicting", [])
                                if evidence_session not in contra:
                                    contra.append(evidence_session)
                        if note:
                            item.setdefault("notes", []).append(note)
                        found = True
                        break
                if not found:
                    warnings.append(
                        f"{section_key}: id {uid!r} not found for update_status"
                    )

            elif op == "close":
                found = False
                for item in items:
                    if isinstance(item, dict) and item.get("id") == uid:
                        result = upd.get("result") or ""
                        sess   = upd.get("session_id") or ""
                        note   = f"[closed by {sess}] {result}".strip()
                        item.setdefault("notes", []).append(note)
                        item["status"] = "closed"
                        found = True
                        break
                if not found:
                    warnings.append(
                        f"{section_key}: id {uid!r} not found for close"
                    )
            else:
                warnings.append(f"unknown op {op!r} for id {uid!r} — skipped")

    new_body = yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
    return f"{prose_before}```yaml\n{new_body}```{prose_after}", warnings


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
