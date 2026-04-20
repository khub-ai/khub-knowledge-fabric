"""Run Round 2 of the ARC-AGI-3 DD loop.

Usage:
  python run_round2.py --round1-session <path>                  # Round-2A
  python run_round2.py --round1-session <path> --prior <file>   # Round-2B

Reads a Round-1 session dir (must contain tutor_reply.json, pupil_reply.json,
tutor_probe_results.json), builds a grounding pack, asks both models to
revise, writes a Round-2 session dir under sessions/ with:
  grounding_pack.json
  tutor_round2_reply.json
  pupil_round2_reply.json
  diagnostic_report.md
  manifest.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import backends
import prompts as round1_prompts                 # noqa: F401 (not used directly)
from round2_prompts import SYSTEM_ROUND2, build_round2_user_message
from grounding_pack import build_grounding_pack
from diagnostic_report import build_markdown


TUTOR_MODEL = "claude-sonnet-4-6"
PUPIL_MODEL = "google/gemma-4-26b-a4b-it"


def extract_json(text: str) -> dict:
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))
    first = text.find("{")
    if first < 0:
        raise ValueError(f"no JSON object in reply: {text[:200]!r}")
    depth = 0
    for i in range(first, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[first:i + 1])
    raise ValueError("unterminated JSON in reply")


def _format_frame_text(grid: list[list[int]]) -> str:
    rows = [", ".join(f"{v:2d}" for v in row) for row in grid]
    return "[\n" + ",\n".join(f"  [{r}]" for r in rows) + "\n]"


def call_model(
    role: str, user_msg: str, *, image_b64: str | None = None,
) -> dict:
    # Round-2 replies are notably longer than Round 1 (revision_notes +
    # re-emitted schema), so bump max_tokens to avoid truncation.
    if role == "TUTOR":
        return backends.call_anthropic(
            model=TUTOR_MODEL, system=SYSTEM_ROUND2, user=user_msg,
            image_b64=image_b64, max_tokens=8000,
        )
    elif role == "PUPIL":
        return backends.call_openrouter(
            model=PUPIL_MODEL, system=SYSTEM_ROUND2, user=user_msg,
            image_b64=image_b64, max_tokens=8000,
        )
    else:
        raise ValueError(role)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round1-session", required=True,
                    help="path to the Round-1 session directory")
    ap.add_argument("--prior", default=None,
                    help="path to a .txt/.md file with prior knowledge "
                         "(triggers Round-2B variant)")
    ap.add_argument("--variant", default=None,
                    help="label for this run (2A or 2B).  Defaults based on --prior")
    ap.add_argument("--sessions-dir", default=str(HERE.parent / "benchmarks" / "sessions"))
    ap.add_argument("--frames-dir",   default=str(HERE.parent / "benchmarks" / "frames"))
    ap.add_argument("--use-image",    action="store_true", default=False)
    a = ap.parse_args()

    round1_dir = Path(a.round1_session)
    if not round1_dir.exists():
        raise SystemExit(f"Round-1 session not found: {round1_dir}")

    prior_knowledge = None
    if a.prior:
        prior_knowledge = Path(a.prior).read_text(encoding="utf-8").strip()
        variant = a.variant or "2B"
    else:
        variant = a.variant or "2A"

    # Load Round 1 artefacts.
    tutor_r1 = json.loads((round1_dir / "tutor_reply.json").read_text(encoding="utf-8"))
    pupil_r1 = json.loads((round1_dir / "pupil_reply.json").read_text(encoding="utf-8"))
    round1_manifest = json.loads((round1_dir / "manifest.json").read_text(encoding="utf-8"))
    game_id = round1_manifest["game_id"]

    # Need fresh frame + env-info for the prompt.  Capture again -- it's idempotent.
    from frame_capture import capture
    payload = capture(game_id, Path(a.frames_dir))
    frame_text = _format_frame_text(payload["grid"])

    # Build grounding pack from TUTOR's executed probes.
    pack = build_grounding_pack(round1_dir)

    # Session dir for Round 2.
    trial_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session_dir = Path(a.sessions_dir) / f"trial_{trial_id}_round{variant}"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "grounding_pack.json").write_text(
        json.dumps(pack, indent=2), encoding="utf-8"
    )
    if prior_knowledge:
        (session_dir / "prior_knowledge.txt").write_text(
            prior_knowledge, encoding="utf-8"
        )

    replies: dict[str, dict] = {}
    for role, r1 in (("TUTOR", tutor_r1), ("PUPIL", pupil_r1)):
        print(f"-- Round 2 calling {role} --")
        user_msg = build_round2_user_message(
            frame_text       = frame_text,
            action_labels    = payload["available_actions"],
            state            = payload["state"],
            levels_completed = payload["levels_completed"],
            win_levels       = payload["win_levels"],
            game_id          = payload["game_id"],
            title            = payload["title"],
            tags             = payload["tags"],
            level            = 1,
            round1_assessment = (r1 or {}).get("assessment") or {},
            grounding_pack   = pack,
            prior_knowledge  = prior_knowledge,
        )
        try:
            rsp = call_model(role, user_msg)
            cache_tag = " (cache hit)" if rsp.get("_cache_hit") else ""
            print(f"   latency: {rsp['latency_ms']} ms, "
                  f"reply chars: {len(rsp['reply'])}{cache_tag}")
            try:
                parsed = extract_json(rsp["reply"])
                parse_err = None
            except Exception as e:  # noqa: BLE001
                parsed = None
                parse_err = f"{type(e).__name__}: {e}"
                print(f"   JSON parse error: {parse_err}")
            entry = {
                "model":       rsp["model"],
                "latency_ms":  rsp["latency_ms"],
                "raw_reply":   rsp["reply"],
                "parse_error": parse_err,
                "assessment":  parsed,
            }
        except Exception as e:  # noqa: BLE001
            print(f"   call error: {e}")
            entry = {"model": "??", "call_error": f"{type(e).__name__}: {e}"}
        replies[role] = entry
        (session_dir / f"{role.lower()}_round2_reply.json").write_text(
            json.dumps(entry, indent=2), encoding="utf-8"
        )

    # Build diagnostic.
    md = build_markdown(
        trial_id      = trial_id,
        variant       = variant,
        prior_knowledge = prior_knowledge,
        round1_tutor  = (tutor_r1 or {}).get("assessment") or {},
        round1_pupil  = (pupil_r1 or {}).get("assessment") or {},
        round2_tutor  = replies["TUTOR"].get("assessment") or {},
        round2_pupil  = replies["PUPIL"].get("assessment") or {},
        facts         = pack,
        tutor_model   = replies["TUTOR"].get("model", "?"),
        pupil_model   = replies["PUPIL"].get("model", "?"),
    )
    (session_dir / "diagnostic_report.md").write_text(md, encoding="utf-8")

    manifest = {
        "trial_id":       trial_id,
        "variant":        variant,
        "round1_session": str(round1_dir),
        "game_id":        game_id,
        "prior_knowledge_file": "prior_knowledge.txt" if prior_knowledge else None,
        "tutor_model":    TUTOR_MODEL,
        "pupil_model":    PUPIL_MODEL,
        "created_at":     datetime.now(timezone.utc).isoformat(),
        "files": [
            "grounding_pack.json",
            "tutor_round2_reply.json",
            "pupil_round2_reply.json",
            "diagnostic_report.md",
        ] + (["prior_knowledge.txt"] if prior_knowledge else []),
    }
    (session_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    # Update the preview panel: frame + Round-1 exchanges + Round-2 revnotes.
    from preview_html import render_index
    render_index(
        frames_dir     = Path(a.frames_dir),
        png_name       = f"{game_id}_L1_init.png",
        frame_payload  = payload,
        round1_session = round1_dir,
        round2_session = session_dir,
    )
    print(f"\nwrote Round-{variant} session: {session_dir}")


if __name__ == "__main__":
    main()
