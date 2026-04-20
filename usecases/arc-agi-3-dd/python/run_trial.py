"""End-to-end first trial:
  1. Capture ls20 L1 initial frame.
  2. Send INITIAL_ASSESSMENT to TUTOR and PUPIL.
  3. Save both JSON replies.
  4. Parse probes, execute DSL-valid ones.
  5. Write a diff report.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import backends
import prompts
from dsl import parse_probe
from dsl_executor import run_probe
from diff_report import build_report


TUTOR_MODEL = "claude-sonnet-4-6"
PUPIL_MODEL = "google/gemma-4-26b-a4b-it"


def extract_json(text: str) -> dict:
    """Model sometimes wraps JSON in code fences.  Recover robustly."""
    # Strip code fences.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))
    # Find the first { ... } balanced object.
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
                return json.loads(text[first:i+1])
    raise ValueError("unterminated JSON in reply")


def call_model(role: str, payload: dict, png_path: Path, *, use_image: bool) -> dict:
    user_msg = prompts.build_user_message(
        frame_text       = prompts.format_frame_text(payload["grid"]),
        action_labels    = payload["available_actions"],
        state            = payload["state"],
        levels_completed = payload["levels_completed"],
        win_levels       = payload["win_levels"],
        game_id          = payload["game_id"],
        title            = payload["title"],
        tags             = payload["tags"],
        level            = 1,
    )
    image_b64 = base64.b64encode(png_path.read_bytes()).decode("ascii") if use_image else None

    if role == "TUTOR":
        return backends.call_anthropic(
            model     = TUTOR_MODEL,
            system    = prompts.SYSTEM,
            user      = user_msg,
            image_b64 = image_b64,
        )
    elif role == "PUPIL":
        return backends.call_openrouter(
            model     = PUPIL_MODEL,
            system    = prompts.SYSTEM,
            user      = user_msg,
            image_b64 = image_b64,
        )
    else:
        raise ValueError(role)


def process_probes(
    assessment:     dict,
    available:      list[str],
    frame_shape:    tuple[int, int],
    game_id:        str,
) -> list[dict]:
    elements = {int(e["id"]): e.get("bbox") for e in assessment.get("elements", [])}
    valid_ids = set(elements.keys())
    results = []
    for probe_json in assessment.get("probes", []) or []:
        parsed = parse_probe(probe_json, available, valid_ids, frame_shape)
        if parsed.errors:
            results.append({
                "probe_id":     parsed.probe_id,
                "executed":     False,
                "parse_errors": parsed.errors,
                "hypothesis":   parsed.hypothesis,
            })
            continue
        results.append(run_probe(parsed, game_id, elements))
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="ls20-9607627b")
    ap.add_argument("--frames-dir",    default=str(HERE.parent / "benchmarks" / "frames"))
    ap.add_argument("--sessions-dir",  default=str(HERE.parent / "benchmarks" / "sessions"))
    ap.add_argument("--use-image",     action="store_true", default=True)
    ap.add_argument("--no-image",      dest="use_image", action="store_false")
    ap.add_argument("--trial-id",      default=None)
    a = ap.parse_args()

    frames_dir   = Path(a.frames_dir)
    sessions_dir = Path(a.sessions_dir)
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # 1. Capture (idempotent — re-captures every run; cheap).
    from frame_capture import capture
    payload = capture(a.game, frames_dir)
    png_path = frames_dir / f"{a.game}_L1_init.png"

    trial_id = a.trial_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session_dir = sessions_dir / f"trial_{trial_id}"
    session_dir.mkdir(parents=True, exist_ok=True)

    # 2. Call both models.
    results: dict = {"trial_id": trial_id, "game_id": a.game, "used_image": a.use_image}
    for role in ("TUTOR", "PUPIL"):
        print(f"-- calling {role} --")
        try:
            rsp = call_model(role, payload, png_path, use_image=a.use_image)
            cache_tag = " (cache hit)" if rsp.get("_cache_hit") else ""
            print(f"   latency: {rsp['latency_ms']} ms, reply chars: {len(rsp['reply'])}{cache_tag}")
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
        results[role] = entry

        # Per-model dump (keeps raw reply for auditing).
        (session_dir / f"{role.lower()}_reply.json").write_text(
            json.dumps(entry, indent=2)
        )

    # 3. Execute probes (where parsable).
    frame_shape = tuple(payload["frame_shape"])
    for role in ("TUTOR", "PUPIL"):
        entry = results[role]
        a_json = entry.get("assessment")
        if not a_json:
            entry["probe_results"] = []
            continue
        entry["probe_results"] = process_probes(
            a_json, payload["available_actions"], frame_shape, a.game
        )
        (session_dir / f"{role.lower()}_probe_results.json").write_text(
            json.dumps(entry["probe_results"], indent=2)
        )

    # 4. Diff report.
    report = build_report(results["TUTOR"], results["PUPIL"])
    (session_dir / "diff_report.json").write_text(json.dumps(report, indent=2))
    (session_dir / "diff_report.md").write_text(
        report_to_markdown(report, results, trial_id), encoding="utf-8"
    )

    # 5. Session manifest.
    manifest = {
        "trial_id":    trial_id,
        "game_id":     a.game,
        "used_image":  a.use_image,
        "tutor_model": TUTOR_MODEL,
        "pupil_model": PUPIL_MODEL,
        "created_at":  datetime.now(timezone.utc).isoformat(),
        "files": [
            "tutor_reply.json",
            "pupil_reply.json",
            "tutor_probe_results.json",
            "pupil_probe_results.json",
            "diff_report.json",
            "diff_report.md",
        ],
    }
    (session_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Update the preview panel (port 8753) so the latest exchanges show
    # under the frame image.
    from preview_html import render_index
    render_index(
        frames_dir     = frames_dir,
        png_name       = png_path.name,
        frame_payload  = payload,
        round1_session = session_dir,
        round2_session = None,
    )
    print(f"\nwrote session: {session_dir}")


def report_to_markdown(report: dict, results: dict, trial_id: str) -> str:
    md = [f"# Trial {trial_id} — ls20 L1 INITIAL_ASSESSMENT diff\n"]
    md.append(f"- TUTOR: `{results['TUTOR'].get('model','?')}`  "
              f"latency={results['TUTOR'].get('latency_ms','?')} ms")
    md.append(f"- PUPIL: `{results['PUPIL'].get('model','?')}`  "
              f"latency={results['PUPIL'].get('latency_ms','?')} ms\n")
    md.append("## Parse status\n")
    for r in ("TUTOR", "PUPIL"):
        md.append(f"- {r}: {'OK' if results[r].get('assessment') else 'FAIL — ' + str(results[r].get('parse_error') or results[r].get('call_error'))}")
    md.append("")
    md.append("## Section summaries\n")
    for section in ("elements", "similar_groups", "initial_strategy", "probes"):
        md.append(f"### {section}")
        md.append("| model | summary |")
        md.append("|---|---|")
        for r in ("TUTOR", "PUPIL"):
            a = results[r].get("assessment") or {}
            summary = report.get("section_summaries", {}).get(section, {}).get(r, "—")
            md.append(f"| {r} | {summary} |")
        md.append("")
    md.append("## Comparison metrics\n")
    for k, v in (report.get("metrics") or {}).items():
        md.append(f"- **{k}**: {v}")
    md.append("")
    md.append("## Probe execution\n")
    for r in ("TUTOR", "PUPIL"):
        md.append(f"### {r}")
        for pr in results[r].get("probe_results", []):
            if not pr.get("executed"):
                md.append(f"- {pr.get('probe_id','?')}: REJECTED ({pr.get('parse_errors') or pr.get('exec_error')})")
            else:
                obs_summary = ", ".join(f"{o['kind']}={_brief(o)}" for o in pr.get("observations", []))
                md.append(f"- {pr.get('probe_id','?')}: OK → {obs_summary}")
        md.append("")
    return "\n".join(md)


def _brief(o: dict) -> str:
    if "value" in o:
        return str(o["value"])
    if o.get("kind") == "ELEMENT_MOVED":
        return f"moved={o.get('moved')} post={o.get('post_bbox')}"
    return "?"


if __name__ == "__main__":
    main()
