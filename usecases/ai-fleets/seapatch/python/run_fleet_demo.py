"""
run_fleet_demo.py — Phase 2 fleet broadcast demo for SeaPatch.

Demonstrates the full SeaPatch loop end-to-end without a rendering engine:

  1. Register 40-drone fleet (38 scouts + 2 commanders)
  2. Inject SeaDronesSee frames into archive via SeaDronesSeebridge
  3. Pick a failure frame (person classified as whitecap by scout S22)
  4. Run DD session → cross-modal TUTOR → tier grounding → pool validation
  5. Broadcast accepted rules fleet-wide
  6. Reprocess 45-minute archive → find previously missed persons
  7. Print summary report

Phase 1 prerequisite:
    python pool_builder.py \\
        --dataset-root data/seadronessee/ \\
        --split val \\
        --pool-dir data/pool/ \\
        --n-positive 10 \\
        --n-negative 20 \\
        --select-hardest \\
        --failure-dir data/failure_frames/

Usage:
    python run_fleet_demo.py \\
        --dataset-root data/seadronessee/ \\
        --pool-dir data/pool/ \\
        --failure-dir data/failure_frames/ \\
        --output results/fleet_demo.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[3]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_THIS_DIR))

from agents import run_maritime_dd_session
from archive import FrameBuffer, reprocess_archive
from fleet import FleetManager
from domain_config import MARITIME_SAR_CONFIG, TIER_OBSERVABILITY, CONFUSABLE_PAIRS
from simulation.seadronessee_bridge import SeaDronesSeebridge
from simulation.thermal_oracle import oracle_for_frame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_failure_frame(failure_dir: Path) -> tuple[Path, str] | None:
    """Return (image_path, ground_truth_label) for the first failure candidate."""
    manifest_path = failure_dir / "failure_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest:
            entry = manifest[0]
            img_path = failure_dir / entry["path"]
            if img_path.exists():
                return img_path, entry["label"]

    # Fallback: first image file in directory
    _EXTS = {".jpg", ".jpeg", ".png", ".webp"}
    for p in sorted(failure_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in _EXTS:
            return p, "person_in_water"
    return None


def load_pool(pool_dir: Path) -> list[tuple[str, str]]:
    manifest = pool_dir / "pool_manifest.json"
    if manifest.exists():
        entries = json.loads(manifest.read_text())
        return [(str(pool_dir / e["path"]), e["label"]) for e in entries]
    _EXTS = {".jpg", ".jpeg", ".png", ".webp"}
    pool = []
    for class_dir in sorted(pool_dir.iterdir()):
        if class_dir.is_dir():
            for p in sorted(class_dir.iterdir()):
                if p.suffix.lower() in _EXTS:
                    pool.append((str(p), class_dir.name))
    return pool


def print_separator(label: str = "", width: int = 60) -> None:
    if label:
        pad = (width - len(label) - 2) // 2
        print(f"\n{'─' * pad} {label} {'─' * pad}")
    else:
        print(f"\n{'─' * width}")


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

async def run_demo(
    dataset_root: Path,
    pool_dir: Path,
    failure_dir: Path,
    coco_annotation: Path | None = None,
    n_scouts: int = 38,
    n_commanders: int = 2,
    lookback_seconds: int = 2700,
    tutor_model: str = "claude-opus-4-6",
    validator_model: str = "claude-sonnet-4-6",
    output_path: Path | None = None,
    quiet: bool = False,
) -> dict:
    demo_start = time.monotonic()
    report: dict = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "phases": {},
    }

    def log(msg: str) -> None:
        if not quiet:
            print(msg)

    # ------------------------------------------------------------------
    # Phase A: Fleet setup
    # ------------------------------------------------------------------
    print_separator("PHASE A: FLEET SETUP")
    fleet = FleetManager()
    scout_ids = fleet.register_scout_fleet(n_scouts)
    commander_ids = fleet.register_commander_fleet(n_commanders)
    log(f"Registered {len(scout_ids)} scouts + {len(commander_ids)} commanders")

    # ------------------------------------------------------------------
    # Phase B: Frame injection
    # ------------------------------------------------------------------
    print_separator("PHASE B: FRAME INJECTION")
    frame_buffer = FrameBuffer()
    bridge = SeaDronesSeebridge(
        dataset_root=dataset_root,
        coco_annotation=coco_annotation,
    )
    all_ids = scout_ids + commander_ids
    bridge.assign_sequences(all_ids)

    n_injected = bridge.inject_to_archive(
        frame_buffer,
        lookback_seconds=lookback_seconds,
        tier="scout",
        original_class="whitecap",
        original_confidence=0.91,
    )
    log(f"Injected {n_injected} confident-negative frames into archive "
        f"(representing {lookback_seconds // 60}m of scout history)")
    log(f"Archive summary: {frame_buffer.summary()}")
    report["phases"]["frame_injection"] = {
        "n_frames_injected": n_injected,
        "lookback_seconds": lookback_seconds,
        "archive_summary": frame_buffer.summary(),
    }

    # ------------------------------------------------------------------
    # Phase C: Pick failure frame + thermal confirmation
    # ------------------------------------------------------------------
    print_separator("PHASE C: FAILURE FRAME SELECTION")
    failure = load_failure_frame(failure_dir)
    if failure is None:
        log(f"ERROR: no failure frames found in {failure_dir}")
        log("Run pool_builder.py with --failure-dir first.")
        return {"error": "No failure frames found."}

    failure_path, ground_truth_label = failure
    oracle = oracle_for_frame(
        frame_path=str(failure_path),
        ground_truth_label=ground_truth_label,
        drone_id="C1",
    )
    log(f"Failure frame : {failure_path.name}")
    log(f"Ground truth  : {oracle.ground_truth_class}")
    log(f"Confirmation  : {oracle.confirmation_details[:100]}...")
    report["phases"]["failure_selection"] = {
        "failure_image": str(failure_path),
        "ground_truth": oracle.ground_truth_class,
        "confirmation_modality": oracle.confirmation_modality,
    }

    # ------------------------------------------------------------------
    # Phase D: DD session
    # ------------------------------------------------------------------
    print_separator("PHASE D: DIALOGIC DISTILLATION SESSION")
    pool_images = load_pool(pool_dir)
    if not pool_images:
        log(f"ERROR: no pool images found in {pool_dir}")
        return {"error": "No pool images found."}
    log(f"Pool: {len(pool_images)} frames")

    pair_info = next(
        (p for p in CONFUSABLE_PAIRS
         if {p["class_a"], p["class_b"]} == {"person_in_water", "whitecap"}),
        {"class_a": "person_in_water", "class_b": "whitecap",
         "pair_id": "person_in_water_vs_whitecap"},
    )

    session_start = time.monotonic()
    transcript = await run_maritime_dd_session(
        failure_image_path=str(failure_path),
        confirmation_modality=oracle.confirmation_modality,
        confirmation_details=oracle.confirmation_details,
        ground_truth_class=oracle.ground_truth_class,
        pupil_classification="whitecap",
        pupil_confidence=0.91,
        pool_images=pool_images,
        pair_info=pair_info,
        config=MARITIME_SAR_CONFIG,
        tier_observability=TIER_OBSERVABILITY,
        tutor_model=tutor_model,
        validator_model=validator_model,
        console=None,
    )
    session_elapsed = int((time.monotonic() - session_start) * 1000)
    outcome = transcript.get("outcome", "unknown")
    log(f"DD session outcome : {outcome.upper()} ({session_elapsed}ms)")

    if outcome != "accepted":
        log("DD session did not produce an accepted rule. Demo ends here.")
        report["phases"]["dd_session"] = {"outcome": outcome, "duration_ms": session_elapsed}
        report["outcome"] = "dd_failed"
        return report

    # Show accepted rules
    for tier, rule in transcript.get("final_rules", {}).items():
        log(f"  [{tier}] {rule.get('rule', '')[:100]}")
        for pc in rule.get("preconditions", [])[:3]:
            log(f"    - {pc[:90]}")

    report["phases"]["dd_session"] = {
        "outcome": outcome,
        "duration_ms": session_elapsed,
        "final_rules": {
            tier: {
                "rule": r.get("rule", ""),
                "preconditions": r.get("preconditions", []),
                "precision": transcript.get("pool_result_after_tighten", {}).get("precision")
                             if transcript.get("pool_result_after_tighten")
                             else transcript.get("pool_result", {}).get("precision"),
            }
            for tier, r in transcript.get("final_rules", {}).items()
        },
        "grounding_reports": transcript.get("grounding_reports", {}),
    }

    # ------------------------------------------------------------------
    # Phase E: Fleet broadcast
    # ------------------------------------------------------------------
    print_separator("PHASE E: FLEET BROADCAST")
    broadcast_start = time.monotonic()

    registered = fleet.integrate_session(transcript, session_id=session_elapsed)
    rule_registry: dict[str, dict] = {}
    broadcast_results: list[dict] = []

    for tier, rule_id in registered.items():
        rule_dict = transcript["final_rules"].get(tier, {})
        rule_registry[rule_id] = rule_dict

        record = await fleet.broadcast_rule(
            rule_id=rule_id,
            tiers=[tier],
        )
        broadcast_elapsed = int((time.monotonic() - broadcast_start) * 1000)
        n_ack = len(record.acknowledged_by)
        log(f"  Broadcast [{tier}] rule {rule_id} → {n_ack} drones acknowledged "
            f"({broadcast_elapsed}ms)")
        broadcast_results.append({
            "tier": tier,
            "rule_id": rule_id,
            "n_acknowledged": n_ack,
            "latency_ms": record.latency_ms,
        })

    total_broadcast_ms = int((time.monotonic() - broadcast_start) * 1000)
    log(f"Fleet-wide broadcast complete: {total_broadcast_ms}ms")
    report["phases"]["broadcast"] = {
        "total_latency_ms": total_broadcast_ms,
        "broadcasts": broadcast_results,
    }

    # ------------------------------------------------------------------
    # Phase F: Retroactive archive reprocessing
    # ------------------------------------------------------------------
    print_separator("PHASE F: RETROACTIVE ARCHIVE REPROCESSING")
    reprocess_start = time.monotonic()

    all_reclassified = []
    for tier, rule_id in registered.items():
        if tier != "scout":
            continue
        rule_dict = rule_registry[rule_id]
        reclassified = await reprocess_archive(
            rule=rule_dict,
            rule_id=rule_id,
            frame_buffer=frame_buffer,
            config=MARITIME_SAR_CONFIG,
            lookback_seconds=lookback_seconds,
            tier="scout",
            reprocess_class="whitecap",
            confidence_min=0.70,
            validator_model=validator_model,
        )
        all_reclassified.extend(reclassified)

        for rc in reclassified:
            fleet.update_track_map(
                coordinates=rc.frame.coordinates or (0.0, 0.0),
                detection_class=rc.new_class,
                confidence=1.0,
                rule_id=rule_id,
                drone_id=rc.frame.drone_id,
                frame_id=rc.frame.frame_id,
                retroactive=True,
            )

    reprocess_elapsed = int((time.monotonic() - reprocess_start) * 1000)
    log(f"Archive reprocessing: {len(all_reclassified)} frames reclassified as "
        f"person_in_water in {reprocess_elapsed}ms")
    for rc in all_reclassified:
        log(f"  [{rc.frame.drone_id}] {Path(rc.frame.image_path).name} — "
            f"was: {rc.frame.original_class} ({rc.frame.original_confidence:.2f}) "
            f"→ now: {rc.new_class}")

    report["phases"]["retroactive_reprocessing"] = {
        "n_reclassified": len(all_reclassified),
        "duration_ms": reprocess_elapsed,
        "reclassified": [rc.to_dict() for rc in all_reclassified],
    }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print_separator("SUMMARY")
    total_elapsed = int((time.monotonic() - demo_start) * 1000)
    swarm_state = fleet.get_swarm_state()

    log(f"Total demo time    : {total_elapsed}ms")
    log(f"DD session time    : {session_elapsed}ms")
    log(f"Broadcast latency  : {total_broadcast_ms}ms (fleet-wide)")
    log(f"Archive reprocessed: {n_injected} frames → {len(all_reclassified)} reclassified")
    log(f"Active rules       : {swarm_state['n_active_rules']}")
    log(f"Track map entries  : {swarm_state['n_track_detections']}")

    report["outcome"] = "success"
    report["totals"] = {
        "total_elapsed_ms": total_elapsed,
        "dd_session_ms": session_elapsed,
        "broadcast_latency_ms": total_broadcast_ms,
        "archive_frames_reprocessed": n_injected,
        "persons_found_retroactively": len(all_reclassified),
        "active_rules": swarm_state["n_active_rules"],
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, default=str))
        log(f"\nReport written to {output_path}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SeaPatch Phase 2 fleet broadcast demo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset-root", required=True, help="SeaDronesSee dataset root.")
    p.add_argument("--pool-dir", required=True, help="Labeled pool directory (from pool_builder.py).")
    p.add_argument("--failure-dir", required=True, help="Failure frames directory (from pool_builder.py --failure-dir).")
    p.add_argument("--coco-annotation", default=None, help="COCO annotation JSON (optional, improves ground truth labels).")
    p.add_argument("--n-scouts", type=int, default=38)
    p.add_argument("--n-commanders", type=int, default=2)
    p.add_argument("--lookback-seconds", type=int, default=2700, help="Archive lookback window in seconds (default: 2700 = 45 min).")
    p.add_argument("--tutor-model", default="claude-opus-4-6")
    p.add_argument("--validator-model", default="claude-sonnet-4-6")
    p.add_argument("--output", default=None, help="Path to write demo report JSON.")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report = asyncio.run(run_demo(
        dataset_root=Path(args.dataset_root),
        pool_dir=Path(args.pool_dir),
        failure_dir=Path(args.failure_dir),
        coco_annotation=Path(args.coco_annotation) if args.coco_annotation else None,
        n_scouts=args.n_scouts,
        n_commanders=args.n_commanders,
        lookback_seconds=args.lookback_seconds,
        tutor_model=args.tutor_model,
        validator_model=args.validator_model,
        output_path=Path(args.output) if args.output else None,
        quiet=args.quiet,
    ))
    outcome = report.get("outcome", "unknown")
    sys.exit(0 if outcome == "success" else 1)
