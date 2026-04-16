"""
mcp_server.py — MCP server exposing ground station tools for SeaPatch.

Exposes the fleet + archive + DD session as MCP tools so a Claude Code session
can drive the swarm with natural language commands.

See DESIGN.md §6 for the full tool interface spec and example session.

Usage:
    # Start the MCP server (Claude Code will connect automatically
    # if configured in .mcp.json or claude_desktop_config.json)
    python mcp_server.py --dataset-root data/seadronessee --pool-dir data/pool

    # Or import directly for programmatic use
    from mcp_server import SeaPatchMCPServer
    server = SeaPatchMCPServer(...)
    await server.broadcast_rule(rule_id="rule_abc123", tiers=["scout"])

MCP tool summary:
    get_camera_frame(drone_id, timestamp, channel)
    broadcast_rule(rule_id, tiers)
    reprocess_archive(rule_id, lookback_seconds, tier)
    update_track_map(coordinates, detection_class, confidence, rule_id, drone_id, frame_id)
    get_swarm_state()
    run_dd_session(failure_image, confirmation, ground_truth, pupil_class, pupil_confidence, pool_dir)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

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
# Server state (singleton per process)
# ---------------------------------------------------------------------------

class SeaPatchMCPServer:
    """Ground station MCP server for the SeaPatch drone swarm.

    Wraps fleet management, frame buffer, and DD session into MCP-callable
    tools. In simulation mode, uses SeaDronesSee bridge for camera frames.
    In production mode, swap get_camera_frame() with real drone API calls.
    """

    def __init__(
        self,
        dataset_root: Path | None = None,
        pool_dir: Path | None = None,
        n_scouts: int = 38,
        n_commanders: int = 2,
        coco_annotation: Path | None = None,
    ) -> None:
        self.fleet = FleetManager()
        self.frame_buffer = FrameBuffer()
        self.pool_dir = pool_dir
        self._rule_registry: dict[str, dict] = {}   # rule_id → rule dict (for reprocess)

        # Register drone fleet
        self._scout_ids = self.fleet.register_scout_fleet(n_scouts)
        self._commander_ids = self.fleet.register_commander_fleet(n_commanders)

        # Set up frame bridge if dataset available
        self._bridge: SeaDronesSeebridge | None = None
        if dataset_root and dataset_root.exists():
            self._bridge = SeaDronesSeebridge(
                dataset_root=dataset_root,
                coco_annotation=coco_annotation,
            )
            all_ids = self._scout_ids + self._commander_ids
            self._bridge.assign_sequences(all_ids)
            # Pre-populate archive with historical frames
            n_injected = self._bridge.inject_to_archive(
                self.frame_buffer,
                lookback_seconds=2700,
                tier="scout",
            )
            print(f"[SeaPatch] Frame bridge ready: {n_injected} frames in archive")

    # -----------------------------------------------------------------------
    # MCP tool: get_camera_frame
    # -----------------------------------------------------------------------

    async def get_camera_frame(
        self,
        drone_id: str,
        timestamp: str = "latest",
        channel: str = "rgb",
    ) -> dict[str, Any]:
        """Return the image path for a drone's camera at a given timestamp.

        Parameters
        ----------
        drone_id:
            Scout or commander drone ID (e.g. "S22", "C1").
        timestamp:
            "latest", "-18m" (relative), or frame index.
        channel:
            "rgb" (all tiers) or "thermal" (commanders only).

        Returns
        -------
        {"drone_id", "timestamp", "channel", "image_path", "available"}
        """
        if channel == "thermal" and not drone_id.startswith("C"):
            return {
                "drone_id": drone_id,
                "timestamp": timestamp,
                "channel": channel,
                "image_path": None,
                "available": False,
                "error": f"Thermal channel only available on commander drones (C*).",
            }

        if self._bridge is None:
            return {
                "drone_id": drone_id,
                "timestamp": timestamp,
                "channel": channel,
                "image_path": None,
                "available": False,
                "error": "No dataset bridge configured. Pass --dataset-root to mcp_server.py.",
            }

        frame_path = self._bridge.get_frame(drone_id, timestamp)
        if frame_path is None:
            return {
                "drone_id": drone_id,
                "timestamp": timestamp,
                "channel": channel,
                "image_path": None,
                "available": False,
                "error": f"No frame available for drone {drone_id} at {timestamp}.",
            }

        return {
            "drone_id": drone_id,
            "timestamp": timestamp,
            "channel": channel,
            "image_path": str(frame_path),
            "available": True,
        }

    # -----------------------------------------------------------------------
    # MCP tool: run_dd_session
    # -----------------------------------------------------------------------

    async def run_dd_session(
        self,
        failure_image: str,
        confirmation: str,
        ground_truth: str = "person_in_water",
        pupil_class: str = "whitecap",
        pupil_confidence: float = 0.91,
        pool_dir: str | None = None,
        tutor_model: str = "claude-opus-4-6",
        validator_model: str = "claude-sonnet-4-6",
    ) -> dict[str, Any]:
        """Run a DD session on a failure frame and return the accepted rules.

        Parameters
        ----------
        failure_image:
            Path to the failure RGB frame.
        confirmation:
            Thermal / ground truth confirmation details (plain English).
        ground_truth:
            Correct class for the failure frame.
        pupil_class:
            Class predicted by the scout at capture time.
        pupil_confidence:
            Scout's confidence for the wrong prediction.
        pool_dir:
            Directory containing labeled pool frames. Falls back to
            self.pool_dir if not specified.

        Returns
        -------
        Session transcript dict with outcome and final_rules.
        """
        resolved_pool_dir = Path(pool_dir) if pool_dir else self.pool_dir
        if resolved_pool_dir is None or not resolved_pool_dir.exists():
            return {"error": "No pool_dir available. Pass pool_dir or configure at startup."}

        # Load pool
        pool_manifest = resolved_pool_dir / "pool_manifest.json"
        if pool_manifest.exists():
            manifest = json.loads(pool_manifest.read_text())
            pool_images = [(str(resolved_pool_dir / e["path"]), e["label"]) for e in manifest]
        else:
            # Walk subdirectories
            _EXTS = {".jpg", ".jpeg", ".png", ".webp"}
            pool_images = []
            for class_dir in sorted(resolved_pool_dir.iterdir()):
                if class_dir.is_dir():
                    for p in sorted(class_dir.iterdir()):
                        if p.suffix.lower() in _EXTS:
                            pool_images.append((str(p), class_dir.name))

        if not pool_images:
            return {"error": f"No pool images found in {resolved_pool_dir}"}

        # Find pair_info
        pair_info = next(
            (p for p in CONFUSABLE_PAIRS
             if {p["class_a"], p["class_b"]} == {ground_truth, pupil_class}),
            {"class_a": ground_truth, "class_b": pupil_class,
             "pair_id": f"{ground_truth}_vs_{pupil_class}"},
        )

        transcript = await run_maritime_dd_session(
            failure_image_path=failure_image,
            confirmation_modality="thermal_FLIR",
            confirmation_details=confirmation,
            ground_truth_class=ground_truth,
            pupil_classification=pupil_class,
            pupil_confidence=pupil_confidence,
            pool_images=pool_images,
            pair_info=pair_info,
            config=MARITIME_SAR_CONFIG,
            tier_observability=TIER_OBSERVABILITY,
            tutor_model=tutor_model,
            validator_model=validator_model,
        )

        # Register accepted rules for downstream broadcast/reprocess
        if transcript.get("outcome") == "accepted":
            registered = self.fleet.integrate_session(transcript)
            transcript["registered_rule_ids"] = registered
            for tier, rule_id in registered.items():
                rule_dict = transcript["final_rules"].get(tier, {})
                self._rule_registry[rule_id] = rule_dict

        return transcript

    # -----------------------------------------------------------------------
    # MCP tool: broadcast_rule
    # -----------------------------------------------------------------------

    async def broadcast_rule(
        self,
        rule_id: str,
        tiers: list[str] | None = None,
        simulate_latency_ms: float = 0.0,
    ) -> dict[str, Any]:
        """Broadcast an accepted rule to all drones of the specified tiers.

        Parameters
        ----------
        rule_id:
            Rule ID returned by run_dd_session (in registered_rule_ids).
        tiers:
            List of tiers to broadcast to ("scout", "commander"). If None,
            broadcasts to the rule's target tier.
        simulate_latency_ms:
            Optional simulated network latency per drone (ms).

        Returns
        -------
        BroadcastRecord summary dict.
        """
        try:
            record = await self.fleet.broadcast_rule(
                rule_id=rule_id,
                tiers=tiers,
                simulate_latency_ms=simulate_latency_ms,
            )
        except ValueError as e:
            return {"error": str(e)}

        return {
            "broadcast_id": record.broadcast_id,
            "rule_id": record.rule_id,
            "tiers": record.tiers,
            "n_drones_acknowledged": len(record.acknowledged_by),
            "acknowledged_by": record.acknowledged_by,
            "latency_ms": record.latency_ms,
            "completed_at": record.completed_at.isoformat() if record.completed_at else None,
        }

    # -----------------------------------------------------------------------
    # MCP tool: reprocess_archive
    # -----------------------------------------------------------------------

    async def reprocess_archive(
        self,
        rule_id: str,
        lookback_seconds: int = 2700,
        tier: str = "scout",
        reprocess_class: str = "whitecap",
        confidence_min: float = 0.70,
        validator_model: str = "claude-sonnet-4-6",
    ) -> dict[str, Any]:
        """Retroactively apply a rule to archived frames.

        Parameters
        ----------
        rule_id:
            Rule ID to apply.
        lookback_seconds:
            How far back to search (default: 2700 = 45 minutes).
        tier:
            Which drone tier's archive to reprocess.
        reprocess_class:
            Only reprocess frames originally classified as this class
            (default: "whitecap" — the confident negatives).
        confidence_min:
            Only reprocess frames where original confidence >= this.

        Returns
        -------
        Summary dict with n_reprocessed, n_reclassified, reclassified frames.
        """
        rule_dict = self._rule_registry.get(rule_id)
        if rule_dict is None:
            return {"error": f"Rule {rule_id} not found in registry."}

        reclassified = await reprocess_archive(
            rule=rule_dict,
            rule_id=rule_id,
            frame_buffer=self.frame_buffer,
            config=MARITIME_SAR_CONFIG,
            lookback_seconds=lookback_seconds,
            tier=tier,
            reprocess_class=reprocess_class,
            confidence_min=confidence_min,
            validator_model=validator_model,
        )

        # Add reclassified detections to track map
        for rc in reclassified:
            self.fleet.update_track_map(
                coordinates=rc.frame.coordinates or (0.0, 0.0),
                detection_class=rc.new_class,
                confidence=1.0,
                rule_id=rule_id,
                drone_id=rc.frame.drone_id,
                frame_id=rc.frame.frame_id,
                retroactive=True,
            )

        n_queried = len(self.frame_buffer.query(
            max_age_seconds=lookback_seconds,
            tier=tier,
            original_class=reprocess_class,
            confidence_min=confidence_min,
        ))

        return {
            "rule_id": rule_id,
            "tier": tier,
            "lookback_seconds": lookback_seconds,
            "n_frames_queried": n_queried,
            "n_reclassified": len(reclassified),
            "reclassified": [rc.to_dict() for rc in reclassified],
        }

    # -----------------------------------------------------------------------
    # MCP tool: update_track_map
    # -----------------------------------------------------------------------

    def update_track_map(
        self,
        coordinates: tuple[float, float],
        detection_class: str,
        confidence: float,
        rule_id: str,
        drone_id: str,
        frame_id: str,
    ) -> dict[str, Any]:
        """Add or update a detection on the semantic track map."""
        detection = self.fleet.update_track_map(
            coordinates=coordinates,
            detection_class=detection_class,
            confidence=confidence,
            rule_id=rule_id,
            drone_id=drone_id,
            frame_id=frame_id,
        )
        return {
            "coordinates": detection.coordinates,
            "detection_class": detection.detection_class,
            "confidence": detection.confidence,
            "detected_at": detection.detected_at.isoformat(),
        }

    # -----------------------------------------------------------------------
    # MCP tool: get_swarm_state
    # -----------------------------------------------------------------------

    def get_swarm_state(self) -> dict[str, Any]:
        """Return current swarm state: drone positions, active rules, track map."""
        state = self.fleet.get_swarm_state()
        state["frame_buffer"] = self.frame_buffer.summary()
        state["track_map"] = self.fleet.get_track_map()
        return state


# ---------------------------------------------------------------------------
# MCP protocol adapter (tool dispatch)
# ---------------------------------------------------------------------------

def _make_tool_dispatcher(server: SeaPatchMCPServer):
    """Return an async dispatcher that maps MCP tool calls to server methods."""

    async def dispatch(tool_name: str, arguments: dict) -> Any:
        if tool_name == "get_camera_frame":
            return await server.get_camera_frame(**arguments)
        elif tool_name == "run_dd_session":
            return await server.run_dd_session(**arguments)
        elif tool_name == "broadcast_rule":
            return await server.broadcast_rule(**arguments)
        elif tool_name == "reprocess_archive":
            return await server.reprocess_archive(**arguments)
        elif tool_name == "update_track_map":
            return server.update_track_map(**arguments)
        elif tool_name == "get_swarm_state":
            return server.get_swarm_state()
        else:
            return {"error": f"Unknown tool: {tool_name}"}

    return dispatch


TOOL_SCHEMAS = [
    {
        "name": "get_camera_frame",
        "description": "Return the image path for a drone camera at a given timestamp.",
        "input_schema": {
            "type": "object",
            "properties": {
                "drone_id": {"type": "string", "description": "e.g. 'S22', 'C1'"},
                "timestamp": {"type": "string", "description": "'latest', '-18m', or frame index", "default": "latest"},
                "channel": {"type": "string", "enum": ["rgb", "thermal"], "default": "rgb"},
            },
            "required": ["drone_id"],
        },
    },
    {
        "name": "run_dd_session",
        "description": (
            "Run a complete DD session on a failure frame. Returns accepted tier rules "
            "and registered rule IDs for broadcast/reprocess."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "failure_image": {"type": "string", "description": "Path to the failure RGB frame."},
                "confirmation": {"type": "string", "description": "Thermal/oracle confirmation details."},
                "ground_truth": {"type": "string", "default": "person_in_water"},
                "pupil_class": {"type": "string", "default": "whitecap"},
                "pupil_confidence": {"type": "number", "default": 0.91},
                "pool_dir": {"type": "string", "description": "Override pool directory path."},
            },
            "required": ["failure_image", "confirmation"],
        },
    },
    {
        "name": "broadcast_rule",
        "description": "Broadcast an accepted rule to all drones of the specified tiers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "rule_id": {"type": "string"},
                "tiers": {"type": "array", "items": {"type": "string"}, "default": None},
                "simulate_latency_ms": {"type": "number", "default": 0.0},
            },
            "required": ["rule_id"],
        },
    },
    {
        "name": "reprocess_archive",
        "description": (
            "Retroactively apply a rule to archived confident-negative frames. "
            "Returns any frames reclassified as person_in_water."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "rule_id": {"type": "string"},
                "lookback_seconds": {"type": "integer", "default": 2700},
                "tier": {"type": "string", "default": "scout"},
                "reprocess_class": {"type": "string", "default": "whitecap"},
                "confidence_min": {"type": "number", "default": 0.70},
            },
            "required": ["rule_id"],
        },
    },
    {
        "name": "update_track_map",
        "description": "Add or update a detection on the semantic track map.",
        "input_schema": {
            "type": "object",
            "properties": {
                "coordinates": {"type": "array", "items": {"type": "number"}, "description": "[lat, lon]"},
                "detection_class": {"type": "string"},
                "confidence": {"type": "number"},
                "rule_id": {"type": "string"},
                "drone_id": {"type": "string"},
                "frame_id": {"type": "string"},
            },
            "required": ["coordinates", "detection_class", "confidence", "rule_id", "drone_id", "frame_id"],
        },
    },
    {
        "name": "get_swarm_state",
        "description": "Return current swarm state: drone count, active rules, archive summary, track map.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


# ---------------------------------------------------------------------------
# Stdio MCP transport (JSON-RPC over stdin/stdout)
# ---------------------------------------------------------------------------

async def run_stdio_server(server: SeaPatchMCPServer) -> None:
    """Run a minimal MCP server over stdio for Claude Code integration.

    Implements the JSON-RPC 2.0 subset that MCP requires:
      initialize → tools/list → tools/call
    """
    dispatch = _make_tool_dispatcher(server)

    async def handle_request(request: dict) -> dict:
        req_id = request.get("id")
        method = request.get("method", "")

        if method == "initialize":
            return {
                "jsonrpc": "2.0", "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "seapatch-ground-station", "version": "0.1.0"},
                },
            }

        if method == "tools/list":
            return {
                "jsonrpc": "2.0", "id": req_id,
                "result": {"tools": TOOL_SCHEMAS},
            }

        if method == "tools/call":
            params = request.get("params", {})
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            try:
                result = await dispatch(tool_name, arguments)
                return {
                    "jsonrpc": "2.0", "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(result, indent=2, default=str)}],
                        "isError": "error" in result if isinstance(result, dict) else False,
                    },
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0", "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Error: {e}"}],
                        "isError": True,
                    },
                }

        # Unrecognised method
        return {
            "jsonrpc": "2.0", "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }

    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    loop = asyncio.get_event_loop()
    await loop.connect_read_pipe(lambda: protocol, sys.stdin.buffer)
    writer_transport, writer_protocol = await loop.connect_write_pipe(
        lambda: asyncio.streams.FlowControlMixin(loop=loop), sys.stdout.buffer
    )
    writer = asyncio.StreamWriter(writer_transport, writer_protocol, reader, loop)

    print("[SeaPatch MCP] Server ready — listening on stdio", file=sys.stderr)

    while True:
        try:
            header = await reader.readline()
            if not header:
                break
            header_str = header.decode().strip()
            if not header_str.startswith("Content-Length:"):
                continue
            content_length = int(header_str.split(":")[1].strip())
            await reader.readline()  # blank line
            body = await reader.readexactly(content_length)
            request = json.loads(body.decode())
            response = await handle_request(request)
            response_body = json.dumps(response).encode()
            writer.write(
                f"Content-Length: {len(response_body)}\r\n\r\n".encode() + response_body
            )
            await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionResetError):
            break
        except Exception as e:
            print(f"[SeaPatch MCP] Error: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SeaPatch ground station MCP server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset-root", default=None, help="SeaDronesSee dataset root.")
    p.add_argument("--pool-dir", default=None, help="Labeled pool directory.")
    p.add_argument("--coco-annotation", default=None, help="COCO annotation JSON.")
    p.add_argument("--n-scouts", type=int, default=38)
    p.add_argument("--n-commanders", type=int, default=2)
    p.add_argument(
        "--mode", default="stdio", choices=["stdio"],
        help="Transport mode (only stdio supported; HTTP planned for Phase 3).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    dataset_root = Path(args.dataset_root) if args.dataset_root else None
    pool_dir = Path(args.pool_dir) if args.pool_dir else None
    coco_annotation = Path(args.coco_annotation) if args.coco_annotation else None

    server = SeaPatchMCPServer(
        dataset_root=dataset_root,
        pool_dir=pool_dir,
        n_scouts=args.n_scouts,
        n_commanders=args.n_commanders,
        coco_annotation=coco_annotation,
    )

    asyncio.run(run_stdio_server(server))
