"""
fleet.py — Swarm state management and rule broadcast for SeaPatch.

Manages the 40-drone heterogeneous fleet:
  - Rule registry (versioned, revocable, tier-specific variants)
  - Per-drone state (position, active rules, last seen)
  - Broadcast engine with per-drone acknowledgement tracking
  - Semantic track map (coordinates → detection result)

Designed to work in simulation (SeaDronesSee frame injection) and
with real drone APIs via the MCP server interface (mcp_server.py).

See DESIGN.md §2 for the fleet architecture.
"""
from __future__ import annotations

import asyncio
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Rule:
    rule_id: str
    rule_text: str
    feature: str
    favors: str
    preconditions: list[str]
    rationale: str
    tier: str                         # "scout" | "commander" | "all"
    precision: float = 0.0
    source_session_id: str = ""
    registered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    revoked: bool = False

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "rule_text": self.rule_text,
            "feature": self.feature,
            "favors": self.favors,
            "preconditions": self.preconditions,
            "rationale": self.rationale,
            "tier": self.tier,
            "precision": self.precision,
            "source_session_id": self.source_session_id,
            "registered_at": self.registered_at.isoformat(),
            "revoked": self.revoked,
        }


@dataclass
class DroneState:
    drone_id: str
    tier: str                         # "scout" | "commander"
    position: tuple[float, float]     # (lat, lon) — may be (0.0, 0.0) in simulation
    altitude_m: float
    active_rule_ids: list[str] = field(default_factory=list)
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged_broadcast_ids: list[str] = field(default_factory=list)


@dataclass
class BroadcastRecord:
    broadcast_id: str
    rule_id: str
    tiers: list[str]
    initiated_at: datetime
    completed_at: datetime | None = None
    acknowledged_by: list[str] = field(default_factory=list)  # drone_ids
    latency_ms: float | None = None


@dataclass
class Detection:
    coordinates: tuple[float, float]
    detection_class: str
    confidence: float
    rule_id: str
    drone_id: str
    frame_id: str
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    retroactive: bool = False


@dataclass
class UrgencyObservation:
    """A single uncertain_investigate report from a scout drone."""
    drone_id: str
    frame_id: str
    urgency: float          # investigation_urgency from classifier (0–1)
    observed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class UrgencyCell:
    """Accumulated urgency state for one grid cell.

    Each time a scout flies over and reports uncertain_investigate, an
    UrgencyObservation is appended.  Accumulated urgency decays exponentially
    so stale observations lose influence over time.
    """
    coordinates: tuple[float, float]
    observations: list[UrgencyObservation] = field(default_factory=list)
    escalated: bool = False
    escalated_at: datetime | None = None
    escalated_drone_id: str | None = None   # commander dispatched

    def accumulated_urgency(self, half_life_s: float = 600.0) -> float:
        """Compute current accumulated urgency with exponential time decay.

        u(t) = Σ_i urgency_i · exp(-λ · (t_now - t_i))
        where λ = ln(2) / half_life_s

        Default half-life: 600 s (10 minutes) — a whitecap-like false
        signal vanishes quickly; a person-consistent signal from repeated
        passes accumulates.
        """
        if not self.observations:
            return 0.0
        lam = math.log(2) / half_life_s
        now = datetime.now(timezone.utc)
        total = 0.0
        for obs in self.observations:
            age_s = (now - obs.observed_at).total_seconds()
            total += obs.urgency * math.exp(-lam * age_s)
        return min(total, 1.0)   # cap at 1.0


@dataclass
class EscalationEvent:
    """Record of a commander dispatch triggered by urgency accumulation."""
    cell_coordinates: tuple[float, float]
    accumulated_urgency: float
    n_observations: int
    triggered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    commander_id: str | None = None
    resolved: bool = False
    resolution: str = ""    # "confirmed_person" | "false_alarm" | "pending"


# ---------------------------------------------------------------------------
# Fleet manager
# ---------------------------------------------------------------------------

class FleetManager:
    """Manages swarm state, rule registry, and broadcast operations.

    In Phase 1/2 (simulation), drones are registered programmatically.
    In Phase 3 (MCP server), drones register via the mesh network.
    """

    # Default escalation parameters
    ESCALATION_THRESHOLD: float = 0.70   # accumulated urgency to trigger dispatch
    URGENCY_HALF_LIFE_S: float = 600.0   # 10-minute decay half-life

    def __init__(self) -> None:
        self._drones: dict[str, DroneState] = {}
        self._rule_pool: dict[str, Rule] = {}
        self._broadcast_log: list[BroadcastRecord] = []
        self._track_map: dict[str, Detection] = {}        # coord_key → Detection
        self._urgency_map: dict[str, UrgencyCell] = {}   # coord_key → UrgencyCell
        self._escalation_log: list[EscalationEvent] = []

    # -----------------------------------------------------------------------
    # Drone registration
    # -----------------------------------------------------------------------

    def register_drone(
        self,
        drone_id: str,
        tier: str,
        position: tuple[float, float] = (0.0, 0.0),
        altitude_m: float = 30.0,
    ) -> DroneState:
        """Register or update a drone in the fleet."""
        if drone_id in self._drones:
            state = self._drones[drone_id]
            state.position = position
            state.altitude_m = altitude_m
            state.last_seen = datetime.now(timezone.utc)
        else:
            state = DroneState(
                drone_id=drone_id,
                tier=tier,
                position=position,
                altitude_m=altitude_m,
            )
            self._drones[drone_id] = state
        return state

    def register_scout_fleet(self, n: int = 38, prefix: str = "S") -> list[str]:
        """Register n scout drones with IDs S01..Sn."""
        ids = [f"{prefix}{i+1:02d}" for i in range(n)]
        for drone_id in ids:
            self.register_drone(drone_id, tier="scout")
        return ids

    def register_commander_fleet(self, n: int = 2, prefix: str = "C") -> list[str]:
        """Register n commander drones with IDs C1..Cn."""
        ids = [f"{prefix}{i+1}" for i in range(n)]
        for drone_id in ids:
            self.register_drone(drone_id, tier="commander", altitude_m=15.0)
        return ids

    # -----------------------------------------------------------------------
    # Rule registry
    # -----------------------------------------------------------------------

    def register_rule(
        self,
        rule_dict: dict,
        tier: str,
        precision: float = 0.0,
        session_id: str = "",
    ) -> str:
        """Register an accepted rule in the pool. Returns rule_id."""
        rule_id = f"rule_{uuid.uuid4().hex[:8]}"
        rule = Rule(
            rule_id=rule_id,
            rule_text=rule_dict.get("rule", ""),
            feature=rule_dict.get("feature", "unknown"),
            favors=rule_dict.get("favors", ""),
            preconditions=rule_dict.get("preconditions", []),
            rationale=rule_dict.get("rationale", ""),
            tier=tier,
            precision=precision,
            source_session_id=session_id,
        )
        self._rule_pool[rule_id] = rule
        return rule_id

    def revoke_rule(self, rule_id: str) -> bool:
        """Revoke a rule from the pool. Returns True if found."""
        if rule_id in self._rule_pool:
            self._rule_pool[rule_id].revoked = True
            return True
        return False

    def get_active_rules(self, tier: str | None = None) -> list[Rule]:
        """Return all non-revoked rules, optionally filtered by tier."""
        rules = [r for r in self._rule_pool.values() if not r.revoked]
        if tier:
            rules = [r for r in rules if r.tier in (tier, "all")]
        return rules

    # -----------------------------------------------------------------------
    # Broadcast
    # -----------------------------------------------------------------------

    async def broadcast_rule(
        self,
        rule_id: str,
        tiers: list[str] | None = None,
        simulate_latency_ms: float = 0.0,
    ) -> BroadcastRecord:
        """Broadcast a rule to all drones of the specified tiers.

        In simulation, this is instantaneous (or simulated with latency).
        In production, this would transmit over the mesh network.

        Returns a BroadcastRecord with per-drone acknowledgements.
        """
        if rule_id not in self._rule_pool:
            raise ValueError(f"Rule {rule_id} not in pool")
        rule = self._rule_pool[rule_id]

        if tiers is None:
            tiers = ["scout", "commander"] if rule.tier == "all" else [rule.tier]

        broadcast_id = f"bc_{uuid.uuid4().hex[:8]}"
        initiated_at = datetime.now(timezone.utc)

        target_drones = [
            drone for drone in self._drones.values()
            if drone.tier in tiers and not rule.revoked
        ]

        if simulate_latency_ms > 0:
            await asyncio.sleep(simulate_latency_ms / 1000.0)

        acknowledged_by = []
        for drone in target_drones:
            if rule_id not in drone.active_rule_ids:
                drone.active_rule_ids.append(rule_id)
            drone.acknowledged_broadcast_ids.append(broadcast_id)
            acknowledged_by.append(drone.drone_id)

        completed_at = datetime.now(timezone.utc)
        elapsed_ms = (completed_at - initiated_at).total_seconds() * 1000

        record = BroadcastRecord(
            broadcast_id=broadcast_id,
            rule_id=rule_id,
            tiers=tiers,
            initiated_at=initiated_at,
            completed_at=completed_at,
            acknowledged_by=acknowledged_by,
            latency_ms=elapsed_ms,
        )
        self._broadcast_log.append(record)
        return record

    # -----------------------------------------------------------------------
    # Semantic track map
    # -----------------------------------------------------------------------

    def update_track_map(
        self,
        coordinates: tuple[float, float],
        detection_class: str,
        confidence: float,
        rule_id: str,
        drone_id: str,
        frame_id: str,
        retroactive: bool = False,
    ) -> Detection:
        """Record a detection on the semantic track map."""
        # Key by rounded coordinates (1m precision at typical latitudes)
        coord_key = f"{coordinates[0]:.5f},{coordinates[1]:.5f}"
        detection = Detection(
            coordinates=coordinates,
            detection_class=detection_class,
            confidence=confidence,
            rule_id=rule_id,
            drone_id=drone_id,
            frame_id=frame_id,
            retroactive=retroactive,
        )
        self._track_map[coord_key] = detection
        return detection

    # -----------------------------------------------------------------------
    # Urgency accumulator + escalation trigger
    # -----------------------------------------------------------------------

    def _coord_key(self, coordinates: tuple[float, float]) -> str:
        """Round coordinates to ~10 m grid (4 decimal places ≈ 11 m at mid-lat)."""
        return f"{coordinates[0]:.4f},{coordinates[1]:.4f}"

    def report_uncertain(
        self,
        coordinates: tuple[float, float],
        investigation_urgency: float,
        drone_id: str,
        frame_id: str,
        observed_at: datetime | None = None,
    ) -> UrgencyCell:
        """Record an uncertain_investigate report from a scout drone.

        Adds an UrgencyObservation to the cell at `coordinates`.  The cell's
        accumulated urgency increases; if it crosses ESCALATION_THRESHOLD and
        the cell has not yet been escalated, returns the cell flagged for
        commander dispatch.

        Parameters
        ----------
        coordinates:
            (lat, lon) of the observed anomaly.
        investigation_urgency:
            The urgency score returned by the PUPIL classifier (0–1).
        drone_id:
            ID of the scout that made the observation.
        frame_id:
            Frame identifier for provenance.
        observed_at:
            Timestamp of the observation (default: now).
        """
        key = self._coord_key(coordinates)
        if key not in self._urgency_map:
            self._urgency_map[key] = UrgencyCell(coordinates=coordinates)

        cell = self._urgency_map[key]
        obs = UrgencyObservation(
            drone_id=drone_id,
            frame_id=frame_id,
            urgency=investigation_urgency,
            observed_at=observed_at or datetime.now(timezone.utc),
        )
        cell.observations.append(obs)
        return cell

    def check_escalations(
        self,
        threshold: float | None = None,
        half_life_s: float | None = None,
    ) -> list[tuple[UrgencyCell, float]]:
        """Return (cell, accumulated_urgency) pairs that exceed the threshold
        and have not yet been escalated.

        Call this after each report_uncertain() to discover cells that need
        a commander dispatch.  Cells are returned sorted by accumulated urgency
        descending (highest priority first).

        Parameters
        ----------
        threshold:
            Override for ESCALATION_THRESHOLD.
        half_life_s:
            Override for URGENCY_HALF_LIFE_S.
        """
        threshold = threshold if threshold is not None else self.ESCALATION_THRESHOLD
        half_life_s = half_life_s if half_life_s is not None else self.URGENCY_HALF_LIFE_S

        candidates = []
        for cell in self._urgency_map.values():
            if cell.escalated:
                continue
            acc = cell.accumulated_urgency(half_life_s=half_life_s)
            if acc >= threshold:
                candidates.append((cell, acc))

        return sorted(candidates, key=lambda x: x[1], reverse=True)

    def escalate(
        self,
        cell: UrgencyCell,
        commander_id: str | None = None,
        threshold: float | None = None,
        half_life_s: float | None = None,
    ) -> EscalationEvent:
        """Mark a cell as escalated and record the dispatch event.

        In simulation this is instantaneous.  In production this would
        issue a waypoint command to the nearest available commander drone.
        """
        threshold = threshold if threshold is not None else self.ESCALATION_THRESHOLD
        half_life_s = half_life_s if half_life_s is not None else self.URGENCY_HALF_LIFE_S
        acc = cell.accumulated_urgency(half_life_s=half_life_s)

        cell.escalated = True
        cell.escalated_at = datetime.now(timezone.utc)
        cell.escalated_drone_id = commander_id

        event = EscalationEvent(
            cell_coordinates=cell.coordinates,
            accumulated_urgency=round(acc, 4),
            n_observations=len(cell.observations),
            commander_id=commander_id,
            resolution="pending",
        )
        self._escalation_log.append(event)
        return event

    def resolve_escalation(
        self,
        coordinates: tuple[float, float],
        resolution: str,  # "confirmed_person" | "false_alarm"
    ) -> bool:
        """Record the outcome of a commander inspection.

        `resolution` should be "confirmed_person" or "false_alarm".
        Returns True if a matching pending escalation was found and updated.
        """
        key = self._coord_key(coordinates)
        for event in reversed(self._escalation_log):
            if (self._coord_key(event.cell_coordinates) == key
                    and event.resolution == "pending"):
                event.resolved = True
                event.resolution = resolution
                return True
        return False

    def get_urgency_map(
        self,
        half_life_s: float | None = None,
        min_urgency: float = 0.0,
    ) -> list[dict]:
        """Return current urgency state for all cells above min_urgency."""
        half_life_s = half_life_s if half_life_s is not None else self.URGENCY_HALF_LIFE_S
        rows = []
        for cell in self._urgency_map.values():
            acc = cell.accumulated_urgency(half_life_s=half_life_s)
            if acc < min_urgency:
                continue
            rows.append({
                "coordinates": cell.coordinates,
                "accumulated_urgency": round(acc, 4),
                "n_observations": len(cell.observations),
                "escalated": cell.escalated,
                "escalated_drone": cell.escalated_drone_id,
                "last_observed": max(
                    (o.observed_at for o in cell.observations),
                    default=None,
                ).isoformat() if cell.observations else None,
            })
        return sorted(rows, key=lambda x: x["accumulated_urgency"], reverse=True)

    def get_escalation_log(self) -> list[dict]:
        return [
            {
                "coordinates": e.cell_coordinates,
                "accumulated_urgency": e.accumulated_urgency,
                "n_observations": e.n_observations,
                "triggered_at": e.triggered_at.isoformat(),
                "commander_id": e.commander_id,
                "resolved": e.resolved,
                "resolution": e.resolution,
            }
            for e in self._escalation_log
        ]

    def get_track_map(self) -> list[dict]:
        return [
            {
                "coordinates": d.coordinates,
                "detection_class": d.detection_class,
                "confidence": d.confidence,
                "rule_id": d.rule_id,
                "drone_id": d.drone_id,
                "frame_id": d.frame_id,
                "detected_at": d.detected_at.isoformat(),
                "retroactive": d.retroactive,
            }
            for d in self._track_map.values()
        ]

    # -----------------------------------------------------------------------
    # State reporting
    # -----------------------------------------------------------------------

    def get_swarm_state(self) -> dict:
        pending_escalations = sum(
            1 for e in self._escalation_log if e.resolution == "pending"
        )
        return {
            "n_drones": len(self._drones),
            "n_scouts": sum(1 for d in self._drones.values() if d.tier == "scout"),
            "n_commanders": sum(1 for d in self._drones.values() if d.tier == "commander"),
            "n_active_rules": len(self.get_active_rules()),
            "n_broadcasts": len(self._broadcast_log),
            "n_track_detections": len(self._track_map),
            "n_urgency_cells": len(self._urgency_map),
            "n_escalations_total": len(self._escalation_log),
            "n_escalations_pending": pending_escalations,
            "drones": {
                drone_id: {
                    "tier": d.tier,
                    "position": d.position,
                    "altitude_m": d.altitude_m,
                    "n_active_rules": len(d.active_rule_ids),
                    "last_seen": d.last_seen.isoformat(),
                }
                for drone_id, d in self._drones.items()
            },
            "active_rules": [r.to_dict() for r in self.get_active_rules()],
            "recent_broadcasts": [
                {
                    "broadcast_id": b.broadcast_id,
                    "rule_id": b.rule_id,
                    "tiers": b.tiers,
                    "n_acknowledged": len(b.acknowledged_by),
                    "latency_ms": b.latency_ms,
                }
                for b in self._broadcast_log[-10:]
            ],
        }

    # -----------------------------------------------------------------------
    # Convenience: full session integration
    # -----------------------------------------------------------------------

    def integrate_session(
        self,
        session_transcript: dict,
        session_id: str = "",
        broadcast: bool = True,
    ) -> dict[str, str]:
        """Register all tier rules from a completed DD session and broadcast.

        Returns dict of tier → rule_id for registered rules.

        Note: broadcast is synchronous here for simplicity; use
        broadcast_rule() directly for async broadcast with latency simulation.
        """
        final_rules = session_transcript.get("final_rules", {})
        pool_result = session_transcript.get("pool_result_after_tighten") or session_transcript.get("pool_result", {})
        precision = pool_result.get("precision", 0.0) if pool_result else 0.0

        registered: dict[str, str] = {}
        for tier, rule_dict in final_rules.items():
            rule_id = self.register_rule(
                rule_dict=rule_dict,
                tier=tier,
                precision=precision,
                session_id=session_id,
            )
            registered[tier] = rule_id

        return registered
