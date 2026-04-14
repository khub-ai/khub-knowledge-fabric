"""
fleet.py — Three-tier fleet state management and rule broadcast for PyroWatch.

Manages the 164-agent heterogeneous fleet:
  - Rule registry (versioned, revocable, tier-specific variants with env context)
  - Per-agent state (position, active rules, condition set)
  - Broadcast engine with per-agent acknowledgement tracking
  - Sector heat map (terrain grid → detection result + urgency)
  - Urgency accumulator (per-cell, exponential decay, escalation trigger)

Three-tier architecture:
  Ground Sentinel (×120) — fixed PTZ cameras, "ground_sentinel" tier
  Scout Drone    (×40)  — patrol drones, "scout_drone" tier
  Commander Aircraft (×4) — MWIR + RGB, "commander_aircraft" tier

See README.md §6 for the fleet architecture.
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
    tier: str                          # "ground_sentinel" | "scout_drone" | "commander_aircraft" | "all"
    context_preconditions: dict = field(default_factory=dict)  # RAWS condition blocks
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
            "context_preconditions": self.context_preconditions,
            "precision": self.precision,
            "source_session_id": self.source_session_id,
            "registered_at": self.registered_at.isoformat(),
            "revoked": self.revoked,
        }


@dataclass
class AgentState:
    agent_id: str
    tier: str                          # "ground_sentinel" | "scout_drone" | "commander_aircraft"
    position: tuple[float, float]      # (lat, lon)
    altitude_m: float
    active_rule_ids: list[str] = field(default_factory=list)
    active_condition_set: str = "normal"  # "red_flag" | "elevated_risk" | "normal"
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged_broadcast_ids: list[str] = field(default_factory=list)


@dataclass
class BroadcastRecord:
    broadcast_id: str
    rule_id: str
    tiers: list[str]
    initiated_at: datetime
    completed_at: datetime | None = None
    acknowledged_by: list[str] = field(default_factory=list)
    latency_ms: float | None = None


@dataclass
class Detection:
    coordinates: tuple[float, float]
    detection_class: str
    confidence: float
    rule_id: str
    agent_id: str
    frame_id: str
    tier: str
    condition_set: str = "normal"
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    retroactive: bool = False


@dataclass
class UrgencyObservation:
    """A single uncertain_investigate report from a sentinel or scout agent."""
    agent_id: str
    tier: str
    frame_id: str
    urgency: float             # investigation_urgency from classifier (0–1)
    condition_set: str = "normal"
    observed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class UrgencyCell:
    """Accumulated urgency state for one terrain grid cell.

    Each time an agent passes over and reports uncertain_investigate, an
    UrgencyObservation is appended.  Accumulated urgency decays exponentially —
    a single shimmer-like false signal fades in minutes; repeated fire-consistent
    observations accumulate above the escalation threshold.
    """
    coordinates: tuple[float, float]
    observations: list[UrgencyObservation] = field(default_factory=list)
    escalated: bool = False
    escalated_at: datetime | None = None
    escalated_agent_id: str | None = None    # commander aircraft dispatched

    def accumulated_urgency(self, half_life_s: float = 300.0) -> float:
        """Compute accumulated urgency with exponential time decay.

        u(t) = Σ_i urgency_i · exp(-λ · (t_now - t_i))
        where λ = ln(2) / half_life_s

        Default half-life: 300 s (5 minutes) — shorter than the maritime case
        because a growing fire front can make early signals irrelevant quickly.
        A shimmer-like single observation at urgency 0.45 decays below 0.10
        in 15 minutes; repeated fire-consistent signals accumulate.
        """
        if not self.observations:
            return 0.0
        lam = math.log(2) / half_life_s
        now = datetime.now(timezone.utc)
        total = 0.0
        for obs in self.observations:
            age_s = (now - obs.observed_at).total_seconds()
            total += obs.urgency * math.exp(-lam * age_s)
        return min(total, 1.0)


@dataclass
class EscalationEvent:
    """Record of a commander aircraft dispatch triggered by urgency accumulation."""
    cell_coordinates: tuple[float, float]
    accumulated_urgency: float
    n_observations: int
    triggered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    commander_id: str | None = None
    resolved: bool = False
    resolution: str = ""    # "confirmed_fire" | "false_alarm" | "pending"


# ---------------------------------------------------------------------------
# Fleet manager
# ---------------------------------------------------------------------------

# Default fleet sizes (operational scenario from README §2)
DEFAULT_N_SENTINELS = 120
DEFAULT_N_SCOUTS = 40
DEFAULT_N_COMMANDERS = 4


class FleetManager:
    """Manages three-tier fleet state, rule registry, and broadcast operations.

    In Phase 1 (simulation), agents are registered programmatically.
    In production, agents register via the mesh/LTE network through the MCP server.
    """

    ESCALATION_THRESHOLD: float = 0.65   # lower than maritime — fire grows fast
    URGENCY_HALF_LIFE_S: float = 300.0   # 5-minute decay (vs 10-min for maritime)

    def __init__(self) -> None:
        self._agents: dict[str, AgentState] = {}
        self._rule_pool: dict[str, Rule] = {}
        self._broadcast_log: list[BroadcastRecord] = []
        self._sector_map: dict[str, Detection] = {}       # coord_key → Detection
        self._urgency_map: dict[str, UrgencyCell] = {}    # coord_key → UrgencyCell
        self._escalation_log: list[EscalationEvent] = []

    # -----------------------------------------------------------------------
    # Agent registration
    # -----------------------------------------------------------------------

    def register_agent(
        self,
        agent_id: str,
        tier: str,
        position: tuple[float, float] = (0.0, 0.0),
        altitude_m: float = 100.0,
    ) -> AgentState:
        """Register or update an agent in the fleet."""
        if agent_id in self._agents:
            state = self._agents[agent_id]
            state.position = position
            state.altitude_m = altitude_m
            state.last_seen = datetime.now(timezone.utc)
        else:
            state = AgentState(
                agent_id=agent_id,
                tier=tier,
                position=position,
                altitude_m=altitude_m,
            )
            self._agents[agent_id] = state
        return state

    def register_sentinel_fleet(
        self, n: int = DEFAULT_N_SENTINELS, prefix: str = "GS"
    ) -> list[str]:
        """Register n ground sentinels with IDs GS-001..GS-n."""
        ids = [f"{prefix}-{i+1:03d}" for i in range(n)]
        for agent_id in ids:
            self.register_agent(agent_id, tier="ground_sentinel", altitude_m=0.0)
        return ids

    def register_scout_fleet(
        self, n: int = DEFAULT_N_SCOUTS, prefix: str = "SD"
    ) -> list[str]:
        """Register n scout drones with IDs SD-01..SD-n."""
        ids = [f"{prefix}-{i+1:02d}" for i in range(n)]
        for agent_id in ids:
            self.register_agent(agent_id, tier="scout_drone", altitude_m=120.0)
        return ids

    def register_commander_fleet(
        self, n: int = DEFAULT_N_COMMANDERS, prefix: str = "CA"
    ) -> list[str]:
        """Register n commander aircraft with IDs CA-1..CA-n."""
        ids = [f"{prefix}-{i+1}" for i in range(n)]
        for agent_id in ids:
            self.register_agent(agent_id, tier="commander_aircraft", altitude_m=800.0)
        return ids

    def update_condition_set(self, agent_id: str, condition_set: str) -> None:
        """Update the active RAWS condition set for an agent.

        Called when ground station broadcasts a weather condition change.
        """
        if agent_id in self._agents:
            self._agents[agent_id].active_condition_set = condition_set

    def update_condition_set_fleet(self, condition_set: str) -> int:
        """Broadcast a condition set update to all agents. Returns count updated."""
        for agent in self._agents.values():
            agent.active_condition_set = condition_set
        return len(self._agents)

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
            context_preconditions=rule_dict.get("context_preconditions", {}),
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
        """Broadcast a rule to all agents of the specified tiers.

        Returns a BroadcastRecord with per-agent acknowledgements.
        """
        if rule_id not in self._rule_pool:
            raise ValueError(f"Rule {rule_id} not in pool")
        rule = self._rule_pool[rule_id]

        if tiers is None:
            if rule.tier == "all":
                tiers = ["ground_sentinel", "scout_drone"]
            else:
                tiers = [rule.tier]

        broadcast_id = f"bc_{uuid.uuid4().hex[:8]}"
        initiated_at = datetime.now(timezone.utc)

        target_agents = [
            agent for agent in self._agents.values()
            if agent.tier in tiers and not rule.revoked
        ]

        if simulate_latency_ms > 0:
            await asyncio.sleep(simulate_latency_ms / 1000.0)

        acknowledged_by = []
        for agent in target_agents:
            if rule_id not in agent.active_rule_ids:
                agent.active_rule_ids.append(rule_id)
            agent.acknowledged_broadcast_ids.append(broadcast_id)
            acknowledged_by.append(agent.agent_id)

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
    # Sector heat map
    # -----------------------------------------------------------------------

    def update_sector_map(
        self,
        coordinates: tuple[float, float],
        detection_class: str,
        confidence: float,
        rule_id: str,
        agent_id: str,
        frame_id: str,
        tier: str,
        condition_set: str = "normal",
        retroactive: bool = False,
    ) -> Detection:
        """Record a detection on the sector heat map."""
        coord_key = f"{coordinates[0]:.5f},{coordinates[1]:.5f}"
        detection = Detection(
            coordinates=coordinates,
            detection_class=detection_class,
            confidence=confidence,
            rule_id=rule_id,
            agent_id=agent_id,
            frame_id=frame_id,
            tier=tier,
            condition_set=condition_set,
            retroactive=retroactive,
        )
        self._sector_map[coord_key] = detection
        return detection

    def get_sector_map(self) -> list[dict]:
        return [
            {
                "coordinates": d.coordinates,
                "detection_class": d.detection_class,
                "confidence": d.confidence,
                "rule_id": d.rule_id,
                "agent_id": d.agent_id,
                "frame_id": d.frame_id,
                "tier": d.tier,
                "condition_set": d.condition_set,
                "detected_at": d.detected_at.isoformat(),
                "retroactive": d.retroactive,
            }
            for d in self._sector_map.values()
        ]

    # -----------------------------------------------------------------------
    # Urgency accumulator + escalation trigger
    # -----------------------------------------------------------------------

    def _coord_key(self, coordinates: tuple[float, float]) -> str:
        """Round to ~10 m grid (4 decimal places ≈ 11 m at mid-lat)."""
        return f"{coordinates[0]:.4f},{coordinates[1]:.4f}"

    def report_uncertain(
        self,
        coordinates: tuple[float, float],
        investigation_urgency: float,
        agent_id: str,
        tier: str,
        frame_id: str,
        condition_set: str = "normal",
        observed_at: datetime | None = None,
    ) -> UrgencyCell:
        """Record an uncertain_investigate report from a sentinel or scout agent.

        Appends an UrgencyObservation to the cell at `coordinates`.

        Parameters
        ----------
        coordinates:
            (lat, lon) of the observed anomaly.
        investigation_urgency:
            The urgency score returned by the classifier (0–1).
            Under red flag conditions, the base_urgency from the rule's
            context_preconditions block should be applied before calling this.
        agent_id, tier, frame_id:
            Provenance for the observation.
        condition_set:
            Active RAWS condition set at time of observation.
        """
        key = self._coord_key(coordinates)
        if key not in self._urgency_map:
            self._urgency_map[key] = UrgencyCell(coordinates=coordinates)

        cell = self._urgency_map[key]
        obs = UrgencyObservation(
            agent_id=agent_id,
            tier=tier,
            frame_id=frame_id,
            urgency=investigation_urgency,
            condition_set=condition_set,
            observed_at=observed_at or datetime.now(timezone.utc),
        )
        cell.observations.append(obs)
        return cell

    def check_escalations(
        self,
        threshold: float | None = None,
        half_life_s: float | None = None,
    ) -> list[tuple[UrgencyCell, float]]:
        """Return (cell, accumulated_urgency) pairs exceeding threshold,
        not yet escalated, sorted highest urgency first.
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
        """Mark a cell as escalated and record the commander dispatch event."""
        half_life_s = half_life_s if half_life_s is not None else self.URGENCY_HALF_LIFE_S
        acc = cell.accumulated_urgency(half_life_s=half_life_s)

        cell.escalated = True
        cell.escalated_at = datetime.now(timezone.utc)
        cell.escalated_agent_id = commander_id

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
        resolution: str,   # "confirmed_fire" | "false_alarm"
    ) -> bool:
        """Record the outcome of a commander MWIR inspection.

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
        """Return urgency state for all cells above min_urgency."""
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
                "escalated_agent": cell.escalated_agent_id,
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

    # -----------------------------------------------------------------------
    # State reporting
    # -----------------------------------------------------------------------

    def get_fleet_state(self) -> dict:
        pending_escalations = sum(
            1 for e in self._escalation_log if e.resolution == "pending"
        )
        tiers = ["ground_sentinel", "scout_drone", "commander_aircraft"]
        return {
            "n_agents": len(self._agents),
            **{f"n_{t}": sum(1 for a in self._agents.values() if a.tier == t) for t in tiers},
            "n_active_rules": len(self.get_active_rules()),
            "n_broadcasts": len(self._broadcast_log),
            "n_sector_detections": len(self._sector_map),
            "n_urgency_cells": len(self._urgency_map),
            "n_escalations_total": len(self._escalation_log),
            "n_escalations_pending": pending_escalations,
            "agents": {
                aid: {
                    "tier": a.tier,
                    "position": a.position,
                    "altitude_m": a.altitude_m,
                    "n_active_rules": len(a.active_rule_ids),
                    "active_condition_set": a.active_condition_set,
                    "last_seen": a.last_seen.isoformat(),
                }
                for aid, a in self._agents.items()
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
    # Convenience: integrate a completed DD session
    # -----------------------------------------------------------------------

    def integrate_session(
        self,
        session_transcript: dict,
        session_id: str = "",
        broadcast: bool = True,
    ) -> dict[str, str]:
        """Register all tier rules from a completed DD session.

        Returns dict of tier → rule_id for registered rules.
        """
        final_rules = session_transcript.get("final_rules", {})
        pool_result = (
            session_transcript.get("pool_result_after_tighten")
            or session_transcript.get("pool_result", {})
        )
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
