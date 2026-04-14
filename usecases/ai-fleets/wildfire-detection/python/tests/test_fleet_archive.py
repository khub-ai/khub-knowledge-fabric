"""
test_fleet_archive.py — Unit tests for fleet.py and archive.py.

No LLM calls. Tests pure Python state management logic.

Run from repo root:
    python -m pytest usecases/ai-fleets/wildfire-detection/python/tests/test_fleet_archive.py -v
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

_PYTHON_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PYTHON_DIR.parents[3]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_PYTHON_DIR))

from fleet import FleetManager, Rule, AgentState
from archive import FrameBuffer, ArchivedFrame, load_frames_from_directory


# ---------------------------------------------------------------------------
# FleetManager tests
# ---------------------------------------------------------------------------

class TestFleetManager:

    def setup_method(self):
        self.fleet = FleetManager()

    def test_register_sentinel_fleet(self):
        ids = self.fleet.register_sentinel_fleet(5)
        assert len(ids) == 5
        assert ids[0] == "GS-001"
        assert ids[4] == "GS-005"
        assert all(self.fleet._agents[i].tier == "ground_sentinel" for i in ids)

    def test_register_scout_fleet(self):
        ids = self.fleet.register_scout_fleet(5)
        assert len(ids) == 5
        assert ids[0] == "SD-01"
        assert ids[4] == "SD-05"
        assert all(self.fleet._agents[i].tier == "scout_drone" for i in ids)

    def test_register_commander_fleet(self):
        ids = self.fleet.register_commander_fleet(4)
        assert len(ids) == 4
        assert ids[0] == "CA-1"
        assert ids[3] == "CA-4"
        assert all(self.fleet._agents[i].tier == "commander_aircraft" for i in ids)

    def test_register_agent_update(self):
        self.fleet.register_agent("GS-001", "ground_sentinel",
                                  position=(34.05, -117.32), altitude_m=0.0)
        self.fleet.register_agent("GS-001", "ground_sentinel",
                                  position=(34.06, -117.33), altitude_m=0.0)
        agent = self.fleet._agents["GS-001"]
        assert agent.position == (34.06, -117.33)

    def test_update_condition_set_single(self):
        self.fleet.register_agent("GS-001", "ground_sentinel")
        self.fleet.update_condition_set("GS-001", "red_flag")
        assert self.fleet._agents["GS-001"].active_condition_set == "red_flag"

    def test_update_condition_set_fleet(self):
        self.fleet.register_sentinel_fleet(3)
        self.fleet.register_scout_fleet(2)
        n = self.fleet.update_condition_set_fleet("red_flag")
        assert n == 5
        assert all(a.active_condition_set == "red_flag" for a in self.fleet._agents.values())

    def test_register_rule_with_context(self):
        rule_dict = {
            "rule": "When blue haze with point source, classify as early_smoke_signature.",
            "feature": "blue_haze_point_source",
            "favors": "early_smoke_signature",
            "confidence": "high",
            "preconditions": ["Blue-gray haze visible", "Concentrated point source"],
            "rationale": "Terpene combustion produces blue smoke",
            "context_preconditions": {
                "red_flag": {"urgency_multiplier": 1.0},
                "elevated_risk": {"urgency_multiplier": 0.7},
                "normal": {"urgency_multiplier": 0.4},
            },
        }
        rule_id = self.fleet.register_rule(rule_dict, tier="ground_sentinel", precision=1.0)
        assert rule_id in self.fleet._rule_pool
        rule = self.fleet._rule_pool[rule_id]
        assert rule.tier == "ground_sentinel"
        assert rule.precision == 1.0
        assert "red_flag" in rule.context_preconditions
        assert not rule.revoked

    def test_revoke_rule(self):
        rule_dict = {
            "rule": "Test",
            "feature": "test",
            "favors": "early_smoke_signature",
            "confidence": "low",
            "preconditions": [],
            "rationale": "",
        }
        rule_id = self.fleet.register_rule(rule_dict, tier="ground_sentinel")
        assert self.fleet.revoke_rule(rule_id)
        assert self.fleet._rule_pool[rule_id].revoked
        assert len(self.fleet.get_active_rules()) == 0

    def test_revoke_nonexistent(self):
        assert not self.fleet.revoke_rule("nonexistent_id")

    def test_get_active_rules_tier_filter(self):
        sentinel_rule = {"rule": "S", "feature": "f1", "favors": "early_smoke_signature",
                         "confidence": "high", "preconditions": [], "rationale": ""}
        scout_rule = {"rule": "D", "feature": "f2", "favors": "early_smoke_signature",
                      "confidence": "high", "preconditions": [], "rationale": ""}
        cmd_rule = {"rule": "C", "feature": "f3", "favors": "early_smoke_signature",
                    "confidence": "high", "preconditions": [], "rationale": ""}
        self.fleet.register_rule(sentinel_rule, tier="ground_sentinel")
        self.fleet.register_rule(scout_rule, tier="scout_drone")
        self.fleet.register_rule(cmd_rule, tier="commander_aircraft")

        sentinel_active = self.fleet.get_active_rules(tier="ground_sentinel")
        assert len(sentinel_active) == 1

        scout_active = self.fleet.get_active_rules(tier="scout_drone")
        assert len(scout_active) == 1

        all_active = self.fleet.get_active_rules()
        assert len(all_active) == 3

    def test_broadcast_rule_to_sentinels(self):
        self.fleet.register_sentinel_fleet(5)
        rule_dict = {"rule": "R", "feature": "f", "favors": "early_smoke_signature",
                     "confidence": "high", "preconditions": [], "rationale": ""}
        rule_id = self.fleet.register_rule(rule_dict, tier="ground_sentinel")

        record = asyncio.run(
            self.fleet.broadcast_rule(rule_id, tiers=["ground_sentinel"])
        )
        assert len(record.acknowledged_by) == 5
        assert record.rule_id == rule_id
        for agent_id, agent in self.fleet._agents.items():
            if agent.tier == "ground_sentinel":
                assert rule_id in agent.active_rule_ids

    def test_broadcast_rule_not_in_pool(self):
        with pytest.raises(ValueError, match="not in pool"):
            asyncio.run(self.fleet.broadcast_rule("bad_id"))

    def test_broadcast_multi_tier(self):
        self.fleet.register_sentinel_fleet(3)
        self.fleet.register_scout_fleet(2)
        rule_dict = {"rule": "Multi", "feature": "f", "favors": "early_smoke_signature",
                     "confidence": "high", "preconditions": [], "rationale": ""}
        rule_id = self.fleet.register_rule(rule_dict, tier="all")
        record = asyncio.run(
            self.fleet.broadcast_rule(rule_id, tiers=["ground_sentinel", "scout_drone"])
        )
        assert len(record.acknowledged_by) == 5

    def test_update_sector_map(self):
        detection = self.fleet.update_sector_map(
            coordinates=(34.0522, -117.3181),
            detection_class="early_smoke_signature",
            confidence=0.90,
            rule_id="rule_abc",
            agent_id="GS-047",
            frame_id="GS047_frame_0023",
            tier="ground_sentinel",
            condition_set="red_flag",
        )
        assert detection.detection_class == "early_smoke_signature"
        assert detection.condition_set == "red_flag"

        sector = self.fleet.get_sector_map()
        assert len(sector) == 1
        assert sector[0]["detection_class"] == "early_smoke_signature"

    def test_sector_map_dedup_by_coordinates(self):
        self.fleet.update_sector_map(
            coordinates=(34.0522, -117.3181), detection_class="heat_shimmer_artifact",
            confidence=0.94, rule_id="r1", agent_id="GS-047", frame_id="f1",
            tier="ground_sentinel",
        )
        self.fleet.update_sector_map(
            coordinates=(34.0522, -117.3181), detection_class="early_smoke_signature",
            confidence=1.0, rule_id="r2", agent_id="GS-047", frame_id="f2",
            tier="ground_sentinel",
        )
        sector = self.fleet.get_sector_map()
        assert len(sector) == 1
        assert sector[0]["detection_class"] == "early_smoke_signature"

    def test_get_fleet_state(self):
        self.fleet.register_sentinel_fleet(120)
        self.fleet.register_scout_fleet(40)
        self.fleet.register_commander_fleet(4)
        state = self.fleet.get_fleet_state()
        assert state["n_agents"] == 164
        assert state["n_ground_sentinel"] == 120
        assert state["n_scout_drone"] == 40
        assert state["n_commander_aircraft"] == 4

    def test_integrate_session(self):
        self.fleet.register_sentinel_fleet(3)
        self.fleet.register_scout_fleet(2)

        transcript = {
            "outcome": "accepted",
            "final_rules": {
                "ground_sentinel": {
                    "rule": "Sentinel rule",
                    "feature": "blue_haze_point_source",
                    "favors": "early_smoke_signature",
                    "confidence": "high",
                    "preconditions": ["Blue-gray haze", "Point source"],
                    "rationale": "Test",
                    "tier": "ground_sentinel",
                    "tier_adapted": True,
                    "context_preconditions": {
                        "red_flag": {"urgency_multiplier": 1.0},
                    },
                },
                "scout_drone": {
                    "rule": "Scout rule",
                    "feature": "wind_elongation",
                    "favors": "early_smoke_signature",
                    "confidence": "high",
                    "preconditions": ["Blue-gray haze", "Wind-axis elongation"],
                    "rationale": "Test",
                    "tier": "scout_drone",
                    "tier_adapted": True,
                    "context_preconditions": {
                        "red_flag": {"urgency_multiplier": 1.0},
                    },
                },
            },
            "pool_result": {"precision": 1.0, "recall": 0.88},
        }

        registered = self.fleet.integrate_session(transcript, session_id="test_session")
        assert "ground_sentinel" in registered
        assert "scout_drone" in registered
        assert len(self.fleet.get_active_rules()) == 2


# ---------------------------------------------------------------------------
# Urgency accumulator tests
# ---------------------------------------------------------------------------

class TestUrgencyAccumulator:

    def setup_method(self):
        self.fleet = FleetManager()

    def test_single_observation_below_threshold(self):
        self.fleet.report_uncertain(
            coordinates=(34.05, -117.32),
            investigation_urgency=0.45,
            agent_id="GS-001",
            tier="ground_sentinel",
            frame_id="f1",
        )
        escalations = self.fleet.check_escalations(threshold=0.65)
        assert len(escalations) == 0

    def test_three_observations_trigger_escalation(self):
        """Three sentinel passes over the same cell should accumulate above threshold."""
        coords = (34.05, -117.32)
        now = datetime.now(timezone.utc)

        # Simulate three passes 3 minutes apart (within 5-min half-life)
        for i, urgency in enumerate([0.45, 0.65, 0.80]):
            observed = now - timedelta(minutes=(2 - i) * 3)
            self.fleet.report_uncertain(
                coordinates=coords,
                investigation_urgency=urgency,
                agent_id=f"GS-{i+1:03d}",
                tier="ground_sentinel",
                frame_id=f"f{i}",
                observed_at=observed,
            )

        escalations = self.fleet.check_escalations(threshold=0.65)
        assert len(escalations) == 1
        cell, acc = escalations[0]
        assert acc >= 0.65

    def test_old_observation_decays(self):
        """An observation 60 minutes old should contribute near zero urgency."""
        coords = (34.05, -117.32)
        old_time = datetime.now(timezone.utc) - timedelta(hours=1)

        self.fleet.report_uncertain(
            coordinates=coords,
            investigation_urgency=0.90,
            agent_id="GS-001",
            tier="ground_sentinel",
            frame_id="f0",
            observed_at=old_time,
        )

        escalations = self.fleet.check_escalations(
            threshold=0.65, half_life_s=300.0
        )
        assert len(escalations) == 0

    def test_escalate_and_resolve(self):
        coords = (34.05, -117.32)
        now = datetime.now(timezone.utc)
        for i, u in enumerate([0.5, 0.7, 0.9]):
            self.fleet.report_uncertain(
                coordinates=coords, investigation_urgency=u,
                agent_id=f"GS-{i+1:03d}", tier="ground_sentinel",
                frame_id=f"f{i}",
                observed_at=now - timedelta(minutes=i),
            )

        escalations = self.fleet.check_escalations(threshold=0.65)
        assert len(escalations) == 1
        cell, acc = escalations[0]

        event = self.fleet.escalate(cell, commander_id="CA-1")
        assert event.commander_id == "CA-1"
        assert event.resolution == "pending"
        assert cell.escalated

        # After escalation, same cell should not appear again
        escalations_after = self.fleet.check_escalations(threshold=0.65)
        assert len(escalations_after) == 0

        # Resolve
        resolved = self.fleet.resolve_escalation(coords, "confirmed_fire")
        assert resolved
        log = self.fleet.get_escalation_log()
        assert log[0]["resolution"] == "confirmed_fire"

    def test_multiple_cells_priority_order(self):
        """Higher urgency cells should sort first."""
        now = datetime.now(timezone.utc)
        for lat, urgency in [(34.05, 0.9), (34.10, 0.7), (34.15, 0.5)]:
            self.fleet.report_uncertain(
                coordinates=(lat, -117.32), investigation_urgency=urgency,
                agent_id="GS-001", tier="ground_sentinel", frame_id=f"f{lat}",
                observed_at=now,
            )
        escalations = self.fleet.check_escalations(threshold=0.45)
        urgencies = [acc for _, acc in escalations]
        assert urgencies == sorted(urgencies, reverse=True)

    def test_urgency_cell_accumulated_urgency_cap(self):
        """Accumulated urgency should never exceed 1.0."""
        coords = (34.05, -117.32)
        now = datetime.now(timezone.utc)
        for i in range(20):
            self.fleet.report_uncertain(
                coordinates=coords, investigation_urgency=1.0,
                agent_id="GS-001", tier="ground_sentinel", frame_id=f"f{i}",
                observed_at=now - timedelta(seconds=i),
            )
        cell = list(self.fleet._urgency_map.values())[0]
        assert cell.accumulated_urgency() <= 1.0


# ---------------------------------------------------------------------------
# FrameBuffer tests
# ---------------------------------------------------------------------------

def _make_frame(
    agent_id: str = "GS-001",
    tier: str = "ground_sentinel",
    original_class: str = "heat_shimmer_artifact",
    original_confidence: float = 0.94,
    age_seconds: float = 0.0,
    image_path: str = "test.jpg",
    raws_condition_set: str = "red_flag",
) -> ArchivedFrame:
    captured_at = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    return ArchivedFrame(
        frame_id=f"{agent_id}_{int(age_seconds)}",
        agent_id=agent_id,
        tier=tier,
        captured_at=captured_at,
        image_path=image_path,
        original_class=original_class,
        original_confidence=original_confidence,
        raws_condition_set=raws_condition_set,
    )


class TestFrameBuffer:

    def setup_method(self):
        self.buf = FrameBuffer()

    def test_add_and_len(self):
        self.buf.add_frame(_make_frame("GS-001"))
        self.buf.add_frame(_make_frame("GS-002"))
        assert len(self.buf) == 2

    def test_query_by_tier(self):
        self.buf.add_frame(_make_frame("GS-001", tier="ground_sentinel"))
        self.buf.add_frame(_make_frame("SD-01", tier="scout_drone"))
        sentinels = self.buf.query(tier="ground_sentinel")
        assert len(sentinels) == 1
        assert sentinels[0].agent_id == "GS-001"

    def test_query_by_class(self):
        self.buf.add_frame(_make_frame("GS-001", original_class="heat_shimmer_artifact"))
        self.buf.add_frame(_make_frame("GS-002", original_class="early_smoke_signature"))
        shimmer = self.buf.query(original_class="heat_shimmer_artifact")
        assert len(shimmer) == 1
        assert shimmer[0].agent_id == "GS-001"

    def test_query_by_confidence(self):
        self.buf.add_frame(_make_frame("GS-001", original_confidence=0.94))
        self.buf.add_frame(_make_frame("GS-002", original_confidence=0.50))
        high_conf = self.buf.query(confidence_min=0.70)
        assert len(high_conf) == 1
        assert high_conf[0].agent_id == "GS-001"

    def test_query_by_age(self):
        self.buf.add_frame(_make_frame("GS-001", age_seconds=100))
        self.buf.add_frame(_make_frame("GS-002", age_seconds=5000))  # too old
        recent = self.buf.query(max_age_seconds=200)
        assert len(recent) == 1
        assert recent[0].agent_id == "GS-001"

    def test_query_by_raws_condition(self):
        self.buf.add_frame(_make_frame("GS-001", raws_condition_set="red_flag"))
        self.buf.add_frame(_make_frame("GS-002", raws_condition_set="normal"))
        red_flag = self.buf.query(raws_condition_set="red_flag")
        assert len(red_flag) == 1
        assert red_flag[0].agent_id == "GS-001"

    def test_query_combined_filters(self):
        self.buf.add_frames([
            _make_frame("GS-001", tier="ground_sentinel",
                        original_class="heat_shimmer_artifact",
                        original_confidence=0.94, age_seconds=100),
            _make_frame("GS-002", tier="ground_sentinel",
                        original_class="heat_shimmer_artifact",
                        original_confidence=0.50, age_seconds=100),  # low conf
            _make_frame("CA-1", tier="commander_aircraft",
                        original_class="heat_shimmer_artifact",
                        original_confidence=0.94, age_seconds=100),  # wrong tier
            _make_frame("GS-003", tier="ground_sentinel",
                        original_class="heat_shimmer_artifact",
                        original_confidence=0.94, age_seconds=9000),  # too old
        ])
        results = self.buf.query(
            max_age_seconds=1800,
            tier="ground_sentinel",
            original_class="heat_shimmer_artifact",
            confidence_min=0.70,
        )
        assert len(results) == 1
        assert results[0].agent_id == "GS-001"

    def test_summary(self):
        self.buf.add_frame(_make_frame("GS-001", tier="ground_sentinel",
                                       original_class="heat_shimmer_artifact",
                                       raws_condition_set="red_flag"))
        self.buf.add_frame(_make_frame("GS-002", tier="ground_sentinel",
                                       original_class="heat_shimmer_artifact",
                                       raws_condition_set="red_flag"))
        self.buf.add_frame(_make_frame("CA-1", tier="commander_aircraft",
                                       original_class="early_smoke_signature",
                                       raws_condition_set="red_flag"))
        summary = self.buf.summary()
        assert summary["total_frames"] == 3
        assert summary["by_tier"]["ground_sentinel"] == 2
        assert summary["by_raws_condition"]["red_flag"] == 3

    def test_load_frames_from_directory(self, tmp_path):
        for i in range(3):
            (tmp_path / f"frame_{i:04d}.jpg").write_bytes(b"JPEG")
        frames = load_frames_from_directory(
            directory=tmp_path,
            agent_id="GS-047",
            tier="ground_sentinel",
            original_class="heat_shimmer_artifact",
            original_confidence=0.94,
            raws_condition_set="red_flag",
            seconds_between_frames=5.0,
        )
        assert len(frames) == 3
        assert all(f.agent_id == "GS-047" for f in frames)
        assert all(f.raws_condition_set == "red_flag" for f in frames)
        times = [f.captured_at for f in frames]
        assert times[0] > times[1] > times[2]


# ---------------------------------------------------------------------------
# Reprocess archive tests (mocked LLM)
# ---------------------------------------------------------------------------

class TestReprocessArchive:

    def test_reprocess_fires_on_matched_frames(self, tmp_path):
        from archive import reprocess_archive
        import asyncio

        smoke_paths = []
        for i in range(3):
            p = tmp_path / f"smoke_{i}.jpg"
            p.write_bytes(b"JPEG")
            smoke_paths.append(p)

        shimmer_paths = []
        for i in range(2):
            p = tmp_path / f"shimmer_{i}.jpg"
            p.write_bytes(b"JPEG")
            shimmer_paths.append(p)

        buf = FrameBuffer()
        for i, path in enumerate(smoke_paths):
            buf.add_frame(ArchivedFrame(
                frame_id=f"smoke_{i}",
                agent_id=f"GS-{i+1:03d}",
                tier="ground_sentinel",
                captured_at=datetime.now(timezone.utc) - timedelta(seconds=i * 60),
                image_path=str(path),
                original_class="heat_shimmer_artifact",
                original_confidence=0.94,
                raws_condition_set="red_flag",
            ))
        for i, path in enumerate(shimmer_paths):
            buf.add_frame(ArchivedFrame(
                frame_id=f"shimmer_{i}",
                agent_id=f"GS-{i+50:03d}",
                tier="ground_sentinel",
                captured_at=datetime.now(timezone.utc) - timedelta(seconds=i * 60),
                image_path=str(path),
                original_class="heat_shimmer_artifact",
                original_confidence=0.94,
                raws_condition_set="red_flag",
            ))

        rule = {
            "rule": "When blue haze with point source, classify as early_smoke_signature.",
            "feature": "blue_haze_point_source",
            "favors": "early_smoke_signature",
            "preconditions": ["Blue-gray haze", "Concentrated point source"],
        }

        call_count = [0]

        async def mock_call_agent(agent_name, content, system_prompt="", model="", max_tokens=512):
            import json
            call_count[0] += 1
            fires = call_count[0] <= len(smoke_paths)
            result = {
                "precondition_met": fires,
                "would_predict": "early_smoke_signature" if fires else None,
                "observations": "Blue-gray haze with point source visible." if fires else "Uniform shimmer, no point source.",
            }
            return json.dumps(result), 100

        reclassified = asyncio.run(reprocess_archive(
            rule=rule,
            rule_id="test_rule",
            frame_buffer=buf,
            lookback_seconds=1800,
            tier="ground_sentinel",
            reprocess_class="heat_shimmer_artifact",
            confidence_min=0.70,
            validator_model="claude-sonnet-4-6",
            call_agent_fn=mock_call_agent,
        ))

        assert len(reclassified) > 0
        assert all(rc.new_class == "early_smoke_signature" for rc in reclassified)
        assert all(rc.rule_id == "test_rule" for rc in reclassified)

    def test_reprocess_respects_confidence_filter(self, tmp_path):
        from archive import reprocess_archive
        import asyncio

        p = tmp_path / "frame.jpg"
        p.write_bytes(b"JPEG")

        buf = FrameBuffer()
        buf.add_frame(ArchivedFrame(
            frame_id="low_conf",
            agent_id="GS-001",
            tier="ground_sentinel",
            captured_at=datetime.now(timezone.utc),
            image_path=str(p),
            original_class="heat_shimmer_artifact",
            original_confidence=0.50,  # below threshold
        ))

        call_count = [0]

        async def mock_call_agent(agent_name, content, system_prompt="", model="", max_tokens=512):
            import json
            call_count[0] += 1
            return json.dumps({"precondition_met": True,
                               "would_predict": "early_smoke_signature",
                               "observations": "test"}), 0

        asyncio.run(reprocess_archive(
            rule={"rule": "R", "feature": "f",
                  "favors": "early_smoke_signature", "preconditions": []},
            rule_id="r1",
            frame_buffer=buf,
            lookback_seconds=1800,
            confidence_min=0.70,
            call_agent_fn=mock_call_agent,
        ))

        assert call_count[0] == 0


# ---------------------------------------------------------------------------
# MWIR oracle tests (no LLM — pure Python)
# ---------------------------------------------------------------------------

class TestMWIROracle:

    def test_oracle_fire_class(self):
        sys.path.insert(0, str(_PYTHON_DIR / "simulation"))
        from mwir_oracle import oracle_for_frame

        result = oracle_for_frame(
            frame_path="test.jpg",
            ground_truth_label="early_smoke_signature",
            agent_id="CA-1",
            coordinates=(34.0522, -117.3181),
        )
        assert result.ground_truth_class == "early_smoke_signature"
        assert "380" in result.confirmation_details       # surface temp
        assert "CA-1" in result.confirmation_details
        assert "combustion threshold" in result.confirmation_details
        assert result.confidence == 1.0
        assert result.confirmation_modality == "MWIR_calibrated_thermal"

    def test_oracle_shimmer_class(self):
        sys.path.insert(0, str(_PYTHON_DIR / "simulation"))
        from mwir_oracle import oracle_for_frame

        result = oracle_for_frame(
            frame_path="test.jpg",
            ground_truth_label="heat_shimmer_artifact",
            agent_id="CA-2",
        )
        assert result.ground_truth_class == "heat_shimmer_artifact"
        assert "Not confirmed" in result.confirmation_details

    def test_oracle_coordinates_in_output(self):
        sys.path.insert(0, str(_PYTHON_DIR / "simulation"))
        from mwir_oracle import oracle_for_frame

        result = oracle_for_frame(
            frame_path="test.jpg",
            ground_truth_label="early_smoke_signature",
            agent_id="CA-1",
            coordinates=(34.0522, -117.3181),
        )
        assert "34.0522" in result.confirmation_details

    def test_real_sensor_raises_not_implemented(self):
        sys.path.insert(0, str(_PYTHON_DIR / "simulation"))
        from mwir_oracle import mwir_from_real_sensor

        with pytest.raises(NotImplementedError, match="Phase 4"):
            mwir_from_real_sensor("thermal.jpg", "rgb.jpg")
