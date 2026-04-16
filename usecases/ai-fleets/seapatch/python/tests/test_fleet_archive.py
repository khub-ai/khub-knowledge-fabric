"""
test_fleet_archive.py — Unit tests for fleet.py and archive.py.

No LLM calls. Tests pure Python state management logic.

Run from repo root:
    python -m pytest usecases/ai-fleets/drone-swarm/python/tests/test_fleet_archive.py -v
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Add python/ dir to path
_PYTHON_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PYTHON_DIR.parents[3]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_PYTHON_DIR))

from fleet import FleetManager, Rule, DroneState
from archive import FrameBuffer, ArchivedFrame, load_frames_from_directory


# ---------------------------------------------------------------------------
# FleetManager tests
# ---------------------------------------------------------------------------

class TestFleetManager:

    def setup_method(self):
        self.fleet = FleetManager()

    def test_register_scout_fleet(self):
        ids = self.fleet.register_scout_fleet(5)
        assert len(ids) == 5
        assert ids[0] == "S01"
        assert ids[4] == "S05"
        assert all(self.fleet._drones[i].tier == "scout" for i in ids)

    def test_register_commander_fleet(self):
        ids = self.fleet.register_commander_fleet(2)
        assert ids == ["C1", "C2"]
        assert all(self.fleet._drones[i].tier == "commander" for i in ids)

    def test_register_drone_update(self):
        self.fleet.register_drone("S01", "scout", position=(51.5, 1.2), altitude_m=30.0)
        self.fleet.register_drone("S01", "scout", position=(51.6, 1.3), altitude_m=35.0)
        drone = self.fleet._drones["S01"]
        assert drone.position == (51.6, 1.3)
        assert drone.altitude_m == 35.0

    def test_register_rule(self):
        rule_dict = {
            "rule": "When oval is uniform, classify as person_in_water.",
            "feature": "brightness_uniform_oval",
            "favors": "person_in_water",
            "confidence": "high",
            "preconditions": ["Uniform brightness", "Clean ellipse"],
            "rationale": "Test",
        }
        rule_id = self.fleet.register_rule(rule_dict, tier="scout", precision=0.95)
        assert rule_id in self.fleet._rule_pool
        rule = self.fleet._rule_pool[rule_id]
        assert rule.tier == "scout"
        assert rule.precision == 0.95
        assert rule.favors == "person_in_water"
        assert not rule.revoked

    def test_revoke_rule(self):
        rule_dict = {
            "rule": "Test rule",
            "feature": "test",
            "favors": "person_in_water",
            "confidence": "low",
            "preconditions": [],
            "rationale": "",
        }
        rule_id = self.fleet.register_rule(rule_dict, tier="scout")
        assert self.fleet.revoke_rule(rule_id)
        assert self.fleet._rule_pool[rule_id].revoked
        assert len(self.fleet.get_active_rules()) == 0

    def test_revoke_nonexistent(self):
        assert not self.fleet.revoke_rule("nonexistent_id")

    def test_get_active_rules_tier_filter(self):
        scout_rule = {"rule": "Scout rule", "feature": "f1", "favors": "person_in_water",
                      "confidence": "high", "preconditions": [], "rationale": ""}
        cmd_rule = {"rule": "Commander rule", "feature": "f2", "favors": "person_in_water",
                    "confidence": "high", "preconditions": [], "rationale": ""}
        self.fleet.register_rule(scout_rule, tier="scout")
        self.fleet.register_rule(cmd_rule, tier="commander")

        scout_active = self.fleet.get_active_rules(tier="scout")
        assert len(scout_active) == 1
        assert scout_active[0].tier == "scout"

        cmd_active = self.fleet.get_active_rules(tier="commander")
        assert len(cmd_active) == 1

        all_active = self.fleet.get_active_rules()
        assert len(all_active) == 2

    def test_broadcast_rule(self):
        self.fleet.register_scout_fleet(5)
        rule_dict = {"rule": "R", "feature": "f", "favors": "person_in_water",
                     "confidence": "high", "preconditions": [], "rationale": ""}
        rule_id = self.fleet.register_rule(rule_dict, tier="scout")

        record = asyncio.run(self.fleet.broadcast_rule(rule_id, tiers=["scout"]))
        assert len(record.acknowledged_by) == 5
        assert record.rule_id == rule_id
        assert record.latency_ms is not None

        # All scouts should have the rule
        for drone_id, drone in self.fleet._drones.items():
            if drone.tier == "scout":
                assert rule_id in drone.active_rule_ids

    def test_broadcast_rule_not_in_pool(self):
        with pytest.raises(ValueError, match="not in pool"):
            asyncio.run(self.fleet.broadcast_rule("bad_id"))

    def test_broadcast_commander_only(self):
        self.fleet.register_scout_fleet(3)
        self.fleet.register_commander_fleet(2)
        rule_dict = {"rule": "Cmd rule", "feature": "f", "favors": "person_in_water",
                     "confidence": "high", "preconditions": [], "rationale": ""}
        rule_id = self.fleet.register_rule(rule_dict, tier="commander")
        record = asyncio.run(self.fleet.broadcast_rule(rule_id, tiers=["commander"]))
        assert len(record.acknowledged_by) == 2
        acked = set(record.acknowledged_by)
        assert acked == {"C1", "C2"}

    def test_update_track_map(self):
        detection = self.fleet.update_track_map(
            coordinates=(51.5074, 1.2345),
            detection_class="person_in_water",
            confidence=0.95,
            rule_id="rule_abc",
            drone_id="S22",
            frame_id="S22_frame_0042",
        )
        assert detection.detection_class == "person_in_water"
        assert detection.drone_id == "S22"

        track = self.fleet.get_track_map()
        assert len(track) == 1
        assert track[0]["detection_class"] == "person_in_water"

    def test_track_map_dedup_by_coordinates(self):
        # Two detections at same coordinate: second should overwrite
        self.fleet.update_track_map(
            coordinates=(51.5074, 1.2345), detection_class="whitecap",
            confidence=0.91, rule_id="r1", drone_id="S22", frame_id="f1",
        )
        self.fleet.update_track_map(
            coordinates=(51.5074, 1.2345), detection_class="person_in_water",
            confidence=1.0, rule_id="r2", drone_id="S22", frame_id="f2",
        )
        track = self.fleet.get_track_map()
        assert len(track) == 1
        assert track[0]["detection_class"] == "person_in_water"

    def test_get_swarm_state(self):
        self.fleet.register_scout_fleet(38)
        self.fleet.register_commander_fleet(2)
        state = self.fleet.get_swarm_state()
        assert state["n_drones"] == 40
        assert state["n_scouts"] == 38
        assert state["n_commanders"] == 2

    def test_integrate_session(self):
        self.fleet.register_scout_fleet(3)
        self.fleet.register_commander_fleet(2)

        transcript = {
            "outcome": "accepted",
            "final_rules": {
                "scout": {
                    "rule": "Scout rule",
                    "feature": "f_scout",
                    "favors": "person_in_water",
                    "confidence": "high",
                    "preconditions": ["Condition A"],
                    "rationale": "Test",
                    "tier": "scout",
                    "tier_adapted": True,
                },
                "commander": {
                    "rule": "Commander rule",
                    "feature": "f_cmd",
                    "favors": "person_in_water",
                    "confidence": "high",
                    "preconditions": ["Condition A", "Stable across 8 frames"],
                    "rationale": "Test",
                    "tier": "commander",
                    "tier_adapted": True,
                },
            },
            "pool_result": {"precision": 1.0, "recall": 0.75},
            "pool_result_after_tighten": None,
        }

        registered = self.fleet.integrate_session(transcript, session_id="test_session")
        assert "scout" in registered
        assert "commander" in registered
        assert len(self.fleet.get_active_rules()) == 2


# ---------------------------------------------------------------------------
# FrameBuffer tests
# ---------------------------------------------------------------------------

def _make_frame(
    drone_id: str = "S01",
    tier: str = "scout",
    original_class: str = "whitecap",
    original_confidence: float = 0.91,
    age_seconds: float = 0.0,
    image_path: str = "test.jpg",
) -> ArchivedFrame:
    captured_at = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    return ArchivedFrame(
        frame_id=f"{drone_id}_{int(age_seconds)}",
        drone_id=drone_id,
        tier=tier,
        captured_at=captured_at,
        image_path=image_path,
        original_class=original_class,
        original_confidence=original_confidence,
    )


class TestFrameBuffer:

    def setup_method(self):
        self.buf = FrameBuffer()

    def test_add_and_len(self):
        self.buf.add_frame(_make_frame("S01"))
        self.buf.add_frame(_make_frame("S02"))
        assert len(self.buf) == 2

    def test_query_by_tier(self):
        self.buf.add_frame(_make_frame("S01", tier="scout"))
        self.buf.add_frame(_make_frame("C1", tier="commander"))
        scouts = self.buf.query(tier="scout")
        assert len(scouts) == 1
        assert scouts[0].drone_id == "S01"

    def test_query_by_class(self):
        self.buf.add_frame(_make_frame("S01", original_class="whitecap"))
        self.buf.add_frame(_make_frame("S02", original_class="person_in_water"))
        whitecaps = self.buf.query(original_class="whitecap")
        assert len(whitecaps) == 1
        assert whitecaps[0].drone_id == "S01"

    def test_query_by_confidence(self):
        self.buf.add_frame(_make_frame("S01", original_confidence=0.91))
        self.buf.add_frame(_make_frame("S02", original_confidence=0.50))
        high_conf = self.buf.query(confidence_min=0.70)
        assert len(high_conf) == 1
        assert high_conf[0].drone_id == "S01"

    def test_query_by_age(self):
        self.buf.add_frame(_make_frame("S01", age_seconds=100))
        self.buf.add_frame(_make_frame("S02", age_seconds=5000))  # too old
        recent = self.buf.query(max_age_seconds=200)
        assert len(recent) == 1
        assert recent[0].drone_id == "S01"

    def test_query_by_drone_id(self):
        self.buf.add_frame(_make_frame("S01"))
        self.buf.add_frame(_make_frame("S22"))
        s22_frames = self.buf.query(drone_id="S22")
        assert len(s22_frames) == 1
        assert s22_frames[0].drone_id == "S22"

    def test_query_combined_filters(self):
        self.buf.add_frames([
            _make_frame("S01", tier="scout", original_class="whitecap",
                        original_confidence=0.91, age_seconds=100),
            _make_frame("S02", tier="scout", original_class="whitecap",
                        original_confidence=0.50, age_seconds=100),  # low conf
            _make_frame("C1", tier="commander", original_class="whitecap",
                        original_confidence=0.91, age_seconds=100),   # wrong tier
            _make_frame("S03", tier="scout", original_class="whitecap",
                        original_confidence=0.91, age_seconds=9000),  # too old
        ])
        results = self.buf.query(
            max_age_seconds=2700,
            tier="scout",
            original_class="whitecap",
            confidence_min=0.70,
        )
        assert len(results) == 1
        assert results[0].drone_id == "S01"

    def test_summary(self):
        self.buf.add_frame(_make_frame("S01", tier="scout", original_class="whitecap"))
        self.buf.add_frame(_make_frame("S02", tier="scout", original_class="whitecap"))
        self.buf.add_frame(_make_frame("C1", tier="commander", original_class="person_in_water"))
        summary = self.buf.summary()
        assert summary["total_frames"] == 3
        assert summary["by_tier"]["scout"] == 2
        assert summary["by_tier"]["commander"] == 1
        assert summary["by_class"]["whitecap"] == 2

    def test_load_frames_from_directory(self, tmp_path):
        # Create fake image files
        for i in range(3):
            (tmp_path / f"frame_{i:04d}.jpg").write_bytes(b"JPEG")
        frames = load_frames_from_directory(
            directory=tmp_path,
            drone_id="S22",
            tier="scout",
            original_class="whitecap",
            original_confidence=0.91,
            seconds_between_frames=2.0,
        )
        assert len(frames) == 3
        assert all(f.drone_id == "S22" for f in frames)
        assert all(f.tier == "scout" for f in frames)
        # Frames should be timestamped in descending order
        times = [f.captured_at for f in frames]
        assert times[0] > times[1] > times[2]


# ---------------------------------------------------------------------------
# Reprocess archive tests (mocked LLM)
# ---------------------------------------------------------------------------

class TestReprocessArchive:
    """Tests retroactive reprocessing using a mocked LLM call."""

    def _make_buffer_with_person_frames(self, n_person: int, n_whitecap: int) -> FrameBuffer:
        buf = FrameBuffer()
        for i in range(n_person):
            buf.add_frame(_make_frame(
                drone_id=f"S{i+1:02d}",
                tier="scout",
                original_class="whitecap",   # misclassified
                original_confidence=0.91,
                age_seconds=i * 60,
                image_path=f"/tmp/person_{i}.jpg",
            ))
        for i in range(n_whitecap):
            buf.add_frame(_make_frame(
                drone_id=f"S{i+50:02d}",
                tier="scout",
                original_class="whitecap",
                original_confidence=0.91,
                age_seconds=i * 60,
                image_path=f"/tmp/actual_whitecap_{i}.jpg",
            ))
        return buf

    def test_reprocess_fires_on_matched_frames(self, tmp_path):
        """Rule fires on person frames, not whitecap frames."""
        from archive import reprocess_archive
        import asyncio

        # Create real dummy image files so the path check passes
        person_paths = []
        for i in range(3):
            p = tmp_path / f"person_{i}.jpg"
            p.write_bytes(b"JPEG")
            person_paths.append(p)

        whitecap_paths = []
        for i in range(2):
            p = tmp_path / f"whitecap_{i}.jpg"
            p.write_bytes(b"JPEG")
            whitecap_paths.append(p)

        buf = FrameBuffer()
        for i, path in enumerate(person_paths):
            buf.add_frame(ArchivedFrame(
                frame_id=f"person_{i}",
                drone_id=f"S{i+1:02d}",
                tier="scout",
                captured_at=datetime.now(timezone.utc) - timedelta(seconds=i * 60),
                image_path=str(path),
                original_class="whitecap",
                original_confidence=0.91,
            ))
        for i, path in enumerate(whitecap_paths):
            buf.add_frame(ArchivedFrame(
                frame_id=f"whitecap_{i}",
                drone_id=f"S{i+50:02d}",
                tier="scout",
                captured_at=datetime.now(timezone.utc) - timedelta(seconds=i * 60),
                image_path=str(path),
                original_class="whitecap",
                original_confidence=0.91,
            ))

        rule = {
            "rule": "When oval is uniform, classify as person_in_water.",
            "feature": "brightness_uniform_oval",
            "favors": "person_in_water",
            "preconditions": ["Uniform brightness", "Clean ellipse"],
        }

        call_count = 0

        async def mock_call_agent(agent_name, content, system_prompt="", model="", max_tokens=512):
            nonlocal call_count
            call_count += 1
            # Person frames fire; whitecap frames don't
            image_path = content[0]["source"]["data"] if isinstance(content, list) else ""
            # Determine by frame_id — we alternate: person frames fire
            fires = call_count <= len(person_paths)
            result = {
                "precondition_met": fires,
                "would_predict": "person_in_water" if fires else None,
                "observations": "Uniform brightness oval observed." if fires else "Fading brightness, irregular boundary.",
            }
            import json
            return json.dumps(result), 100

        reclassified = asyncio.run(reprocess_archive(
            rule=rule,
            rule_id="test_rule",
            frame_buffer=buf,
            lookback_seconds=2700,
            tier="scout",
            reprocess_class="whitecap",
            confidence_min=0.70,
            validator_model="claude-sonnet-4-6",
            call_agent_fn=mock_call_agent,
        ))

        assert len(reclassified) > 0
        assert all(rc.new_class == "person_in_water" for rc in reclassified)
        assert all(rc.rule_id == "test_rule" for rc in reclassified)

    def test_reprocess_respects_confidence_filter(self, tmp_path):
        """Frames below confidence_min are not reprocessed."""
        from archive import reprocess_archive
        import asyncio

        p = tmp_path / "frame.jpg"
        p.write_bytes(b"JPEG")

        buf = FrameBuffer()
        buf.add_frame(ArchivedFrame(
            frame_id="low_conf",
            drone_id="S01",
            tier="scout",
            captured_at=datetime.now(timezone.utc),
            image_path=str(p),
            original_class="whitecap",
            original_confidence=0.50,  # below threshold
        ))

        call_count = 0

        async def mock_call_agent(agent_name, content, system_prompt="", model="", max_tokens=512):
            nonlocal call_count
            call_count += 1
            return '{"precondition_met": true, "would_predict": "person_in_water", "observations": "test"}', 0

        asyncio.run(reprocess_archive(
            rule={"rule": "R", "feature": "f", "favors": "person_in_water", "preconditions": []},
            rule_id="r1",
            frame_buffer=buf,
            lookback_seconds=2700,
            confidence_min=0.70,  # 0.50 < 0.70, so frame excluded
            call_agent_fn=mock_call_agent,
        ))

        # Low-confidence frame should not have been processed
        assert call_count == 0
