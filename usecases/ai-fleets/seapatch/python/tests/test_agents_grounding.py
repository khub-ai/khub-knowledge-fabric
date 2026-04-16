"""
test_agents_grounding.py — Unit tests for tier grounding check and temporal
reformulation logic in agents.py.

Uses a mocked LLM call — no API calls required.

Run from repo root:
    python -m pytest usecases/ai-fleets/drone-swarm/python/tests/test_agents_grounding.py -v
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

_PYTHON_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PYTHON_DIR.parents[3]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_PYTHON_DIR))

from agents import (
    run_tier_grounding_check,
    run_temporal_reformulation,
    adapt_rule_for_tier,
)
from domain_config import MARITIME_SAR_CONFIG, TIER_OBSERVABILITY


# ---------------------------------------------------------------------------
# Mock call_agent factory
# ---------------------------------------------------------------------------

def _make_call_agent(response_json: dict):
    """Return an async call_agent mock that always returns response_json."""
    async def mock_call_agent(agent_name, content, system_prompt="", model="", max_tokens=512):
        return json.dumps(response_json), 50
    return mock_call_agent


def _make_sequential_call_agent(responses: list[dict]):
    """Return a call_agent that returns responses in sequence."""
    call_count = [0]

    async def mock_call_agent(agent_name, content, system_prompt="", model="", max_tokens=512):
        idx = min(call_count[0], len(responses) - 1)
        call_count[0] += 1
        return json.dumps(responses[idx]), 50

    return mock_call_agent


# ---------------------------------------------------------------------------
# run_tier_grounding_check
# ---------------------------------------------------------------------------

class TestRunTierGroundingCheck:

    def _make_rule(self, preconditions: list[str]) -> dict:
        return {
            "rule": "When conditions are met, classify as person_in_water.",
            "feature": "test_feature",
            "favors": "person_in_water",
            "confidence": "high",
            "preconditions": preconditions,
        }

    def test_all_observable(self):
        preconditions = ["Uniform brightness across oval", "Clean elliptical boundary"]
        response = {
            "tier": "scout",
            "criteria": [
                {"precondition": preconditions[0], "classification": "observable", "reason": "Visible at 1-2cm/px"},
                {"precondition": preconditions[1], "classification": "observable", "reason": "Shape visible at altitude"},
            ],
            "summary": "accept_all",
        }
        result, ms = asyncio.run(run_tier_grounding_check(
            candidate_rule=self._make_rule(preconditions),
            tier="scout",
            tier_observability=TIER_OBSERVABILITY["scout"],
            config=MARITIME_SAR_CONFIG,
            call_agent_fn=_make_call_agent(response),
        ))
        assert result["summary"] == "accept_all"
        assert len(result["observable"]) == 2
        assert len(result["temporal"]) == 0
        assert len(result["unobservable"]) == 0

    def test_temporal_criterion_flagged(self):
        preconditions = [
            "Uniform brightness across oval",
            "Oval maintains position across 15 consecutive frames",
        ]
        response = {
            "tier": "scout",
            "criteria": [
                {"precondition": preconditions[0], "classification": "observable", "reason": "Single-frame visible"},
                {"precondition": preconditions[1], "classification": "temporal", "reason": "Requires multi-frame comparison"},
            ],
            "summary": "reformulate_temporal",
        }
        result, _ = asyncio.run(run_tier_grounding_check(
            candidate_rule=self._make_rule(preconditions),
            tier="scout",
            tier_observability=TIER_OBSERVABILITY["scout"],
            config=MARITIME_SAR_CONFIG,
            call_agent_fn=_make_call_agent(response),
        ))
        assert result["summary"] == "reformulate_temporal"
        assert len(result["temporal"]) == 1
        assert "consecutive frames" in result["temporal"][0]
        assert len(result["observable"]) == 1

    def test_unobservable_criterion_flagged(self):
        preconditions = [
            "Uniform brightness across oval",
            "Ridge curvature detail 2mm scale visible",
        ]
        response = {
            "tier": "scout",
            "criteria": [
                {"precondition": preconditions[0], "classification": "observable", "reason": "OK"},
                {"precondition": preconditions[1], "classification": "unobservable", "reason": "Requires sub-5cm resolution"},
            ],
            "summary": "remove_some",
        }
        result, _ = asyncio.run(run_tier_grounding_check(
            candidate_rule=self._make_rule(preconditions),
            tier="scout",
            tier_observability=TIER_OBSERVABILITY["scout"],
            config=MARITIME_SAR_CONFIG,
            call_agent_fn=_make_call_agent(response),
        ))
        assert len(result["unobservable"]) == 1
        assert len(result["observable"]) == 1

    def test_parse_failure_fallback(self):
        """On parse failure, all preconditions treated as observable."""
        async def bad_call_agent(agent_name, content, system_prompt="", model="", max_tokens=512):
            return "Not valid JSON at all", 50

        preconditions = ["Condition A", "Condition B"]
        result, _ = asyncio.run(run_tier_grounding_check(
            candidate_rule=self._make_rule(preconditions),
            tier="scout",
            tier_observability=TIER_OBSERVABILITY["scout"],
            config=MARITIME_SAR_CONFIG,
            call_agent_fn=bad_call_agent,
        ))
        # Fallback: all treated as observable
        assert len(result["observable"]) == 2
        assert len(result["temporal"]) == 0
        assert len(result["unobservable"]) == 0

    def test_commander_tier_receives_different_context(self):
        """Verify commander tier passes its own observability context."""
        received_system_prompt = []

        async def capture_system_prompt(agent_name, content, system_prompt="", model="", max_tokens=512):
            received_system_prompt.append(system_prompt)
            return json.dumps({
                "tier": "commander",
                "criteria": [{"precondition": "P", "classification": "observable", "reason": "OK"}],
                "summary": "accept_all",
            }), 50

        asyncio.run(run_tier_grounding_check(
            candidate_rule={"rule": "R", "feature": "f", "favors": "piw",
                            "confidence": "high", "preconditions": ["P"]},
            tier="commander",
            tier_observability=TIER_OBSERVABILITY["commander"],
            config=MARITIME_SAR_CONFIG,
            call_agent_fn=capture_system_prompt,
        ))

        assert received_system_prompt
        assert "commander" in received_system_prompt[0]
        assert "stabilised" in received_system_prompt[0]  # commander tier description


# ---------------------------------------------------------------------------
# run_temporal_reformulation
# ---------------------------------------------------------------------------

class TestRunTemporalReformulation:

    def test_returns_proxy(self):
        response = {
            "temporal_criterion": "oval maintains position across 15 consecutive frames",
            "proxy": "brightness is approximately uniform across the full oval extent — unlike a whitecap which fades from crest outward",
            "rationale": "Uniform brightness in a single frame is the static correlate of temporal stability.",
            "confidence": "high",
        }
        result, ms = asyncio.run(run_temporal_reformulation(
            temporal_criterion="oval maintains position across 15 consecutive frames",
            ground_truth_class="person_in_water",
            wrong_class="whitecap",
            config=MARITIME_SAR_CONFIG,
            call_agent_fn=_make_call_agent(response),
        ))
        assert result["proxy"] is not None
        assert "brightness" in result["proxy"]
        assert result["confidence"] == "high"

    def test_parse_failure_returns_none_proxy(self):
        async def bad_call(agent_name, content, system_prompt="", model="", max_tokens=512):
            return "not json", 50

        result, _ = asyncio.run(run_temporal_reformulation(
            temporal_criterion="stable across frames",
            ground_truth_class="person_in_water",
            wrong_class="whitecap",
            config=MARITIME_SAR_CONFIG,
            call_agent_fn=bad_call,
        ))
        assert result["proxy"] is None
        assert result["confidence"] == "low"


# ---------------------------------------------------------------------------
# adapt_rule_for_tier
# ---------------------------------------------------------------------------

class TestAdaptRuleForTier:

    def _base_rule(self) -> dict:
        return {
            "rule": "When conditions are met, classify as person_in_water.",
            "feature": "brightness_uniform_oval",
            "favors": "person_in_water",
            "confidence": "high",
            "preconditions": [
                "Uniform brightness across oval",
                "Oval stable across 15 frames",         # temporal
                "Sub-millimetre surface texture visible", # unobservable for scout
            ],
        }

    def test_removes_unobservable_adds_proxy(self):
        grounding_response = {
            "tier": "scout",
            "criteria": [
                {"precondition": "Uniform brightness across oval",
                 "classification": "observable", "reason": "OK"},
                {"precondition": "Oval stable across 15 frames",
                 "classification": "temporal", "reason": "Multi-frame"},
                {"precondition": "Sub-millimetre surface texture visible",
                 "classification": "unobservable", "reason": "Too fine"},
            ],
            "summary": "remove_some",
        }
        reformulation_response = {
            "temporal_criterion": "Oval stable across 15 frames",
            "proxy": "Boundary is clean ellipse without foam dispersal",
            "rationale": "Static correlate of temporal stability.",
            "confidence": "medium",
        }

        responses = [grounding_response, reformulation_response]
        adapted_rule, report = asyncio.run(adapt_rule_for_tier(
            candidate_rule=self._base_rule(),
            tier="scout",
            tier_observability=TIER_OBSERVABILITY["scout"],
            ground_truth_class="person_in_water",
            wrong_class="whitecap",
            config=MARITIME_SAR_CONFIG,
            call_agent_fn=_make_sequential_call_agent(responses),
        ))

        # Unobservable removed, temporal replaced by proxy, observable kept
        assert "Sub-millimetre surface texture visible" not in adapted_rule["preconditions"]
        assert "Oval stable across 15 frames" not in adapted_rule["preconditions"]
        assert "Uniform brightness across oval" in adapted_rule["preconditions"]
        assert "Boundary is clean ellipse without foam dispersal" in adapted_rule["preconditions"]

        # Report should document what happened
        assert len(report["unobservable_removed"]) == 1
        assert len(report["temporal_reformulated"]) == 1
        assert report["temporal_reformulated"][0]["proxy"] == "Boundary is clean ellipse without foam dispersal"

    def test_all_observable_no_reformulation_calls(self):
        call_count = [0]

        async def counting_call_agent(agent_name, content, system_prompt="", model="", max_tokens=512):
            call_count[0] += 1
            return json.dumps({
                "tier": "scout",
                "criteria": [
                    {"precondition": "Uniform brightness across oval",
                     "classification": "observable", "reason": "OK"},
                ],
                "summary": "accept_all",
            }), 50

        rule = {
            "rule": "R",
            "feature": "f",
            "favors": "person_in_water",
            "confidence": "high",
            "preconditions": ["Uniform brightness across oval"],
        }
        adapted_rule, report = asyncio.run(adapt_rule_for_tier(
            candidate_rule=rule,
            tier="scout",
            tier_observability=TIER_OBSERVABILITY["scout"],
            ground_truth_class="person_in_water",
            wrong_class="whitecap",
            config=MARITIME_SAR_CONFIG,
            call_agent_fn=counting_call_agent,
        ))

        # Only 1 call (grounding check), 0 reformulation calls
        assert call_count[0] == 1
        assert adapted_rule["preconditions"] == ["Uniform brightness across oval"]
        assert report["unobservable_removed"] == []
        assert report["temporal_reformulated"] == []

    def test_adapted_rule_metadata(self):
        async def call_agent(agent_name, content, system_prompt="", model="", max_tokens=512):
            return json.dumps({
                "tier": "scout",
                "criteria": [
                    {"precondition": "Uniform brightness", "classification": "observable", "reason": "OK"},
                ],
                "summary": "accept_all",
            }), 50

        rule = {"rule": "R", "feature": "f", "favors": "person_in_water",
                "confidence": "high", "preconditions": ["Uniform brightness"]}
        adapted_rule, _ = asyncio.run(adapt_rule_for_tier(
            candidate_rule=rule,
            tier="scout",
            tier_observability=TIER_OBSERVABILITY["scout"],
            ground_truth_class="person_in_water",
            wrong_class="whitecap",
            config=MARITIME_SAR_CONFIG,
            call_agent_fn=call_agent,
        ))

        assert adapted_rule["tier"] == "scout"
        assert adapted_rule["tier_adapted"] is True
        assert adapted_rule["favors"] == "person_in_water"


# ---------------------------------------------------------------------------
# Thermal oracle tests (no LLM — pure Python)
# ---------------------------------------------------------------------------

class TestThermalOracle:

    def test_oracle_person_in_water(self):
        sys.path.insert(0, str(_PYTHON_DIR / "simulation"))
        from thermal_oracle import oracle_for_frame

        result = oracle_for_frame(
            frame_path="test.jpg",
            ground_truth_label="person_in_water",
            drone_id="C1",
            coordinates=(51.5074, 1.2345),
        )
        assert result.ground_truth_class == "person_in_water"
        assert "37" in result.confirmation_details  # temperature
        assert "C1" in result.confirmation_details
        assert result.confidence == 1.0

    def test_oracle_whitecap(self):
        sys.path.insert(0, str(_PYTHON_DIR / "simulation"))
        from thermal_oracle import oracle_for_frame

        result = oracle_for_frame(
            frame_path="test.jpg",
            ground_truth_label="whitecap",
            drone_id="C2",
        )
        assert result.ground_truth_class == "whitecap"
        assert "no persistent" in result.confirmation_details.lower()

    def test_thermal_from_flir_raises_not_implemented(self):
        sys.path.insert(0, str(_PYTHON_DIR / "simulation"))
        from thermal_oracle import thermal_from_flir_image

        with pytest.raises(NotImplementedError, match="Phase 4"):
            thermal_from_flir_image("thermal.jpg", "rgb.jpg")
