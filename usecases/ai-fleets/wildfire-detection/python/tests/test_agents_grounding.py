"""
test_agents_grounding.py — Unit tests for tier grounding check, temporal
reformulation, environmental context injection, and adapt_rule_for_tier.

No LLM calls — uses mocked call_agent functions.

Run from repo root:
    python -m pytest usecases/ai-fleets/wildfire-detection/python/tests/test_agents_grounding.py -v
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
    run_env_context_injection,
    _evaluate_condition_set,
)
from domain_config import WILDFIRE_DETECTION_CONFIG, TIER_OBSERVABILITY


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _make_call_agent(response_json: dict):
    async def mock(agent_name, content, system_prompt="", model="", max_tokens=512):
        return json.dumps(response_json), 50
    return mock


def _make_sequential_call_agent(responses: list[dict]):
    call_count = [0]

    async def mock(agent_name, content, system_prompt="", model="", max_tokens=512):
        idx = min(call_count[0], len(responses) - 1)
        call_count[0] += 1
        return json.dumps(responses[idx]), 50

    return mock


def _base_rule(preconditions=None) -> dict:
    return {
        "rule": "When blue haze with point source, classify as early_smoke_signature.",
        "feature": "blue_haze_point_source",
        "favors": "early_smoke_signature",
        "confidence": "high",
        "preconditions": preconditions or [
            "Blue-gray haze visible in the frame",
            "Concentrated point source at the base of the haze",
        ],
        "rationale": "Terpene smoke from chaparral combustion has a blue color.",
    }


# ---------------------------------------------------------------------------
# _evaluate_condition_set (pure Python, no mock needed)
# ---------------------------------------------------------------------------

class TestEvaluateConditionSet:

    def test_red_flag_all_criteria(self):
        raws = {"wind_speed_mph": 28, "relative_humidity_pct": 8,
                "temperature_f": 95, "fire_weather_watch": False}
        assert _evaluate_condition_set(raws) == "red_flag"

    def test_red_flag_via_watch(self):
        raws = {"wind_speed_mph": 5, "relative_humidity_pct": 40,
                "temperature_f": 70, "fire_weather_watch": True}
        assert _evaluate_condition_set(raws) == "red_flag"

    def test_elevated_risk(self):
        raws = {"wind_speed_mph": 15, "relative_humidity_pct": 20,
                "temperature_f": 80}
        assert _evaluate_condition_set(raws) == "elevated_risk"

    def test_normal(self):
        raws = {"wind_speed_mph": 5, "relative_humidity_pct": 60,
                "temperature_f": 65}
        assert _evaluate_condition_set(raws) == "normal"

    def test_empty_raws_returns_normal(self):
        assert _evaluate_condition_set({}) == "normal"


# ---------------------------------------------------------------------------
# run_tier_grounding_check
# ---------------------------------------------------------------------------

class TestRunTierGroundingCheck:

    def test_all_observable(self):
        preconditions = [
            "Blue-gray haze visible in the frame",
            "Concentrated point source at the haze base",
        ]
        response = {
            "tier": "ground_sentinel",
            "criteria": [
                {"precondition": preconditions[0], "classification": "observable",
                 "reason": "Color visible in 4K RGB at 350m range"},
                {"precondition": preconditions[1], "classification": "observable",
                 "reason": "Spatial point visible in single frame"},
            ],
            "summary": "accept_all",
        }
        result, ms = asyncio.run(run_tier_grounding_check(
            candidate_rule=_base_rule(preconditions),
            tier="ground_sentinel",
            tier_observability=TIER_OBSERVABILITY["ground_sentinel"],
            config=WILDFIRE_DETECTION_CONFIG,
            call_agent_fn=_make_call_agent(response),
        ))
        assert result["summary"] == "accept_all"
        assert len(result["observable"]) == 2
        assert len(result["temporal"]) == 0
        assert len(result["unobservable"]) == 0

    def test_temporal_criterion_flagged_for_scout(self):
        """Drift direction is temporal for scout drone (single-pass)."""
        preconditions = [
            "Blue-gray haze visible",
            "Haze drifts southeast consistently across consecutive frames",
        ]
        response = {
            "tier": "scout_drone",
            "criteria": [
                {"precondition": preconditions[0], "classification": "observable",
                 "reason": "Single-frame color visible"},
                {"precondition": preconditions[1], "classification": "temporal",
                 "reason": "Requires consecutive frames — single-pass drone"},
            ],
            "summary": "reformulate_temporal",
        }
        result, _ = asyncio.run(run_tier_grounding_check(
            candidate_rule=_base_rule(preconditions),
            tier="scout_drone",
            tier_observability=TIER_OBSERVABILITY["scout_drone"],
            config=WILDFIRE_DETECTION_CONFIG,
            call_agent_fn=_make_call_agent(response),
        ))
        assert result["summary"] == "reformulate_temporal"
        assert len(result["temporal"]) == 1
        assert "consecutive frames" in result["temporal"][0]
        assert len(result["observable"]) == 1

    def test_temporal_criterion_observable_for_sentinel(self):
        """Same drift feature is observable for ground sentinel (fixed PTZ)."""
        preconditions = [
            "Blue-gray haze visible",
            "Haze drifts southeast consistently across consecutive frames",
        ]
        response = {
            "tier": "ground_sentinel",
            "criteria": [
                {"precondition": preconditions[0], "classification": "observable",
                 "reason": "Color visible"},
                {"precondition": preconditions[1], "classification": "observable",
                 "reason": "Fixed PTZ covers same area — temporal features available"},
            ],
            "summary": "accept_all",
        }
        result, _ = asyncio.run(run_tier_grounding_check(
            candidate_rule=_base_rule(preconditions),
            tier="ground_sentinel",
            tier_observability=TIER_OBSERVABILITY["ground_sentinel"],
            config=WILDFIRE_DETECTION_CONFIG,
            call_agent_fn=_make_call_agent(response),
        ))
        assert len(result["temporal"]) == 0
        assert len(result["observable"]) == 2

    def test_unobservable_criterion_flagged(self):
        preconditions = [
            "Blue-gray haze visible",
            "Particle size distribution in smoke column measurable",
        ]
        response = {
            "tier": "scout_drone",
            "criteria": [
                {"precondition": preconditions[0], "classification": "observable", "reason": "OK"},
                {"precondition": preconditions[1], "classification": "unobservable",
                 "reason": "Requires spectrometric analysis — not available in RGB"},
            ],
            "summary": "remove_some",
        }
        result, _ = asyncio.run(run_tier_grounding_check(
            candidate_rule=_base_rule(preconditions),
            tier="scout_drone",
            tier_observability=TIER_OBSERVABILITY["scout_drone"],
            config=WILDFIRE_DETECTION_CONFIG,
            call_agent_fn=_make_call_agent(response),
        ))
        assert len(result["unobservable"]) == 1
        assert len(result["observable"]) == 1

    def test_parse_failure_fallback(self):
        async def bad_call(agent_name, content, system_prompt="", model="", max_tokens=512):
            return "Not valid JSON at all", 50

        preconditions = ["Condition A", "Condition B"]
        result, _ = asyncio.run(run_tier_grounding_check(
            candidate_rule=_base_rule(preconditions),
            tier="scout_drone",
            tier_observability=TIER_OBSERVABILITY["scout_drone"],
            config=WILDFIRE_DETECTION_CONFIG,
            call_agent_fn=bad_call,
        ))
        assert len(result["observable"]) == 2
        assert len(result["temporal"]) == 0
        assert len(result["unobservable"]) == 0

    def test_sentinel_tier_receives_correct_context(self):
        received = []

        async def capture(agent_name, content, system_prompt="", model="", max_tokens=512):
            received.append(system_prompt)
            return json.dumps({
                "tier": "ground_sentinel",
                "criteria": [{"precondition": "P", "classification": "observable", "reason": "OK"}],
                "summary": "accept_all",
            }), 50

        asyncio.run(run_tier_grounding_check(
            candidate_rule={"rule": "R", "feature": "f", "favors": "early_smoke_signature",
                            "confidence": "high", "preconditions": ["P"]},
            tier="ground_sentinel",
            tier_observability=TIER_OBSERVABILITY["ground_sentinel"],
            config=WILDFIRE_DETECTION_CONFIG,
            call_agent_fn=capture,
        ))
        assert received
        assert "ground_sentinel" in received[0]
        assert "PTZ" in received[0]

    def test_scout_tier_receives_correct_context(self):
        received = []

        async def capture(agent_name, content, system_prompt="", model="", max_tokens=512):
            received.append(system_prompt)
            return json.dumps({
                "tier": "scout_drone",
                "criteria": [{"precondition": "P", "classification": "observable", "reason": "OK"}],
                "summary": "accept_all",
            }), 50

        asyncio.run(run_tier_grounding_check(
            candidate_rule={"rule": "R", "feature": "f", "favors": "early_smoke_signature",
                            "confidence": "high", "preconditions": ["P"]},
            tier="scout_drone",
            tier_observability=TIER_OBSERVABILITY["scout_drone"],
            config=WILDFIRE_DETECTION_CONFIG,
            call_agent_fn=capture,
        ))
        assert received
        assert "single frames only" in received[0]


# ---------------------------------------------------------------------------
# run_temporal_reformulation
# ---------------------------------------------------------------------------

class TestRunTemporalReformulation:

    def test_returns_wind_axis_proxy(self):
        response = {
            "temporal_criterion": "haze drifts southeast consistently across consecutive frames",
            "proxy": "haze column is elongated along a southeast axis from the point source — asymmetric extension in the prevailing wind direction",
            "rationale": "Consistent drift in one direction leaves a spatial asymmetry in a single frame.",
            "confidence": "high",
        }
        result, ms = asyncio.run(run_temporal_reformulation(
            temporal_criterion="haze drifts southeast consistently across consecutive frames",
            ground_truth_class="early_smoke_signature",
            wrong_class="heat_shimmer_artifact",
            config=WILDFIRE_DETECTION_CONFIG,
            call_agent_fn=_make_call_agent(response),
        ))
        assert result["proxy"] is not None
        assert "elongated" in result["proxy"] or "axis" in result["proxy"]
        assert result["confidence"] == "high"

    def test_parse_failure_returns_none_proxy(self):
        async def bad_call(agent_name, content, system_prompt="", model="", max_tokens=512):
            return "not json", 50

        result, _ = asyncio.run(run_temporal_reformulation(
            temporal_criterion="drift consistent across frames",
            ground_truth_class="early_smoke_signature",
            wrong_class="heat_shimmer_artifact",
            config=WILDFIRE_DETECTION_CONFIG,
            call_agent_fn=bad_call,
        ))
        assert result["proxy"] is None
        assert result["confidence"] == "low"


# ---------------------------------------------------------------------------
# adapt_rule_for_tier
# ---------------------------------------------------------------------------

class TestAdaptRuleForTier:

    def _rule_with_temporal(self) -> dict:
        return {
            "rule": "When conditions are met, classify as early_smoke_signature.",
            "feature": "blue_haze_point_source",
            "favors": "early_smoke_signature",
            "confidence": "high",
            "preconditions": [
                "Blue-gray haze visible in the frame",
                "Haze drifts consistently southeast across frames",  # temporal
                "MWIR temperature absolute reading available",        # unobservable
            ],
        }

    def test_removes_unobservable_reformulates_temporal(self):
        grounding_response = {
            "tier": "scout_drone",
            "criteria": [
                {"precondition": "Blue-gray haze visible in the frame",
                 "classification": "observable", "reason": "Color visible"},
                {"precondition": "Haze drifts consistently southeast across frames",
                 "classification": "temporal", "reason": "Single-pass drone"},
                {"precondition": "MWIR temperature absolute reading available",
                 "classification": "unobservable", "reason": "MWIR not on scout"},
            ],
            "summary": "remove_some",
        }
        reformulation_response = {
            "temporal_criterion": "Haze drifts consistently southeast across frames",
            "proxy": "Haze column elongated along southeast axis from the point source",
            "rationale": "Consistent drift produces spatial asymmetry in single frame.",
            "confidence": "high",
        }

        adapted_rule, report = asyncio.run(adapt_rule_for_tier(
            candidate_rule=self._rule_with_temporal(),
            tier="scout_drone",
            tier_observability=TIER_OBSERVABILITY["scout_drone"],
            ground_truth_class="early_smoke_signature",
            wrong_class="heat_shimmer_artifact",
            config=WILDFIRE_DETECTION_CONFIG,
            call_agent_fn=_make_sequential_call_agent([grounding_response, reformulation_response]),
        ))

        assert "MWIR temperature absolute reading available" not in adapted_rule["preconditions"]
        assert "Haze drifts consistently southeast across frames" not in adapted_rule["preconditions"]
        assert "Blue-gray haze visible in the frame" in adapted_rule["preconditions"]
        assert "Haze column elongated along southeast axis from the point source" in adapted_rule["preconditions"]

        assert len(report["unobservable_removed"]) == 1
        assert len(report["temporal_reformulated"]) == 1
        assert report["temporal_reformulated"][0]["proxy"] is not None

    def test_all_observable_no_extra_calls(self):
        call_count = [0]

        async def counting_call(agent_name, content, system_prompt="", model="", max_tokens=512):
            call_count[0] += 1
            return json.dumps({
                "tier": "ground_sentinel",
                "criteria": [
                    {"precondition": "Blue-gray haze visible",
                     "classification": "observable", "reason": "OK"},
                ],
                "summary": "accept_all",
            }), 50

        rule = {
            "rule": "R", "feature": "f", "favors": "early_smoke_signature",
            "confidence": "high", "preconditions": ["Blue-gray haze visible"],
        }
        adapted_rule, report = asyncio.run(adapt_rule_for_tier(
            candidate_rule=rule,
            tier="ground_sentinel",
            tier_observability=TIER_OBSERVABILITY["ground_sentinel"],
            ground_truth_class="early_smoke_signature",
            wrong_class="heat_shimmer_artifact",
            config=WILDFIRE_DETECTION_CONFIG,
            call_agent_fn=counting_call,
        ))

        assert call_count[0] == 1
        assert adapted_rule["preconditions"] == ["Blue-gray haze visible"]
        assert report["unobservable_removed"] == []
        assert report["temporal_reformulated"] == []

    def test_adapted_rule_metadata(self):
        async def call_agent(agent_name, content, system_prompt="", model="", max_tokens=512):
            return json.dumps({
                "tier": "ground_sentinel",
                "criteria": [
                    {"precondition": "Blue-gray haze visible", "classification": "observable", "reason": "OK"},
                ],
                "summary": "accept_all",
            }), 50

        rule = {"rule": "R", "feature": "f", "favors": "early_smoke_signature",
                "confidence": "high", "preconditions": ["Blue-gray haze visible"]}
        adapted_rule, _ = asyncio.run(adapt_rule_for_tier(
            candidate_rule=rule,
            tier="ground_sentinel",
            tier_observability=TIER_OBSERVABILITY["ground_sentinel"],
            ground_truth_class="early_smoke_signature",
            wrong_class="heat_shimmer_artifact",
            config=WILDFIRE_DETECTION_CONFIG,
            call_agent_fn=call_agent,
        ))

        assert adapted_rule["tier"] == "ground_sentinel"
        assert adapted_rule["tier_adapted"] is True
        assert adapted_rule["favors"] == "early_smoke_signature"


# ---------------------------------------------------------------------------
# run_env_context_injection
# ---------------------------------------------------------------------------

class TestRunEnvContextInjection:

    def _accepted_rule(self) -> dict:
        return {
            "rule": "When blue haze with point source, classify as early_smoke_signature.",
            "feature": "blue_haze_point_source",
            "favors": "early_smoke_signature",
            "confidence": "high",
            "preconditions": ["Blue-gray haze", "Point source"],
            "rationale": "Terpene combustion",
        }

    def test_context_blocks_added(self):
        response = {
            "red_flag": {"raws_wind_speed_mph": "> 20", "urgency_multiplier": 1.0},
            "elevated_risk": {"raws_wind_speed_mph": "> 10", "urgency_multiplier": 0.7},
            "normal": {"urgency_multiplier": 0.4},
        }
        augmented_rule, ms = asyncio.run(run_env_context_injection(
            accepted_rule=self._accepted_rule(),
            raws_conditions=None,
            config=WILDFIRE_DETECTION_CONFIG,
            call_agent_fn=_make_call_agent(response),
        ))
        assert "context_preconditions" in augmented_rule
        assert "red_flag" in augmented_rule["context_preconditions"]
        assert "elevated_risk" in augmented_rule["context_preconditions"]
        assert "normal" in augmented_rule["context_preconditions"]

    def test_active_condition_set_evaluated_from_raws(self):
        response = {
            "red_flag": {"urgency_multiplier": 1.0},
            "elevated_risk": {"urgency_multiplier": 0.7},
            "normal": {"urgency_multiplier": 0.4},
        }
        raws = {"wind_speed_mph": 28, "relative_humidity_pct": 8, "temperature_f": 95}
        augmented_rule, _ = asyncio.run(run_env_context_injection(
            accepted_rule=self._accepted_rule(),
            raws_conditions=raws,
            config=WILDFIRE_DETECTION_CONFIG,
            call_agent_fn=_make_call_agent(response),
        ))
        assert augmented_rule.get("active_condition_set") == "red_flag"
        assert augmented_rule.get("raws_snapshot") == raws

    def test_fallback_on_parse_error(self):
        async def bad_call(agent_name, content, system_prompt="", model="", max_tokens=512):
            return "not json", 50

        augmented_rule, _ = asyncio.run(run_env_context_injection(
            accepted_rule=self._accepted_rule(),
            raws_conditions=None,
            config=WILDFIRE_DETECTION_CONFIG,
            call_agent_fn=bad_call,
        ))
        # Should fall back to static defaults
        assert "context_preconditions" in augmented_rule
        ctx = augmented_rule["context_preconditions"]
        assert "red_flag" in ctx
        assert "normal" in ctx

    def test_no_active_condition_set_without_raws(self):
        response = {
            "red_flag": {"urgency_multiplier": 1.0},
            "elevated_risk": {"urgency_multiplier": 0.7},
            "normal": {"urgency_multiplier": 0.4},
        }
        augmented_rule, _ = asyncio.run(run_env_context_injection(
            accepted_rule=self._accepted_rule(),
            raws_conditions=None,
            config=WILDFIRE_DETECTION_CONFIG,
            call_agent_fn=_make_call_agent(response),
        ))
        assert "active_condition_set" not in augmented_rule
