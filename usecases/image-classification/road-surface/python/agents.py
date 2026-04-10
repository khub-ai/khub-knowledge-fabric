"""
agents.py — Road surface condition agent runners.

Thin delegation layer: re-exports backend infrastructure from dermatology
and wires the core dialogic distillation agents to ROAD_SURFACE_CONFIG.

Multi-backend:
  Set ACTIVE_MODEL to switch between Anthropic and OpenAI backends.
  Inherited from dermatology/agents.py infrastructure.
"""

from __future__ import annotations
import importlib.util as _ilu
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Import backend infrastructure from dermatology (avoids code duplication)
# ---------------------------------------------------------------------------
_DERM_AGENTS_PATH = (
    Path(__file__).resolve().parents[2] / "dermatology" / "python" / "agents.py"
)
_derm_spec = _ilu.spec_from_file_location("derm_agents", _DERM_AGENTS_PATH)
_derm_agents = _ilu.module_from_spec(_derm_spec)
_derm_spec.loader.exec_module(_derm_agents)

# Re-export backend infrastructure
call_agent         = _derm_agents.call_agent
reset_cost_tracker = _derm_agents.reset_cost_tracker
get_cost_tracker   = _derm_agents.get_cost_tracker
ACTIVE_MODEL       = _derm_agents.ACTIVE_MODEL
DEFAULT_MODEL      = _derm_agents.DEFAULT_MODEL

# ---------------------------------------------------------------------------
# KF root on sys.path (needed if this file is run directly)
# ---------------------------------------------------------------------------
_KF_ROOT = Path(__file__).resolve().parents[4]
if str(_KF_ROOT) not in sys.path:
    sys.path.insert(0, str(_KF_ROOT))

# ---------------------------------------------------------------------------
# Domain config + core agents
# ---------------------------------------------------------------------------
from domain_config import ROAD_SURFACE_CONFIG  # noqa: E402
from core.dialogic_distillation import agents as _dd_agents  # noqa: E402


# ---------------------------------------------------------------------------
# Wrappers — delegate to core with ROAD_SURFACE_CONFIG
# ---------------------------------------------------------------------------

async def run_expert_rule_author(
    task: dict,
    wrong_prediction: str,
    correct_label: str,
    model_reasoning: str = "",
    model: str = "",
    prior_context: str = "",
) -> tuple[dict, int]:
    """Author a corrective rule from a road surface failure case."""
    return await _dd_agents.run_expert_rule_author(
        task,
        wrong_prediction,
        correct_label,
        config=ROAD_SURFACE_CONFIG,
        model_reasoning=model_reasoning,
        model=model or ACTIVE_MODEL,
        prior_context=prior_context,
        call_agent_fn=call_agent,
    )


async def run_rule_validator_on_image(
    image_path: str,
    ground_truth: str,
    candidate_rule: dict,
    model: str = "",
) -> tuple[dict, int]:
    """Test whether a candidate rule applies to a single labeled road image."""
    return await _dd_agents.run_rule_validator_on_image(
        image_path=image_path,
        ground_truth=ground_truth,
        candidate_rule=candidate_rule,
        config=ROAD_SURFACE_CONFIG,
        model=model or ACTIVE_MODEL,
        call_agent_fn=call_agent,
    )


async def validate_candidate_rule(
    candidate_rule: dict,
    validation_images: list,
    trigger_image_path: str,
    trigger_correct_label: str,
    model: str = "",
) -> dict:
    """Validate a candidate rule against the pool (precision gate)."""
    return await _dd_agents.validate_candidate_rule(
        candidate_rule=candidate_rule,
        validation_images=validation_images,
        trigger_image_path=trigger_image_path,
        trigger_correct_label=trigger_correct_label,
        config=ROAD_SURFACE_CONFIG,
        model=model or ACTIVE_MODEL,
        call_agent_fn=call_agent,
    )


async def run_contrastive_feature_analysis(
    tp_cases: list,
    fp_cases: list,
    candidate_rule: dict,
    pair_info: dict,
    model: str = "",
) -> tuple[dict, int]:
    """Identify discriminating feature between TP and FP road images."""
    return await _dd_agents.run_contrastive_feature_analysis(
        tp_cases=tp_cases,
        fp_cases=fp_cases,
        candidate_rule=candidate_rule,
        pair_info=pair_info,
        config=ROAD_SURFACE_CONFIG,
        model=model or ACTIVE_MODEL,
        call_agent_fn=call_agent,
    )


async def run_rule_spectrum_generator(
    candidate_rule: dict,
    tp_cases: list,
    fp_cases: list,
    contrastive_result: dict,
    pair_info: dict,
    model: str = "",
) -> tuple[list, int]:
    """Generate 4-level specificity spectrum of a candidate rule."""
    return await _dd_agents.run_rule_spectrum_generator(
        candidate_rule=candidate_rule,
        tp_cases=tp_cases,
        fp_cases=fp_cases,
        contrastive_result=contrastive_result,
        pair_info=pair_info,
        config=ROAD_SURFACE_CONFIG,
        model=model or ACTIVE_MODEL,
        call_agent_fn=call_agent,
    )


async def run_semantic_rule_validator(
    candidate_rule: dict,
    pair_info: dict,
    model: str = "",
) -> tuple[dict, int]:
    """Text-only semantic check of a road surface rule's domain logic."""
    return await _dd_agents.run_semantic_rule_validator(
        candidate_rule=candidate_rule,
        pair_info=pair_info,
        config=ROAD_SURFACE_CONFIG,
        model=model or ACTIVE_MODEL,
        call_agent_fn=call_agent,
    )
