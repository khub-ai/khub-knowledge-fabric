"""Explorer — fallback action selection when the planner has no plan.

When :func:`planner.compute_plan` returns ``None`` — because the active
goal has no known path forward given current hypotheses — the runner
hands control to the explorer.  The explorer's job is to produce a
*useful* next action: one that either reduces epistemic uncertainty
(info-gain) or probes an entity the engine knows nothing about
(curiosity).

Design
------
Two drivers, blended by the tunable
:attr:`ExplorerConfig.curiosity_level`:

1. **Information-gain** — prefer actions that would differentiate
   between currently *competing* hypotheses (those sharing a
   canonical key but diverging on full keys).  Each such group
   represents an unresolved question in the agent's model; an action
   that would cause different competitors to predict different
   outcomes has positive info-gain.

2. **Curiosity** — prefer actions that involve entities or action
   types the engine knows little about.  Each entity has a
   :func:`claim_coverage` in ``[0, 1]`` counting how many of the
   standard claim slots (property / relational / causal / transition /
   structure-mapping) are populated for it.  Low coverage → high
   curiosity draw.

The explorer also *generates goals*: it proposes
:class:`Goal`\\s of the form "probe entity X" or "try action A" that
the runner can feed into the :class:`GoalForest` at LOW priority.
This lets curiosity integrate with the normal goal / planner loop
rather than existing as a parallel system.

Capability audit (standing invariant 7)
----------------------------------------
* **Debugging** — PRIMARY.  Info-gain exploration is the deliberate
  design of an experiment that would disambiguate competing
  hypotheses.  This is the "form hypothesis → test → iterate" loop
  made explicit at the action-selection level.
* **Problem-solving** — secondary.  Curiosity-generated goals
  augment the goal forest with unknowns the primary goal doesn't
  cover, which often surfaces the preconditions the main goal
  needs.
* **Tool creation** — minor.  ``detect_generalization_candidates``
  from the hypothesis store drives info-gain priorities when many
  related hypotheses share a pattern; resolving the dispute is a
  prerequisite for eventually compressing them into a tool.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set, Tuple

from .claims import (
    CausalClaim,
    Claim,
    PropertyClaim,
    RelationalClaim,
    StructureMappingClaim,
    TransitionClaim,
)
from .conditions import (
    Condition,
    ActionTried,
    EntityProbed,
)
from . import hypothesis_store as _store
from .types import (
    Action,
    EntityModel,
    Goal,
    GoalNode,
    GoalStatus,
    NodeType,
    WorldState,
)


# ---------------------------------------------------------------------------
# Claim-coverage metric
# ---------------------------------------------------------------------------


# Standard slot types an entity "could" have hypotheses about.  Coverage
# is the fraction of these slots with at least one hypothesis (of any
# credence) mentioning the entity.  Used for curiosity ranking.
_STANDARD_SLOTS = ("property", "relational", "causal", "transition", "structure")


def claim_coverage(entity_id: str, ws: WorldState) -> float:
    """Return fraction of :data:`_STANDARD_SLOTS` covered by at least
    one hypothesis referencing the entity.

    The metric deliberately does not require *committed* hypotheses —
    even a provisional claim means the entity has some structure the
    engine has noticed.  Only fully unknown entities score 0.0.
    """
    filled: Set[str] = set()
    for h in ws.hypotheses.values():
        claim = h.claim
        if entity_id not in claim.referenced_entities():
            continue
        if isinstance(claim, PropertyClaim):
            filled.add("property")
        elif isinstance(claim, RelationalClaim):
            filled.add("relational")
        elif isinstance(claim, CausalClaim):
            filled.add("causal")
        elif isinstance(claim, TransitionClaim):
            filled.add("transition")
        elif isinstance(claim, StructureMappingClaim):
            filled.add("structure")
    return len(filled) / len(_STANDARD_SLOTS)


# ---------------------------------------------------------------------------
# Info-gain estimate for a candidate action
# ---------------------------------------------------------------------------


def info_gain(action: Action,
              ws:     WorldState) -> float:
    """Estimate how much uncertainty the action would resolve.

    Simple model: count the number of distinct canonical_keys among
    contested hypothesis groups where the action appears in a
    :class:`TransitionClaim` or in an entity referenced by the group.
    Each such group contributes ``1.0 / n_competitors`` — more
    competitors means more information per disambiguating outcome.

    This is a coarse but tractable proxy; Phase 4 may refine it with a
    KL-divergence estimate once transition predictions from committed
    hypotheses accumulate enough.
    """
    gain = 0.0
    groups = _store.contested_groups(ws)
    for group in groups:
        canonical = group[0].claim.canonical_key()
        # Touched if the action appears in any of the group's claims
        touches = False
        for h in group:
            claim = h.claim
            if isinstance(claim, TransitionClaim) and claim.action in (action.name, "*"):
                touches = True
                break
        if touches:
            gain += 1.0 / max(1, len(group))
    return gain


# ---------------------------------------------------------------------------
# Curiosity: generate exploration goals
# ---------------------------------------------------------------------------


def propose_curiosity_goals(ws: WorldState,
                            *,
                            step: int = 0,
                            action_space: Optional[List[Action]] = None) -> List[Goal]:
    """Produce low-priority :class:`Goal`\\s targeting unknowns.

    Two kinds:

    * **Entity-probing** — for every known entity whose
      :func:`claim_coverage` is below ``curiosity_threshold``,
      generate a Goal whose ATOM condition is
      :class:`EntityProbed`\\ ``(entity_id)``.
    * **Action-trial** — for every action in ``action_space`` that
      does not appear in any committed TransitionClaim, generate a
      Goal whose ATOM condition is
      :class:`ActionTried`\\ ``(action_id)``.

    Both kinds are returned regardless of the explorer's master switch;
    the caller (runner) decides whether to add them based on
    :attr:`ExplorerConfig.generate_curiosity_goals`.

    Parameters
    ----------
    step
        Used as ``created_at`` on newly minted Goals.
    action_space
        Needed to generate action-trial goals.  If omitted, only
        entity-probing goals are produced.
    """
    cfg = _explorer_cfg(ws)
    if not cfg.generate_curiosity_goals:
        return []

    goals: List[Goal] = []
    threshold = cfg.curiosity_threshold
    base_prio = cfg.novelty_base

    # Entity probing
    for entity_id, entity in ws.entities.items():
        cov = claim_coverage(entity_id, ws)
        if cov >= threshold:
            continue
        priority = base_prio * (1.0 - cov)
        goal_id = f"explore:entity:{entity_id}"
        if goal_id in ws.goal_forest.goals:
            continue  # already proposed
        goals.append(Goal(
            id         = goal_id,
            root       = GoalNode(
                id         = f"{goal_id}::atom",
                node_type  = NodeType.ATOM,
                condition  = EntityProbed(entity_id, coverage=threshold),
                status     = GoalStatus.OPEN,
                source     = "explorer:curiosity",
                created_at = step,
            ),
            priority   = priority,
            source     = "explorer:curiosity",
            created_at = step,
        ))

    # Action trials
    if action_space:
        tried_actions = _actions_in_transitions(ws)
        for action in action_space:
            if action.name in tried_actions:
                continue
            goal_id = f"explore:action:{action.id}"
            if goal_id in ws.goal_forest.goals:
                continue
            goals.append(Goal(
                id         = goal_id,
                root       = GoalNode(
                    id         = f"{goal_id}::atom",
                    node_type  = NodeType.ATOM,
                    condition  = ActionTried(action.id),
                    status     = GoalStatus.OPEN,
                    source     = "explorer:curiosity",
                    created_at = step,
                ),
                priority   = base_prio * 0.5,
                source     = "explorer:curiosity",
                created_at = step,
            ))

    return goals


def _actions_in_transitions(ws: WorldState) -> Set[str]:
    """Action names that appear in any TransitionClaim hypothesis,
    regardless of credence."""
    names: Set[str] = set()
    for h in ws.hypotheses.values():
        if isinstance(h.claim, TransitionClaim):
            names.add(h.claim.action)
    return names


# ---------------------------------------------------------------------------
# Master: choose an exploration action given no plan
# ---------------------------------------------------------------------------


def choose_exploration_action(ws:           WorldState,
                              action_space: List[Action]) -> Optional[Action]:
    """Pick an action when the planner cannot produce a plan.

    Scoring per action:

        score(a) = info_gain_weight * info_gain(a, ws)
                 + novelty_base   * curiosity_bonus(a, ws)

    where ``curiosity_bonus`` equals
    ``1 - claim_coverage(most_relevant_entity)`` — relevant entity is
    the one the action most plausibly affects (currently a proxy:
    the agent itself).

    When multiple actions tie, prefer actions whose name has never
    been seen in a TransitionClaim, then lexicographic name order for
    determinism.  Returns ``None`` if ``action_space`` is empty.
    """
    if not action_space:
        return None

    cfg = _explorer_cfg(ws)
    tried = _actions_in_transitions(ws)
    agent_coverage = claim_coverage("agent", ws)

    scored: List[Tuple[float, int, str, Action]] = []
    for action in action_space:
        ig = info_gain(action, ws)
        curiosity = (1.0 - agent_coverage) * 0.5
        score = cfg.info_gain_weight * ig + cfg.novelty_base * curiosity
        untried_bonus = 0 if action.name in tried else 1  # lower primary key wins
        scored.append((-score, -untried_bonus, action.name, action))

    scored.sort()
    return scored[0][3]


# ---------------------------------------------------------------------------
# Config access
# ---------------------------------------------------------------------------


def _explorer_cfg(ws: WorldState):
    if ws.config is not None and hasattr(ws.config, "explorer"):
        return ws.config.explorer
    from .config import ExplorerConfig
    return ExplorerConfig()
