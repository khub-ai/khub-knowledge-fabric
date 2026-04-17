"""Goal forest — operations on the GoalForest / GoalNode / Goal types.

The :class:`GoalForest` holds all active top-level goals.  Each goal is
rooted at an AND-OR-CHANCE tree (:class:`GoalNode`).  This module
provides the operations a runner needs to manage the forest during an
episode:

* Adding a new goal (adapter-seeded primary or engine-derived subgoal)
* Selecting which goal to pursue next — priority, deadlines, conflicts
* Expanding a goal with engine-derived subgoals using committed
  :class:`CausalClaim`\\s from the hypothesis store
* Walking the tree and checking achievement / failure
* Detecting conflicts across concurrent goals

Phase 3 scope
-------------
* Full operations for ATOM / AND / OR / CHANCE nodes.
* Conflict detection for MUTEX (logically incompatible conditions) and
  TEMPORAL (deadline overlap).  RESOURCE and ADVERSARIAL conflicts
  are detected structurally (pattern in place) but their resolution
  policies rely on a full rule/principal system best exercised in
  robotics adapters (Phase 5+).
* Robotics-extension node kinds (OPTION / MAINTAIN / LOOP /
  ADVERSARIAL / INFO_SET) are recognised but pass through unchanged —
  the planner/explorer will acquire handling for them when the
  corresponding adapter is built.

Capability audit (standing invariant 7)
----------------------------------------
* **Problem-solving** — PRIMARY.  Goal decomposition is how the
  engine turns a single high-level objective into a plan-generating
  structure.
* **Debugging** — secondary.  Subgoal derivation is read-only against
  committed hypotheses, so hypothesis demotions automatically
  invalidate their corresponding subgoals at the next selection pass.
* **Tool creation** — deferred.  An Option injected into the forest
  becomes a GoalNode with ``node_type=OPTION``; Phase 7 will add
  Option-producing miners and the planner handler.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Tuple

from .claims import CausalClaim
from .conditions import AtPosition, Condition, EntityInState
from .types import (
    ConflictType,
    Goal,
    GoalConflict,
    GoalForest,
    GoalNode,
    GoalStatus,
    NodeType,
    Ordering,
    ResolutionPolicy,
    WorldState,
)
from . import hypothesis_store as _store


# ---------------------------------------------------------------------------
# Registration and selection
# ---------------------------------------------------------------------------


def add_goal(ws: WorldState, goal: Goal) -> None:
    """Register a top-level goal in the forest.

    If no goal is currently active, the newly-added goal becomes
    active.  Otherwise the forest's active pointer is unchanged; a
    subsequent :func:`select_active_goal` call will arbitrate.
    """
    ws.goal_forest.goals[goal.id] = goal
    if ws.goal_forest.active_goal_id is None:
        ws.goal_forest.active_goal_id = goal.id


def select_active_goal(ws: WorldState) -> Optional[str]:
    """Pick the goal to pursue next.

    Selection policy (Phase 3):

    1. Filter out goals whose status is ACHIEVED, PRUNED, or ABANDONED.
    2. Filter out goals blocked by unresolved conflicts whose
       resolution policy is ``FAIL``.
    3. Among survivors, prefer higher priority; tiebreak on earlier
       deadline; final tiebreak on insertion order (dict iteration).

    Returns the selected goal ID, or ``None`` if no candidate exists.
    The forest's ``active_goal_id`` is updated in place.
    """
    candidates = [g for g in ws.goal_forest.goals.values()
                  if g.root.status not in
                  (GoalStatus.ACHIEVED, GoalStatus.PRUNED, GoalStatus.ABANDONED)]
    if not candidates:
        ws.goal_forest.active_goal_id = None
        return None

    blocked_ids = {c.goal_a for c in ws.goal_forest.conflicts
                   if c.resolution_policy == ResolutionPolicy.FAIL}
    blocked_ids |= {c.goal_b for c in ws.goal_forest.conflicts
                    if c.resolution_policy == ResolutionPolicy.FAIL}
    candidates = [g for g in candidates if g.id not in blocked_ids] or candidates

    def sort_key(g: Goal) -> Tuple:
        # higher priority first; earlier deadline first; stable tiebreak
        deadline = g.deadline if g.deadline is not None else float("inf")
        return (-g.priority, deadline, g.created_at)

    candidates.sort(key=sort_key)
    chosen = candidates[0]
    ws.goal_forest.active_goal_id = chosen.id
    return chosen.id


# ---------------------------------------------------------------------------
# Subgoal derivation from committed CausalClaims
# ---------------------------------------------------------------------------


def derive_subgoals_from_causal(ws:      WorldState,
                                goal_id: str,
                                *,
                                max_depth: int = 3,
                                step:       int = 0) -> List[str]:
    """Expand ATOM leaves of a goal by consulting committed
    :class:`CausalClaim`\\s.

    For each ATOM leaf with condition ``C``:

    1. Scan committed hypotheses for a ``CausalClaim`` whose
       ``effect.canonical_key() == C.canonical_key()``.  If present,
       the trigger condition becomes a new ATOM sibling under an AND
       node that replaces the original leaf: achieve trigger → effect
       becomes true.

    2. If multiple causal claims point at the same effect, an OR node
       is inserted with one AND branch per alternative trigger.

    Only claims with credence above the engine's commit threshold are
    used.  Contradicted or undecided claims are ignored, so the derived
    structure naturally reflects current confidence.

    Parameters
    ----------
    max_depth
        Cap on recursion depth to prevent unbounded expansion if the
        hypothesis graph contains near-cycles.
    step
        Current step, used as ``created_at`` on newly created nodes.

    Returns
    -------
    list[str]
        IDs of nodes that were newly expanded.  Empty if nothing
        changed — either no applicable CausalClaims or every leaf
        already has derived children.
    """
    goal = ws.goal_forest.goals.get(goal_id)
    if goal is None:
        return []

    expanded: List[str] = []
    _expand_node(goal.root, ws, max_depth, step, expanded)
    return expanded


def _expand_node(node:     GoalNode,
                 ws:       WorldState,
                 depth:    int,
                 step:     int,
                 expanded: List[str]) -> None:
    """Recursively expand ATOM leaves using committed CausalClaims."""
    if depth <= 0:
        return

    # Recurse into composites first so we don't modify a structure while
    # iterating it at a higher level.
    if node.node_type in (NodeType.AND, NodeType.OR):
        for child in list(node.children):
            _expand_node(child, ws, depth - 1, step, expanded)
        return

    if node.node_type != NodeType.ATOM or node.condition is None:
        return

    causal_claims = _committed_causals_for_condition(ws, node.condition)
    if not causal_claims:
        return

    # Build subgoal structure:
    #   If one causal claim:  replace ATOM leaf with AND(trigger_atom, original_atom_condition_check)
    #     — the simplest decomposition
    #   If multiple:          OR over AND branches
    new_children: List[GoalNode] = []
    for cc_hypothesis in causal_claims:
        causal: CausalClaim = cc_hypothesis.claim
        trig_atom = GoalNode(
            id         = f"{node.id}::trig::{cc_hypothesis.id}",
            node_type  = NodeType.ATOM,
            condition  = causal.trigger,
            status     = GoalStatus.OPEN,
            supporting_hypothesis_ids = [cc_hypothesis.id],
            source     = "engine:derived-from-causal",
            created_at = step,
        )
        new_children.append(trig_atom)

    if len(new_children) == 1:
        # Simple decomposition: achieve the trigger, then the effect holds.
        # Represent by converting this ATOM into an AND whose children are
        # the trigger atom followed by a verifier atom (the original effect).
        verifier = GoalNode(
            id         = f"{node.id}::verify",
            node_type  = NodeType.ATOM,
            condition  = node.condition,
            status     = GoalStatus.OPEN,
            source     = "engine:derived-from-causal",
            created_at = step,
        )
        node.node_type = NodeType.AND
        node.children  = [new_children[0], verifier]
        node.condition = None  # AND nodes have no condition
        node.ordering  = Ordering.SEQUENTIAL
        node.supporting_hypothesis_ids = list(new_children[0].supporting_hypothesis_ids)
        expanded.append(node.id)
    else:
        # Alternatives: OR over AND branches.  Each AND has the trigger atom
        # followed by a verifier for the original effect condition.
        verifier_template = node.condition
        or_children: List[GoalNode] = []
        for trig in new_children:
            verifier = GoalNode(
                id         = f"{trig.id}::verify",
                node_type  = NodeType.ATOM,
                condition  = verifier_template,
                status     = GoalStatus.OPEN,
                source     = "engine:derived-from-causal",
                created_at = step,
            )
            and_node = GoalNode(
                id         = f"{trig.id}::and",
                node_type  = NodeType.AND,
                children   = [trig, verifier],
                ordering   = Ordering.SEQUENTIAL,
                supporting_hypothesis_ids = list(trig.supporting_hypothesis_ids),
                source     = "engine:derived-from-causal",
                created_at = step,
            )
            or_children.append(and_node)
        node.node_type = NodeType.OR
        node.children  = or_children
        node.condition = None
        node.supporting_hypothesis_ids = [h for c in new_children
                                          for h in c.supporting_hypothesis_ids]
        expanded.append(node.id)


def _committed_causals_for_condition(ws:        WorldState,
                                     condition: Condition) -> List:
    """Return committed CausalClaim hypotheses whose effect matches
    the given condition (by canonical key)."""
    target_key = condition.canonical_key()
    out = []
    for h in _store.committed(ws):
        if isinstance(h.claim, CausalClaim):
            if h.claim.effect.canonical_key() == target_key:
                out.append(h)
    return out


# ---------------------------------------------------------------------------
# Achievement / walk helpers
# ---------------------------------------------------------------------------


def atomic_leaves(node: GoalNode) -> Iterator[GoalNode]:
    """Yield every ATOM descendant of the given node.

    Used by the planner to determine what concrete sub-conditions the
    current goal reduces to, and by the explorer to find
    exploration-worthy leaves (ones with open status and no plan).
    """
    if node.node_type == NodeType.ATOM:
        yield node
        return
    for child in node.children:
        yield from atomic_leaves(child)


def is_achieved(ws: WorldState, goal_id: str) -> bool:
    """Check whether a goal's tree is satisfied by the current
    :class:`WorldState`.

    Recursively:

    * ATOM: its condition evaluates to ``True``.
    * AND : all children are achieved.
    * OR  : any child is achieved; ``active_branch`` is recorded.
    * CHANCE: treated as achieved if the selected outcome branch is
      achieved (best effort without runtime randomness).
    * OPTION / MAINTAIN / LOOP / ADVERSARIAL / INFO_SET: not yet
      implemented — treated as not-achieved (planner/runner handles
      them separately).

    Unknown truth values (``condition.evaluate`` returns ``None``)
    are treated as not-achieved — we require positive confirmation.
    """
    goal = ws.goal_forest.goals.get(goal_id)
    if goal is None:
        return False
    return _node_achieved(goal.root, ws)


def _node_achieved(node: GoalNode, ws: WorldState) -> bool:
    if node.node_type == NodeType.ATOM:
        if node.condition is None:
            return False
        return node.condition.evaluate(ws) is True
    if node.node_type == NodeType.AND:
        return all(_node_achieved(c, ws) for c in node.children)
    if node.node_type == NodeType.OR:
        # Prefer active_branch if set
        if node.active_branch is not None:
            for c in node.children:
                if c.id == node.active_branch:
                    return _node_achieved(c, ws)
        return any(_node_achieved(c, ws) for c in node.children)
    if node.node_type == NodeType.CHANCE:
        # Optimistic: if any branch achieves the goal, treat as achieved
        return any(_node_achieved(c, ws) for c in node.children)
    # Reserved types not yet handled
    return False


def mark_status(ws: WorldState, goal_id: str, status: GoalStatus) -> None:
    """Set the status on the root and propagate achievement downward
    when the new status is ACHIEVED."""
    goal = ws.goal_forest.goals.get(goal_id)
    if goal is None:
        return
    goal.root.status = status
    if status == GoalStatus.ACHIEVED:
        _cascade_achieved(goal.root)


def _cascade_achieved(node: GoalNode) -> None:
    node.status = GoalStatus.ACHIEVED
    for child in node.children:
        _cascade_achieved(child)


def refresh_status(ws: WorldState, goal_id: str) -> GoalStatus:
    """Re-evaluate a goal's root status against the current
    WorldState.  Returns the updated status.

    Called by the runner each step to detect achievement or to
    surface newly-open subgoals as children complete.
    """
    goal = ws.goal_forest.goals.get(goal_id)
    if goal is None:
        return GoalStatus.ABANDONED
    if is_achieved(ws, goal_id):
        mark_status(ws, goal_id, GoalStatus.ACHIEVED)
    return goal.root.status


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------


def detect_conflicts(ws: WorldState, step: int) -> List[GoalConflict]:
    """Scan active goals for conflicts and update
    ``ws.goal_forest.conflicts`` in place.  Returns the conflict list.

    Phase 3 implements:

    * **MUTEX** — two goals whose ATOM-leaf conditions are logical
      negations (``C`` and ``Negation(C)``) appear as leaves of
      simultaneously-active goals.
    * **TEMPORAL** — two goals with deadlines tighter than the sum
      of their minimum achievement time would require (estimated
      heuristically by leaf count).
    * **RESOURCE** — structural pattern noted (both goals reference
      the same resource in ATOM conditions); resolution policy
      defaults to PRIORITY.
    * **ADVERSARIAL** — two goals from different principals where
      one principal's ``context`` explicitly excludes the other's.
      Detected structurally; resolution delegated to the principal-
      authority arbitration that arrives with the robotics rule
      system (Phase 5).
    """
    active = [g for g in ws.goal_forest.goals.values()
              if g.root.status in (GoalStatus.OPEN, GoalStatus.ACTIVE)]
    conflicts: List[GoalConflict] = []

    for i, ga in enumerate(active):
        for gb in active[i + 1:]:
            c = _pair_conflict(ga, gb, step)
            if c is not None:
                conflicts.append(c)

    ws.goal_forest.conflicts = conflicts
    return conflicts


def _pair_conflict(ga: Goal, gb: Goal, step: int) -> Optional[GoalConflict]:
    """Return a :class:`GoalConflict` describing why two goals
    conflict, or ``None`` if they don't."""
    leaves_a = list(atomic_leaves(ga.root))
    leaves_b = list(atomic_leaves(gb.root))

    # MUTEX — any pair of leaves that are logical negations
    for la in leaves_a:
        if la.condition is None:
            continue
        for lb in leaves_b:
            if lb.condition is None:
                continue
            if _is_negation_of(la.condition, lb.condition):
                return GoalConflict(
                    goal_a            = ga.id,
                    goal_b            = gb.id,
                    conflict_type     = ConflictType.MUTEX,
                    resolution_policy = ResolutionPolicy.PRIORITY,
                    detected_at       = step,
                    rationale         = (f"ATOM leaves {la.id} and {lb.id} "
                                         f"have logically incompatible conditions"),
                )

    # TEMPORAL — overlapping deadlines with cumulative leaf count too high
    if ga.deadline is not None and gb.deadline is not None:
        leaf_total = len(leaves_a) + len(leaves_b)
        earliest = min(ga.deadline, gb.deadline)
        if earliest - step < leaf_total:  # rough heuristic: 1 step per leaf
            return GoalConflict(
                goal_a            = ga.id,
                goal_b            = gb.id,
                conflict_type     = ConflictType.TEMPORAL,
                resolution_policy = ResolutionPolicy.PRIORITY,
                detected_at       = step,
                rationale         = (f"combined leaf count ({leaf_total}) exceeds "
                                     f"steps to earliest deadline ({earliest - step})"),
            )

    # RESOURCE — shared resource reference in atoms
    res_a = _resource_refs(leaves_a)
    res_b = _resource_refs(leaves_b)
    shared = res_a & res_b
    if shared:
        return GoalConflict(
            goal_a            = ga.id,
            goal_b            = gb.id,
            conflict_type     = ConflictType.RESOURCE,
            resolution_policy = ResolutionPolicy.PRIORITY,
            detected_at       = step,
            rationale         = f"shared resources referenced: {sorted(shared)}",
        )

    # ADVERSARIAL — distinct principals with excluding contexts
    if (ga.principal is not None and gb.principal is not None
            and ga.principal.id != gb.principal.id):
        if (ga.principal.context is not None
                and _is_negation_of(ga.principal.context, gb.principal.context
                                    if gb.principal.context is not None
                                    else ga.principal.context)):
            return GoalConflict(
                goal_a            = ga.id,
                goal_b            = gb.id,
                conflict_type     = ConflictType.ADVERSARIAL,
                resolution_policy = ResolutionPolicy.USER_ARBITRATE,
                detected_at       = step,
                rationale         = (f"principals {ga.principal.id} and {gb.principal.id} "
                                     f"have mutually excluding contexts"),
            )

    return None


def _is_negation_of(a: Condition, b: Condition) -> bool:
    """True if ``a == Negation(b)`` or ``b == Negation(a)`` at the
    canonical-key level."""
    from .conditions import Negation
    if isinstance(a, Negation) and a.condition.canonical_key() == b.canonical_key():
        return True
    if isinstance(b, Negation) and b.condition.canonical_key() == a.canonical_key():
        return True
    return False


def _resource_refs(leaves: List[GoalNode]) -> set:
    """Collect resource IDs referenced by ResourceAbove/ResourceBelow
    conditions within the given ATOM leaves."""
    from .conditions import ResourceAbove, ResourceBelow
    out = set()
    for leaf in leaves:
        cond = leaf.condition
        if isinstance(cond, (ResourceAbove, ResourceBelow)):
            out.add(cond.resource_id)
    return out
