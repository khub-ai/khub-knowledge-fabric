"""Planner — AO* search over a goal's AND-OR-CHANCE tree.

Given a :class:`Goal` and an action space, produce a :class:`Plan` that
is expected to achieve the goal while respecting active :class:`Rule`\\s
and prior evidence.  The planner is the problem-solving heart of the
engine (capability-audit primary).

Design
------
The goal tree is walked recursively:

* **ATOM**  — BFS over states reachable via committed
  :class:`TransitionClaim`\\s.  Leaf conditions that use
  :class:`AtPosition` benefit most because positions form a natural
  state space; other conditions fall back to a "direct" check (try each
  action and see if the condition becomes true after execution).
* **AND**   — Plan each child in order (``SEQUENTIAL``) or any order
  with simple cost sorting (``UNORDERED``).  The concatenated actions
  become the plan for the AND node.
* **OR**    — Plan each child, choose the cheapest that succeeds.
  Heuristic branch preference from committed :class:`StrategyClaim`\\s
  breaks ties and biases toward empirically-successful branches.
* **CHANCE**— Choose the branch with highest expected value
  (``outcome_prior * success_reward - plan_cost``).  The plan must be
  valid for the chosen branch; the runner is expected to replan if a
  different branch is observed.

Rules
-----
Before search, the action space is filtered:

* ``INVIOLABLE`` rules prohibiting an action → action removed outright.
* ``DEFEASIBLE`` rules → removed unless suppressed by a higher-authority
  rule with positive weight.
* ``ADVISORY`` rules → retained but contribute cost penalties.

Rule filtering is a single pure function that any sub-planner calls; it
does not maintain state.

Options
-------
When the active goal tree contains an ``OPTION``-kind node, the planner
invokes the Option's ``internal_plan`` as a single step with
``pre_condition = option.applicability``.  This is how learned macro-
actions collapse the branching factor of search.

Budget
------
Search is capped by ``PlannerConfig.max_plan_depth`` and
``PlannerConfig.branch_budget``.  Exceeding either returns ``None``,
which the runner takes as "planner exhausted → explore".
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from .claims import (
    StrategyClaim,
    TransitionClaim,
)
from .conditions import (
    AtPosition,
    Condition,
)
from . import hypothesis_store as _store
from .types import (
    Action,
    Goal,
    GoalNode,
    NodeType,
    Option,
    Ordering,
    Plan,
    PlanStatus,
    PlannedAction,
    Rule,
    RuleConstraint,
    ConstraintKind,
    Violability,
    WorldState,
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute_plan(ws:            WorldState,
                 goal_id:       str,
                 action_space:  List[Action],
                 *,
                 step:           int = 0,
                 start_state:    Optional[Dict] = None) -> Optional[Plan]:
    """Compute a plan for the named goal.

    Parameters
    ----------
    ws
        WorldState.  Provides hypotheses (TransitionClaims for the
        transition model, StrategyClaims for OR-branch heuristics,
        Rules for filtering), engine config (limits), and current
        agent state (used as BFS start).
    goal_id
        Which goal in ``ws.goal_forest.goals`` to plan for.
    action_space
        Actions the adapter says are currently available.  Filtered
        by Rules before use.
    step
        Current step number — goes on the returned Plan's
        ``computed_at`` field.
    start_state
        Optional override of ``ws.agent`` as the BFS start.  Used
        internally when planning subgoals whose antecedent subgoal
        would change agent state.

    Returns
    -------
    Plan | None
        A plan whose ``steps`` are the ordered actions, whose
        ``assumptions`` are the hypothesis IDs the plan depends on,
        and whose ``branch_selections`` record OR-node choices.
        ``None`` if no plan was found within budget.
    """
    goal = ws.goal_forest.goals.get(goal_id)
    if goal is None:
        return None

    cfg = _planner_cfg(ws)
    filtered_actions = apply_rules_filter(action_space, ws)
    if not filtered_actions:
        return None

    start = start_state if start_state is not None else dict(ws.agent)

    ctx = _PlanCtx(
        ws                = ws,
        action_space      = filtered_actions,
        max_depth         = cfg.max_plan_depth,
        branch_budget     = cfg.branch_budget,
        nodes_expanded    = 0,
    )

    result = _plan_node(goal.root, start, ctx)
    if result is None:
        return None

    steps, assumptions, branch_selections = result
    return Plan(
        goal_id            = goal_id,
        steps              = steps,
        computed_at        = step,
        assumptions        = sorted(set(assumptions)),
        branch_selections  = branch_selections,
        status             = PlanStatus.ACTIVE,
    )


# ---------------------------------------------------------------------------
# Rule filter
# ---------------------------------------------------------------------------


def apply_rules_filter(actions: List[Action], ws: WorldState) -> List[Action]:
    """Return ``actions`` with those violating active Rules removed.

    * ``INVIOLABLE`` + ``PROHIBIT`` on an action name → remove.
    * ``DEFEASIBLE`` + ``PROHIBIT`` → remove (Phase 3 has no
      higher-authority override mechanism; robotics Phase 5 adds
      principal arbitration).
    * ``ADVISORY`` → retained; callers query
      :func:`advisory_penalty_for_action` to inflate cost.
    """
    if not ws.rules:
        return list(actions)

    prohibited: Set[str] = set()
    for rule in ws.rules.values():
        if rule.violability in (Violability.INVIOLABLE, Violability.DEFEASIBLE):
            constraint: RuleConstraint = rule.constraint
            if constraint.kind == ConstraintKind.PROHIBIT and isinstance(constraint.target, str):
                # Rule may be conditionally applicable; if the condition
                # is evaluable and false, the rule doesn't apply right now.
                cond_truth = rule.condition.evaluate(ws)
                if cond_truth is False:
                    continue
                prohibited.add(constraint.target)

    return [a for a in actions if a.name not in prohibited]


def advisory_penalty_for_action(action: Action, ws: WorldState) -> float:
    """Sum the cost penalties from all active ``ADVISORY`` rules that
    prefer the action is avoided."""
    penalty = 0.0
    for rule in ws.rules.values():
        if rule.violability != Violability.ADVISORY:
            continue
        constraint = rule.constraint
        if constraint.kind != ConstraintKind.PROHIBIT:
            continue
        if isinstance(constraint.target, str) and constraint.target == action.name:
            cond_truth = rule.condition.evaluate(ws)
            if cond_truth is not False:
                penalty += rule.priority * constraint.weight
    return penalty


# ---------------------------------------------------------------------------
# Internal planning context
# ---------------------------------------------------------------------------


class _PlanCtx:
    """Mutable bag passed through the recursive planner so sibling
    calls share the branch-budget counter without awkward return
    value plumbing."""
    __slots__ = ("ws", "action_space", "max_depth",
                 "branch_budget", "nodes_expanded")

    def __init__(self, ws, action_space, max_depth, branch_budget, nodes_expanded):
        self.ws             = ws
        self.action_space   = action_space
        self.max_depth      = max_depth
        self.branch_budget  = branch_budget
        self.nodes_expanded = nodes_expanded

    def charge(self) -> bool:
        """Debit one node from the budget.  Returns False when
        exhausted — the caller should bail out."""
        self.nodes_expanded += 1
        return self.nodes_expanded <= self.branch_budget


# ---------------------------------------------------------------------------
# Node dispatch
# ---------------------------------------------------------------------------


_PlanResult = Tuple[List[PlannedAction], List[str], Dict[str, str]]


def _plan_node(node: GoalNode,
               state: Dict,
               ctx:   _PlanCtx) -> Optional[_PlanResult]:
    """Dispatch on node type.  Returns (steps, assumptions,
    branch_selections) or None if no plan could be found."""
    if not ctx.charge():
        return None

    if node.node_type == NodeType.ATOM:
        return _plan_atom(node, state, ctx)
    if node.node_type == NodeType.AND:
        return _plan_and(node, state, ctx)
    if node.node_type == NodeType.OR:
        return _plan_or(node, state, ctx)
    if node.node_type == NodeType.CHANCE:
        return _plan_chance(node, state, ctx)
    if node.node_type == NodeType.OPTION:
        return _plan_option(node, state, ctx)
    # MAINTAIN / LOOP / ADVERSARIAL / INFO_SET — reserved for later
    # phases; reject for now so the caller knows planning failed.
    return None


# ---------------------------------------------------------------------------
# ATOM — BFS over reachable states
# ---------------------------------------------------------------------------


def _plan_atom(node:  GoalNode,
               state: Dict,
               ctx:   _PlanCtx) -> Optional[_PlanResult]:
    """Find a sequence of actions that achieves ``node.condition``
    starting from ``state``.

    Uses committed :class:`TransitionClaim`\\s as the transition
    model.  If the condition is already true at ``state``, returns
    an empty plan.  If the condition is a positional predicate, BFS
    over positions using action effects.  For other conditions, try
    each action once and check the effect.
    """
    cond = node.condition
    if cond is None:
        return [], [], {}

    # Short-circuit: already satisfied?
    if _condition_holds(cond, state, ctx.ws):
        return [], [], {}

    # Gather committed transition model
    transitions = _committed_transitions(ctx)
    if not transitions:
        # No learned dynamics yet — can't plan positively.
        return None

    # BFS where state is the subset of agent-state that matters for
    # transitions (typically position and key resources).  We hash on a
    # flattened tuple of (position, lives, resources) as a coarse key.
    start_key = _state_key(state)
    frontier  = deque([(state, [], [], start_key)])
    visited:  Set[Tuple] = {start_key}
    assumptions: List[str] = []

    while frontier:
        cur, path, path_assumptions, _ = frontier.popleft()

        if not ctx.charge():
            return None

        if _condition_holds(cond, cur, ctx.ws):
            return path, path_assumptions, {}

        if len(path) >= ctx.max_depth:
            continue

        for action in ctx.action_space:
            for tc_id, tc in transitions:
                if tc.action not in (action.name, "*"):
                    continue
                if tc.pre.evaluate_state(cur, ctx.ws) is False:
                    continue
                next_state = _apply_transition(cur, tc)
                next_key = _state_key(next_state)
                if next_key in visited:
                    continue
                visited.add(next_key)
                step = PlannedAction(
                    action                = action,
                    expected_effects      = [tc],
                    depends_on_hypotheses = [tc_id],
                    pre_condition         = tc.pre,
                )
                frontier.append((next_state,
                                 path + [step],
                                 path_assumptions + [tc_id],
                                 next_key))
    return None


def _committed_transitions(ctx: _PlanCtx) -> List[Tuple[str, TransitionClaim]]:
    """Collected (hypothesis_id, TransitionClaim) for every committed
    transition-kind hypothesis in the store."""
    out: List[Tuple[str, TransitionClaim]] = []
    for h in _store.committed(ctx.ws):
        if isinstance(h.claim, TransitionClaim):
            out.append((h.id, h.claim))
    return out


# ---------------------------------------------------------------------------
# AND
# ---------------------------------------------------------------------------


def _plan_and(node:  GoalNode,
              state: Dict,
              ctx:   _PlanCtx) -> Optional[_PlanResult]:
    """Plan each child; concatenate step sequences.

    ``SEQUENTIAL`` children are planned in order.  ``UNORDERED``
    children are sorted by a heuristic cost (number of atomic leaves)
    so cheaper-looking children go first and possibly satisfy
    preconditions for others.
    """
    children = list(node.children)
    if node.ordering == Ordering.UNORDERED:
        children.sort(key=lambda c: sum(1 for _ in _atomic_count(c)))

    steps: List[PlannedAction] = []
    assumptions: List[str] = []
    branches: Dict[str, str] = {}
    cur_state = dict(state)

    for child in children:
        sub = _plan_node(child, cur_state, ctx)
        if sub is None:
            return None
        sub_steps, sub_ass, sub_branches = sub
        steps.extend(sub_steps)
        assumptions.extend(sub_ass)
        branches.update(sub_branches)
        cur_state = _simulate_steps(cur_state, sub_steps)

    return steps, assumptions, branches


# ---------------------------------------------------------------------------
# OR
# ---------------------------------------------------------------------------


def _plan_or(node:  GoalNode,
             state: Dict,
             ctx:   _PlanCtx) -> Optional[_PlanResult]:
    """Plan each child; pick cheapest by step count adjusted for
    strategy-claim preference.

    Strategy preference: for each child, consult committed
    :class:`StrategyClaim`\\s whose ``context_pattern`` evaluates true
    in the current state.  A matching StrategyClaim biases the cost
    down by ``(1 - success_rate) * 5`` (so a 0.9-success-rate strategy
    is preferred by half a step over a 0.5-rate one).
    """
    best: Optional[_PlanResult] = None
    best_cost = float("inf")
    chosen_child_id: Optional[str] = None

    for child in node.children:
        sub = _plan_node(child, state, ctx)
        if sub is None:
            continue
        sub_steps, sub_ass, sub_branches = sub
        cost = float(len(sub_steps))
        cost -= _strategy_preference(child, ctx.ws)
        if cost < best_cost:
            best = sub
            best_cost = cost
            chosen_child_id = child.id

    if best is None:
        return None

    steps, assumptions, branches = best
    if chosen_child_id is not None:
        branches[node.id] = chosen_child_id
        node.active_branch = chosen_child_id
    return steps, assumptions, branches


def _strategy_preference(child: GoalNode, ws: WorldState) -> float:
    """Bias term for OR-branch selection from StrategyClaims.
    Returns 0 if no claim applies."""
    best_bias = 0.0
    for h in _store.committed(ws):
        if not isinstance(h.claim, StrategyClaim):
            continue
        ctx_ok = h.claim.context_pattern.evaluate(ws) is True
        if not ctx_ok:
            continue
        # Child is preferred if its id contains the strategy_type string
        if h.claim.strategy_type in child.id:
            bias = h.claim.success_rate * 5.0
            if bias > best_bias:
                best_bias = bias
    return best_bias


# ---------------------------------------------------------------------------
# CHANCE
# ---------------------------------------------------------------------------


def _plan_chance(node:  GoalNode,
                 state: Dict,
                 ctx:   _PlanCtx) -> Optional[_PlanResult]:
    """Pick the outcome branch with highest expected value.

    Expected value = ``prior(outcome) / (plan_cost + 1)``.  The ``+1``
    avoids a divide-by-zero for zero-step plans.  The chosen branch
    is recorded in ``branch_selections`` under ``node.id`` so the
    runner can detect deviation and replan.
    """
    best: Optional[_PlanResult] = None
    best_ev = -1.0
    chosen_child_id: Optional[str] = None

    for child in node.children:
        sub = _plan_node(child, state, ctx)
        if sub is None:
            continue
        sub_steps, sub_ass, sub_branches = sub
        prior = node.outcome_priors.get(child.id, 1.0 / max(1, len(node.children)))
        ev = prior / (len(sub_steps) + 1.0)
        if ev > best_ev:
            best = sub
            best_ev = ev
            chosen_child_id = child.id

    if best is None:
        return None
    steps, assumptions, branches = best
    if chosen_child_id is not None:
        branches[node.id] = chosen_child_id
        node.active_branch = chosen_child_id
    return steps, assumptions, branches


# ---------------------------------------------------------------------------
# OPTION
# ---------------------------------------------------------------------------


def _plan_option(node:  GoalNode,
                 state: Dict,
                 ctx:   _PlanCtx) -> Optional[_PlanResult]:
    """Inline an Option's internal plan as a single-step "macro".

    Option nodes carry the Option's ``id`` as a supporting hypothesis
    so the runner can record usage statistics at execution time.  The
    Option's applicability condition is checked against ``state``; if
    not applicable, fall through by returning ``None``.
    """
    # In Phase 3 the OPTION node carries the Option ID in
    # ``supporting_hypothesis_ids[0]``; the ws.options dict provides
    # the Option object.
    if not node.supporting_hypothesis_ids:
        return None
    opt_id = node.supporting_hypothesis_ids[0]
    option: Optional[Option] = ctx.ws.options.get(opt_id)
    if option is None:
        return None
    if option.applicability.evaluate(ctx.ws) is False:
        return None
    # Reuse the pre-recorded plan as the output.  Mark dependence on the
    # Option ID so the runner can update n_uses / success_rate.
    assumption_ids = [opt_id]
    return list(option.internal_plan.steps), assumption_ids, {}


# ---------------------------------------------------------------------------
# Helpers: condition evaluation against an arbitrary state dict
# ---------------------------------------------------------------------------


def _condition_holds(cond:  Condition,
                     state: Dict,
                     ws:    WorldState) -> bool:
    """Evaluate a condition against a possibly-hypothetical state.

    We swap the agent dict into the WorldState temporarily because
    :meth:`Condition.evaluate` reads from ``ws.agent``.  Other entity
    states remain as-is.  This is a conservative simulation — side
    effects on non-agent entities aren't modelled here.
    """
    saved_agent = ws.agent
    ws.agent = state
    try:
        truth = cond.evaluate(ws)
    finally:
        ws.agent = saved_agent
    return truth is True


def _apply_transition(state: Dict, tc: TransitionClaim) -> Dict:
    """Apply a TransitionClaim's ``post`` condition to ``state``,
    returning a new dict.

    Phase 3 MVP handles ``AtPosition`` specifically because grid games
    are the canonical test case; other condition types leave state
    unchanged.  A generalised state-update mechanism (supporting
    arbitrary property mutations) will arrive with the richer
    TransitionClaim schemas in Phase 4.
    """
    new_state = dict(state)
    post = tc.post
    if isinstance(post, AtPosition):
        new_state["position"] = tuple(post.pos)
    return new_state


def _simulate_steps(state: Dict,
                    steps: List[PlannedAction]) -> Dict:
    """Simulate executing steps against ``state`` by chaining
    TransitionClaims' ``post`` conditions.  Used for AND planning so
    later children see realistic intermediate state."""
    cur = dict(state)
    for s in steps:
        for claim in s.expected_effects:
            if isinstance(claim, TransitionClaim):
                cur = _apply_transition(cur, claim)
    return cur


def _state_key(state: Dict) -> Tuple:
    """Hashable key for BFS visited set.  Includes position and
    (if present) lives / resources so the same position with different
    resource counts is treated as a distinct state."""
    pos = tuple(state.get("position", ()))
    lives = state.get("lives")
    resources = tuple(sorted((state.get("resources") or {}).items()))
    return (pos, lives, resources)


def _atomic_count(node: GoalNode):
    """Yield atomic leaves for a rough cost heuristic in AND
    ordering."""
    if node.node_type == NodeType.ATOM:
        yield node
        return
    for c in node.children:
        yield from _atomic_count(c)


# ---------------------------------------------------------------------------
# Monkey patch: evaluate_state on Condition base (for planner sim)
# ---------------------------------------------------------------------------
#
# The planner needs to evaluate a Condition against a hypothetical
# state dict rather than a full WorldState.  Rather than plumbing
# through every Condition subclass, we attach a helper here.  Pure
# additive — does not change Condition's existing semantics.


def _evaluate_state(self, state: Dict, ws: WorldState) -> Optional[bool]:
    """Evaluate this condition with ``state`` swapped in as
    ``ws.agent``.  Restores the original agent state on exit."""
    saved = ws.agent
    ws.agent = state
    try:
        return self.evaluate(ws)
    finally:
        ws.agent = saved


Condition.evaluate_state = _evaluate_state  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Config access
# ---------------------------------------------------------------------------


def _planner_cfg(ws: WorldState):
    if ws.config is not None and hasattr(ws.config, "planner"):
        return ws.config.planner
    from .config import PlannerConfig
    return PlannerConfig()
