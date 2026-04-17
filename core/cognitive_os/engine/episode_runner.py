"""Episode runner — the main loop that orchestrates an episode.

One call to :func:`run_episode` consumes a fully-constructed
:class:`Adapter` and drives it through a single episode.  The runner
owns the step cadence; every subsystem in the engine is invoked from
here in a fixed order that preserves the invariants each phase
established.

Main loop structure
-------------------

::

    1. Adapter.initialize() / reset()          — populate tool registry,
                                                  seed primary goal,
                                                  full_scan frame
    2. Observer.full_scan() if not yet done    — initial visual priors
    3. Loop until done or budget exhausted:
       a. obs = adapter.observe()
       b. ingest(ws, obs)                      — append to history,
                                                  update agent state,
                                                  refresh entity snapshots
       c. miners.step()                        — PropertyObserved,
                                                  Transition, FutilePattern,
                                                  Surprise
       d. update_credence_from_events()        — evidence pass
       e. apply_staleness_decay_all()
       f. prune_abandoned()
       g. derive_subgoals_from_causal()        — expand goal tree
          detect_conflicts()
       h. select_active_goal()
       i. plan = compute_plan()                — or None if exhausted
       j. if plan invalid → replan
          elif plan is None → explore
          elif plan exhausted → add curiosity goal, retry
       k. action = plan.steps[0] or exploration fallback
       l. adapter.execute(action)              — and record _last_action
       m. step += 1
    4. run_post_mortem()                       — extract lessons,
                                                  synthesise Options,
                                                  persist cross-episode
    5. Adapter.on_episode_end()

Capability audit
----------------
* **Problem-solving** — PRIMARY.  The runner is where planning,
  exploration, and execution are actually connected to the
  environment.  Without this the engine is a collection of parts.
* **Debugging** — PRIMARY.  The runner drives the hypothesis
  lifecycle every step: ingest events, run miners, update credence,
  detect surprises, trigger specialisation.  Phase 2's machinery
  fires here per step.
* **Tool creation** — secondary.  :func:`run_post_mortem` is called
  at episode end; the :class:`OptionSynthesiser` hook is exercised,
  ready for Phase 7's fuller synthesis.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, List, Optional

from .adapters import Adapter
from .config import EngineConfig, PlannerConfig
from .miners import Miner, default_miners
from .types import (
    Action,
    AgentMoved,
    EntityModel,
    Event,
    GoalStatus,
    Observation,
    Plan,
    PlanStatus,
    PostMortem,
    SurpriseEvent,
    WorldState,
)
from . import hypothesis_store as _store
from . import goal_forest as _gf
from . import planner as _planner
from . import explorer as _explorer
from .postmortem import run_post_mortem


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_episode(adapter:     Adapter,
                ws:          WorldState,
                cfg:         EngineConfig,
                *,
                episode_id:  Optional[str] = None,
                max_steps:   int = 10_000,
                miners:      Optional[List[Miner]] = None) -> PostMortem:
    """Run one episode end-to-end against ``adapter``.

    Parameters
    ----------
    adapter
        A fully-constructed :class:`Adapter` subclass instance.
    ws
        WorldState to populate.  A fresh instance is typical for
        competition runs; a preloaded instance (with persisted
        hypotheses + options) is typical for training runs.
    cfg
        :class:`EngineConfig` to drive thresholds, budgets, and
        planner cadence.  Attached to ``ws.config`` for the
        duration.
    episode_id
        Stable identifier for logging and :class:`PostMortem`.
        Defaults to ``"{adapter.env_id}::ep_auto"``.
    max_steps
        Hard cap on iterations.  Exceeding triggers ``"timeout"``.
    miners
        Override the default miner suite.  Defaults to
        :func:`miners.default_miners()`.

    Returns
    -------
    PostMortem
        The episode retrospective.  Callers persist its
        ``lessons`` and ``options_synthesised`` if operating in a
        mode that allows cross-episode accumulation.

    Side effects
    ------------
    Mutates ``ws`` extensively — hypotheses, goal forest, observation
    history, agent state all accumulate.  Calls every
    :class:`Adapter` hook in order.
    """
    ws.config = cfg
    episode_id = episode_id or f"{adapter.env_id}::ep_auto"
    miners = miners or default_miners()

    # Phase 1: init
    adapter.initialize(ws)
    obs = adapter.reset()
    _ingest_observation(ws, obs)
    adapter.on_episode_start(ws)

    start_time = perf_counter()
    step = 0
    failed_plans: List[Plan] = []
    current_plan: Optional[Plan] = None
    final_status = "in_progress"

    while step < max_steps:
        if adapter.is_done():
            final_status = _done_status(ws, adapter)
            break

        # Mining + credence update pass
        events = list(obs.events) if obs else []
        for miner in miners:
            miner.step(ws, events, step)
        # Miners may have appended surprise events; re-read.
        events = list(ws.observation_history[-1].events) if ws.observation_history else events

        _store.update_credence_from_events(ws, events, step)
        _store.apply_staleness_decay_all(ws, step)
        _store.prune_abandoned(ws, step)

        # Subgoal derivation + conflict detection
        for goal_id in list(ws.goal_forest.goals.keys()):
            _gf.derive_subgoals_from_causal(ws, goal_id, step=step)
            _gf.refresh_status(ws, goal_id)
        _gf.detect_conflicts(ws, step=step)

        # Plan validity
        current_plan = _check_plan_validity(ws, current_plan, cfg)

        # Goal selection + planning
        if current_plan is None or current_plan.status != PlanStatus.ACTIVE:
            active = _gf.select_active_goal(ws)
            if active is None:
                # No active goal → try curiosity goals
                cur_goals = _explorer.propose_curiosity_goals(
                    ws, step=step, action_space=adapter.action_space())
                for g in cur_goals:
                    _gf.add_goal(ws, g)
                active = _gf.select_active_goal(ws)

            if active is not None:
                current_plan = _planner.compute_plan(
                    ws, active, adapter.action_space(), step=step)

        # Action selection
        action: Optional[Action] = None
        if current_plan is not None and current_plan.steps:
            action = current_plan.steps[0].action
        else:
            action = _explorer.choose_exploration_action(
                ws, adapter.action_space())

        if action is None:
            # Nothing to do — exhausted.
            final_status = "abandoned:no_action"
            break

        # Execute and record
        ws.agent["_last_action"] = action.name
        adapter.execute(action)

        # Advance plan pointer
        if current_plan is not None and current_plan.steps:
            current_plan.current_step_index += 1
            current_plan.steps = current_plan.steps[1:]
            if not current_plan.steps:
                current_plan.status = PlanStatus.COMPLETE
                failed_plans.append(current_plan)
                current_plan = None

        step += 1
        obs = adapter.observe()
        _ingest_observation(ws, obs)

    else:
        # Loop exited via max_steps guard
        final_status = "timeout"

    # Episode cleanup: pending plan becomes a failed plan for analysis
    if current_plan is not None:
        current_plan.status = PlanStatus.INVALIDATED
        failed_plans.append(current_plan)

    wall_time = perf_counter() - start_time
    pm = run_post_mortem(
        ws                = ws,
        episode_id        = episode_id,
        final_step        = step,
        final_status      = final_status,
        failed_plans      = failed_plans,
        wall_time_seconds = wall_time,
    )
    adapter.on_episode_end(ws)
    return pm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ingest_observation(ws: WorldState, obs: Observation) -> None:
    """Integrate a new Observation into WorldState.

    * Appends to ``observation_history``.
    * Updates ``ws.step`` and ``ws.agent`` (preserving engine-owned
      private keys starting with ``_``).
    * Refreshes or creates :class:`EntityModel`\\s from
      ``entity_snapshots``.
    """
    if obs is None:
        return
    ws.observation_history.append(obs)
    ws.step = obs.step

    # Preserve private keys (engine-owned) while updating public ones
    private = {k: v for k, v in ws.agent.items() if k.startswith("_")}
    ws.agent = dict(obs.agent_state)
    ws.agent.update(private)

    for entity_id, snapshot in obs.entity_snapshots.items():
        ent = ws.entities.get(entity_id)
        if ent is None:
            ent = EntityModel(
                id              = entity_id,
                properties      = dict(snapshot),
                first_seen_step = obs.step,
                last_seen_step  = obs.step,
                kind            = snapshot.get("_kind"),
            )
            ws.entities[entity_id] = ent
        else:
            ent.properties.update(snapshot)
            ent.last_seen_step = obs.step


def _check_plan_validity(ws:           WorldState,
                         plan:         Optional[Plan],
                         cfg:          EngineConfig) -> Optional[Plan]:
    """Return the plan unchanged if still valid; ``None`` if
    invalidated.

    Validity checks:

    * Plan assumptions: every hypothesis ID in ``plan.assumptions``
      must still be committed.  Any demotion invalidates.
    * Plan status: ACTIVE plans continue; COMPLETE/INVALIDATED/FAILED
      require replanning.
    """
    if plan is None:
        return None
    if plan.status != PlanStatus.ACTIVE:
        return None

    cred_cfg = cfg.credence
    for h_id in plan.assumptions:
        h = ws.hypotheses.get(h_id)
        if h is None or not h.credence.is_committed(cred_cfg):
            plan.status = PlanStatus.INVALIDATED
            return None
    return plan


def _done_status(ws: WorldState, adapter: Adapter) -> str:
    """Infer a terminal status label from the adapter + goal state.

    Uses :func:`goal_forest.is_achieved` so the check reflects the
    current :class:`WorldState` (e.g. position just updated by the
    last :meth:`Adapter.execute` → :meth:`Adapter.observe`) rather
    than a stale cached status from before the final step.
    """
    for goal_id in ws.goal_forest.goals.keys():
        if _gf.is_achieved(ws, goal_id):
            # Cascade the status so post-mortem sees it correctly.
            _gf.mark_status(ws, goal_id, GoalStatus.ACHIEVED)
            return "success"
    return "failure"
