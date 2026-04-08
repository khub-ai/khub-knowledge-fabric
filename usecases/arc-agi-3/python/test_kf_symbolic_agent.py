"""
test_kf_symbolic_agent.py — End-to-end smoke tests for the orchestration loop.
"""

from __future__ import annotations

import asyncio
import sys
import traceback

from kf_symbolic_agent import KFSymbolicAgent
from mock_proposer import MockProposer
from planner import match_color_goal, reach_cell_goal
from rule_library import RuleLibrary
from synthetic_games import (
    AGENT,
    generate_navigation_game,
    generate_toggle_puzzle,
)


def _toggle_goal(g):
    cells = set()
    cr, cc = g.slot_pos
    for r in range(cr - g.slot_radius + 1, cr + g.slot_radius):
        for c in range(cc - g.slot_radius + 1, cc + g.slot_radius):
            if 0 <= r < g.grid_h and 0 <= c < g.grid_w:
                cells.add((r, c))
    return match_color_goal(cells, g.cycle[g.target_index])


def test_navigation_episode_wins() -> None:
    wins = 0
    n = 20
    for seed in range(n):
        g = generate_navigation_game(seed=seed)
        agent = KFSymbolicAgent(
            explore_steps=120, max_steps=300, agent_color=AGENT, seed=seed,
        )
        goal = reach_cell_goal(g.goal_pos, AGENT)
        result = asyncio.run(agent.play_episode(g, goal,
                                                source_tag=f"nav-{seed}"))
        if result.won:
            wins += 1
    # 70% bound: end-to-end loop is harder than the planner-only test
    # because exploration/replan/budgeting interact. The planner-only and
    # convergence tests pin the higher 80-95% bounds for the components.
    assert wins >= int(0.70 * n), f"only {wins}/{n} navigation episodes won"


def test_toggle_episode_wins() -> None:
    wins = 0
    n = 20
    for seed in range(n):
        g = generate_toggle_puzzle(seed=seed)
        agent = KFSymbolicAgent(
            explore_steps=40, max_steps=120, seed=seed,
        )
        result = asyncio.run(agent.play_episode(g, _toggle_goal(g),
                                                source_tag=f"toggle-{seed}"))
        if result.won:
            wins += 1
    assert wins >= int(0.80 * n), f"only {wins}/{n} toggle episodes won"


def test_rule_library_records_on_win_and_seeds_next_run() -> None:
    lib = RuleLibrary()
    g = generate_navigation_game(seed=0)
    agent = KFSymbolicAgent(
        rule_library=lib, explore_steps=80, max_steps=200,
        agent_color=AGENT, seed=0,
    )
    r1 = asyncio.run(agent.play_episode(g, reach_cell_goal(g.goal_pos, AGENT)))
    assert r1.won
    assert r1.library_recorded
    assert len(lib) == 1

    # A second run on the same seed should now be seeded from the library.
    g2 = generate_navigation_game(seed=0)
    agent2 = KFSymbolicAgent(
        rule_library=lib, explore_steps=80, max_steps=200,
        agent_color=AGENT, seed=1,
    )
    r2 = asyncio.run(agent2.play_episode(g2, reach_cell_goal(g2.goal_pos, AGENT)))
    assert r2.won
    assert r2.library_seeded >= 1


def test_proposer_stall_consultation_invoked() -> None:
    """Force a stall by giving an unsolvable goal and verify the
    Proposer is consulted (mock records the calls)."""
    g = generate_toggle_puzzle(seed=0)
    mock = MockProposer(mode="constrained")
    # Goal that no plan can satisfy: a cell color outside the cycle.
    bad_goal = lambda frame: False
    agent = KFSymbolicAgent(
        proposer=mock, explore_steps=10, max_steps=80,
        stall_threshold=1, seed=0,
    )
    asyncio.run(agent.play_episode(g, bad_goal))
    assert any(getattr(r, "request_kind", None) is not None
               for r in mock.received_requests)


def main() -> int:
    tests = [v for k, v in globals().items()
             if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL  {fn.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
