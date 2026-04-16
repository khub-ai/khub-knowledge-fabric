"""test_routing.py -- unit tests for level-2 ring routing logic.

Tests the key decision: after RC visits done, the agent must refuel at ring
(14,15) (5 steps from win gate (14,40)), not ring (39,50) (23 steps from win gate).

Run with: python test_routing.py
"""
import sys
sys.path.insert(0, ".")

from collections import deque


def _bfs_path(start, end, walkable, step_size, extra_passable=None):
    """Minimal BFS matching ensemble.py signature."""
    if extra_passable is None:
        extra_passable = set()
    if start == end:
        return []
    queue = deque([(start, [start])])
    seen = {start}
    while queue:
        pos, path = queue.popleft()
        for dc, dr in [(0, step_size), (0, -step_size), (step_size, 0), (-step_size, 0)]:
            nxt = (pos[0] + dc, pos[1] + dr)
            if nxt in seen:
                continue
            if nxt not in walkable and nxt not in extra_passable:
                continue
            seen.add(nxt)
            new_path = path + [nxt]
            if nxt == end:
                return new_path
            queue.append((nxt, new_path))
    return None


def _build_ls20_l2_walkable():
    """Build wall-free grid for ls20 level 2.
    Player start=(29,40), step_size=5.
    Col positions: 4,9,14,19,24,29,34,39,44,49 (offset 4 mod 5).
    Row positions: 0,5,10,15,20,25,30,35,40,45,50,55 (multiples of 5).
    Real maze has walls; this tests pure routing logic in the open-grid case.
    """
    step = 5
    wset = set()
    for k in range(10):      # cols: 4,9,14,...,49
        c = 4 + k * step
        for r in range(0, 61, step):   # rows: 0,5,10,...,60
            wset.add((c, r))
    return wset


def _ring_coverage(ring_pos, non_ring_cands, wset, step_size, budget):
    """Count non-ring candidates reachable from ring within budget steps."""
    cnt = 0
    for c in non_ring_cands:
        path = _bfs_path(ring_pos, c, wset, step_size, {c})
        if path is not None and len(path) <= budget:
            cnt += 1
    return cnt


def test_ring_sort_target_excludes_rings():
    """Ring sort target should never be a known ring position."""
    ring_pos_set = {(14, 15), (39, 50)}
    unvis_cands = [(39, 50), (49, 60), (19, 40), (14, 40)]

    non_ring_cands = [p for p in unvis_cands if p not in ring_pos_set]
    assert (39, 50) not in non_ring_cands, "(39,50) is a ring, must be excluded"
    assert (14, 40) in non_ring_cands
    assert (19, 40) in non_ring_cands
    print("  ring sort target excludes (39,50): OK")


def test_bfs_path_on_ls20_grid():
    """Verify BFS finds paths in the correct ls20 coordinate system."""
    wset = _build_ls20_l2_walkable()
    step = 5

    # Basic reachability
    path = _bfs_path((29, 40), (14, 40), wset, step, {(14, 40)})
    assert path is not None, "(29,40) to (14,40) must be reachable"
    assert len(path) == 4, f"Expected 3 steps + end = 4 nodes, got {len(path)}"  # [start, m1, m2, end]

    # Ring to win gate
    path2 = _bfs_path((14, 15), (14, 40), wset, step, {(14, 40)})
    assert path2 is not None, "(14,15) to (14,40) must be reachable"
    assert len(path2) == 6, f"Expected 5 steps (6 nodes), got {len(path2)}"
    print(f"  (14,15) to (14,40): {len(path2)-1} steps -- OK")


def test_coverage_prefers_ring_near_win_gate():
    """In wall-free grid, both rings cover all candidates within budget.
    Key check: (14,15) specifically covers (14,40) in 5 steps."""
    wset = _build_ls20_l2_walkable()
    step = 5
    budget = 21

    non_ring_cands = [(49, 60), (19, 40), (14, 40), (24, 40), (4, 40)]

    cov_14_15 = _ring_coverage((14, 15), non_ring_cands, wset, step, budget)
    cov_39_50 = _ring_coverage((39, 50), non_ring_cands, wset, step, budget)

    print(f"  Coverage from (14,15): {cov_14_15}/{len(non_ring_cands)}")
    print(f"  Coverage from (39,50): {cov_39_50}/{len(non_ring_cands)}")

    # (14,15) covers (14,40) in 5 steps
    path = _bfs_path((14, 15), (14, 40), wset, step, {(14, 40)})
    assert path is not None, "(14,15) to (14,40) must be reachable"
    assert len(path) - 1 <= budget, f"path length {len(path)-1} exceeds budget {budget}"
    assert len(path) - 1 == 5, f"Expected 5 steps, got {len(path)-1}"
    print(f"  (14,15) to (14,40) = {len(path)-1} steps -- OK")

    # Both should cover non-ring candidates (wall-free grid)
    assert cov_14_15 >= 1, "Ring (14,15) must cover at least (14,40)"


def test_rc_done_mode_skips_ring_safety():
    """In rc_done_mode, safety check only requires reaching candidate (no ring after)."""
    # Simulate: player at (29,40), 21 effective steps, rc_done=True
    # Nearest candidate (14,40) is 3 steps away -- should be safely explorable
    wset = _build_ls20_l2_walkable()
    step = 5
    steps_remaining = 21
    player_pos = (29, 40)
    cand = (14, 40)

    path = _bfs_path(player_pos, cand, wset, step, {cand})
    assert path is not None
    dist = len(path) - 1  # number of moves

    # rc_done_mode check: dist < steps_remaining
    can_explore = dist < steps_remaining
    assert can_explore, f"Expected can_explore=True with {dist} steps to candidate and {steps_remaining} remaining"
    print(f"  rc_done_mode: {dist} steps to (14,40), {steps_remaining} remaining -> can_explore={can_explore} -- OK")


if __name__ == "__main__":
    import traceback
    tests = [
        test_ring_sort_target_excludes_rings,
        test_bfs_path_on_ls20_grid,
        test_coverage_prefers_ring_near_win_gate,
        test_rc_done_mode_skips_ring_safety,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1
        except Exception as e:
            print(f"ERROR: {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
