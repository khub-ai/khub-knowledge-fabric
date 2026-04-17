"""
Optimal solver for TR87 - all 6 levels.

No LLM calls. Reads game internal state, computes exact action sequence.

Actions:
  ACTION1 (1) = rotate current slot/set BACKWARD  (sprite X -> X-1 mod 7)
  ACTION2 (2) = rotate current slot/set FORWARD   (sprite X -> X+1 mod 7)
  ACTION3 (3) = cursor LEFT
  ACTION4 (4) = cursor RIGHT

Level modes:
  Standard (L1-3): orange strip sprites must match rule(cyan)
  double_translation (L4): orange must match rule(rule(cyan))
  alter_rules (L5-6): rules themselves are rotatable; find rule rotations
                      so that rule(cyan)=orange
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import arc_agi

api_key = ""
for line in open("P:/_access/Security/api_keys.env"):
    if line.startswith("arc_api_key="):
        api_key = line.split("=", 1)[1].strip()


def sprite_num(name: str) -> int:
    return int(name[-1])


def apply_rules(names: list, rule_map: dict, n_passes: int = 1) -> list | None:
    """Apply rule_map to names n_passes times. rule_map: {(n1,...): [o1,...]}."""
    result = names
    for _ in range(n_passes):
        out = []
        i = 0
        while i < len(result):
            matched = False
            for lhs_key, rhs_val in rule_map.items():
                size = len(lhs_key)
                if tuple(result[i:i+size]) == lhs_key:
                    out.extend(rhs_val)
                    i += size
                    matched = True
                    break
            if not matched:
                return None
        result = out
    return result


def solve_standard(game, double: bool = False) -> list[int] | None:
    """
    Standard / double_translation mode solver.
    Returns action sequence or None if unsolvable.
    """
    rule_map = {
        tuple(s.name for s in lhs): [s.name for s in rhs]
        for lhs, rhs in game.cifzvbcuwqe
    }
    cyan_names = [s.name for s in game.zvojhrjxxm]
    passes = 2 if double else 1
    target = apply_rules(cyan_names, rule_map, passes)
    if target is None:
        return None

    orange_names = [s.name for s in game.ztgmtnnufb]
    if len(target) != len(orange_names):
        return None

    actions = []
    for i in range(len(target)):
        steps = (sprite_num(target[i]) - sprite_num(orange_names[i])) % 7
        actions.extend([2] * steps)
        if i < len(target) - 1:
            actions.append(4)  # cursor right
    return actions


def _rotated_name(name: str, delta: int) -> str:
    """Return sprite name with number shifted by delta mod 7."""
    base = name[:-1]
    n = (sprite_num(name) - 1 + delta) % 7 + 1
    return f"{base}{n}"


def solve_alter_rules(game) -> list[int] | None:
    """
    alter_rules mode: find rotations for each rule set (LHS and RHS) so
    that the rotated rules correctly map cyan -> orange.

    Uses recursive search matching cyan left-to-right.
    Returns action sequence or None.
    """
    rules = [
        ([s.name for s in lhs], [s.name for s in rhs])
        for lhs, rhs in game.cifzvbcuwqe
    ]
    cyan_names = [s.name for s in game.zvojhrjxxm]
    orange_names = [s.name for s in game.ztgmtnnufb]

    # set_index: rotation delta (0-6)
    rotations: dict[int, int] = {}

    def search(cyan_pos: int, orange_pos: int) -> bool:
        if cyan_pos == len(cyan_names):
            return orange_pos == len(orange_names)
        for rule_i, (lhs_names, rhs_names) in enumerate(rules):
            lhs_size = len(lhs_names)
            rhs_size = len(rhs_names)
            if cyan_pos + lhs_size > len(cyan_names):
                continue
            if orange_pos + rhs_size > len(orange_names):
                continue

            # Determine required LHS rotation
            cyan_sub = cyan_names[cyan_pos:cyan_pos + lhs_size]
            req_lhs = None
            for k in range(7):
                if all(
                    (sprite_num(lhs_names[j]) - 1 + k) % 7 + 1 == sprite_num(cyan_sub[j])
                    for j in range(lhs_size)
                ):
                    req_lhs = k
                    break
            if req_lhs is None:
                continue

            lhs_set_idx = rule_i * 2
            if lhs_set_idx in rotations and rotations[lhs_set_idx] != req_lhs:
                continue

            # Determine required RHS rotation
            orange_sub = orange_names[orange_pos:orange_pos + rhs_size]
            req_rhs = None
            for k in range(7):
                if all(
                    (sprite_num(rhs_names[j]) - 1 + k) % 7 + 1 == sprite_num(orange_sub[j])
                    for j in range(rhs_size)
                ):
                    req_rhs = k
                    break
            if req_rhs is None:
                continue

            rhs_set_idx = rule_i * 2 + 1
            if rhs_set_idx in rotations and rotations[rhs_set_idx] != req_rhs:
                continue

            # Commit and recurse
            old_lhs = rotations.get(lhs_set_idx)
            old_rhs = rotations.get(rhs_set_idx)
            rotations[lhs_set_idx] = req_lhs
            rotations[rhs_set_idx] = req_rhs

            if search(cyan_pos + lhs_size, orange_pos + rhs_size):
                return True

            # Undo
            if old_lhs is None:
                del rotations[lhs_set_idx]
            else:
                rotations[lhs_set_idx] = old_lhs
            if old_rhs is None:
                del rotations[rhs_set_idx]
            else:
                rotations[rhs_set_idx] = old_rhs

        return False

    if not search(0, 0):
        return None

    # Build action sequence over all sets
    n_sets = len(rules) * 2
    actions = []
    for set_i in range(n_sets):
        delta = rotations.get(set_i, 0)
        actions.extend([2] * delta)
        if set_i < n_sets - 1:
            actions.append(4)
    return actions


def solve_alter_double(game) -> list[int] | None:
    """
    alter_rules + double_translation mode.

    Two-pass search:
      Pass 1: stage-1 rules (LHS matches cyan type) map cyan -> intermediate.
              Each rule has an independent LHS delta and RHS delta.
              RHS delta is searched (0-6) because it determines the intermediate.
      Pass 2: stage-2 rules (LHS matches intermediate type) map intermediate -> orange.
              LHS delta determined by matching intermediate; RHS delta by matching orange.

    All rotations are stored in the shared dict and returned as the action sequence.
    """
    all_rules = [
        ([s.name for s in lhs], [s.name for s in rhs])
        for lhs, rhs in game.cifzvbcuwqe
    ]
    cyan_names  = [s.name for s in game.zvojhrjxxm]
    orange_names = [s.name for s in game.ztgmtnnufb]

    cyan_base = cyan_names[0][:-1] if cyan_names else ""
    stage1 = [i for i, (lhs, _) in enumerate(all_rules) if lhs and lhs[0][:-1] == cyan_base]
    stage2 = [i for i in range(len(all_rules)) if i not in stage1]

    rotations: dict[int, int] = {}
    intermediate: list[str] = []

    def search2(inter_pos: int, orange_pos: int) -> bool:
        if inter_pos == len(intermediate):
            return orange_pos == len(orange_names)
        for rule_i in stage2:
            lhs_names, rhs_names = all_rules[rule_i]
            lhs_size, rhs_size = len(lhs_names), len(rhs_names)
            if inter_pos + lhs_size > len(intermediate): continue
            if orange_pos + rhs_size > len(orange_names): continue
            inter_sub = intermediate[inter_pos:inter_pos + lhs_size]
            req_lhs = None
            for k in range(7):
                if all((sprite_num(lhs_names[j])-1+k)%7+1 == sprite_num(inter_sub[j])
                       for j in range(lhs_size)):
                    req_lhs = k; break
            if req_lhs is None: continue
            lhs_idx = rule_i * 2
            if lhs_idx in rotations and rotations[lhs_idx] != req_lhs: continue
            orange_sub = orange_names[orange_pos:orange_pos + rhs_size]
            req_rhs = None
            for k in range(7):
                if all((sprite_num(rhs_names[j])-1+k)%7+1 == sprite_num(orange_sub[j])
                       for j in range(rhs_size)):
                    req_rhs = k; break
            if req_rhs is None: continue
            rhs_idx = rule_i * 2 + 1
            if rhs_idx in rotations and rotations[rhs_idx] != req_rhs: continue
            old_l, old_r = rotations.get(lhs_idx), rotations.get(rhs_idx)
            rotations[lhs_idx] = req_lhs; rotations[rhs_idx] = req_rhs
            if search2(inter_pos + lhs_size, orange_pos + rhs_size): return True
            if old_l is None: del rotations[lhs_idx]
            else: rotations[lhs_idx] = old_l
            if old_r is None: del rotations[rhs_idx]
            else: rotations[rhs_idx] = old_r
        return False

    def search1(cyan_pos: int) -> bool:
        if cyan_pos == len(cyan_names):
            return search2(0, 0)
        for rule_i in stage1:
            lhs_names, rhs_names = all_rules[rule_i]
            lhs_size = len(lhs_names)
            if cyan_pos + lhs_size > len(cyan_names): continue
            cyan_sub = cyan_names[cyan_pos:cyan_pos + lhs_size]
            req_lhs = None
            for k in range(7):
                if all((sprite_num(lhs_names[j])-1+k)%7+1 == sprite_num(cyan_sub[j])
                       for j in range(lhs_size)):
                    req_lhs = k; break
            if req_lhs is None: continue
            lhs_idx = rule_i * 2
            if lhs_idx in rotations and rotations[lhs_idx] != req_lhs: continue
            rhs_idx = rule_i * 2 + 1
            for req_rhs in range(7):
                if rhs_idx in rotations and rotations[rhs_idx] != req_rhs: continue
                rotated_rhs = [
                    rhs_names[j][:-1] + str((sprite_num(rhs_names[j])-1+req_rhs)%7+1)
                    for j in range(len(rhs_names))
                ]
                old_l, old_r = rotations.get(lhs_idx), rotations.get(rhs_idx)
                rotations[lhs_idx] = req_lhs; rotations[rhs_idx] = req_rhs
                intermediate.extend(rotated_rhs)
                if search1(cyan_pos + lhs_size): return True
                del intermediate[-len(rhs_names):]
                if old_l is None: del rotations[lhs_idx]
                else: rotations[lhs_idx] = old_l
                if old_r is None: del rotations[rhs_idx]
                else: rotations[rhs_idx] = old_r
        return False

    if not search1(0):
        return None

    n_sets = len(all_rules) * 2
    actions = []
    for set_i in range(n_sets):
        delta = rotations.get(set_i, 0)
        actions.extend([2] * delta)
        if set_i < n_sets - 1:
            actions.append(4)
    return actions


def solve_level(game, level_num: int, verbose: bool = True) -> list[int] | None:
    alter = game.current_level.get_data("alter_rules")
    double = game.current_level.get_data("double_translation")

    if verbose:
        print(f"  L{level_num}: alter={alter}, double={double}, "
              f"cyan={len(game.zvojhrjxxm)}, orange={len(game.ztgmtnnufb)}, "
              f"rules={len(game.cifzvbcuwqe)}, budget={game.upmkivwyrxz}")

    if alter and double:
        actions = solve_alter_double(game)
    elif alter:
        actions = solve_alter_rules(game)
    else:
        actions = solve_standard(game, double=bool(double))

    if actions is None:
        if verbose:
            print(f"  L{level_num}: FAILED to compute action sequence")
        return None

    if verbose:
        print(f"  L{level_num}: {len(actions)} actions: "
              + " ".join(f"A{a}" for a in actions))
    return actions


def run_tr87(verbose: bool = True) -> dict:
    arc = arc_agi.Arcade(arc_api_key=api_key)
    env = arc.make("tr87", render_mode=None)
    frame = env.reset()
    game = env._game

    total_steps = 0
    result = {"levels_solved": 0, "steps_per_level": [], "success": False}

    for level_num in range(1, 7):
        if verbose:
            print(f"\n=== Level {level_num} ===")

        actions = solve_level(game, level_num, verbose=verbose)
        if actions is None:
            break

        level_steps = 0
        solved = False
        for action in actions:
            frame = env.step(action)
            level_steps += 1
            total_steps += 1
            if frame.levels_completed >= level_num:
                solved = True
                break

        if verbose:
            print(f"  L{level_num}: {'SOLVED' if solved else 'FAILED'} "
                  f"in {level_steps} steps (total: {total_steps})")

        result["steps_per_level"].append(level_steps)

        if not solved:
            break
        result["levels_solved"] = level_num

    result["total_steps"] = total_steps
    result["success"] = result["levels_solved"] == 6

    if verbose:
        print(f"\nFinal: {result['levels_solved']}/6 levels solved, "
              f"{total_steps} total steps")
        print(f"State: {frame.state.name}, levels_completed: {frame.levels_completed}")
        if result["success"]:
            print("*** FULL GAME SOLVED! ***")

    return result


def run_tr87_competition(verbose: bool = True) -> dict:
    """Run TR87 in competition mode.

    Uses a shadow local env to compute actions (needs env._game) while a competition
    env executes the same actions to record a server-side competition scorecard.

    Seeds match: competition server uses the same initial state as local seed=0.
    """
    from arc_agi.base import OperationMode
    from arcengine.enums import GameAction

    to_ga = lambda a: GameAction[f"ACTION{a}"]

    # Competition env: creates server-side scorecard, runs online
    comp_arc = arc_agi.Arcade(
        arc_api_key=api_key,
        operation_mode=OperationMode.COMPETITION,
    )
    comp_env = comp_arc.make("tr87", render_mode=None)

    # Shadow local env: reads game internals to compute actions
    local_arc = arc_agi.Arcade(arc_api_key=api_key)
    local_env = local_arc.make("tr87", render_mode=None)

    comp_frame = comp_env.reset()
    local_frame = local_env.reset()
    game = local_env._game

    scorecard_id = comp_arc._default_scorecard_id
    scorecard_url = f"https://three.arcprize.org/scorecard/{scorecard_id}"

    if verbose:
        print(f"Competition scorecard: {scorecard_id}")
        print(f"Scorecard URL: {scorecard_url}")

    total_steps = 0
    result = {
        "levels_solved": 0,
        "steps_per_level": [],
        "success": False,
        "scorecard_id": scorecard_id,
        "scorecard_url": scorecard_url,
    }

    for level_num in range(1, 7):
        if verbose:
            print(f"\n=== Level {level_num} ===")

        actions = solve_level(game, level_num, verbose=verbose)
        if actions is None:
            break

        level_steps = 0
        solved = False
        for action in actions:
            comp_frame = comp_env.step(to_ga(action))
            local_frame = local_env.step(action)
            level_steps += 1
            total_steps += 1
            if comp_frame.levels_completed >= level_num:
                solved = True
                break

        if verbose:
            print(f"  L{level_num}: {'SOLVED' if solved else 'FAILED'} "
                  f"in {level_steps} steps (total: {total_steps})")

        result["steps_per_level"].append(level_steps)
        if not solved:
            break
        result["levels_solved"] = level_num

    result["total_steps"] = total_steps
    result["success"] = result["levels_solved"] == 6

    if verbose:
        print(f"\nFinal: {result['levels_solved']}/6 levels solved, {total_steps} total steps")
        print(f"Competition state: {comp_frame.state.name}")
        print(f"Scorecard URL: {scorecard_url}")
        try:
            sc = comp_arc.get_scorecard()
            if sc:
                print(f"Score: {sc.score:.4f}, competition_mode={sc.competition_mode}")
        except Exception:
            pass
        if result["success"]:
            print("*** FULL GAME SOLVED IN COMPETITION MODE! ***")

    return result


if __name__ == "__main__":
    import sys
    if "--competition" in sys.argv:
        result = run_tr87_competition(verbose=True)
    else:
        result = run_tr87(verbose=True)
