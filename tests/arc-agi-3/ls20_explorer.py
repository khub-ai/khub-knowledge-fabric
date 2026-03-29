#!/usr/bin/env python3
"""
Novelty-driven ARC-AGI-3 LS20 explorer.

This is not a guaranteed solver. It is a sample research agent that tries to
figure out how the environment works by:

1. Observing which actions are available.
2. Preferring actions that lead to visually novel states.
3. Treating increases in `levels_completed` as strong progress signals.
4. Learning simple action preferences from prior episodes.

Requirements:
    pip install arc-agi

Usage:
    python ls20_explorer.py
    python ls20_explorer.py --episodes 25 --steps 120 --render terminal
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import arc_agi


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def digest(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()[:16]


def to_plain(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [to_plain(x) for x in value]
    if isinstance(value, tuple):
        return [to_plain(x) for x in value]
    if isinstance(value, dict):
        return {str(k): to_plain(v) for k, v in value.items()}
    if hasattr(value, "model_dump"):
        return to_plain(value.model_dump())
    if hasattr(value, "dict"):
        return to_plain(value.dict())
    if hasattr(value, "__dict__"):
        return {
            k: to_plain(v)
            for k, v in vars(value).items()
            if not k.startswith("_")
        }
    return str(value)


def obs_field(obs: Any, name: str, default: Any = None) -> Any:
    if obs is None:
        return default
    if isinstance(obs, dict):
        return obs.get(name, default)
    return getattr(obs, name, default)


def obs_state_name(obs: Any) -> str:
    state = obs_field(obs, "state")
    if state is None:
        return "UNKNOWN"
    if hasattr(state, "name"):
        return state.name
    return str(state)


def obs_frame(obs: Any) -> Any:
    return to_plain(obs_field(obs, "frame"))


def frame_dimensions(frame: Any) -> tuple[int, int]:
    node = frame
    while isinstance(node, list) and node:
        if isinstance(node[0], list) and node[0] and not isinstance(node[0][0], list):
            return len(node), len(node[0])
        node = node[0]
    return 30, 30


@dataclass
class StepRecord:
    episode: int
    step: int
    state_key: str
    state_name: str
    action: str
    data: dict[str, Any]
    next_key: str
    next_state_name: str
    levels_completed: int
    delta_levels: int
    novel: bool


@dataclass
class ExplorerMemory:
    seen_states: set[str] = field(default_factory=set)
    state_visits: defaultdict[str, int] = field(default_factory=lambda: defaultdict(int))
    action_visits: defaultdict[tuple[str, str], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    coord_visits: defaultdict[tuple[str, str, int, int], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    records: list[StepRecord] = field(default_factory=list)
    best_levels_completed: int = 0


class LS20Explorer:
    def __init__(
        self,
        episodes: int,
        max_steps: int,
        render_mode: str | None,
        seed: int,
        output_dir: Path,
    ) -> None:
        self.episodes = episodes
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.arc = arc_agi.Arcade()
        self.env = self.arc.make("ls20", render_mode=render_mode)
        if self.env is None:
            raise RuntimeError("Failed to create LS20 environment.")
        self.memory = ExplorerMemory()
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def available_actions(self) -> list[Any]:
        return list(getattr(self.env, "action_space", []))

    def state_key(self, obs: Any) -> str:
        summary = {
            "state": obs_state_name(obs),
            "levels_completed": obs_field(obs, "levels_completed", 0),
            "win_levels": obs_field(obs, "win_levels", 0),
            "frame": obs_frame(obs),
        }
        return digest(summary)

    def candidate_coords(self, state_key: str, action_name: str, frame: Any) -> list[tuple[int, int]]:
        height, width = frame_dimensions(frame)
        points: list[tuple[int, int]] = []
        xs = sorted(set([0, width // 4, width // 2, (3 * width) // 4, max(0, width - 1)]))
        ys = sorted(set([0, height // 4, height // 2, (3 * height) // 4, max(0, height - 1)]))
        points.extend((x, y) for y in ys for x in xs)
        for _ in range(12):
            points.append((self.rng.randrange(width), self.rng.randrange(height)))
        points.sort(
            key=lambda xy: self.memory.coord_visits[(state_key, action_name, xy[0], xy[1])]
        )
        return points

    def action_name(self, action: Any) -> str:
        return getattr(action, "name", str(action))

    def score_action(
        self,
        state_key: str,
        action_name: str,
        next_predicted_bonus: float = 0.0,
    ) -> float:
        visits = self.memory.action_visits[(state_key, action_name)]
        state_visits = self.memory.state_visits[state_key]
        ucb = math.sqrt(math.log(state_visits + 2) / (visits + 1))
        return next_predicted_bonus + 2.0 * ucb - 0.03 * visits

    def choose_action(self, obs: Any) -> tuple[Any, dict[str, Any]]:
        state_key = self.state_key(obs)
        frame = obs_frame(obs)
        best_action = None
        best_data: dict[str, Any] = {}
        best_score = float("-inf")

        for action in self.available_actions():
            name = self.action_name(action)
            is_complex = bool(getattr(action, "is_complex", lambda: False)())

            if not is_complex:
                score = self.score_action(state_key, name)
                if score > best_score:
                    best_action = action
                    best_data = {}
                    best_score = score
                continue

            for x, y in self.candidate_coords(state_key, name, frame)[:20]:
                probe_bonus = -0.02 * self.memory.coord_visits[(state_key, name, x, y)]
                score = self.score_action(state_key, name, probe_bonus)
                if score > best_score:
                    best_action = action
                    best_data = {"x": x, "y": y}
                    best_score = score

        if best_action is None:
            raise RuntimeError("No available action found.")
        return best_action, best_data

    def step_once(
        self,
        episode: int,
        step: int,
        obs: Any,
    ) -> tuple[Any, tuple[str, dict[str, Any]]]:
        state_key = self.state_key(obs)
        state_name = obs_state_name(obs)
        levels_before = int(obs_field(obs, "levels_completed", 0) or 0)

        action, data = self.choose_action(obs)
        action_name = self.action_name(action)
        self.memory.state_visits[state_key] += 1
        self.memory.action_visits[(state_key, action_name)] += 1
        if "x" in data and "y" in data:
            self.memory.coord_visits[(state_key, action_name, data["x"], data["y"])] += 1

        next_obs = self.env.step(action, data=data)
        next_key = self.state_key(next_obs)
        next_name = obs_state_name(next_obs)
        levels_after = int(obs_field(next_obs, "levels_completed", 0) or 0)
        delta_levels = levels_after - levels_before
        novel = next_key not in self.memory.seen_states
        self.memory.seen_states.add(next_key)
        self.memory.best_levels_completed = max(self.memory.best_levels_completed, levels_after)

        self.memory.records.append(
            StepRecord(
                episode=episode,
                step=step,
                state_key=state_key,
                state_name=state_name,
                action=action_name,
                data=data,
                next_key=next_key,
                next_state_name=next_name,
                levels_completed=levels_after,
                delta_levels=delta_levels,
                novel=novel,
            )
        )

        print(
            f"[ep {episode:02d} step {step:03d}] "
            f"{action_name} {data if data else ''} -> {next_name} "
            f"levels={levels_after} delta={delta_levels} novel={novel}"
        )

        return next_obs, (action_name, data)

    def run(self) -> None:
        best_trace: list[tuple[str, dict[str, Any]]] = []

        for episode in range(1, self.episodes + 1):
            print(f"\n=== Episode {episode}/{self.episodes} ===")
            obs = self.env.reset()
            trace: list[tuple[str, dict[str, Any]]] = []
            start = time.time()

            for step in range(1, self.max_steps + 1):
                obs, action_taken = self.step_once(episode, step, obs)
                trace.append(action_taken)

                state_name = obs_state_name(obs)
                levels_completed = int(obs_field(obs, "levels_completed", 0) or 0)
                if levels_completed > self.memory.best_levels_completed:
                    best_trace = trace[:]

                if state_name == "WIN":
                    print(f"WIN in episode {episode}, step {step}")
                    self.write_outputs(best_trace or trace)
                    return
                if state_name == "GAME_OVER":
                    print(f"GAME_OVER in episode {episode}, step {step}")
                    break

            elapsed = time.time() - start
            print(
                f"Episode summary: best_levels={self.memory.best_levels_completed} "
                f"elapsed={elapsed:.1f}s seen_states={len(self.memory.seen_states)}"
            )

        self.write_outputs(best_trace)

    def write_outputs(self, best_trace: list[tuple[str, dict[str, Any]]]) -> None:
        summary = {
            "best_levels_completed": self.memory.best_levels_completed,
            "seen_states": len(self.memory.seen_states),
            "records": [r.__dict__ for r in self.memory.records],
            "best_trace": [
                {"action": action, "data": data}
                for action, data in best_trace
            ],
        }
        out_file = self.output_dir / "ls20_exploration_summary.json"
        out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nWrote exploration summary to {out_file}")

        try:
            scorecard = self.arc.close_scorecard()
            if scorecard is not None:
                score_file = self.output_dir / "ls20_scorecard.json"
                if hasattr(scorecard, "model_dump_json"):
                    score_file.write_text(
                        scorecard.model_dump_json(indent=2),
                        encoding="utf-8",
                    )
                else:
                    score_file.write_text(
                        json.dumps(to_plain(scorecard), indent=2),
                        encoding="utf-8",
                    )
                print(f"Wrote scorecard to {score_file}")
        except Exception as exc:
            print(f"Could not close scorecard cleanly: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--render",
        default="terminal-fast",
        choices=["terminal", "terminal-fast", "human", "none"],
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="ls20_runs")
    args = parser.parse_args()

    render_mode = None if args.render == "none" else args.render
    agent = LS20Explorer(
        episodes=args.episodes,
        max_steps=args.steps,
        render_mode=render_mode,
        seed=args.seed,
        output_dir=Path(args.output_dir),
    )
    agent.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
