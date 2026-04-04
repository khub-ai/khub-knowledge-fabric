# ARC-AGI-3 Ensemble

A Knowledge Fabric specialization for interactive ARC-AGI-3 environments.

ARC-AGI-3 is the next evolution of the ARC benchmark family: instead of
static input/output puzzle pairs, the agent must learn a hidden environment by
acting inside it, observing the results, and discovering progress signals
through sequential decision-making.

## What ARC-AGI-3 is about

ARC-AGI-3 is an interactive benchmark for general reasoning and learning. In
contrast to ARC-AGI-2, where the task is to infer a transformation rule from a
small set of demo grids, ARC-AGI-3 tasks are:

- sequential and episodic: the agent acts repeatedly in a live environment;
- hidden-state driven: much of the game logic is implicit in how actions change
the grid;
- exploration-heavy: progress is discovered through action and observation,
not provided directly by a labeled training pair;
- measured by game success: the benchmark rewards level completion and
meaningful state progress, not just a single predicted output grid.

A strong ARC-AGI-3 system must combine:

- accurate visual interpretation of the current game frame,
- hypothesis generation about action effects,
- goal-directed planning under uncertainty,
- tool/rule reuse across episodes,
- and revision when a plan does not produce progress.

## What makes ARC-AGI-3 special

ARC-AGI-3 is harder than ARC-AGI-2 in ways that matter for real-world AI.

- **No fixed I/O demos.** The only signal comes from the environment and
  progress metrics such as `levels_completed`.
- **Sequential interaction.** Actions can have delayed or context-dependent
  consequences that only become visible after multiple steps.
- **Environment mechanics are latent.** The system must infer hidden rules like
  which action toggles a switch or whether a move is blocked by a wall.
- **State persists across time.** The agent must remember previous observations
  and adapt goals as the episode unfolds.
- **Experimentation matters.** A good agent balances exploration with exploit-
  able progress, rather than simply guessing the next best move.

This makes ARC-AGI-3 a valuable testbed for AI systems that need to operate in
open-ended, interactive domains rather than just one-shot classification or
pattern-matching tasks.

## Why Knowledge Fabric is significant here

Knowledge Fabric (KF) is designed to support reusable, general-purpose
reasoning across domains. ARC-AGI-3 is an important use case because it shows
how KF can move from static puzzle solving to live, sequential environments.

KF contributes in several ways:

- **Modular reasoning rounds.** The same pipeline structure used in ARC-AGI-2
  can be specialized for ARC-AGI-3: observe the world, reason about it, plan
  actions, execute them, and revise based on outcomes.
- **Persistent knowledge.** KF stores rules and tools that can be reused across
  episodes, enabling the system to accumulate understanding of environment
  mechanics over time.
- **Goal and state management.** KF is built with `GoalManager` and
  `StateManager` abstractions, which are essential when progress depends on
  multi-step plans and hidden objectives.
- **Domain-general patterns.** ARC-AGI-3 requires the same high-level capability
  that other interactive domains need: converting raw observations into
  structured knowledge, then using that knowledge to guide action.

That means ARC-AGI-3 is not just a benchmark for this repo — it is a proof
point for KF's broader claim that reusable, learned knowledge can power many
different applications.

## How this repo approaches ARC-AGI-3

The `usecases/arc-agi-3/` folder contains the current ARC-AGI-3 ensemble
specialization.

- `usecases/arc-agi-3/DESIGN.md` — detailed design decisions for the interactive
  ensemble.
- `usecases/arc-agi-3/python/harness.py` — episode harness and CLI entry point.
- `usecases/arc-agi-3/python/ensemble.py` — episode orchestrator, rule matching,
  and action execution flow.
- `usecases/arc-agi-3/python/agents.py` — OBSERVER and MEDIATOR runner logic.
- `usecases/arc-agi-3/python/rules.py` / `tools.py` — ARC-AGI-3 specializations of
  the KF rule and tool shims.
- `usecases/arc-agi-3/prompts/observer.md` — OBSERVER system prompt.
- `usecases/arc-agi-3/prompts/mediator.md` — MEDIATOR system prompt.

### Experimental support

For live ARC-AGI-3 experimentation, the repo also includes:

- `tests/arc-agi-3/README.md` — exploratory test material for LS20.
- `tools/kf-session-viewer/README.md` — session replay support for ARC-AGI-3
  playlogs.

## Current status

This ARC-AGI-3 specialization is an early research prototype.

- The repository contains the KF ensemble scaffolding, prompt templates, and
  an episode harness designed to wrap ARC-AGI-3 environments.
- The implementation is focused on exploratory integration rather than a
  production-ready game solver.
- Useful outputs today include playlogs, observation summaries, and evidence of
  how the KF pipeline can be adapted to interactive domains.
- The main goal is to validate the architecture and gather evidence for
  reusing KF knowledge abstractions in sequential, action-driven tasks.

## Why ARC-AGI-3 matters beyond this benchmark

The ARC-AGI-3 pattern is relevant to many other domains that require
observation-driven action and knowledge accumulation, such as:

- robotics (`usecases/robotics/DESIGN.md`),
- cybersecurity and attack defense (`usecases/cyber-security/README.md`),
- expert knowledge transfer and clinical workflows
  (`usecases/expert-knowledge-transfer-for-image-classification/README.md`).

The same KF primitives used for ARC-AGI-3 also underlie these other use
cases:

- `docs/ensemble-pipeline.md` — the core KF ensemble architecture and how it
  supports multiple specializations.
- `docs/architecture.md` — core knowledge artifact schema, storage, and retrieval
  design.
- `usecases/arc-agi-2/README.md` and `usecases/arc-agi-2/DESIGN.md` — the
  static puzzle predecessor that shares the same foundational pipeline.

## Get started

1. Read `usecases/arc-agi-3/DESIGN.md` for the architecture and assumptions.
2. Review `tests/arc-agi-3/README.md` for ARC-AGI-3 test harness setup.
3. Explore the prompt templates in `usecases/arc-agi-3/prompts/`.
4. Run the ARC-AGI-3 harness with the local Python CLI once the environment is
   configured.

## Related materials

- [Root README](../../README.md)
- [KF Ensemble Pipeline](../../docs/ensemble-pipeline.md)
- [KF Architecture](../../docs/architecture.md)
- [ARC-AGI-2 Ensemble](../arc-agi-2/README.md)
- [ARC-AGI-2 Design](../arc-agi-2/DESIGN.md)
- [ARC-AGI-3 Design](DESIGN.md)
- [ARC-AGI-3 Prompts](prompts/observer.md)
- [ARC-AGI-3 Prompts](prompts/mediator.md)
- [ARC-AGI-3 Test Material](../../tests/arc-agi-3/README.md)
- [KF Session Viewer](../../tools/kf-session-viewer/README.md)
- [Robotics Use Case](../robotics/README.md)
- [Image Classification UC200](../expert-knowledge-transfer-for-image-classification/README.md)
