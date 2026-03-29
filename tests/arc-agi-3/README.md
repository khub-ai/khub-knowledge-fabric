# ARC-AGI-3 Test Material

This folder contains early test material for experimenting with Knowledge Fabric
against ARC-AGI-3 environments.

Current focus:
- `LS20`, a public ARC-AGI-3 game that emphasizes conditional interaction,
  exploration, memory, and latent-state reasoning.

## Files

- `ls20_explorer.py`
  A novelty-driven exploration script intended to probe the LS20 environment,
  observe state changes, and build an action trace that may reveal how the game
  works.

## Goal

The goal is not to provide a guaranteed LS20 solver. The goal is to create a
repeatable exploration harness that can help us answer questions like:

- Which actions are available in different states?
- Which actions visibly change the environment?
- Are there consistent action consequences?
- Does `levels_completed` increase under identifiable conditions?
- Can action sequences be reused across runs?

This is useful as a first step toward evaluating whether Knowledge Fabric can
learn reusable rules or tools in an interactive environment rather than a static
input/output puzzle setting.

## Prerequisites

- A conda environment named `arc`
- The ARC-AGI-3 Python package installed in that environment
- Access to the `ls20` environment through the ARC toolkit

## Suggested Usage

From PowerShell or `cmd`:

```bash
conda run -n arc python ls20_explorer.py
conda run -n arc python ls20_explorer.py --episodes 25 --steps 120 --render terminal
```

## Outputs

The script writes:

- `ls20_runs/ls20_exploration_summary.json`
- `ls20_runs/ls20_scorecard.json` when available

These outputs can be inspected later to understand which abstract states were
visited, which actions were attempted, and whether any progress signals were
observed.

## Notes

- This is a research harness, not a benchmark submission agent.
- The script prefers novelty and light coverage rather than brute-force random
  play.
- If the installed ARC SDK differs slightly from the version assumed here, some
  field names may need adjustment.
