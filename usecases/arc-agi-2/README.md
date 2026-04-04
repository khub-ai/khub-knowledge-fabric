# ARC-AGI Ensemble

A 5-agent ensemble that solves [ARC-AGI-2](https://arcprize.org/) abstract reasoning puzzles through structured adversarial debate. The ensemble exposes a single synchronous REST endpoint — from the outside, it behaves like one agent.

## The idea

ARC-AGI puzzles require discovering a transformation rule from a few input→output demonstration pairs, then applying that rule to a new input. Single LLMs struggle because they commit to one interpretation early and can't escape it. An ensemble of agents with different reasoning styles can:

1. **Propose multiple hypotheses** — three solvers approach the same puzzle from different angles
2. **Catch errors** — a dedicated CRITIC verifies each proposal against ALL demo pairs
3. **Refine through debate** — solvers read each other's proposals and CRITIC's feedback, then revise
4. **Make a final decision** — a MEDIATOR weighs the evidence and produces the answer

## Agents

| Agent | Approach | Role in debate |
|---|---|---|
| **SOLVER-SPATIAL** | Visual/geometric reasoning (symmetry, rotation, gravity) | Proposes a rule, defends or revises after criticism |
| **SOLVER-PROCEDURAL** | Step-by-step algorithms (for-each-cell rules, coordinate math) | Proposes a rule, defends or revises after criticism |
| **SOLVER-ANALOGICAL** | Pattern classification (flood-fill? stamp? sort? extraction?) | Proposes a rule, defends or revises after criticism |
| **CRITIC** | Verification — tests each rule against every demo pair | Reports PASS/FAIL per solver, identifies specific failing cells |
| **MEDIATOR** | Final decision + knowledge generalization | Picks the best-supported answer, extracts lessons for future tasks, can inject human insights into the debate |

The MEDIATOR was originally named "JUDGE" but was renamed to reflect its broader mandate: it does not just decide — it also **manages a production rule system** that accumulates puzzle-solving knowledge, applies that knowledge to future tasks, guides debate direction, and can escalate to a human when the ensemble is stuck.

## Debate protocol

```
Round 0:  Rule matching — evaluate active rules against puzzle         (one LLM call)
          → Matched rules' actions injected into solver prompts
Round 1:  Three solvers independently propose a rule + output grid     (parallel)
          → Convergence check: if all three agree, CRITIC confirms
Round 2:  CRITIC evaluates each proposal against all demo pairs
Round 3:  Solvers revise based on CRITIC feedback + each other's work  (parallel)
Round 4:  MEDIATOR reads full debate, produces final answer + updates rules
```

If all solvers converge in Round 1 and CRITIC confirms, the debate short-circuits to save cost.

## Design decisions

### Reasoning only — no code execution (for now)

The solvers reason about grids in natural language + JSON. Python tooling exists (see `python/grid_tools.py`) for post-hoc analysis and evaluation, but the agents themselves do not execute code during the debate. This was chosen to:
- Keep the debate fast and cost-predictable
- Avoid the complexity of sandboxed execution in Round 1
- Let the ensemble validate itself — if code were the oracle, it would undercut the debate

Python code execution for solvers can be added in a later phase once the pure-reasoning baseline is established.

### Human is part of the ensemble

For research and evaluation runs, a human operator may inject insights into the debate at any point. Three mechanisms:
1. **Debate checkpoints** — before R1 (hypothesis), after CRITIC (insight), before MEDIATOR (final guidance)
2. **Stalemate detection** — if the ensemble fails to converge after Round 3, the harness can pause and prompt the operator for a hint
3. **Rule management** — humans can add, edit, deprecate, or reactivate rules directly in `rules.json`; these are picked up at the next puzzle

The UI shows all rule state changes after each task: which rules fired (with updated stats), which were created/generalized/specialized (with lineage and reasoning).

### Production Rule System

Instead of free-form knowledge entries, the ensemble maintains a **rule base** of condition-action pairs (`rules.json`). Each rule's condition identifies a puzzle type; its action provides solving guidance. Rules are matched against each new puzzle (Round 0), and their actions are injected into solver and MEDIATOR prompts.

```json
{
  "id": "r_001",
  "condition": "Grid contains isolated colored cells above empty space with a solid base row",
  "action": "Apply column-wise gravity: each non-background cell falls to the lowest empty cell in its column.",
  "stats": { "fired": 5, "succeeded": 4, "failed": 1 },
  "source": "mediator",
  "source_task": "1e0a9b12",
  "tags": ["gravity", "column", "spatial"],
  "lineage": { "type": "new", "parent_ids": [], "reason": "" },
  "status": "active"
}
```

**Rule evolution**: When rules fail, the MEDIATOR can **generalize** (condition too narrow), **specialize** (condition too broad), or **merge** rules. Each derivation is tracked via `lineage`, forming an auditable knowledge tree. Stats (fired/succeeded/failed) drive ranking so well-tested rules surface first.

**Rule matching**: One cheap LLM call per puzzle evaluates all active conditions. Matches are ranked by `match_confidence × success_rate`. Top-N rules (default 5) are injected into agent prompts.

See `specs/design-spec.md` Section 29 for the full specification.

### Python for research tooling, Node.js for the API server

| Layer | Language | Reason |
|---|---|---|
| Agentorum UI + `/api/ensemble` REST endpoint | Node.js | Already exists; integrates with the broader platform |
| Test harness, grid tools, evaluation, visualization | Python | numpy, matplotlib, rich — better fit for data science work |

When deeper UI integration is needed, the Python tooling can be exposed as a microservice with the Node server proxying to it.

## API endpoint (Node.js / Agentorum)

```
POST /api/ensemble
Content-Type: application/json

{
  "task": {
    "train": [
      { "input": [[0,1],[1,0]], "output": [[1,0],[0,1]] }
    ],
    "test": [
      { "input": [[0,0,1],[0,1,0],[1,0,0]] }
    ]
  },
  "context": "optional: learnings from previous tasks",
  "config": {
    "maxRounds": 4,
    "convergenceEnabled": true
  }
}
```

**Response:**
```json
{
  "answer": [[1,1,0],[1,0,1],[0,1,1]],
  "debate": [
    { "round": 1, "agent": "SOLVER-SPATIAL", "content": "..." },
    { "round": 1, "agent": "SOLVER-PROCEDURAL", "content": "..." },
    "..."
  ],
  "metadata": {
    "rounds": 4,
    "converged": false,
    "durationMs": 28400,
    "agents": 5
  }
}
```

## Running the Node test harness

Prerequisites:
- Agentorum server running (`npm start` from `packages/server/`)
- `ANTHROPIC_API_KEY` environment variable set
- ARC-AGI-2 training data in JSON format

```bash
# Run a single task (default)
node usecases/arc-agi-ensemble/test-harness.mjs

# Run a specific task by ID
node usecases/arc-agi-ensemble/test-harness.mjs --task-id 1e0a9b12

# Run first 10 tasks
node usecases/arc-agi-ensemble/test-harness.mjs --limit 10

# Custom data directory and server
node usecases/arc-agi-ensemble/test-harness.mjs \
  --data-dir /path/to/arctest2025/data/training \
  --server http://localhost:4800 \
  --limit 5 --output my-results.json
```

## Running the Python harness (research)

Prerequisites:
- Python 3.10+ with packages from `python/requirements.txt`
- `ANTHROPIC_API_KEY` environment variable set

```bash
cd usecases/arc-agi-ensemble/python

# Install dependencies (first time)
pip install -r requirements.txt

# Run a single task
python harness.py --task-id 1e0a9b12

# Run 10 tasks with charts saved per task
python harness.py --limit 10 --charts --charts-dir charts/

# Enable human-in-the-loop
python harness.py --limit 20 --human --output results.json

# Use a specific rules file
python harness.py --limit 10 --rules my-rules.json
```

### Training mode vs. test mode

The harness operates in one of two modes, controlled by `--mode`:

| | `--mode train` (default) | `--mode test` |
|---|---|---|
| **Rules** | New rules created and persisted | Read-only — existing rules used, no new rules saved |
| **Tools** | New tools generated and persisted | Read-only — existing tools used, no new tools saved |
| **Human checkpoints** | Available via `--human` | Disabled (ignored if passed) |
| **Revisions on failure** | Up to `MAX_REVISIONS` (default 5) | At most 1 revision |
| **Use case** | Gap-repair, learning new puzzle types | Benchmark evaluation, leaderboard runs |

```bash
# Benchmark a single task without touching the knowledge base
python harness.py --mode test --task-id 1e0a9b12

# Evaluate a batch of tasks (read-only, no learning side-effects)
python harness.py --mode test --limit 100 --output benchmark.json

# Train on a specific failed task (default mode — explicit for clarity)
python harness.py --mode train --task-id 14b8e18c --max-revisions 5
```

In test mode, tools may still be **generated in-memory** during the run (the LLM can still propose new tools to solve a task), but they are discarded when the process exits and never written to `tools.json`.

### Visualizations

The Python harness can generate four chart types:

| Chart | Description |
|---|---|
| **Hypothesis Grid Evolution** | Each solver's proposed grid at R1 and R3, next to the expected output |
| **Debate Flow Diagram** | Agent × round timeline with confidence/PASS/FAIL coloring |
| **Learning Curve** | Cumulative accuracy and cell accuracy over tasks, with KB pattern count overlay |
| **Ensemble vs Solo** | Bar chart comparing ensemble accuracy against individual solver baselines |

## First test result

Task `1e0a9b12` (column-wise gravity, 5×5 grid, 3 demo pairs):
- All three solvers **converged in Round 1**
- CRITIC **confirmed** — Round 3 skipped (convergence shortcut triggered)
- MEDIATOR produced the correct answer
- Duration: **40.7 seconds**  ✓ CORRECT

## Cost estimate

Each task requires 8 API calls in the full 4-round protocol (3 solvers × 2 rounds + CRITIC + MEDIATOR). Early convergence reduces this to 5 calls. With Claude Sonnet at ~$3/MTok input, ~$15/MTok output:

| Scale | Estimated cost | Duration |
|---|---|---|
| 1 task | ~$0.10–0.25 | ~30–60 seconds |
| 10 tasks | ~$1–2.50 | ~5–10 minutes |
| 100 tasks | ~$10–25 | ~50–100 minutes |
| 400 tasks (full eval) | ~$40–100 | ~3–7 hours |

Convergence shortcut reduces cost on easy tasks (where all solvers agree immediately).

## Files

```
usecases/arc-agi-ensemble/
├── README.md                          ← this file
├── arc-agi-ensemble.scenario.json     ← scenario configuration (Agentorum)
├── test-harness.mjs                   ← Node.js automated test runner
├── prompts/
│   ├── solver-spatial.md              ← visual/geometric reasoning
│   ├── solver-procedural.md           ← algorithmic step-by-step reasoning
│   ├── solver-analogical.md           ← pattern classification reasoning
│   ├── critic.md                      ← verification and challenge
│   └── mediator.md                    ← final decision + knowledge extraction
└── python/                            ← research tooling (standalone)
    ├── requirements.txt
    ├── harness.py                     ← CLI test runner
    ├── ensemble.py                    ← debate orchestrator
    ├── agents.py                      ← async Anthropic API calls per agent
    ├── grid_tools.py                  ← numpy grid operations
    ├── rules.py                       ← production rule system (condition-action pairs)
    ├── metadata.py                    ← structured per-round metadata capture
    ├── display.py                     ← rich terminal display for human participation
    └── visualize.py                   ← matplotlib/plotly charts
```
