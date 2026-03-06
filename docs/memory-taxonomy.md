# Memory Taxonomy and Generalization

This document describes the four types of knowledge that the PIL pipeline is designed to capture, and explains why **generalization** — not just recording — is what separates knowledge management from mere memory.

## The four memory types

| Memory type | Question answered | Raw form | Generalized form |
|---|---|---|---|
| **Episodic** | *What happened?* | Event logs, conversation history | Feeds generalization of other types; not itself stored long-term |
| **Semantic** | *What is true?* | Individual facts, observations | Rules, conventions, constraints applicable across contexts |
| **Procedural** | *How do you do it?* | Step-by-step descriptions | Structured recipes, or optionally executable programs |
| **Evaluative** | *What counts as "good"?* | Individual preference signals, choices | Judgment heuristics, aesthetic principles, value frameworks |

Most agent "memory" today is **episodic** — compacted conversation logs carried forward between sessions. Some advanced agents index and summarize prior sessions, which is a step beyond purely ephemeral context, but the result is still episodic in nature: a record of what was said, not a distillation of what was learned.

This project focuses on producing **semantic, procedural, and evaluative** knowledge in their **generalized forms**: rules that apply across contexts, procedures that can be reused or automated, and judgment frameworks that guide decision-making in novel situations.

## Why generalization matters

Recording that the user corrected a summary format five times is episodic memory. Distilling those five corrections into a rule — "always use bullet points, max 5 items, no filler phrases" — is semantic generalization. The rule is more compact, more useful, and applicable to situations the user has never explicitly commented on.

Generalization is what gives the agent **predictive behaviour**: the ability to act correctly in new situations based on principles learned from past ones, rather than requiring explicit instruction every time.

Each memory type has its own generalization path:

- **Episodic → Semantic**: repeated observations become general rules
- **Episodic → Procedural**: repeated step sequences become structured recipes
- **Episodic/Semantic → Evaluative**: patterns of choice and preference become judgment heuristics

## Semantic memory

Semantic knowledge captures what is true — facts, preferences, conventions, constraints. In its raw form, it's a specific observation: "this user prefers TypeScript." In its generalized form, it becomes a rule with scope: "all code produced for this user should use TypeScript with strict mode, ES modules, and async/await."

The generalization path:

```
specific observation → pattern across observations → rule with defined scope
```

Semantic artifacts are the most straightforward to capture and the most immediately useful. They reduce the need for repeated prompting and allow the agent to apply known preferences automatically.

## Procedural memory

Procedural knowledge captures how to do things. It sits on a spectrum:

```
text description  →  structured recipe  →  executable program
```

**The structured recipe is the primary artifact.** Not every procedure should — or can — become a program. Many use cases benefit from the flexibility of a recipe that the agent interprets with judgment. The recipe remains human-readable and editable, and the agent can adapt it to context.

However, for procedures that are **repeated identically**, optionally turning the recipe into an executable program has compounding benefits:

- **Perfect repeatability** — a program produces the same output every time. In enterprise contexts, this matters: certain actions must have perfectly predictable results, with no AI-introduced deviation regardless of how many times the procedure runs.
- **Cost efficiency** — identifying when a procedure applies and running a program is vastly cheaper than having an LLM re-derive a complex workflow from scratch each time.
- **Auditability** — code can be reviewed, tested, and version-controlled.
- **Scalability** — a library of programs scales better than a library of prompts.

Generalizing procedural knowledge is **harder** than generalizing episodic or semantic memory. Turning "here's roughly how I do X" into reliable, tested code requires validation, edge-case handling, and user trust. The agent must learn not just *what* to do, but *when* it is safe to automate.

Over time, this produces a powerful effect: the agent learns to **build and maintain a library of executable programs** that serve the user's needs — essentially programming its own tools. The structured recipe is retained alongside the program as documentation and as a fallback for cases where flexibility is needed.

## Evaluative memory

Evaluative knowledge is the most subtle and arguably the most important of the four types. It answers the question: **what counts as "good"?**

This is the knowledge that guides selection among valid alternatives. When the rules don't determine a single answer — when multiple options are technically correct — something else must guide the choice. That "something else" is evaluative knowledge: learned judgment about what humans find natural, elegant, correct, or appropriate.

### Why it's distinct from semantic memory

You could argue that "user prefers concise summaries" is a semantic fact. At the surface level it is. But evaluative knowledge operates at a **meta-level**: it doesn't just tell you what is true — it tells you how to *choose* when multiple truths are available. It's the implicit scoring function that ranks valid options.

This distinction matters because evaluative knowledge is **harder to learn** than semantic knowledge. Semantic facts can often be stated explicitly ("the capital of France is Paris"). Evaluative knowledge is typically **tacit** — the person who holds it may not be able to articulate it fully, but they can demonstrate it through repeated choices. It must be induced from patterns of preference, not just recorded from statements.

### The ARC-AGI-v2 example

The ARC-AGI-v2 benchmark illustrates this clearly. Each visual puzzle presents input-output examples from which the solver must infer a transformation rule and apply it to a new input. Multiple transformations may be logically consistent with the examples, but only those that align with human visual intuitions — symmetry, simplicity, compositional regularity — are scored as correct.

No amount of semantic knowledge (facts about grids) or procedural knowledge (algorithms for pattern matching) is sufficient. The solver must acquire evaluative knowledge: *what do humans consider a natural, elegant transformation?* This can only be learned by exposure to human judgments, not derived from rules alone.

ARC-AGI-v2, in this view, can only be fully solved by a system that learns evaluative knowledge — the general visual principles favoured by humans.

### Professional domains

The same structure appears in every domain where decisions require judgment:

- **Law**: Multiple legal arguments may be technically valid; evaluative knowledge determines which one a judge would find persuasive, which precedent is most relevant, what constitutes "reasonable."
- **Medicine**: Multiple treatments may be clinically indicated; evaluative knowledge captures how a senior clinician weighs risk tolerance, patient quality of life, and institutional norms.
- **Engineering**: Multiple architectures solve the problem; evaluative knowledge reflects which tradeoffs the team considers acceptable, what "clean code" means in this codebase.
- **Ethics**: Competing principles may apply; evaluative knowledge encodes how a person or culture prioritises among them.

In each case, the knowledge is **not reducible to rules or procedures** — it's a learned sense of what constitutes a good outcome in context.

### The generalization path for evaluative knowledge

```
individual preference → pattern of preferences → judgment heuristic → value framework
```

For example:
1. "User chose the shorter summary" *(single observation)*
2. "User consistently chooses concise options over detailed ones" *(pattern)*
3. "Favour brevity; include only information the reader would act on" *(heuristic)*
4. "This user's communication philosophy: respect the reader's time, lead with the actionable takeaway, omit hedging" *(value framework)*

The generalized form is powerful because it allows the agent to make good choices in **novel situations** — situations the user has never explicitly commented on, but where the value framework still applies. This is what makes evaluative knowledge the closest analogue to what we colloquially call "wisdom."
