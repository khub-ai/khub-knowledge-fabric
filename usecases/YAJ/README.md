# Knowledge Fabric Use Case: YAJ (Yet Another Jarvis)
## A User-Owned Runtime Adaptation Layer For Personal And Work Assistants

---

> **Status**: Design spec only — not yet implemented  
> **Theme**: Knowledge Fabric (KF) as a personalization and adaptation layer for AI assistants  
> **Last updated**: 2026.04.01  

[Knowledge Fabric (KF)](../../docs/what-is-kf.md) is well suited to personal and work assistants because the biggest gap in current assistants is not generic intelligence, but **stable adaptation**: remembering how a specific user works, what they prefer, what they care about, what patterns they follow, and how those patterns should influence future behavior. YAJ is a KF use case that makes that adaptation explicit, portable, reviewable, and user-owned.

---

## 1. The Problem

Modern assistants are often impressive in a single interaction, but weak over time.

They repeatedly fail to:

- remember stable user preferences across sessions
- learn repeated procedures without being re-taught
- preserve naming conventions, work habits, and recurring workflows
- adapt their judgment to a user's specific standards
- survive model switches or vendor changes without losing personalization
- let the user inspect, edit, or remove what was learned

As a result, the user keeps spending effort on the same corrections:

- "Use this naming format."
- "Do not summarize like that."
- "When I say monthly reports, I mean these three institutions."
- "Ask before sending messages to external parties."
- "When there is uncertainty here, I prefer conservative handling."

Current assistant memory systems often make this worse because what they learn is:

- opaque
- vendor-controlled
- hard to edit
- hard to transfer
- hard to govern

YAJ proposes a different path: the assistant learns in a way the user can see, revise, and keep.

---

## 2. Why This Is A Strong KF Use Case

YAJ is not interesting because "Jarvis" sounds futuristic. It is interesting because it makes one of KF's deepest claims concrete:

**personalization should be an explicit, user-owned runtime knowledge layer, not an opaque side effect inside a vendor platform.**

This use case is a strong fit for KF because:

1. **The problem is universal**. Nearly every assistant product struggles with long-term personalization and workflow adaptation.

2. **The value is immediately legible**. Users understand the pain of having to repeat themselves.

3. **The required expertise is naturally available**. The user is the domain expert on their own habits, preferences, and standards. No special external talent is required.

4. **The artifact story is strong**. User preferences, procedures, conventions, and evaluative standards are exactly the kinds of [knowledge artifacts](../../docs/glossary.md#knowledge-artifact) KF is designed to store.

5. **The market need is real**. Personal assistants, enterprise copilots, executive assistants, agentic productivity tools, and customer-facing copilots all need better personalization.

This means YAJ can serve as both:

- a compelling end-user demonstration
- a flagship explanation of why KF matters as a broader runtime adaptation layer

---

## 3. What YAJ Actually Is

YAJ should not be positioned as:

- a generic "super assistant"
- a vague personality layer
- a bundle of prompts pretending to be memory

It should be positioned as:

**an assistant whose adaptation is powered by explicit KF artifacts that capture how the user works, communicates, decides, and revises their own procedures over time**

In other words, YAJ is not primarily about sounding more personal. It is about becoming more operationally aligned to one user's real patterns.

---

## 4. What KF Would Learn In YAJ

YAJ is a natural home for multiple KF artifact types:

- **Preference artifacts**: formatting, tone, communication style, pacing, priorities
- **Convention artifacts**: file naming, folder structure, recurring labels, terminology
- **Procedure artifacts**: repeated task sequences such as downloads, report preparation, or CRM updates
- **Judgment artifacts**: what the user considers good, risky, acceptable, too verbose, too casual, too aggressive, incomplete, or sufficient
- **Boundary artifacts**: when the assistant should ask before acting, when it should abstain, and when it should escalate

Example learned knowledge:

- "When drafting outbound messages, lead with the action item and keep tone professional but not stiff."
- "Monthly financial statements are saved as `YYYY-MM-institution-account.pdf`."
- "When I ask for a summary, prefer bullets over prose unless I explicitly request a narrative."
- "When a task might affect external stakeholders, ask for confirmation before sending or modifying anything."
- "For ambiguous analytical conclusions, I prefer cautious wording over confident extrapolation."

These are small things individually, but collectively they determine whether an assistant feels genuinely aligned or persistently off.

---

## 5. Why YAJ Could Stand Out

The consumer and enterprise markets already have many assistant products. Most are difficult to distinguish.

YAJ can stand out only if it is framed correctly.

The distinctive idea is not:

- "another smart assistant"

The distinctive idea is:

- **user-owned personalization**
- **portable assistant adaptation**
- **explicit runtime learning without fine-tuning**
- **inspectable and revisable memory rather than opaque memory**

This matters because most assistants today force a bad tradeoff:

- either they do not adapt enough
- or they adapt in opaque ways the user cannot govern

YAJ can demonstrate a better alternative:

**the assistant learns visibly, incrementally, and under user control**

That is a more durable and differentiated story than "Jarvis but better."

---

## 6. Market Relevance

There is a real market need here.

Relevant product categories include:

- personal productivity assistants
- executive assistants
- AI writing assistants
- enterprise knowledge workers' copilots
- customer-support assistants
- CRM and workflow assistants
- developer assistants that adapt to team conventions

In all of these categories, one of the hardest unsolved problems is:

**how to make the assistant adapt deeply without making it uncontrollable**

KF is directly relevant because it offers:

- adaptation without retraining
- persistence across sessions
- explicit governance and deletion
- portability across models and vendors
- incremental accumulation of user-specific knowledge

So yes, there is real market demand. The key is to present YAJ as a **better personalization architecture**, not just a feature-rich assistant.

---

## 7. Recommended First Demonstration

The first YAJ experiment should be narrow and concrete.

Recommended scenario:

**A recurring personal/work assistant workflow that combines conventions, procedures, and judgment**

Example:

- managing monthly financial statement downloads
- organizing files by the user's naming conventions
- preparing summaries in the user's preferred communication style
- asking for confirmation at the user's preferred risk boundaries

Why this is a good first demo:

- the workflow is easy to understand
- the repeated corrections are obvious
- the value of persistence is immediate
- the required expertise comes from the user, not outside specialists
- the learned artifacts are easy to inspect and explain

This is much stronger than a vague "general assistant" demo.

---

## 8. Candidate Experiment Structure

### Phase 1: Preference and convention learning

**Task**:
- Let the assistant interact with the user across repeated sessions.
- Capture stable preferences and conventions from repeated corrections.

**Examples**:
- response format
- naming patterns
- default tone
- domain vocabulary
- default summarization depth

**Goal**:
- show that KF can consolidate repeated user corrections into reusable artifacts

### Phase 2: Procedure learning

**Task**:
- Observe or elicit a repeated multi-step workflow.
- Turn it into a structured procedure artifact.

**Examples**:
- monthly downloads
- recurring report preparation
- inbox triage
- recurring scheduling workflow

**Goal**:
- show that the assistant improves operationally, not just stylistically

### Phase 3: Judgment and boundary learning

**Task**:
- Learn what the user considers acceptable, risky, or in need of confirmation.

**Examples**:
- when to ask before acting
- how conservative to be in uncertain conclusions
- when to escalate
- what level of polish is "good enough"

**Goal**:
- show that KF captures not only procedure, but evaluative standards

### Phase 4: Optional tool compilation

**Task**:
- When a repeated procedure becomes precise enough, generate a tool or script for reliable execution.

**Goal**:
- demonstrate the bridge from learned user procedure to executable automation

> Compilation is governed by the safety contract in §8a. It is not an optimization the system performs on its own; it is a deliberate user-approved promotion across a phase boundary.

---

## 8a. Safety Contract For Procedure Compilation

Compilation is the point at which YAJ stops being a memory layer and starts being an automation layer. The safety properties of the system change discontinuously at that point: a Regime A artifact that produces a bad sentence is annoying; a Regime C artifact that mis-files a financial statement, overwrites a folder, or contacts a real person is not recoverable through the normal correction loop. This section defines the contract that governs that phase change.

### Three regimes

Every artifact in YAJ exists in one of three regimes. Default is A. Promotion is always explicit, reviewable, and reversible.

- **Regime A — Suggestion.** Influences LLM output only. No tool calls, no writes, no network. Failure mode is bad output; recovery is the user's next correction. Preferences, conventions, judgments, and most boundaries live here.

- **Regime B — Supervised action.** Drives tool calls, but each call passes through the assistant's normal in-loop reasoning and is visible in the session. Side effects happen, but under per-call LLM judgment, not under a pre-baked script. Procedures live here by default.

- **Regime C — Compiled execution.** The artifact has been compiled into a tool or script that can run a sequence of side-effecting steps without per-step LLM review. This is the phase change. Promotion to C is never automatic.

### Promotion gate (B → C)

Promotion requires, in this order:

1. **Provenance**: a complete trace of which sessions, corrections, and revision-triggers produced the procedure, presented to the user in human-readable form. Approval without provenance is approval of a black box and is refused.
2. **Side-effect classification**: every tool call the compiled procedure can emit is tagged as `reversible`, `reversible-with-effort`, `socially-reversible`, or `irreversible`. The classification is shown at the gate.
3. **Dry-run evidence**: at least one successful dry-run that prints intended side effects without performing them. For procedures with `irreversible` steps, dry-run is mandatory on every promotion and every staleness re-validation.
4. **Scope binding**: the compiled tool inherits the union of its source artifacts' scopes and is structurally refused execution outside them.
5. **Eligibility check**: the procedure does not touch any never-compile category (see below). If it does, promotion is refused and the procedure stays in Regime B.
6. **Explicit user approval**: a deliberate one-time action, not a click-through.

### Never-compile categories

Some artifact scopes are **structurally ineligible** for Regime C, regardless of how stable or well-tested they appear. These stay in Regime B forever:

- anything carrying a `boundary` artifact
- anything touching credentials, secrets, or authentication flows
- anything that writes to a financial system
- anything that sends communication to an external recipient
- anything tagged with a sensitive scope per `security.md`

This list is intentionally rigid. The most compelling automation demos in this domain are exactly the ones whose failures are unrecoverable, and the only honest response is to refuse to compile them. They can still run in Regime B with full assistant supervision; they cannot run unattended.

### Properties that hold inside Regime C

Compiled tools that pass the gate must:

- **Carry scope at runtime.** A tool compiled in `personal` refuses to run when the assistant is operating in `work`, even if the planner selects it.
- **Resolve secrets at run time** from a vault. Secrets are never inlined into the compiled artifact.
- **Be idempotent or carry an explicit rollback** for every `reversible` and `reversible-with-effort` step. Steps that are `socially-reversible` or `irreversible` cannot be batched without per-invocation confirmation.
- **Honor a staleness budget.** Compiled tools expire on a schedule, and on observed environment change (e.g., a target site's structure shifts, an upstream API version changes). Expired tools require re-validation before next use, including a fresh dry-run.
- **Log every invocation** in a user-inspectable audit log, with enough context to undo `reversible` actions after the fact.
- **Respect a global kill switch.** A single user action disables all Regime C tools system-wide, immediately.

### Composition boundary

Two individually approved Regime C tools chained by the planner can produce a *composition* whose risk exceeds either component (e.g., download statements → summarize → email summary). When the planner chains compiled tools across a side-effect boundary, the **composition itself** requires confirmation, distinct from the per-tool approvals. Composition approvals are not cached across sessions.

### The approval economy

Confirmation is a finite resource. If the system asks "are you sure?" on every run, users click through and the safety layer becomes theater. The contract therefore budgets approval interruptions explicitly:

- Promotion: always confirmed.
- Staleness re-validation: always confirmed.
- Scope or composition boundary crossing: always confirmed.
- Routine invocation of an approved, in-budget, in-scope, non-stale Regime C tool whose steps are all `reversible`: **not** confirmed.
- Any step classified `irreversible`: confirmed every time, no exceptions, no "remember this choice."

Approval fatigue is treated as a safety failure mode in its own right, not as a UX nuisance.

### What this costs

This contract makes some demos impossible. A "YAJ sends my monthly summary email automatically" headline demo is precisely the thing the contract refuses. That is the intended outcome. The YAJ thesis is that explicit, user-owned personalization is more valuable *because* it is governed; a YAJ that optimizes for impressive autonomous side effects is a different product and should not borrow this one's safety story.

---

## 9. What Makes YAJ Distinctive Under KF

YAJ should emphasize that KF gives the assistant capabilities that most personalization systems do not combine well:

- **runtime learning**
- **explicit artifact storage**
- **human-readable adaptation**
- **portable knowledge across model changes**
- **revision and deletion**
- **governance over what has been learned**
- **optional compilation of repeated procedures into tools**

The strongest claim is:

**YAJ turns personalization from a hidden vendor-side memory feature into an explicit user-owned asset.**

That is the core differentiator.

---

## 10. Example KF Artifacts For YAJ

### Preference artifact

```json
{
  "artifact_type": "preference",
  "scope": "writing_style",
  "condition": "When summarizing for the user",
  "action": "lead with the actionable takeaway, prefer bullets, minimize hedging",
  "rationale": "The user values concise communication and quick extraction of decisions."
}
```

### Convention artifact

```json
{
  "artifact_type": "convention",
  "scope": "file_naming",
  "condition": "When saving monthly financial statements",
  "action": "use the format YYYY-MM-institution-account.pdf",
  "rationale": "The user uses chronological sorting and institution grouping for later retrieval."
}
```

### Procedure artifact

```json
{
  "artifact_type": "procedure",
  "scope": "monthly_statement_downloads",
  "steps": [
    "Check whether the current month folder exists",
    "Download Chase statement",
    "Download Fidelity statement",
    "Download Amex statement",
    "Rename files using the stored naming convention",
    "Place files into the current month folder"
  ]
}
```

### Boundary artifact

```json
{
  "artifact_type": "boundary",
  "scope": "external_actions",
  "condition": "When a task affects external people or sends a message outside the organization",
  "action": "ask for explicit confirmation before proceeding",
  "rationale": "The user prefers conservative control over outward-facing actions."
}
```

---

## 11. Evaluation Plan

The YAJ use case should make a narrow and defensible claim first.

### Primary claim

KF can make an assistant measurably more aligned to one user's recurring patterns, conventions, procedures, and judgment over time, without fine-tuning the underlying model.

### Baselines

- Assistant with no memory
- Assistant with naive session-history carryover
- Assistant with raw retrieved notes
- Assistant with KF artifacts

### Metrics

- reduction in repeated user corrections
- task completion quality over repeated sessions
- consistency with user conventions
- correctness of procedure execution
- number of interactions needed before a reusable artifact is formed
- user editability and auditability of what was learned

### Strong first success criterion

A convincing first result would be:

- fewer repeated corrections than the baselines
- better adherence to naming, formatting, and task conventions
- one or more procedures improved or partially automated through learned artifacts
- all improvements achieved without retraining the base model

---

## 12. Risks And Boundaries

This use case also has real risks:

- **category vagueness**: "Jarvis" can sound generic and overhyped
- **attention dilution**: a broad assistant story can become unfocused
- **evaluation fuzziness**: personalization quality is harder to benchmark than closed tasks
- **privacy sensitivity**: personal knowledge must be stored and governed carefully
- **overclaim risk**: the assistant will still have limits even if it learns better

So YAJ should be framed as:

- a KF flagship for user-owned personalization
- not as a magical general assistant

---

## 13. Implementation Sketch

Suggested repo structure:

```
usecases/
  YAJ/
    README.md                 ← this design spec
    python/
      harness.py              ← session-based evaluation harness
      ensemble.py             ← runtime KF orchestration for assistant tasks
      rules.py                ← shim for RuleEngine
      tools.py                ← shim for ToolRegistry
      scenarios/
        monthly_reports.json
        inbox_triage.json
        outbound_messages.json
```

Suggested `dataset_tag`:

```text
yaj
```

Suggested first benchmark scenarios:

1. repeated writing-style and formatting corrections
2. recurring file naming and organization workflow
3. recurring multi-step administrative procedure
4. boundary-sensitive action where confirmation preference matters

---

## 14. Why This Matters To Vendors

YAJ also has vendor relevance.

Assistant vendors are all competing on similar dimensions:

- model quality
- tool access
- UI polish
- latency

But a large part of user value comes from a harder thing:

**how well the assistant becomes *your* assistant over time**

KF gives vendors a different story:

- personalization without retraining
- portable user-owned knowledge
- explicit governance and deletion
- lower switching friction across base models
- deeper user retention through accumulated adaptation

That is strategically important because personalization can become a durable moat only if it survives model churn and stays under user control.

---

## 15. Recommended Next Step

The next practical step is to build one narrow, visible, repeatable YAJ scenario:

**show an assistant learning a user's formatting conventions, naming scheme, confirmation boundary, and one repeated procedure over several sessions, with all learned artifacts visible and editable.**

That would be enough to demonstrate:

- why YAJ is more than a generic assistant
- why KF is the right architecture for personalization
- why user-owned adaptation could matter in the market
