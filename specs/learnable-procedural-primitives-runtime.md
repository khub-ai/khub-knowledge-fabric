# Spec: LLM-Centered Procedural Learning Runtime

## Purpose

This spec defines an optional PIL extension for solving a fixed known set of structured problems through an LLM-centered loop of attempt, user guidance, revision, generalization, validation, and storage.

The runtime is not designed as a benchmark-specific solver. ARC-v2 and ARC-v3 are optional adapters that provide one useful proving ground. The same runtime should also support other domains such as spreadsheets, symbolic tasks, and workflow problems.

## Core Design Decision

The runtime treats the LLM as the primary all-purpose processor. The system relies on the LLM to interpret tasks, propose solution procedures, apply stored knowledge artifacts, propose plausible generalizations, and revise its own approach based on user guidance and validation results.

Deterministic helpers may still be used, but they are subordinate tools inside an LLM-centered runtime rather than the core of the runtime itself.

## Core Rules

- Preserve PIL principles: local, inspectable, portable, artifact-based knowledge.
- Do not change default repo behavior when the runtime is not configured.
- Load ARC adapters only when explicitly requested.
- Treat the canonical model as language-independent. TypeScript, Python, or other language bindings are implementations of the model, not the model itself.
- Represent saved solutions in a well-defined natural-language DSL or pseudo-code form rather than a host-language-specific program definition.
- Save only generalized solutions that have been validated against known problems beyond the originating problem.

## Fixed-Set Operating Loop

The runtime operates over a fixed set of problems known in advance for the current run.

1. The agent is given the overall goal for the target problem set.
2. The agent uses a domain adapter to retrieve one problem from the fixed set.
3. The agent uses accumulated knowledge artifacts to attempt a solution.
4. If the attempt fails, the agent solicits user guidance and revises its solution attempt. This repeats until that problem is solved.
5. Once the problem is solved, the agent asks the LLM to produce a list of plausible generalizations of the solution.
6. The agent immediately validates those plausible generalizations against additional known problems from the fixed set.
7. The best validated generalization is saved to the store as a generalized solution artifact.
8. The runtime moves to the next unsolved problem and repeats the loop.

The runtime is complete for the current run when all problems in the fixed set are solved or explicitly marked unresolved.

## Architecture

1. Core PIL handles persistence, provenance, retrieval, lifecycle, revision, confidence, and feedback.
2. Procedural Learning Runtime handles task selection, LLM prompting, solution attempts, user-guided revision, generalization proposals, validation, and artifact promotion.
3. Domain Adapters expose the fixed problem set, provide task representations, and supply domain-specific rendering or checking utilities.
4. Benchmark Harness measures progress over the fixed known set but does not define the runtime model.

## Role Of The Adapter

A domain adapter should do only the domain-specific work needed to let the runtime operate.

Adapter responsibilities:
- enumerate or retrieve problems from the fixed set
- render a problem into the form expected by the runtime
- provide any domain-specific reference materials or views
- provide validation utilities when available
- provide problem IDs and split membership when relevant

Adapter responsibilities should not include embedding benchmark-specific solution heuristics into the runtime.

## Role Of The LLM

The LLM is responsible for:
- interpreting the current problem
- retrieving and applying prior knowledge artifacts
- writing an initial solution in the runtime DSL
- revising the solution after user feedback
- proposing plausible generalizations of a solved solution
- selecting among multiple generalization candidates
- explaining why a candidate should or should not transfer

The runtime should explicitly feed the LLM the requirement that only human-like judgments are acceptable when operating in domains such as ARC. In practice this means the prompt context should include domain-level judgment guidance rather than relying on pure pattern fitting alone.

## Knowledge Use During Solving

Before attempting a new problem, the runtime retrieves relevant artifacts from the store. These may include:
- prior generalized solutions
- prior failure cases
- user guidance artifacts
- domain-specific judgment artifacts
- previously promoted DSL procedures

The LLM should be prompted to use those artifacts explicitly when drafting the next attempt, rather than treating them as passive background text.

## Solution Representation: Natural-Language DSL

Saved solutions should be expressed in well-defined natural language pseudo-code rather than host-language-specific code. The goal is to make the solution directly readable by humans, directly usable by the LLM, and portable across implementation languages.

A solution artifact in this DSL should include the following sections when applicable:
- name
- intent
- domain
- preconditions
- steps
- decision points
- success criteria
- failure signals
- validation history
- provenance

A typical step should read like a controlled instruction rather than free-form prose. Example style: identify repeated structures, compare them for common pattern, keep the shared transformation, discard incidental variation.

The DSL should be natural-language-first but structured enough that an LLM can reliably parse, apply, critique, and generalize it.

## Canonical Records

All records below are language-independent conceptual records, not TypeScript or Python classes.

### Problem Reference
Required fields: `adapter`, `problem_id`.
Optional fields: `split`, `variant`, `sequence_index`.

### Solution Attempt
Required fields: `id`, `problem_ref`, `attempt_text`, `attempt_status`, `created_at`.
Optional fields: `used_artifact_ids`, `trace_id`, `review_notes`.

### User Guidance
Required fields: `id`, `problem_ref`, `guidance_text`, `created_at`, `author`.
Optional fields: `target_attempt_id`, `judgment_type`, `rationale`.

### Solved Procedure
Required fields: `id`, `problem_ref`, `dsl_text`, `created_at`.
Optional fields: `supporting_attempt_ids`, `supporting_guidance_ids`, `notes`.

### Generalization Candidate
Required fields: `id`, `source_procedure_id`, `candidate_text`, `created_at`.
Optional fields: `rationale`, `expected_transfer_scope`, `judgment_basis`.

### Validation Record
Required fields: `id`, `generalization_candidate_id`, `problem_ref`, `result`, `created_at`.
Optional fields: `failure_summary`, `score_summary`, `review_notes`.

### Generalized Solution Artifact
Required fields: `id`, `name`, `dsl_text`, `scope`, `created_at`.
Optional fields: `preconditions`, `validation_summary`, `source_problem_refs`, `source_guidance_ids`, `source_candidate_ids`.

### Judgment Artifact
Required fields: `id`, `domain`, `judgment_text`, `created_at`.
Optional fields: `adapter`, `source`, `priority`.

### Penalty Profile
Required fields: `id`, `name`, `complexity_weight`, `fragility_weight`, `overfit_weight`, `runtime_weight`.
Optional fields: `adapter`, `notes`, `author`.

### Penalty Decision
Required fields: `id`, `profile_id`, `decision_type`, `author`, `created_at`.
Optional fields: `target_id`, `delta_complexity`, `delta_fragility`, `delta_overfit`, `delta_runtime`, `delta_total`, `rationale`.

## Generalization Step

After a problem is solved, the runtime must ask the LLM to produce multiple plausible generalizations of the solved procedure. The runtime should not assume that the first generalization is the best one.

Each generalization candidate should attempt to separate:
- essential transformation logic
- likely domain-invariant judgment
- incidental details tied only to the originating problem

In domains such as ARC, the runtime should explicitly instruct the LLM that acceptable generalization must align with human-like judgment rather than mere benchmark-specific curve fitting. This guidance should be part of the prompt context for generalization and validation.

## Validation Before Save

A generalized solution must be immediately validated on additional known problems from the fixed set before it is saved.

Validation should answer at least three questions:
- does the generalized procedure work beyond the original problem?
- does it fail in a way that suggests overfitting or fragility?
- does it still reflect the intended human-like judgment for the domain?

The runtime should prefer a generalized solution that is slightly less broad but clearly validated over one that sounds broader but has weak transfer evidence.

## Search, Revision, And Guidance Loop

The runtime should not be framed as blind program search. It is a guided iterative learning loop.

The loop for one problem is:
1. retrieve one problem
2. retrieve relevant artifacts
3. ask the LLM for a DSL-style attempt
4. validate the attempt
5. if failed, request user guidance
6. revise the attempt using guidance and prior artifacts
7. repeat until solved
8. ask for plausible generalizations
9. validate those generalizations on other known problems
10. save the strongest validated generalized solution

## Human-Tunable Penalties

Candidate evaluation should include penalties for complexity, fragility, overfitting, and runtime cost, plus explicit human adjustment.

Those penalties should influence:
- which solution attempts are retained as useful
- which generalization candidates are promoted or rejected
- how the LLM is guided in later rounds

A penalty decision should record exactly what changed, why it changed, who changed it, and which target it affected.

## Optional ARC Adapters

ARC-v2 and ARC-v3 adapters remain optional packages. They provide problem access and domain framing, but they must not introduce hardcoded ARC-specific solving logic into the runtime core.

## Success Criteria

- Existing PIL flows remain unchanged by default.
- The LLM uses stored artifacts as active procedural knowledge, not as passive notes.
- Solved procedures are represented in a structured natural-language DSL.
- Generalizations are proposed by the LLM and validated before save.
- User guidance can iteratively improve problem solving until the fixed set is solved.
- Saved generalized solutions transfer across multiple known problems and remain inspectable by humans.
