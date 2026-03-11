# Spec: Expert-to-Agent Dialogic Learning

## Purpose

This document specifies a learning pattern in which a [Knowledge Fabric](../docs/glossary.md#knowledge-fabric-kf) agent acquires deep, reusable knowledge from a human domain expert through structured dialogue.

The goal is not merely to record what the expert says. The goal is to help the agent learn [knowledge artifacts](../docs/glossary.md#knowledge-artifact) that are transferable, bounded, revisable, and useful in future cases.

This spec is written for two audiences:
- developers who may implement the mechanism
- educated readers who want to understand the core idea without needing to read the source code first

## Project Context

This spec is part of a six-phase development roadmap. A brief orientation:

- **Phase 1 — Personal Knowledge Store** ✅ complete: the agent passively accumulates facts, preferences, and procedures from ordinary conversation, and injects relevant knowledge into future sessions.
- **Phase 2 — Generalization**: automatic consolidation of patterns observed across multiple sessions into reusable general rules.
- **Phase 3 — Procedural Learning**: structured learning of repeatable multi-step procedures from interaction.
- **Phase 4 — Expert-to-Agent Dialogic Learning** ← this spec: active, structured elicitation of deep knowledge from a domain expert through purposeful back-and-forth dialogue.
- **Phase 5 — Portability**: exporting and importing knowledge packages across agents and platforms, enabling knowledge to be shared and sold.
- **Phase 6 — Governance**: provenance, audit trails, and lifecycle management at organizational scale.

Phase 4 builds directly on Phase 1's knowledge store, artifact types, and relation graph. It does not require Phases 2 or 3 to be complete first. Where this spec refers to "Phase 1" it means the passive accumulation pipeline; where it refers to "Phase 4" it means the active dialogic learning mechanism described here.

For the full roadmap with milestone details see [docs/roadmap.md](../docs/roadmap.md).

## Why This Matters

Many systems can store facts or summarize conversations. That is not enough. An expert is valuable not only because they know many facts, but because they know how to judge, how to prioritize, how to notice exceptions, and how to revise their thinking when evidence changes.

A large share of valuable human knowledge is tacit. It does not exist as a clean manual waiting to be retrieved. It lives in examples, corrections, edge cases, habits of attention, and the reasons an expert gives when they say a tempting conclusion is wrong. If an agent cannot learn that layer, then it will remember surface facts while missing the deeper know-how that actually matters.

Dialogue is one of the best ways to surface this kind of knowledge because experts often reveal what they know while explaining, correcting, contrasting cases, and reacting to overgeneralization. Expert-to-agent [dialogic learning](../docs/glossary.md#dialogic-learning) is meant to capture that deeper layer. The agent learns not just answers, but methods, boundaries, failure modes, and standards of judgment.

## Why We Think This Approach Can Succeed

The case for this approach is not that no one else is working on memory, teachable agents, or dialogic systems. Many are. The case is that this particular combination may solve an important part of the problem more effectively than several adjacent approaches on their own.

First, it turns what is learned into explicit [knowledge artifacts](../docs/glossary.md#knowledge-artifact) rather than leaving it buried in conversation history or implicit in model behavior. That makes the result inspectable, revisable, and reusable.

Second, it is better suited than full fine-tuning for many person-specific, team-specific, and expert-specific forms of adaptation. Fine-tuning is powerful, but it is slow to revise, expensive to iterate, and poorly matched to the incremental teaching of changing conventions, judgment criteria, and boundary conditions.

Third, it gives the system a credible path to practical value even before general intelligence is solved. If the agent can reliably learn a domain experts own procedures, judgment patterns, and revision triggers in an inspectable form, then the result can already be useful in power-user, team, and high-accountability settings.

This does not guarantee success. The main risks remain real: extracting the right abstractions, validating them, avoiding overgeneralization, and proving practical value in live use. But the reason to pursue this path is that it targets a gap that other mechanisms often leave open: the accumulation of generalized know-how through guided interaction.

## Core Idea

The agent learns best when it does not behave like a passive note-taker. It should instead behave like a careful student: asking for concrete examples, proposing tentative generalizations, testing those generalizations, and asking when they fail.

In other words, the dialogue itself is part of the learning mechanism.

## Design Principles

- Start from concrete cases before abstracting.
- Distinguish general rules from one-off anecdotes.
- Seek boundaries, exceptions, and failure modes.
- Treat revision as part of learning rather than as a sign of failure.
- Store knowledge in forms that remain human-readable and inspectable.
- Prefer generalized procedures and judgments over raw transcripts.

## Scope

This spec is domain-general. It can apply to investing, debugging, legal reasoning, operations, medicine, or any other domain where an expert has learned stable patterns of thought.

For explanatory purposes, examples in this spec assume a long-term fundamental investing expert, because that domain is rich enough to show judgment, strategy, procedure, and revision, while still being understandable to a broad audience.

## What Counts As Deep Knowledge

For the purposes of this system, deep knowledge includes:
- procedures: how to do something
- judgments: what counts as good, bad, attractive, dangerous, or insufficient
- strategies: how to approach a class of situations
- boundaries: when a rule does not apply
- revision triggers: what evidence should change the conclusion
- failure cases: past mistakes that refine future judgment

## The Learning Problem

A raw expert conversation contains several different things mixed together: anecdotes, facts, habits, domain language, emotional reactions, and general methods. A useful learning system must separate these strands instead of treating them as equal.

The agent therefore needs to do more than ask open-ended questions. It needs to ask questions for specific learning purposes.

## Dialogic Learning Pattern

The core pattern is iterative:

1. ask for a concrete case
2. extract the expert's reasoning process
3. propose a tentative generalization
4. ask for corrections, exceptions, and failure modes
5. refine the generalized rule
6. store the refined rule with scope and caveats

This pattern can repeat at many levels, from small habits to broad strategic principles.

## Agent Question Taxonomy

The agent should internally classify its questions by learning objective. This taxonomy should remain open-ended rather than fixed.


A mature implementation should allow new question types to be added during learning itself. For example, an expert may explicitly tell the agent that a certain kind of question is necessary for learning well in that domain. When that happens, the system should be able to record the new question type, its purpose, and an example of when to use it.

The goal is not to lock the agent into a frozen list of question classes. The goal is to give it a growing repertoire of effective learning moves.
### 1. Case Elicitation Questions
Used to obtain a real example.
Example: Walk me through a real investment you seriously considered.

### 2. Process Extraction Questions
Used to recover sequence and decision logic.
Example: What did you check first, and what made you continue instead of rejecting it early?

### 3. Priority Questions
Used to determine which signals matter most.
Example: If you had only ten minutes, what are the first three things you would check?

### 4. Abstraction Questions
Used to separate the general method from the details of one case.
Example: What part of this is specific to this company, and what part reflects your general method?

### 5. Boundary Questions
Used to determine where a rule stops applying.
Example: When does this rule fail? What kind of business makes it unreliable?

### 6. Counterexample Questions
Used to test whether the rule survives contact with difficult cases.
Example: Can you think of a case that looked similar but where this reasoning would have been wrong?

### 7. Revision Questions
Used to learn how the expert changes their mind.
Example: What evidence would make you abandon this thesis?

### 8. Transfer Questions
Used to test whether knowledge applies beyond the original case.
Example: Would this approach still work in another sector or in a weaker market?

### 9. Confidence Questions
Used to calibrate certainty and scope.
Example: Is this one of your strongest rules, or more of a useful tendency that sometimes works?

## Why The Questions Matter

A good question does not just gather more text. It reduces uncertainty about one of the following:
- what the expert actually does
- why they do it
- how generally it applies
- what would make it fail
- what would make them revise it

An implementation should therefore track not just the content of answers, but also which gaps in knowledge each question is trying to close.

## Question Selection

### Default Selection Algorithm

Question selection follows a deterministic priority order based on which consolidation gaps remain open for the current candidate rule. This keeps most session turns cheap — no LLM call is needed to choose the next question.

Default priority order:

1. If no concrete case has been provided → ask a case-elicitation question.
2. If a case exists but no reasoning sequence has been extracted → ask a process-extraction question.
3. If a sequence exists but no generalized rule has been proposed → ask an abstraction question.
4. If a rule has been proposed but no boundary has been tested → ask a boundary question.
5. If a boundary has been noted but no exception or failure mode has been recorded → ask a counterexample question.
6. If no revision trigger has been captured → ask a revision question.
7. If confidence has not been calibrated → ask a confidence question.
8. If all five consolidation criteria are met → synthesize: propose the rule for expert correction rather than asking another question.

Transfer questions are optional and should be used when the agent has reason to believe the rule may apply beyond the domain in which it was learned.

### LLM-Backed Question Selection

LLM selection is used in three situations:

- The expert's last answer was rich enough that it may have closed multiple gaps at once. The agent should call the LLM to re-assess gap status before choosing the next question type.
- The rule-based priority would select a question type already asked about this candidate rule, with an inconclusive answer. The agent should use the LLM to craft a better-targeted version rather than repeating the same question mechanically.
- The session has multiple candidate rules in progress simultaneously and the agent must decide which to advance. The agent should call the LLM to select the most important open thread.

### Preventing Repetition

Before selecting a question type, the agent checks the question history for the current candidate rule. If that type was already asked and the corresponding gap is still open, the agent should rephrase rather than re-ask. If rephrase attempts exceed two for the same gap, the agent should record the gap as unresolved and move forward rather than asking a third time.

### Cost Guidance

- Default path using rule-based selection: zero additional LLM calls per turn for question selection.
- Complex path using LLM-assisted selection: one LLM call per turn.
- Synthesis proposal: one LLM call per candidate rule.
- Expert correction parsing: one LLM call per correction.
- Gap detection, meaning whether an answer filled one of the five consolidation gaps, can be batched with the extraction call for that turn.

Target: ten to twenty LLM calls for a focused session of four to six turns per candidate rule.

## Working Loop

A practical implementation should follow a controlled loop rather than an unstructured conversation.

1. Select a learning objective.
Example: learn how this expert filters investment opportunities.

2. Ask the next best question based on current knowledge gaps.

3. Parse the answer into candidate knowledge units.
These may include case facts, procedures, judgments, priorities, boundaries, and revision triggers.

4. Check what is still missing.
The system should ask: do we have a concrete example? a general rule? an exception? a revision trigger? a confidence estimate?

5. Ask the next question that closes the most important gap.

6. Periodically synthesize.
The agent should restate what it believes it has learned and let the expert accept, refine, or reject that synthesis.

7. Consolidate only when the minimum learning criteria are met.

## Minimum Consolidation Criteria

A candidate rule should not be treated as deep learned knowledge unless the dialogue has captured at least:
- one concrete case
- one generalized restatement
- one scope or boundary statement
- one exception, failure mode, or counterexample
- one revision trigger or disconfirming signal

This prevents the system from mistaking a polished anecdote for a robust principle.

## What The Agent Should Store

The system should store structured [artifacts](../docs/glossary.md#knowledge-artifact) rather than only keeping the full transcript.

Typical outputs include:
- a procedural artifact describing a repeatable method
- a judgment artifact describing what counts as good or dangerous
- a strategy artifact describing an approach to a class of situations
- a boundary artifact describing when the knowledge should not be applied
- a revision artifact describing what evidence should change the conclusion
- a failure artifact describing a past mistake that refined later judgment

## Artifact Type Mapping

Phase 4 produces six artifact types. Three map directly to existing `KnowledgeKind` values defined in Phase 1. Three are new and require extending the kind taxonomy.

### Existing Kinds Reused By Phase 4

- `procedure` — a repeatable method or sequence. Maps directly to the Phase 1 kind of the same name.
- `judgment` — an evaluative principle or quality criterion. Maps directly to the Phase 1 kind of the same name.
- `strategy` — a general approach to a class of situations. Maps directly to the Phase 1 kind of the same name.

### New KnowledgeKind Values

The following three values should be added to `KnowledgeKind` in `types.ts`. They are produced only through dialogic sessions and are not created by passive Phase 1 extraction.

- `boundary` — a statement of when a rule does not apply. Example: "this survivability check is weaker in commodity-driven businesses where balance sheet structure reflects industry norms rather than fragility."
- `revision-trigger` — a condition that should cause the expert to revise or abandon a conclusion. Example: "downgrade the thesis if confidence depends mainly on management promises rather than observable structure."
- `failure-case` — a past mistake that refined the expert's later judgment. Example: "applied the fragility rule too mechanically and rejected a business that was structurally sound."

### Relations Between Artifact Types

Dialogic artifacts are typically linked using the relation types already defined in Phase 1:

- A `boundary` artifact `constrains` the `judgment` or `procedure` it limits.
- A `revision-trigger` artifact `supports` the `procedure` or `judgment` it qualifies.
- A `failure-case` artifact `supersedes` the earlier incorrect `judgment` or `procedure` it corrected.

A complete dialogic learning outcome for one topic typically produces: one `procedure`, one or two `judgment` artifacts, one or two `boundary` artifacts, one `revision-trigger`, and zero or one `failure-case`. These are linked via the relation graph into a coherent knowledge cluster.

### Session Provenance

All artifacts produced in a dialogic session carry provenance in the form `session:<session-id>` so their origin is traceable back to the specific exchange that produced them. The session transcript is the authoritative audit record for the artifact.

## Session Model

A `DialogueSession` tracks the full state of one expert learning session from start to finish. Sessions are persisted at `~/.openclaw/knowledge/sessions/<session-id>.json`. They are retained permanently after completion as the audit record.

### Core Fields

- `id` — unique identifier for the session.
- `objective` — the declared learning goal, stated in natural language. Example: "learn how this expert screens investment opportunities before committing research time."
- `domain` — the topic area, used to match this session to prior sessions in the same field. Example: `long-term-fundamental-investing`.
- `stage` — the current phase of the session. See stages below.
- `createdAt` and `lastActiveAt` — ISO 8601 timestamps.
- `turns` — the full transcript in order. See turn structure below.
- `candidateRules` — the rules being developed in this session, each with a gap status record. See candidate rule structure below.
- `questionHistory` — which question types were asked during this session and for which candidate rule, used to prevent repetition.
- `artifactIds` — the IDs of artifacts that were promoted to the main store at session end.
- `priorSessionIds` — IDs of earlier sessions in the same domain, used for multi-session continuity.
- `inheritedArtifactIds` — IDs of artifacts loaded from prior sessions that informed this session's starting state.
- `customQuestionTypes` — question types added by the expert during this session. See custom question type structure below.

### Session Stages

- `eliciting-case` — asking for a concrete example; no rule yet proposed.
- `extracting-process` — unpacking the expert's reasoning sequence.
- `abstracting` — generalizing from the case to a tentative rule.
- `testing-boundaries` — asking for exceptions, failures, and limits.
- `synthesizing` — agent proposes a rule for the expert to correct.
- `complete` — session ended; all candidate rules have been either consolidated or archived as incomplete.

### Candidate Rule Fields

Each candidate rule being developed during the session has:

- `id` — unique identifier within the session.
- `content` — the current best statement of the rule, updated as corrections are applied.
- `kind` — the expected `KnowledgeKind` of the artifact when consolidated.
- `gaps` — a record of which of the five consolidation criteria have been met. A candidate rule is ready for synthesis when all five are satisfied.
- `relatedTurnIds` — the turn IDs that contributed evidence to this rule.

### Consolidation Gap Status Fields

The five consolidation criteria are tracked as individual boolean fields:

- `hasConcretCase` — at least one concrete real-world example has been provided.
- `hasGeneralizedRestatement` — the agent has proposed a generalized version and the expert has accepted or corrected it.
- `hasScopeOrBoundary` — a scope statement or boundary condition has been captured.
- `hasExceptionOrFailureMode` — at least one exception, counterexample, or failure mode has been recorded.
- `hasRevisionTrigger` — at least one condition that would cause the expert to revise the rule has been stated.

### Dialogue Turn Fields

Each turn in the session transcript records:

- `turnId`, `role` (agent or expert), `content`, and `timestamp`.
- `questionType` — for agent turns, which question type from the taxonomy was used.
- `candidateRuleId` — which candidate rule this turn was primarily advancing.
- `extractedUnits` — candidate knowledge fragments parsed from this turn.
- `correctionType` — for expert turns that contain a correction: `rule-revision`, `scope-adjustment`, or `counterexample-added`.

### Custom Question Type Fields

When an expert introduces a question type the agent should use in future:

- `id`, `name` (short label), `purpose` (what learning gap it addresses), `exampleQuestion`, `addedAt`, and `addedBySessionId`.

Custom question types are stored in the session and propagated to future sessions in the same domain.

### Session End

A session ends when the user explicitly closes it, all candidate rules are consolidated or archived as incomplete, or the session has been inactive beyond a configurable timeout (default: 24 hours).

On session end:
1. All candidate rules that satisfy all five consolidation criteria are promoted to `artifacts.jsonl` with provenance `session:<session-id>`.
2. Incomplete rules remain in the session file with their gap status noted, available for continuation in a future session.
3. The session file is retained permanently as the audit record.

## Multi-Session Continuity

### Starting A New Session In The Same Domain

When a new session is started with a domain that matches a prior session, the agent:

1. Loads all artifacts whose provenance links them to prior sessions in that domain.
2. Presents a brief knowledge summary: a restatement of what it believes it has learned from prior sessions in that domain.
3. Asks the expert to confirm or correct the summary before the new session proceeds.

This prevents the agent from re-asking questions already well-answered in prior sessions, and gives the expert an opportunity to flag outdated or incorrect prior artifacts before new knowledge is built on top of them.

### Gap Inheritance

When selecting the next question type at the start of a new session, the agent checks whether the corresponding gap is already closed by a prior artifact:

- If a solid `procedure` artifact already exists for the current topic, skip case-elicitation and process-extraction questions for that procedure unless the expert signals the procedure has changed.
- If `boundary` artifacts already exist for a rule, skip boundary questions for that rule.
- If a `revision-trigger` artifact already exists for a rule, skip revision questions for that rule.

### Depth Over Breadth

The agent should default to deepening existing knowledge — testing known rules against new cases, refining boundary conditions, adding failure examples — rather than opening new threads. A new topic thread should only be introduced when the expert explicitly introduces a new case or objective.

### Session Linking

Each session records `priorSessionIds` and `inheritedArtifactIds` so the provenance chain across sessions is fully traceable.

## Example Domain Framing

In a long-term fundamental investing dialogue, the agent may learn things such as:
- procedure: check balance sheet survivability before spending time on upside stories
- judgment: a cheap stock is unattractive if survival depends on heroic assumptions
- strategy: prefer businesses where downside can be bounded with ordinary reasoning
- boundary: this rule is weaker in commodity-driven businesses
- revision trigger: downgrade the thesis if confidence depends mainly on management promises

## Effective Synthesis Behavior

The agent should regularly say things like:
- Here is the rule I think you are using. Please correct me if I have made it too broad.
- It sounds as though this applies mainly to stable cash-generating businesses. Is that right?
- I may be overgeneralizing from one case. Can you give me an example where this would fail?

This kind of explicit synthesis is important because it makes the expert an active critic of the agent's emerging model.

## Failure Modes To Avoid

An implementation should actively avoid the following patterns:
- endless open-ended questions with no clear learning objective
- storing anecdotes as if they were general rules
- storing broad rules without boundaries or exceptions
- assuming expert confidence is the same as evidence
- treating revision as inconsistency rather than progress
- optimizing for a pleasant interview instead of usable knowledge

## Integration Notes

### Phase 1 Passive Extraction During A Session

When a `DialogueSession` is active, Phase 1 passive extraction via `processMessage()` is suspended for messages that are part of the session dialogue. The reason is that during a teaching session the expert's messages are processed by the Phase 4 pipeline, which applies the minimum consolidation criteria before any artifact is promoted. Running Phase 1 extraction in parallel would produce shallow candidate artifacts from half-formed rules that have not yet passed those criteria.

Phase 1 extraction resumes automatically when the session ends.

Exception: if the expert's message is clearly out of scope for the session domain — for example, a preference about communication style rather than the domain being taught — Phase 1 may capture it as a normal candidate. When in doubt, Phase 4 takes precedence.

### Expert Correction Processing

A correction is detected when the expert explicitly rejects a synthesis proposal, qualifies a rule the agent has proposed, or contradicts an answer given earlier in the session.

When a correction is detected:

1. The prior candidate rule content is archived in its evidence array as a superseded version.
2. The correction is parsed by the LLM into a revised rule statement.
3. The gap status field `hasGeneralizedRestatement` is reset — the new restatement must be re-validated against the expert before the rule can be promoted.
4. A `supersedes` relation is prepared linking the new rule to the old.
5. The correction turn is tagged in the transcript with a `correctionType` value.

If the rule was already promoted to the main store from a prior session, the correction produces a new artifact with a `supersedes` relation to the old one. The old artifact is marked `retired: true`. The full audit trail is preserved in both the session transcript and the artifact relation graph.

### Mode Invocation

Dialogic sessions are started explicitly by the user, not inferred from conversation tone. This is important: automatic detection of teaching intent is unreliable and risks opening sessions during normal conversation.

A session should begin with an explicit command that specifies a learning objective and domain. Example:

```
/teach "how you screen investment opportunities" investing
```

The explicit invocation model also means the user retains control over when knowledge elicitation is active, which is appropriate for a user-owned, local-first system.

## Readability And Communication Guidance

Because this mechanism may also be explained to non-developers, examples and outputs should remain readable. A good demonstration should use only a few concrete cases and keep the focus on the expert's reasoning moves rather than on excessive domain jargon.

A useful format for public-facing writeups is:
- short framing of what the agent is trying to learn
- the dialogue itself
- commentary explaining why each important question worked
- a short list of the artifacts the system would store

## Success Criteria

This mechanism is working well when:
- the agent asks questions that progressively narrow uncertainty
- the agent can distinguish case-specific detail from general method
- the agent explicitly surfaces boundaries and revision triggers
- the expert meaningfully corrects the agent's tentative generalizations
- the resulting artifacts are reusable, inspectable, and more useful than a transcript alone

## Relation To PIL

This mechanism is consistent with the general principles of [PIL](../docs/glossary.md#pil-persistable-interactive-learning). It assumes that knowledge is acquired through interaction, generalized into reusable artifacts, revised through feedback, and stored in a user-owned inspectable form.

The exact implementation may evolve, but the learning pattern described here should remain valid even as storage, prompting, or runtime components change.

## Suggested Follow-On Document

A natural companion to this spec would be a worked example such as: learning long-term investing judgment from an expert investor. That example should include the dialogue, commentary on why the agent's questions were effective, and the final artifacts that KF would store.

## See Also

- Worked example: [Expert-to-Agent Dialogic Learning With An Investment Expert](./expert-to-agent-dialogic-learning-example-investing.md)
- Positioning: [Expert-to-Agent Dialogic Learning In The Current Landscape](../docs/dialogic-learning-positioning.md)

## Summary

Expert-to-agent dialogic learning is the process by which an agent learns deep knowledge from an expert through structured dialogue. The central idea is simple: the agent learns effectively not by asking for more information in general, but by asking the right question at the right moment in order to surface procedures, judgments, boundaries, and revision rules that can later be reused. 
