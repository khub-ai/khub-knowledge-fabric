# Spec: Expert-to-Agent Dialogic Learning

## Purpose

This document specifies a learning pattern in which a [Knowledge Fabric](../docs/glossary.md#knowledge-fabric-kf) agent acquires deep, reusable knowledge from a human domain expert through structured dialogue.

The goal is not merely to record what the expert says. The goal is to help the agent learn [knowledge artifacts](../docs/glossary.md#knowledge-artifact) that are transferable, bounded, revisable, and useful in future cases.

This spec is written for two audiences:
- developers who may implement the mechanism
- educated readers who want to understand the core idea without needing to read the source code first

## Why This Matters

Many systems can store facts or summarize conversations. That is not enough. An expert is valuable not only because they know many facts, but because they know how to judge, how to prioritize, how to notice exceptions, and how to revise their thinking when evidence changes.

Expert-to-agent [dialogic learning](../docs/glossary.md#dialogic-learning) is meant to capture that deeper layer. The agent learns not just answers, but methods, boundaries, failure modes, and standards of judgment.

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
