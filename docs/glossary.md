 
# Glossary 
 
## PIL (Persistable Interactive Learning) 
 
[PIL](../docs/glossary.md#pil-persistable-interactive-learning) is the core idea behind this repository: an agent learns from interaction, turns what it learns into reusable knowledge artifacts, and applies that knowledge later under confidence-aware controls. 
 
## Knowledge Fabric (KF) 
 
[Knowledge Fabric](../docs/glossary.md#knowledge-fabric-kf), or KF, is the broader knowledge layer built around PIL. In plain terms, it is the mechanism by which knowledge is elicited, generalized, stored, revised, and reused across sessions or agents. 
 
## Knowledge Artifact 
 
A [knowledge artifact](../docs/glossary.md#knowledge-artifact) is a stored unit of reusable knowledge. It may capture a fact, preference, procedure, judgment, strategy, boundary condition, or failure case. 
 
## Generalized Knowledge Artifact 
 
A [generalized knowledge artifact](../docs/glossary.md#generalized-knowledge-artifact) is an artifact that captures a pattern or rule extending beyond a single episode. It is more reusable than a raw transcript excerpt or one-off memory. 
 
## Procedure Artifact 
 
A [procedure artifact](../docs/glossary.md#procedure-artifact) describes how to do something in a repeatable way. In some contexts this may later be compiled into code, but the canonical form remains human-readable. 
 
## Judgment Artifact 
 
A [judgment artifact](../docs/glossary.md#judgment-artifact) captures an evaluative principle, such as what counts as good evidence, a dangerous pattern, or an attractive opportunity. 
 
## Strategy Artifact 
 
A [strategy artifact](../docs/glossary.md#strategy-artifact) captures a general approach to a class of problems rather than a single procedure for one task. 
 
## Boundary Artifact 
 
A [boundary artifact](../docs/glossary.md#boundary-artifact) describes when a rule, strategy, or procedure should not be applied, or when its confidence should be reduced. 
 
## Revision Trigger 
 
A [revision trigger](../docs/glossary.md#revision-trigger) is a condition or kind of evidence that should cause the agent to revise or abandon a previously learned rule. 
 
## Failure Artifact 
 
A [failure artifact](../docs/glossary.md#failure-artifact) records a past mistake, failure mode, or misleading case that helps refine future judgment. 
 
## Domain Adapter 
 
A [domain adapter](../docs/glossary.md#domain-adapter) is the domain-specific layer that exposes tasks, renders domain inputs, and provides validation or support utilities without hardcoding domain-specific solution logic into the core runtime. 
 
## Dialogic Learning 
 
[Dialogic learning](../docs/glossary.md#dialogic-learning) is learning through structured back-and-forth exchange rather than passive observation alone. In this repo, it refers to an agent learning through purposeful dialogue with a user or expert. 
 
## Expert-to-Agent Dialogic Learning 
 
[Expert-to-agent dialogic learning](../specs/expert-to-agent-dialogic-learning.md) is the specific pattern in which an agent learns deep, reusable knowledge from a domain expert through carefully structured questioning, synthesis, correction, and consolidation. 
 
## Natural-Language DSL 
 
A [natural-language DSL](../docs/glossary.md#natural-language-dsl) is a constrained, well-defined pseudo-code style expressed in ordinary language. It is designed to be readable by humans and consistently interpretable by an LLM. 
 
## Validation Record 
 
A [validation record](../docs/glossary.md#validation-record) captures the outcome of testing a candidate rule, solution, or generalization against one or more cases. 
