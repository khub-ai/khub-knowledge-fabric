# Planning Notes: KF Productization Direction

As of April 8, 2026.

## Purpose

This note captures the current product-planning view of Knowledge Fabric (KF) as it evolves toward a more productized implementation. It is meant to guide prioritization, product positioning, and design tradeoffs, with Dialogic Learning treated as one of KF's most important differentiating capabilities rather than as the product itself.

This is not the main technical specification. It is a planning companion to the deeper design and Dialogic Learning materials already in the repository.

## Working Definition

In KF, Dialogic Learning is the process by which an AI system learns reusable know-how through structured dialogue with a human expert or a stronger model. The goal is not only to remember what was said, but to convert teaching interactions into governed knowledge artifacts such as procedures, judgments, boundaries, revision triggers, and failure cases.

## Role Of Dialogic Learning Within KF

KF should be framed as a broader knowledge layer for AI systems: a way to capture, structure, govern, reuse, and eventually transfer learned knowledge across sessions, workflows, and agents.

Within that broader product vision, Dialogic Learning matters because it is one of the strongest ways for KF to acquire high-value knowledge that is otherwise difficult to extract from documents or ordinary memory systems. It is therefore a core feature and a likely differentiator, but not the full scope of KF.

## Why This Matters Now

The market signal appears strong even if the category name is still emerging.

- Enterprise AI adoption is high, but scaled value still lags.
- Many organizations can deploy models, but still struggle to make them domain-specific, trustworthy, and operationally governable.
- A large share of valuable expertise is tacit and cannot be captured well by document retrieval alone.
- Public web data is becoming less available, which increases the value of learning from proprietary human interaction.
- Organizations increasingly need AI systems that are auditable, revisable, and usable in high-accountability workflows.

The practical gap is clear: many systems can store context, summarize chat, or retrieve documents, but fewer systems are designed to turn expert dialogue into reusable, inspectable, portable capability.

## Market Need Assessment For KF

KF appears most valuable where knowledge is:

- tacit rather than fully documented
- judgment-heavy rather than purely factual
- proprietary or organization-specific
- changeable over time
- costly to label through traditional data pipelines
- safety-critical or accountability-sensitive

Likely high-value settings include:

- healthcare and clinical support
- finance and investment workflows
- legal and compliance review
- manufacturing and operations
- security analysis
- technical support and expert troubleshooting
- internal analyst and operator training

## Potential Benefits For KF

If KF executes this well, the combination of artifact-based knowledge capture, governed memory, and Dialogic Learning could offer the following advantages:

- Capture tacit expert knowledge that would otherwise remain trapped in individuals or teams.
- Turn one-off teaching into reusable knowledge assets.
- Improve domain performance without requiring full retraining for every update.
- Make expert reasoning more inspectable for review, editing, and reuse.
- Support human-in-the-loop validation as part of the product, not as an afterthought.
- Create faster onboarding paths for junior practitioners by exposing expert reasoning patterns.
- Enable a future marketplace or portability layer in which knowledge artifacts can move across agents.
- Strengthen KF's differentiation versus ordinary memory features by focusing on governed learning rather than recall alone.

## Things To Watch Out For

Dialogic Learning is promising, but it is only one part of whether KF becomes a strong product.

- Human-AI collaboration does not always outperform the better of human-only or AI-only performance.
- Experts can teach outdated practices, local biases, or overconfident heuristics.
- Stronger models can also act as flawed teachers if their reasoning is persuasive but wrong.
- Dialogue can cause anchoring, imitation, or authority-following instead of genuine abstraction.
- Informal chat is difficult to audit unless the system stores provenance, evidence, and revision history.
- Different experts may disagree, requiring conflict handling and governance.
- Sensitive information may appear in teaching sessions, especially in regulated domains.
- A system optimized for "agreement with teacher" can become sycophantic rather than truth-seeking.
- Overclaiming is a strategic risk; many adjacent products already provide memory, persistence, or personalization.

## Product Implications For KF

The product opportunity is not "an AI that chats with experts." The stronger framing is:

KF can become a knowledge infrastructure layer that helps AI systems accumulate reusable, auditable, portable know-how.

That suggests several product implications:

- The primary unit of value should be the knowledge artifact, not the transcript.
- KF should support multiple knowledge acquisition paths, with Dialogic Learning as a flagship path rather than the only path.
- Provenance, scope, revision, and review should be first-class features.
- The system should reward learning quality, not just conversation smoothness.
- Dialogic sessions should be explicit workflows with clear objectives and outputs.
- The product should support both human experts and stronger-model teachers, but keep their provenance distinct.
- Product success should be measured by transfer and reuse, not only by session satisfaction.

## Near-Term Product Direction

For KF's next planning horizon, the most defensible direction appears to be:

1. Focus KF on high-value workflows where governed reusable knowledge matters more than generic chat convenience.
2. Use Dialogic Learning as a flagship capability for acquiring tacit expert knowledge, especially in expert-facing and high-accountability workflows.
3. Treat image-classification and similar use cases as proving grounds for a broader KF learning and governance pattern.
4. Standardize the session loop so teaching produces consistent artifact types and metadata.
5. Build review and correction into the learning flow so artifacts can be challenged and revised.
6. Preserve model-agnosticism where possible so KF remains a knowledge layer rather than a model-locked feature.
7. Plan for portability early, even if full cross-agent packaging arrives later.

## Questions That Should Guide Productization

These questions should stay active as design checkpoints for KF productization:

- What is the smallest repeatable KF workflow that clearly beats ordinary chat plus notes?
- Which KF capabilities besides Dialogic Learning need to mature in parallel for the product to feel complete?
- Which artifact types create the most value first: procedures, judgments, boundaries, or revision triggers?
- How should KF distinguish "raw teaching transcript" from "accepted learned knowledge"?
- What evidence threshold should be required before a candidate rule is promoted?
- How should conflicting expert views be represented?
- When should a stronger model be allowed to teach, critique, or validate another model?
- What governance requirements are mandatory before claiming readiness for medicine, law, or finance?
- What product metrics best reflect learning quality rather than user delight alone?

## Early Success Criteria

KF is moving in a good product direction if it can show that:

- repeated sessions produce reusable artifacts rather than isolated chat logs
- learned artifacts improve performance on later tasks in the same domain
- experts can review and correct what the system learned
- provenance remains clear enough for audit and revision
- knowledge can survive beyond a single model session or tool surface

## Naming And Positioning Note

The repository filename is intentionally simple: `planning_notes.md`.

Within the document, the clearest title for now is "Planning Notes: KF Productization Direction" because it signals three things at once:

- this is a planning document rather than a final spec
- the topic is KF productization overall
- the purpose is to guide productization decisions inside KF

## Related Documents

- [Dialogic learning positioning](../docs/dialogic-learning-positioning.md)
- [Demo dialogic learning](../docs/demo-dialogic-learning.md)
- [Expert-to-agent dialogic learning spec](../specs/expert-to-agent-dialogic-learning.md)
- [Roadmap](../docs/roadmap.md)
- [Glossary](../docs/glossary.md)

## External Sources Referenced In This Planning Discussion

- [McKinsey: The State of AI, 2025](https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai/)
- [Stanford HAI: AI Index 2025, Responsible AI](https://hai.stanford.edu/ai-index/2025-ai-index-report/responsible-ai)
- [OpenAI: The State of Enterprise AI 2025](https://openai.com/business/guides-and-resources/the-state-of-enterprise-ai-2025-report/)
- [Nature Human Behaviour: Human-AI combinations in decision making](https://www.nature.com/articles/s41562-024-02024-1)
- [NIST AI RMF Generative AI Profile](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence)
- [OECD AI Principles](https://www.oecd.org/en/topics/ai-principles.html)
