
# Worked Example: Expert-to-Agent Dialogic Learning With An Investment Expert

## Purpose

This example shows how a Knowledge Fabric agent can learn deep, reusable knowledge from a domain expert through dialogue. It is written as a blueprint for developers, but also as a readable demonstration for broader audiences.

The example is feasible under the general principles of the Knowledge Fabric framework (PIL — Persistable Interactive Learning), even if every supporting implementation detail is not yet complete. 
 
## Domain And Participants 
 
- Expert: Mara Chen, a long-term fundamental investing expert. 
- Agent: KF, a Knowledge Fabric agent designed to learn reusable procedures, judgments, and revision rules. 
- Domain focus: deciding whether a public company is worth deeper research, with emphasis on avoiding weak businesses dressed up as cheap opportunities. 
 
## Why This Domain Works 
 
Investing is useful here because it naturally contains procedure, judgment, boundaries, and revision. To keep the dialogue readable, the example avoids technical finance jargon and focuses on how the expert thinks rather than on the details of a specific company model. 
 
## Session 1: Concrete Case Elicitation 
 
**KF:** I want to learn how you decide whether a company is worth serious study. Can you walk me through a real case where you rejected an investment early? 
 
**Mara:** Sure. A few years ago I looked at a small industrial distributor. On the surface it looked cheap. The stock had fallen a lot, the valuation looked low, and management was talking about a turnaround. 
 
**KF:** What did you check first? 
 
**Mara:** I checked whether the business could survive ordinary bad conditions without needing luck. I looked at debt maturity, how much cash it burned in a weak year, and whether customers would keep buying even in a downturn. 
 
**KF:** Why did you start there instead of asking how much upside the turnaround might have? 
 
**Mara:** Because upside does not matter if the company cannot survive long enough to enjoy it. Cheap stocks attract people into thinking about recovery before they have established survivability. 
 
**KF:** So one of your rules is: before studying upside, test whether the business survives ordinary stress without heroic assumptions. 
 
**Mara:** That is close. I would phrase it a little more sharply: before studying upside, ask whether the company can get through a normal rough patch using only believable assumptions. If the answer is no, I usually stop. 
 
**KF:** What made you stop in this case? 
 
**Mara:** The company needed three things to go right at once: refinancing, margin recovery, and customer retention. That is already too many moving parts for something that only looks cheap.
 
### Commentary 
 
- The agent starts with a concrete case rather than asking for abstract wisdom. This grounds the later generalization in evidence. 
- The question What did you check first is a process extraction question. It surfaces order, which is often more reusable than a loose list of factors. 
- The agent proposes a tentative rule early. This is important because it gives the expert something to refine rather than leaving the abstraction implicit. 
 
### Candidate Artifacts Emerging After Session 1 
 
- Procedure artifact candidate: Before studying upside, test whether the business can survive a normal rough patch under believable assumptions. 
- Judgment artifact candidate: A stock that looks cheap but requires several independent favorable events is likely not truly safe or attractive. 
- Episodic support: industrial distributor case rejected because survivability depended on refinancing, margin recovery, and customer retention all improving. 
 
At this stage these are only candidates. They are promising, but the agent does not yet know their boundaries or failure modes. 
 
## Session 2: Boundary And Failure Learning 
 
**KF:** I think I learned a broad rule: if a company has debt and a weak year could hurt it badly, reject it before studying upside. Is that too broad? 
 
**Mara:** Yes, that is too broad. Plenty of good businesses use debt responsibly. The issue is not debt alone. The issue is whether survival depends on optimistic assumptions. 
 
**KF:** So the important distinction is not leverage by itself, but fragility under realistic stress? 
 
**Mara:** Exactly. A stable toll-road business with manageable debt is very different from a cyclical business that needs a perfect recovery. 
 
**KF:** When does your rule become less reliable? 
 
**Mara:** It becomes harder in businesses where earnings are temporarily depressed but the asset base is unusually strong, or where there is hidden optionality that the market is missing. It is also harder in sectors where the accounting smooths over real risk. 
 
**KF:** Have you ever been wrong by applying this too harshly? 
 
**Mara:** Yes. I once rejected a niche manufacturer because I thought the debt made it too risky. I was too mechanical. I later realized the customer relationships were sticky, capital spending was flexible, and the balance sheet only looked tight because demand was at a temporary low. 
 
**KF:** What lesson did that mistake teach you? 
 
**Mara:** Ask whether the business can self-correct through ordinary operations before assuming the capital structure makes it fragile.
 
### Commentary 
 
- The agent intentionally overgeneralizes, which is useful because it invites correction. 
- The expert teaches a sharper concept: fragility under realistic stress, not debt in isolation. 
- The question Have you ever been wrong by applying this too harshly is a revision question. It reveals a failure case and improves the generality of the rule. 
 
### Artifact Updates After Session 2 
 
- Procedure artifact revised: Before studying upside, test whether the business can survive a normal rough patch using believable assumptions and ordinary operating self-correction. 
- Boundary artifact created: Do not treat debt alone as disqualifying; the real concern is fragility, not leverage in the abstract. 
- Failure artifact created: Past error came from applying the fragility rule too mechanically to a business with strong customer retention and flexible operations. 
- Revision trigger created: If the business can self-correct through ordinary operations, reduce the penalty attached to a seemingly tight balance sheet. 
 
These artifacts are now more useful because they include both the rule and the condition under which the rule should be softened. 
 
## Session 3: Agent Applies Stored Knowledge To A New Case 
 
Assume the agent later encounters a new research case: Harbor Tools, a small manufacturer whose shares have fallen sharply after two weak quarters. The agent retrieves the artifacts above before beginning its discussion with Mara. 
 
**KF:** Before I look at upside, I want to test survivability using the rule you taught me. Harbor Tools has meaningful debt, one large customer representing 35 percent of revenue, and a credit facility that must be renegotiated within eighteen months. On the other hand, maintenance capital needs are modest. My first-pass judgment is that this deserves caution because survival appears to depend on both refinancing and customer retention. 
 
**Mara:** That is a good first pass. What would you want to know next before rejecting it? 
 
**KF:** I would want to know whether the business can self-correct through ordinary operations. Specifically: can it cut costs without damaging customer relationships, and is the large customer genuinely at risk or just temporarily destocking? 
 
**Mara:** Good. You are using the rule correctly now. The original version would have led you to reject the company too quickly. The revised version tells you what to test next. 
 
### Commentary 
 
- This section shows application rather than mere storage. 
- The agent is not repeating a memorized line. It is using the stored rule, the boundary, and the revision trigger together. 
- The expert confirms that the revised artifact produces better reasoning than the original one. 
 
## Session 4: Agent Attempts Generalization 
 
At this point the agent has one successful case, one corrected abstraction, one boundary, and one failure-based revision. It now attempts a more general restatement.
 
**KF:** I want to test a generalized version of what I have learned. Tell me if this is too broad: when a company appears cheap after a decline, first test whether it can survive a normal rough patch using believable assumptions and ordinary operating self-correction. If survival depends on several favorable events happening together, treat the cheapness as suspect and delay upside analysis. 
 
**Mara:** That is much better. I would still add one thing. 
 
**KF:** What should I add? 
 
**Mara:** Add that the rule is strongest when the business model is understandable and the main risks are visible. In some situations the market may be discounting hidden assets or unusual recovery paths that are not obvious from a first pass. 
 
**KF:** So the general rule should carry a scope condition: it is strongest in businesses where downside can be judged with ordinary reasoning and the key risks are visible. 
 
**Mara:** Yes. That is an important limit. 
 
**KF:** Should I also record a new question type for this domain: after identifying fragility, ask whether the risk is visible and legible through ordinary analysis? 
 
**Mara:** Yes. That is a very good investing question. Some businesses are dangerous not only because they are weak, but because the real risk cannot be seen clearly in time. 
 
### Commentary 
 
- This session shows the taxonomy remaining open-ended. The expert explicitly helps the agent add a new type of question. 
- The agent is no longer learning only a single rule. It is learning how to ask better domain questions in the future. 
- The resulting knowledge is more than a summary. It includes a refined procedure, a scope condition, and a new questioning move that can improve later learning. 
 
## What KF Would Create, Store, And Apply 
 
### 1. Created During Dialogue 
 
During the dialogue, the system would create provisional candidate artifacts such as: 
 
- Case artifact: industrial distributor rejected because survivability depended on several favorable developments happening together. 
- Procedure candidate: test survivability before upside. 
- Judgment candidate: apparent cheapness is suspicious when the thesis depends on multiple things going right. 
- Boundary candidate: debt alone is not the real issue. 
- Failure candidate: rule once applied too mechanically to a temporarily weak but self-correcting business. 
- New question type candidate: visibility probe, used to ask whether downside is understandable through ordinary reasoning. 
 
These are not yet all trusted equally. Some are still tentative until validated through correction, application, or later reuse. 
 
### 2. Stored After Consolidation 
 
After enough evidence and correction, KF would store a more mature set of reusable artifacts.
 
#### Stored Procedure Artifact 
 
Title: `Screen survivability before upside` 
 
Natural-language DSL form: 
 
1. When a company appears cheap after distress or decline, do not begin with upside. 
2. First ask whether the business can survive a normal rough patch using believable assumptions. 
3. Check whether the business can self-correct through ordinary operations. 
4. If survival depends on several favorable events happening together, lower confidence and delay upside work. 
5. If the business can self-correct and key risks are visible, continue to deeper analysis. 
 
#### Stored Judgment Artifact 
 
Title: `Cheapness with stacked contingencies is suspect` 
 
Rule: 
A low valuation is less meaningful when the investment case depends on several separate positive outcomes occurring together. 
 
#### Stored Boundary Artifact 
 
Title: `Do not treat debt alone as fragility` 
 
Boundary: 
Debt is not by itself disqualifying. The stronger signal is fragility under realistic stress. 
 
#### Stored Revision Trigger 
 
Title: `Self-correction can soften apparent fragility` 
 
Trigger: 
If the company can preserve customer relationships, flex costs, and stabilize operations without extraordinary help, revise an initially harsh fragility judgment upward. 
 
#### Stored Failure Artifact 
 
Title: `Fragility rule applied too mechanically` 
 
Lesson: 
A past rejection was wrong because the company's operating resilience was stronger than the balance sheet first suggested. 
 
#### Stored Question Type Artifact 
 
Title: `Visibility probe` 
 
Purpose: 
Ask whether the downside can be understood through ordinary reasoning or whether important risk is hidden, delayed, or hard to see. 
 
Example use: 
Is the real downside visible enough that a first-pass survivability judgment is trustworthy? 
 
### 3. Applied Later 
 
When the agent later evaluates a new company or learns from another expert, it can retrieve these artifacts and use them in several ways: 
 
- guide which questions to ask first 
- generate an initial structured judgment 
- identify what evidence is still missing 
- avoid repeating a previously recorded mistake 
- decide whether a new case confirms, refines, or contradicts the stored rule
 
This means the stored knowledge changes future behavior. It is not passive memory. It becomes part of the agent's working method. 
 
## Why This Example Matters For Implementation 
 
A successful implementation of Expert-to-Agent Dialogic Learning should not merely save transcripts and call that learning. It should support: 
 
- deliberate question selection based on missing knowledge 
- provisional artifact creation during dialogue 
- explicit expert correction of tentative abstractions 
- storage of both rules and their boundaries 
- reuse of learned artifacts in later cases 
- extension of the question taxonomy when the domain itself teaches better ways to ask 
 
The example above is intentionally modest in scope. That is a strength. It shows a realistic path to deep learning without requiring magic. The agent learns effectively because it asks for a case, extracts process, proposes a rule, gets corrected, asks for failure conditions, applies the revised knowledge, and records a new kind of question introduced by the expert. 
 
## Related Documents

- Main spec: [Expert-to-Agent Dialogic Learning](./expert-to-agent-dialogic-learning.md)
- Glossary: [Glossary](../docs/glossary.md) 
 
## Summary 
 
This worked example shows how a [Knowledge Fabric](../docs/glossary.md#knowledge-fabric-kf) agent can learn from an expert in a way that is concrete, generalizable, and implementable. The important mechanism is not simply that the expert talks. The mechanism is that the agent asks targeted questions, proposes generalizations, receives correction, turns the result into [knowledge artifacts](../docs/glossary.md#knowledge-artifact), and later applies those artifacts to a new case. 
