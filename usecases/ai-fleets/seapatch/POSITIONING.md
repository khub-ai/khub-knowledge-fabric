# SeaPatch: Marketing Narrative and Positioning

> **Status**: Pre-experiment working document. This file records marketing
> angles, narrative framings, and anticipated claims developed prior to
> experimental validation. Architectural and structural claims are stated as
> facts where they hold by construction. Performance claims and outcome
> predictions are explicitly marked as design targets or hypotheses. Do not
> use unhedged performance claims in external communications until
> experimental results are available.

---

## Overview

SeaPatch demonstrates three distinct but related capabilities that, taken
together, represent a novel contribution to deployed AI systems:

1. **A compelling SAR scenario** — a drowning person missed at 0.91 confidence,
   found in 52 seconds, retroactively recovered from 45 minutes of prior footage
2. **Cognitive patching for tiered AI fleets** — the first demonstrated mechanism
   for instantly propagating validated knowledge across a large heterogeneous
   fleet without retraining
3. **Human-governed AI at scale** — a governance architecture in which humans
   remain meaningfully in the loop by governing policies, not individual decisions

Each hook addresses a distinct audience. This document records how to use each
one and how to frame claims responsibly.

---

## Hook 1: The SAR Scenario

### The story

A coast guard commander drone identifies what 38 optical scouts confidently
missed for 45 minutes. A rescue swimmer explains why in plain language.
Dialogic Distillation turns that explanation into a rule and broadcasts it
fleet-wide — including retroactively reclassifying 45 minutes of already-captured
sea-surface footage — without retraining a single model.

### Why it works as a hook

- High-stakes, life-or-death context makes the time-to-fix gap visceral
- The "confident miss" paradox: 0.91 confidence, wrong answer, frame never
  queued for human review — the system was confidently blind
- Retroactive reprocessing: knowledge the system didn't have at capture time
  is applied to footage that already passed
- A concrete speed comparison between design target and conventional pipeline

### What can be stated directly (architectural — true by construction)

- The classifier returned 0.91 confidence "no person" on a drowning person; the
  frame was never queued for human review because it was above the review threshold
- The rule is a natural language string, architecture-agnostic, deployable over
  any communication channel to any classifier architecture
- Retroactive reprocessing is a first-class operation in the protocol — the same
  rule governs both future and archived frames
- A rule can be revoked with a single delete operation; the fleet immediately reverts

### What must be hedged (performance targets — not yet measured)

- "We expect fleet-wide rule deployment in under 60 seconds." [design target]
- "Retroactive recall target: > 0.80 on archived person-in-water frames." [design target]
- "Rule precision target: ≥ 0.90 on SeaDronesSee pool validation." [design target]
- "Conventional retraining pipeline: 6–24 hours." [estimated baseline; not yet
  measured against this specific use case]

### Audience framing

| Audience | Lead with |
|---|---|
| General tech press | The 0.91 confidence miss; the retroactive reprocessing; the human story |
| SAR / maritime operators | Operational gap; time-to-fix comparison; 45-minute retroactive window |
| AI/ML researchers | Cross-modal TUTOR→PUPIL knowledge transfer; temporal feature reformulation as within-frame proxies |
| Robotics engineers | Architecture-agnostic rule broadcast; heterogeneous tier support; no per-tier pipeline |

---

## Hook 2: Cognitive Patching for Tiered AI Fleets

### The concept

In any tiered AI deployment — drone swarms, camera networks, factory sensors,
medical imaging devices — when a superior unit or human expert discovers
something the lower tier doesn't know, there is currently no mechanism to
propagate that knowledge instantly, safely, and architecture-agnostically
across the entire lower tier.

SeaPatch is the first demonstration of this mechanism working at fleet scale.

The analogy to software patching is direct: software gets patched when a
vulnerability is found; AI deployed in the field currently cannot be. One
expert dialogue, one validated rule, one broadcast — the fleet is patched.

### Why it works as a hook

- Names a structural gap that field engineers have encountered but not named
- The software patch analogy is immediately understood
- Generalizes beyond drones: the same pattern applies to camera networks,
  factory sensors, medical imaging, edge compute networks
- Architecture-agnostic is the key differentiator — one rule, two hardware
  tiers (Cortex-M MCU and Jetson Orin NX), no per-architecture pipeline

### What can be stated directly (architectural — true by construction)

- Rules are natural language strings (~200 tokens), deployable over any
  communication channel to any classifier architecture
- The same rule reaches MobileNetV3 scouts and Qwen3-VL-8B commanders with
  no architecture-specific adaptation
- The grounding check validates per-tier observability before broadcast —
  criteria that the scout tier cannot observe are removed or reformulated
  into within-frame proxies
- Any rule can be revoked; the fleet reverts immediately; no retrain required

### What must be hedged (working hypotheses — require verification)

- "To our knowledge, no existing framework combines instant in-field
  deployment, architecture-agnosticism, retroactive reprocessing, and
  natural language rules for tiered AI fleets." [working hypothesis;
  formal literature review pending]
- Fleet broadcast latency targets are design goals, not yet measured

### Audience framing

| Audience | Lead with |
|---|---|
| Robotics / embedded systems | Architecture-agnostic; no per-tier retraining pipeline |
| Enterprise AI in safety-critical domains | Structural gap; operational tempo; no cold-start requirement |
| Defense / government | Fleet patching at mission tempo; no return-to-base required |
| Academic (ICRA / IROS / NeurIPS) | Novel system contribution; cross-modal synthesis; retroactive reprocessing |

---

## Hook 3: Human-Governed AI at Scale

### The concept

The medium of the patch is natural language. That single fact changes the
governance story entirely:

- A human can read every rule the fleet is currently operating on
- A human can originate a rule through ordinary conversation
- A human can revoke any rule instantly
- Every classification traces to a specific rule, authored by a specific
  person, at a specific time

This is **human-in-the-loop at the policy level, not the decision level** —
a qualitatively different and scalable form of meaningful human oversight.

### Why it works as a hook

- Addresses anti-AI sentiment directly: positions the technology as amplifying
  human expertise, not displacing it
- The governance triad (transparency, provenance, revocability) is
  architectural — true by construction, not a bolt-on compliance feature
- Aligns with regulatory direction: high-risk autonomous systems face
  increasing requirements for human oversight, explainability, and auditability
- The three modes of knowledge flow make the human role explicit and
  structurally necessary, not optional

### The governance triad (all three are architectural — true by construction)

**Transparency** — Every active rule governing fleet behavior is a human-readable
natural language string. Any authorized person can read the complete operative
rule set without ML expertise. The question "what does this fleet currently
know?" has a legible answer.

**Provenance** — Every rule has a logged author, timestamp, dialogue transcript,
and validation record. Every classification traces to a specific rule. The
question "why did this drone classify this frame this way?" has a traceable
answer terminating in a human dialogue.

**Revocability** — Any rule can be deleted. The fleet immediately reverts. A
rule generating false positives in a new operational context can be pulled in
seconds without a retrain cycle.

### Reframing the human-AI relationship (for anti-AI audiences)

| AI concern | SeaPatch reality |
|---|---|
| AI displaces human expertise | Rescue swimmer's 30 years of experience governs 38 drones simultaneously; she is scaled, not replaced |
| AI operates opaquely | Fleet knowledge base is a document any domain expert can read without ML background |
| AI cannot be corrected after deployment | Any rule is one delete command from revocation, fleet-wide, in seconds |
| AI is unaccountable when it fails | Every classification has a full provenance chain terminating in a named human dialogue |

### What can be stated directly (architectural — true by construction)

- Rules are natural language; any authorized person can read the complete set
  of rules governing the fleet's current behavior
- Every rule has a logged author, dialogue transcript, and validation record
- Any rule can be revoked; the fleet immediately reverts
- The rescue swimmer's description becomes the operative rule through
  dialogue — not a summary, not a translation, but a grounded and validated
  version of her own expertise

### What must be hedged

- "This architecture satisfies EU AI Act requirements for high-risk autonomous
  systems." [alignment is architecturally plausible; formal legal review
  required before claiming compliance]
- "Rules are stable and rarely require revocation in practice." [observed in
  prior DD experiments; not yet demonstrated at fleet scale]

### Audience framing

| Audience | Lead with |
|---|---|
| Policy makers / regulators | Governance-by-design; human oversight structural not procedural |
| Enterprise risk and compliance | Auditability, provenance, revocability as design properties |
| Press covering AI safety | Counter-narrative: amplifier not replacer; policy-level oversight |
| SAR / maritime organizations | Expert authority explicit and preserved; rescue swimmer's knowledge persists |

---

## The Three Modes of Knowledge Flow

This is a structural property of the system — true by construction:

| Mode | Knowledge source | Knowledge recipient | SeaPatch example |
|---|---|---|---|
| AI → AI | Commander drone (thermal FLIR) | Scout fleet (RGB classifiers) | Cross-modal detection rule derived from thermal confirmation |
| Human → AI | Rescue swimmer (ground station) | Entire fleet | Instinctive drowning response rule |
| Human + AI → AI | Expert + commander together | Scout fleet | Cross-modal rule grounded through dialogue, with per-tier reformulation |

The third mode is what SeaPatch actually demonstrates. The human provides domain
knowledge; the commander provides cross-modal evidence; the grounding check
determines per-tier observability; the human can approve, modify, or reject
before broadcast. No single source was sufficient alone.

This is the most important mode to communicate: it is not "AI replacing the
human" nor "human correcting the AI" but a genuine synthesis where both
contribute what the other cannot.

---

## What Not to Say in External Communications

Until experimental results are available, avoid these in unhedged form:

- Specific latency figures (52 seconds, < 60 seconds) stated as measured facts
- Retroactive recall figures stated as measured results
- "The only framework" or "no other system" without a completed literature review
- EU AI Act compliance claims without formal legal review
- Scenario outcomes (person recovered, two further incidents detected) presented
  as measured experimental results rather than design scenario descriptions

---

## Recommended Publication Sequence

| Stage | What's ready | Framing |
|---|---|---|
| Now | Architecture, governance properties, three modes, the structural gap | Design properties, stated as such |
| After Phase 1 | Rule precision on SeaDronesSee pool, grounding check results | Measured results on real data |
| After Phase 2 | Fleet broadcast latency, retroactive recall, cross-tier consistency | Full quantitative claims |
| After Phase 2 | arXiv preprint with measured results | Academic credibility before press |
| After preprint | Press, conference outreach, vertical market engagement | Validated claims throughout |

The architectural and governance hooks (Hooks 2 and 3) can be communicated
before experimental results, clearly framed as design properties. The
performance and scenario hooks (Hook 1 numbers) should wait for data.

---

## See Also

- [README.md](README.md) — user-facing scenario and motivation
- [DESIGN.md](DESIGN.md) — implementation architecture
- [Cognitive Patching](../../../docs/cognitive-patching.md) — the general concept this use case demonstrates
- [DD Patchability](../../../docs/patchability.md) — predicting whether DD will work in a new domain
