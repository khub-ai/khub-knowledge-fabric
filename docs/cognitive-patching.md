# Cognitive Patching: Runtime Knowledge Propagation for Tiered AI Systems

> **Status**: Conceptual framework document. The ideas here are grounded in
> the architecture of the SeaPatch use case but have not yet been validated
> at scale. Claims about novelty ("to our knowledge", "currently lacking")
> are working hypotheses pending a formal literature review. Performance
> claims are design targets, not measured results.

---

## The Gap

As AI systems scale into large heterogeneous deployments — drone swarms,
camera networks, factory floor sensors, medical imaging devices, edge compute
networks — a fundamental operational capability is currently missing:

> When a superior unit or human expert discovers something the lower tier
> doesn't know, there is no mechanism to propagate that knowledge instantly,
> safely, and architecture-agnostically across the entire lower tier.

This is not a gap in any one product. To our knowledge, it is a gap in the
field.

---

## The Problem in Concrete Terms

A commander drone (Jetson Orin NX, Qwen3-VL-8B, thermal FLIR) discovers that
38 scout drones (Cortex-M MCU, MobileNetV3, RGB only) have been misclassifying
a drowning person as whitecap/foam at 0.91 confidence for 45 minutes.

What are the options?

| Approach | Why it fails here |
|---|---|
| Retrain and redeploy | Hours to days; requires labeled data; cannot happen in the field; architecture-specific |
| Fine-tune scouts | Same problems; also infeasible on Cortex-M MCUs |
| Update commander's LLM prompt | Commander already knows; scouts run a different architecture |
| Human review queue | Frames above the confidence threshold are never queued; confident misses are invisible |
| Broadcast a new procedure to scouts | Scouts have no mechanism to ingest natural language instructions at inference time |

None of these work within an operational timeframe where minutes matter.

---

## What Cognitive Patching Does

Cognitive patching is the application of Dialogic Distillation (DD) to runtime
fleet knowledge propagation. It produces a natural language rule — validated
against ground truth frames, checked for per-tier observability, and approved
before broadcast — that updates every unit in the fleet simultaneously.

Properties that make this work where other approaches fail:

**Architecture-agnostic** — A natural language rule is approximately 200 tokens
of text. It can be injected into any classifier architecture as prompt context.
No weight update, no gradient computation, no per-architecture retraining
pipeline.

**In-field deployable** — The rule is transmitted over the existing communication
channel. No deployment pipeline, no staging environment, no engineer required.

**Validated before broadcast** — The grounding check confirms every criterion
in the rule is observable by the receiving tier's sensor at its operational
parameters. Criteria that fail grounding are removed or reformulated as
within-frame proxies. The pool validation gate (precision ≥ 0.90, max FP = 0)
confirms the rule's performance before it reaches the fleet.

**Retroactive** — The same rule that governs future classifications can be
applied immediately to the archived frame buffer, re-examining footage that
was classified before the blind spot was identified.

**Revocable** — Any rule can be deleted with a single operation. The fleet
immediately reverts to its previous behavior. No retrain required to undo.

---

## Three Modes of Knowledge Flow

Cognitive patching supports three distinct knowledge propagation paths:

### Mode 1: AI → AI

A superior AI unit (richer sensor, higher compute, more capable model) identifies
a classification gap in lower-tier units and the system derives a corrective rule
from its cross-modal evidence.

*SeaPatch example*: Commander drone C2 detects a 37°C thermal signature. Scout
S22's RGB frame from the same coordinates is retrieved. The system identifies
the optical features in S22's frame that correspond to the thermal confirmation
and produces a rule grounded in what the scout tier can observe.

### Mode 2: Human → AI

A domain expert provides knowledge that no AI unit in the fleet currently holds.
Their description, in natural language, becomes the operative rule after
grounding and validation.

*SeaPatch example*: The rescue swimmer describes the instinctive drowning
response — head position, brightness uniformity, arm V-darkening — in her own
words. Her description becomes the rule. Every drone in the fleet now knows
what she knows. Her expertise persists beyond her shift.

### Mode 3: Human + AI → AI (collaborative synthesis)

The most powerful mode. Human domain knowledge and AI cross-modal evidence
are combined in dialogue to produce a rule that neither could produce alone.

*SeaPatch example*: The commander's thermal confirmation tells the swimmer what
is in the water. She articulates what to look for in RGB to detect it. The
grounding check determines what scouts can actually observe at their tier,
reformulating temporal features as within-frame proxies. The result is a rule
that requires the thermal evidence, the human expertise, and the grounding
constraint simultaneously — no single source was sufficient.

This mode is not "AI replacing the human" nor "human correcting the AI." It is
a genuine synthesis where both contribute what the other cannot.

---

## Governance by Construction

Because the medium of propagation is natural language, cognitive patching
provides governance properties that weight-update approaches structurally
cannot offer.

### Transparency

Every rule governing fleet behavior is a human-readable natural language string.
Any authorized person — domain expert, safety officer, incident investigator —
can read the complete operative rule set without ML expertise. The question
"what does this fleet currently know?" has a legible answer.

### Provenance

Every rule has a logged author, timestamp, dialogue transcript, and validation
record. Every classification traces to a specific rule. The question "why did
this drone classify this frame this way?" has a traceable answer: rule R47,
authored by [name] at [time], validated against 30 pool frames, precision 1.0.

### Revocability

Any rule can be deleted. The fleet immediately reverts. A rule generating false
positives in a new operational context can be pulled in seconds without a
retrain cycle. Governance operates at operational tempo.

---

## Human-in-the-Loop at the Policy Level

The standard human-in-the-loop model — a human reviewing individual AI
decisions — does not scale to large autonomous fleets. A 38-drone swarm
classifying frames continuously cannot be meaningfully supervised
decision-by-decision.

Cognitive patching offers a different model: **human-in-the-loop at the
policy level**.

A human cannot review every frame. A human *can* review every rule that
governs how frames are classified — because rules are short, readable, and
authored in natural language by domain experts. One human-approved rule governs
millions of subsequent decisions. That is scalable oversight.

This is not a weaker form of oversight. For large autonomous systems, it may
be the only form of oversight that is both meaningful and operationally feasible.

The key distinction from the conventional use of "human-in-the-loop":

> **"Human-in-the-loop at the policy level, not the decision level."**

In the conventional framing, a human watches AI decisions stream past and
occasionally clicks approve. That doesn't scale, and the human often cannot
meaningfully evaluate what they're approving. In cognitive patching, the human
governs the policies that drive all decisions — and those policies are legible.

---

## Relationship to AI Governance Frameworks

High-risk AI systems in safety-critical contexts (autonomous vehicles, maritime
SAR, medical devices, industrial automation) face increasing regulatory pressure
for human oversight, explainability, and auditability.

Cognitive patching's architecture aligns with these requirements by construction:

- **Human oversight**: rules require approval before broadcast; any rule can be
  revoked by an authorized human instantly; no technical barrier to human
  intervention
- **Explainability**: rules are natural language; every classification traces to
  a specific human-readable rule; no black-box inference chain
- **Auditability**: every rule has a full provenance record; every classification
  has a rule trace; the complete history of what the fleet knew and when is logged

*Note*: whether this satisfies specific regulatory requirements (e.g., EU AI Act
Article 14 on human oversight for high-risk systems) requires formal legal review.
The architectural alignment is noted here as a design property, not a compliance
claim.

---

## What Is Not Yet Validated

The following are design targets or working hypotheses, not measured results:

- Fleet-wide rule deployment in under 60 seconds [design target; not yet measured]
- Retroactive recall > 0.80 on archived frames [design target]
- Rule precision ≥ 0.90 on SeaDronesSee pool validation [design target]
- Cross-tier consistency > 0.85 (scout and commander agree on the same frame) [design target]
- The assertion that this combination of properties is unique in the field
  [working hypothesis; formal literature review pending]

---

## Scope of Applicability

The cognitive patching pattern applies to any deployment combining:

1. A large number of lower-tier AI units (edge devices, sensors, lightweight classifiers)
2. A smaller number of higher-tier units or human experts with richer knowledge
3. A need to propagate knowledge from higher to lower tiers faster than retraining allows
4. Heterogeneous hardware making per-architecture retraining pipelines impractical

Candidate domains beyond maritime SAR:

| Domain | Lower-tier units | Higher-tier knowledge source |
|---|---|---|
| Factory automation | Edge vision sensors, defect classifiers | Supervisory AI, quality engineer |
| Smart city / traffic | Intersection cameras, vehicle classifiers | Traffic management AI, operations |
| Medical imaging | Screening devices at clinic tier | Specialist radiologist, diagnostic AI |
| Agriculture | Field drones, crop health classifiers | Agronomist, satellite imagery AI |
| Security / surveillance | Perimeter cameras, anomaly detectors | SOC analyst, threat intelligence AI |

The maritime SAR scenario is the initial demonstration because the time-to-fix
constraint is maximally visceral, the hardware heterogeneity is maximal (MCU vs
Jetson), and the cross-modal gap (thermal to RGB) requires all three roles of DD
— Patch, Synthesizer, and Propagator — simultaneously.

---

## Relationship to Dialogic Distillation

Cognitive patching is the application of Dialogic Distillation (DD) to a tiered
multi-agent deployment. The core DD mechanism — expert dialogue, grounding check,
pool validation, rule acceptance — is unchanged. The SeaPatch use case introduces
four extensions:

| Extension | Standard DD | SeaPatch addition |
|---|---|---|
| TUTOR prompt | Single modality throughout | Cross-modal: expert bridges thermal knowledge to RGB observables |
| Temporal features | Not addressed | Temporal discriminators reformulated as within-frame proxies for single-frame classifiers |
| Grounding check | Single PUPIL context | Runs per hardware tier; produces tier-specific rule variants |
| Rule application | Forward-only | Retroactive reprocessing of archived frame buffer after acceptance |

These extensions make DD applicable to fleet patching without changing its core
protocol.

---

## See Also

- [SeaPatch README](../usecases/ai-fleets/seapatch/README.md) — scenario and motivation
- [SeaPatch DESIGN](../usecases/ai-fleets/seapatch/DESIGN.md) — implementation architecture
- [SeaPatch POSITIONING](../usecases/ai-fleets/seapatch/POSITIONING.md) — marketing narrative and claim framing
- [DD Patchability](./patchability.md) — predicting whether DD will work in a new domain
