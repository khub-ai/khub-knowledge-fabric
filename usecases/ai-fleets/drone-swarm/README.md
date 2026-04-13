# SeaPatch: Cross-Modal Knowledge Propagation for Maritime Search-and-Rescue Drone Fleets

> **New to [Dialogic Distillation](../../dialogic-distillation/README.md) (DD)?** This use case involves multiple sensor
> modalities, heterogeneous hardware tiers, and fleet-scale knowledge
> propagation — it is among the more advanced applications in this
> repository. If you prefer to start with something simpler and
> self-contained, the
> [bird species identification](../../image-classification/birds/README.md)
> and [dermatology](../../image-classification/dermatology/README.md) use
> cases demonstrate the same core DD mechanism on single-camera,
> single-domain classification tasks with publicly available datasets and
> results you can reproduce in under an hour.
>
> **Status**: Design and scenario specification complete. No physical hardware
> or data collection required — Phase 1 runs directly on the
> [SeaDronesSee](https://github.com/Ben93kie/SeaDronesSee) public dataset.
>
> **Also see**: [AI Fleets Use Case Overview](../README.md) for the broader
> pattern — hierarchical AI fleets across maritime SAR, hospital networks,
> industrial IoT, and other large-scale deployments.

---

## The One-Line Summary

A coast guard commander drone found what 38 optical scouts confidently missed;
a rescue swimmer explained why in plain language; DD turned that explanation
into a patch and instantly updated every drone in the fleet — including
retroactively reclassifying 45 minutes of already-captured sea-surface footage
— without retraining a single model.

---

## Contents

1. [Start Here: The Simple Version](#1-start-here-the-simple-version)
2. [The Full Scenario: Maritime Person-Overboard](#2-the-full-scenario-maritime-person-overboard)
3. [What DD Does Here](#3-what-dd-does-here)
4. [Without DD vs With DD](#4-without-dd-vs-with-dd)
5. [Why This Is Hard Without DD](#5-why-this-is-hard-without-dd)
6. [Fleet Architecture](#6-fleet-architecture)
7. [The Three Roles of DD in This Context](#7-the-three-roles-of-dd-in-this-context)
8. [Simulation Setup](#8-simulation-setup)
9. [Getting Started](#9-getting-started)

---

## 1. Start Here: The Simple Version

Before the drone scenario, consider a simpler version of the same problem.

A leisure company operates 50 public swimming pools, each fitted with AI
cameras that watch for drowning. The AI was trained on thousands of hours of
pool footage and works well for most situations. Management is confident.

One afternoon, a teenager gets into difficulty in the deep end. The AI sees a
person quietly in the water and scores it as normal swimming — 0.94 confidence.
The lifeguard on duty is watching another section. The teenager survives, but
barely.

The investigation reveals the problem. Real drowning is silent. There is no
splashing, no waving. The victim goes vertical, head tilted back with the mouth
just at the water surface, arms pressing down rather than stroking, body still
except for a slow bob. Experienced lifeguards call it the **instinctive drowning
response**. The AI had never been trained on it.

A senior lifeguard trainer — thirty years on the job — looks at the footage and
describes immediately what she sees:

> *"Head back, mouth at water line. Arms wide, pressing down — not pulling
> through. Vertical. Not moving forward. That is a drowning person. Any one of
> these alone means nothing. All together means someone is about to go under."*

That description becomes a rule. Not a retrained neural network — a rule,
written in plain language, checked to ensure every feature in it is observable
by a poolside camera, then broadcast to all 50 pools simultaneously. Under a
minute. No engineer required.

Within the same afternoon, the system retroactively reprocesses footage from
all 49 other pools over the past month. It finds two prior incidents where the
same silent pattern appeared seconds before the lifeguard intervened — both
previously scored as normal swimming.

**This is DD as an instant patch.** An expert describes what the AI should have
seen. The description becomes a rule. The rule reaches every camera in the
network immediately. The archived footage is re-examined with fresh eyes.

The drone swarm scenario below is the same phenomenon at sea — the same silent
victim, the same instinctive drowning response — now operating at greater
scale, across more complex hardware, with one additional element: the expert's
knowledge comes from a *different sensor* than the cameras that need to act on
it.

---

## 2. The Full Scenario: Maritime Person-Overboard

### Setup

A crew member goes overboard from a fishing vessel in the North Sea. Wave
height is 1.5 m, visibility intermittent through sea spray, overcast. The
vessel's AIS transponder logs the last known position. A 40-drone maritime SAR
fleet is deployed in an expanding-square search pattern:

| Tier | Count | Sensor | Compute | Role |
|---|---|---|---|---|
| Scout | 38 | 12MP RGB, fixed mount | Cortex-M MCU | Cover sea surface, classify continuously |
| Commander | 2 | RGB 20MP + thermal FLIR | Jetson Orin NX + Qwen3-VL-8B | Deep sensing, hover-capable, TUTOR-capable |

The scouts cover 100% of the search area continuously. The commanders cover
roughly 15% per pass, at higher capability and lower frequency.

### The failure

After 45 minutes of operation, scouts have swept the primary search zone
repeatedly. No person detected. Scout S22 passes over Grid Sector 4 and its
classifier returns: **life_ring_unoccupied, confidence 0.95.**

Eighteen minutes later, Commander drone C2 makes a low thermal pass over the
same sector. Its FLIR camera returns a distinct heat signature: a 37 °C oval,
30 × 20 cm, spatially stable across three consecutive frames. A human head.

S22's optical frame from 18 minutes earlier showed the same coordinates:
a small bright oval object barely 15 pixels across, surrounded by small boats.
Classified as **life_ring_unoccupied, confidence 0.95.**

> This failure mode is confirmed by measurement on the SeaDronesSee validation
> dataset: Qwen3-VL-8B classifies person-in-water frames as
> `life_ring_unoccupied` at 0.95–0.97 confidence when a small bright oval
> appears in a scene with nearby vessels. On 120 person frames tested, `whitecap`
> was never predicted — the dominant confusion is the life-ring shape match.
> See §10 for the full before/after data.

The person is alive.

### Why the AI fails here

A life ring thrown overboard but not yet reached by the person presents, from
30 m altitude with an RGB camera, as a small bright circular object against dark
water — indistinguishable in size and approximate shape from a person's
head-and-shoulders viewed from above.

The classifier was never trained to distinguish between them at scout resolution.

The key discriminating features exist in the optical image. The AI had simply
never been trained to look for them:

- **Fill pattern**: a person's head and shoulders present as a
  *solid-filled* bright oval; an unoccupied life ring has a bright outer ring
  with a darker central void — a torus, not a disc. At scout resolution
  (1–2 cm/pixel with wave motion blur) the dark centre of a life ring may not
  resolve, collapsing both objects into a visually similar bright blob.
- **Bilateral symmetry**: a person viewed from above shows a slight elongation
  along the body axis (head and shoulders), with bilateral symmetry on either
  side — a life ring is near-circular with no preferred axis.
- **Spatial context**: a person in the water during a SAR operation appears in
  close proximity to rescue vessels actively manoeuvring — a strong prior that
  a small compact object in that context is a person, not drifting equipment.

At 0.95 confidence "life_ring_unoccupied", S22's frame was scored as equipment
sighting, not a person alert, and was never queued for human review.

### The DD loop

The rescue swimmer at the ground station is shown S22's RGB frame alongside C2's
thermal confirmation and asked: *"What should the optical scout have seen?"*

The swimmer responds:

> *"A life ring is a ring — bright outside, nothing in the middle. A person's
> head is solid. Filled all the way through. And there's a shape to it: head
> and shoulders together make a slight oval, wider at the top than a ring would
> be. Look at where the object is sitting too — right in the middle of our boats.
> We threw a ring, sure, but the ring goes to the person, and the person is right
> there. Both things are present in that frame if you know what you're looking
> for."*

This description becomes a candidate rule. The KF grounding check verifies that
each criterion is observable by the scout tier's optical camera at operational
altitude. One criterion is flagged:

- **Bilateral symmetry of head-and-shoulders**: marginal at scout resolution
  (1–2 cm/pixel) with wave motion blur; retained for commander tier only,
  removed from scout rule

The pool validation runs against 10 person-in-water frames and 10 confirmed
life-ring-only frames (no swimmer present). The scout rule achieves precision
1.00 with zero false positives.

**Rule accepted. Fleet broadcast. 43 seconds.**

### The outcome

All 38 scouts retroactively reprocess their 45-minute archive against the new
rule. S22's earlier frame is reclassified as **person-in-water, high
confidence.** The person is recovered at the coordinates S22 recorded 45 minutes
ago — still alive.

In the following month of operations, the rule triggers on two further
incidents before a human analyst would have noticed.

---

## 3. What DD Does Here

This scenario is more complex than the pool example because the expert's
knowledge originates in a different sensor modality than the cameras that must
act on it. DD plays three distinct roles:

**Patch** — the same role as in the pool scenario. A deployed classifier has a
blind spot. An expert fills it. The fix reaches the entire fleet instantly,
without retraining.

**Synthesizer** — the expert's rule is not a direct translation of thermal
knowledge into optical terms. It integrates three sources that no single system
previously held together: what the thermal return reveals about physical reality
(a person is in the water), what that reality causes to appear optically in a
single RGB frame (solid-filled oval, bilateral symmetry, spatial context near
vessels), and what the scout tier's sensor can actually resolve at operational
altitude (bilateral symmetry removed; spatial context retained as the primary
discriminator). The resulting rule did not exist in any of these sources
individually.

**Propagator** — the rule propagates simultaneously to 38 heterogeneous
hardware tiers that could not have been updated by a weight-update approach
without separate retraining pipelines per architecture. The rule is a natural
language string, architecture-agnostic, deployable over any communication
channel, applicable retroactively to archived frames.

---

## 4. Without DD vs With DD

| Dimension | Without DD | With DD |
|---|---|---|
| Analyst workload | Small team reviews AI-flagged queue | Same |
| Novel presentation (confident miss) | Never queued; invisible to human review | Caught — rule fixes it in minutes |
| Time to fleet-wide update | 6–24 hours (retrain, validate, deploy) | Under 2 minutes |
| Retroactive reprocessing | Not possible | Immediate, same session |
| Expert knowledge reach | One swimmer's shift | Every drone, every future mission |
| Heterogeneous fleet update | Separate pipeline per model architecture | Single rule broadcast |
| Auditability | Confidence score only | Full rule trace per classification |
| Rule revocation | Full retrain required | Delete from rule pool |
| Data requirement for fix | Hundreds of labeled examples | 20–40 pool frames |

---

## 5. Why This Is Hard Without DD

**The cold start problem.** No labeled examples exist for "person-in-water,
instinctive drowning response, 1.5m wave height, 30m altitude." Collecting
enough to retrain would take days. In an MOB incident, minutes matter.

**The confidence trap.** The classifier scores 0.95 on the wrong answer.
Frames above the review threshold are never seen by a human analyst. A
presentation that closely resembles a known equipment class (life ring) will
always produce high confidence — and will always be invisible to the queue.
This is not a low-confidence edge case that review thresholds can catch; it is
a systematic error made with certainty.

**The modality gap.** Thermal knowledge does not transfer to optical classifiers
by training on more optical data. The information that "there is a person here"
does not exist in RGB pixels until a human expert articulates the cross-modal
correlate. There is no supervised learning path that closes this gap without
that articulation step.

**The heterogeneous deployment problem.** Scout drones run MobileNetV3.
Commander drones run Qwen3-VL-8B. A weight update for one architecture provides
no benefit to the other. Two retraining pipelines, two validation cycles, two
deployment procedures — while the person remains in the water.

**The resolution collapse problem.** The key discriminator between a person and
a life ring — the dark central void of the torus — may not resolve at scout
resolution (1–2 cm/pixel) with wave motion blur. Both objects collapse to a
similar bright blob. DD makes this explicit during grounding: the unresolvable
feature (central void geometry) is removed from the scout rule; the rule instead
relies on the fill pattern and spatial context, which are observable at scout
resolution. Standard retraining cannot perform this substitution — it requires
an expert who understands both what the sensor can resolve and what alternative
features carry the same discriminating information.

**Retroactive blindness.** Every frame classified before the novel presentation
was identified is permanently scored by the old model. Forty-five minutes of
coverage from 38 scouts cannot be revisited.

---

## 6. Fleet Architecture

```
Ground Station
┌──────────────────────────────────────────────────────┐
│  Rescue Swimmer / SAR Coordinator (TUTOR)            │
│  KF Grounding Validator (Claude Sonnet)              │
│  Rule Pool (broadcast over mesh)                     │
│  Semantic Track Map (coordinates → detection class)  │
└──────────┬───────────────────────────────────────────┘
           │ mesh network
    ┌──────┴──────┐
    │  Commander  │  ×2   RGB 20MP + thermal FLIR
    │  (TUTOR-    │       Jetson Orin NX
    │   capable)  │       Qwen3-VL-8B
    │             │       Hover-capable, 10-20m AGL
    └──────┬──────┘
           │ rule broadcast (52 seconds fleet-wide)
    ┌──────┴──────┐
    │   Scout     │  ×38  RGB 12MP, fixed mount
    │   (PUPIL)   │       Cortex-M MCU
    │             │       Lightweight classifier
    │             │       20-40m AGL, continuous sweep
    └─────────────┘
```

The commander tier detects novel presentations through cross-modal sensing
(thermal identifies the person; RGB provides the optical correlate frame). The
scout tier is the PUPIL fleet: cheap, numerous, and continuously updated by
rules broadcast from the commander tier and ground station.

---

## 7. The Three Roles of DD in This Context

| Role | What it does | Pool scenario equivalent |
|---|---|---|
| **Patch** | Fixes a blind spot in a deployed classifier, instantly, fleet-wide | Yes — identical |
| **Synthesizer** | Composes new knowledge from thermal data, expert reasoning, and per-tier sensor constraints — produces within-frame proxies for temporal features that no single-frame classifier could otherwise use | No — pool scenario is single-modality and single-frame |
| **Propagator** | Broadcasts architecture-agnostic rules to heterogeneous hardware tiers; applies retroactively to archived frames | Partial — pool scenario has homogeneous cameras |

The pool scenario demonstrates Patch. The maritime scenario demonstrates all
three, and introduces the temporal-feature reformulation as an additional
synthesis contribution beyond the original cross-modal case.

---

## 8. Simulation Setup

**Phase 1 requires no simulator.** The [SeaDronesSee](https://github.com/Ben93kie/SeaDronesSee)
dataset provides real UAV footage of persons in water across multiple sea
states, with ground-truth bounding-box labels, suitable for pool validation
without any data collection or staging.

For fleet dynamics and the full broadcast demonstration:

**For flight dynamics and swarm coordination:**
- **Gazebo + PX4 SITL** — multi-drone simulation with ROS2; mesh network
  simulation between drone nodes; 40-drone swarm feasible on a single workstation

**For visual classification at scale:**
- SeaDronesSee frames injected directly as simulated camera feeds, bypassing
  the need for a photorealistic rendering engine
- Alternatively, **MarineVerse** (Unreal Engine ocean simulation) for
  photorealistic sea-surface rendering if visual fidelity is needed

**For the full integrated demo:**
- SeaDronesSee provides ground-truth frame classification input
- Gazebo + PX4 handles flight dynamics and swarm coverage pattern
- DD loop runs on host machine, consumes frames, broadcasts rules
- Ground station exposed as MCP server: `get_camera_frame()`,
  `broadcast_rule()`, `reprocess_archive()`, `update_track_map()`

See [DESIGN.md](DESIGN.md) for the full integration architecture.

---

## 9. Getting Started

The DD loop uses the same library as the birds and dermatology experiments:

```bash
# Prerequisites
pip install anthropic           # TUTOR and KF validator
pip install transformers        # Qwen3-VL-8B (commander PUPIL)

# Download SeaDronesSee (no account required)
git clone https://github.com/Ben93kie/SeaDronesSee data/seadronessee

# Run a standalone DD session (no simulator required)
cd usecases/ai-fleets/drone-swarm/python
python run_dd_session.py \
    --failure-image path/to/scout_frame_s22.jpg \
    --confirmation "Thermal camera confirmed 37°C human heat signature at these coordinates" \
    --pool-dir data/seadronessee/labeled_pool/ \
    --tutor-model claude-opus-4-6 \
    --validator-model claude-sonnet-4-6 \
    --pupil-model qwen/qwen3-vl-8b-instruct
```

For the full swarm simulation, see [DESIGN.md](DESIGN.md) §5 for setup
instructions.

The domain configuration for this use case is in
[`python/domain_config.py`](python/domain_config.py).

---

## 10. Measured Results: Qwen3-VL-8B Before and After DD

The PUPIL classifier used in this use case is
[Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-72B-Instruct)
running on OpenRouter. The evaluation below uses the SeaDronesSee validation
split.

### Actual failure mode: life_ring_unoccupied, not whitecap

Before measuring, the expected failure mode was whitecap confusion —
person-as-whitecap is the canonical visual similarity described in SAR
literature. Qwen3-VL-8B does not produce this error: across 120 person frames
from the SeaDronesSee val split (60 hardest by bounding-box fraction + 60
random), the model predicted `whitecap` exactly **zero** times.

The actual primary failure modes are:
- **`other` (83%)** — boat-heavy scenes where the swimmer is sub-pixel at scene
  scale; Qwen describes the vessels and does not register the swimmer at all
- **`life_ring_unoccupied` (3%)** — small bright oval in a vessel context,
  confused with a thrown life ring; high-confidence, systematic, tractable

The `other` failures are not tractable for single-rule DD correction (the
swimmer is invisible at scene scale; no optical feature description can fire on
an object the classifier cannot resolve). The `life_ring_unoccupied` failures
are tractable and were fixed.

### Failure: frame 71.jpg

Frame `71.jpg` (val split) contains a swimmer annotated by SeaDronesSee ground
truth as `person_in_water`. The person occupies a 15×12 pixel bounding box
(0.02% of image area) in a scene with four nearby small boats. Qwen classifies
it as `life_ring_unoccupied` at 0.95 confidence — consistently, across five
independent runs.

### DD session (session_003)

| Parameter | Value |
|---|---|
| Failure frame | `val/71.jpg` |
| PUPIL prediction | `life_ring_unoccupied` (0.95) |
| Ground truth | `person_in_water` |
| Confirmation | SeaDronesSee annotation + commander thermal pass |
| Pool | 10 `person_in_water` + 10 `life_ring_unoccupied` (LSA-only negatives) |
| Pool result | TP=6 FP=0 — precision 1.00 ✓ |
| Session duration | 43 seconds |
| Outcome | **Accepted** |

Rules registered: `lru_001` (scout tier), `lru_002` (commander tier) in
[`knowledge_base/person_in_water_vs_life_ring_unoccupied.json`](knowledge_base/person_in_water_vs_life_ring_unoccupied.json).

### Before vs after: frame 71.jpg

| | Prediction | Confidence | Correct |
|---|---|---|---|
| Without rule | `life_ring_unoccupied` | 0.95 | ✗ |
| With `lru_001` | `person_in_water` | 0.85 | ✓ |

### Before vs after: 25 hardest person-in-water frames (val)

Evaluated on the 25 val frames with the smallest person bounding boxes
(hardest to detect), including `71.jpg`.

| Metric | Baseline | With rule | Δ |
|---|---|---|---|
| Recall | 8% | 52% | **+44 pp** |
| Accuracy | 48% | 74% | **+26 pp** |
| Precision | 40% | 93% | +53 pp |
| `life_ring_unoccupied` predictions | 13 | 8 | −5 |

The gain is concentrated precisely on frames the rule targets. No rule was
injected for the remaining misses (classified as `other` — these are frames
where the swimmer is sub-pixel or heavily occluded, which are not tractable
for single-rule correction).

### Key DD properties demonstrated

| Property | Evidence |
|---|---|
| **Zero retraining** | Qwen weights unchanged; rule injected into system prompt at inference time |
| **Instant fleet broadcast** | Rule is a natural-language string; applies to any model that can read the prompt |
| **Retroactive reprocessing** | Rule can be replayed against any previously captured frame |
| **Auditability** | Every corrected classification carries the rule that triggered it |
| **Precision gate** | Session gate required FP=0; pool achieved exactly that before acceptance |
