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
> **Also see**: [Robotics Use Case Overview](../README.md) for the broader
> context of Knowledge Fabric in embodied systems.

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
classifier returns: **no person, confidence 0.91.**

Eighteen minutes later, Commander drone C2 makes a low thermal pass over the
same sector. Its FLIR camera returns a distinct heat signature: a 37 °C oval,
30 × 20 cm, spatially stable across three consecutive frames. A human head.

S22's optical frame from 18 minutes earlier showed the same coordinates:
a pale oval region barely distinguishable from surrounding whitecaps.
Classified as **whitecap/foam, confidence 0.91.**

The person is alive.

### Why the AI fails here

A person in cold-water distress exhibits the **instinctive drowning response**:
body near-vertical, head tilted back to keep the mouth at the waterline, arms
pressing laterally outward and downward rather than stroking, minimal forward
movement. From 30 m altitude with an RGB camera, this presents as a small pale
oval — nearly identical in size, brightness, and shape to a breaking whitecap.

The key discriminating features exist in the optical image. The AI had simply
never been trained to look for them.

**Temporal stability** is the strongest discriminator: a whitecap appears and
disperses within 2–3 frames (under 1 second at typical drone frame rates); a
floating person's head maintains its position and shape across 10–20 frames.
But the classifier processes single frames and has no access to temporal
context.

Within a single frame, subtler discriminators remain:
- Brightness uniformity: a whitecap is bright at its peak and fades radially
  outward; a floating head has approximately uniform brightness across its
  full oval extent
- Bilateral V-shaped darkening flanking the oval, caused by arms pressing
  outward just below the surface — not present in floating debris of similar
  size
- Oval geometry: a clean, stable ellipse maintained above the wave crests,
  unlike the irregular, asymmetric profile of foam or debris

At 0.91 confidence "no person", S22's frame was never queued for human review.

### The DD loop

The rescue swimmer at the ground station is shown S22's RGB frame alongside C2's
thermal confirmation and asked: *"What should the optical scout have seen?"*

The swimmer responds:

> *"The head is the anchor. At this sea state, a whitecap is gone in two seconds
> — it blooms and breaks up. A person's head doesn't move like that. In a single
> frame look for the oval: a clean ellipse above the chop, consistent brightness
> all the way across — a whitecap fades from the crest outward, the head doesn't.
> And look on either side of the oval just at the waterline — you'll see a
> slight darkening in a V, that's the arms pressing down. Debris doesn't have
> that V. The body position is the thing: vertical in the water, not going
> anywhere."*

This description becomes a candidate rule. The KF grounding check verifies that
each criterion is observable by the scout tier's optical camera at operational
altitude. Two criteria are flagged:

- **Arm V-darkening**: marginal at scout resolution (1–2 cm/pixel) and wave
  chop; retained for commander tier only
- **Temporal stability**: cannot be expressed as a within-frame feature for
  single-frame classifiers; removed from scout rule, retained as commander rule

After one contrastive tightening round against 18 SeaDronesSee pool frames
(6 person-in-water, 12 whitecap/foam/debris), the scout rule achieves
precision 1.0.

**Rule accepted. Fleet broadcast. 52 seconds.**

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
single RGB frame (brightness uniformity, oval geometry, flanking V), and what
the scout tier's sensor can actually resolve at operational altitude (V-darkening
is removed; temporal stability is reformulated as within-frame proxies). The
resulting rule did not exist in any of these sources individually.

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

**The confidence trap.** The classifier scores 0.91 on the wrong answer.
Frames above the review threshold are never seen by a human analyst. A novel
presentation that closely resembles a known negative class (whitecap) will
always produce high confidence — and will always be invisible to the queue.

**The modality gap.** Thermal knowledge does not transfer to optical classifiers
by training on more optical data. The information that "there is a person here"
does not exist in RGB pixels until a human expert articulates the cross-modal
correlate. There is no supervised learning path that closes this gap without
that articulation step.

**The heterogeneous deployment problem.** Scout drones run MobileNetV3.
Commander drones run Qwen3-VL-8B. A weight update for one architecture provides
no benefit to the other. Two retraining pipelines, two validation cycles, two
deployment procedures — while the person remains in the water.

**The temporal feature problem.** The most discriminating feature (spatial
stability over time) is invisible to single-frame classifiers. DD makes this
explicit during grounding: the rule is reformulated into within-frame proxies
that approximate the temporal signal in a single image. This is a synthesis step
that standard retraining cannot perform — it requires an expert who understands
both what the temporal signal means and what within-frame proxies capture it.

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
cd usecases/robotics/drone-swarm/python
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
