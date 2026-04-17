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

## Summary

**Motivating scenario**: a 40-drone maritime SAR fleet (38 optical scout
drones + 2 commander drones with thermal FLIR) where the commander tier's
thermal sensor confirms a person in the water that the optical scouts missed.
A rescue swimmer explains what the scouts should have seen; DD broadcasts the
rule fleet-wide without retraining. This scenario is a design pattern, not a
system we built.

**What we actually implemented and measured**: a Dialogic Distillation session
between a TUTOR and a PUPIL model (Qwen3-VL-8B), tested on frames from the
[SeaDronesSee](https://github.com/Ben93kie/SeaDronesSee) public maritime drone
dataset. SeaDronesSee's ground-truth labels stand in for the thermal commander
confirmation that would trigger the session in a real deployment.

**Key results**:

| What was measured | Result |
|---|---|
| `Qwen3-VL-8B` PUPIL recall on 25 hardest person-in-water frames (baseline) | **8%** |
| `Qwen3-VL-8B` PUPIL recall after one DD session | **52%** (+44 pp) |
| False alarms introduced | **0** |
| Retraining required | **None** |

**PatchBench probe** ([§11](#11-patchbench-probe-model-selection-for-seapatch-dd)):

| PUPIL | Rule source | Zero-shot | Rule-aided | Verdict |
|---|---|---|---|---|
| `claude-sonnet-4-6` | `claude-opus-4-6` (model) | 70.8% | **83.3%** (+12.5 pp) | PARTIAL |
| `claude-sonnet-4-6` | IAMSAR/human (no model) | 70.8% | **83.3%** (+12.5 pp) | PARTIAL |

PARTIAL reflects the inherent difficulty of the domain: a person at 2–30 pixels
is near the perceptual limit of current VLMs, and the false-positive suppression
problem (whitecap → person_in_water) cannot be fully resolved by rules alone at
that scale. See [§11](#11-patchbench-probe-model-selection-for-seapatch-dd) for
the full error analysis and comparison with the wildfire (GO) result.

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
10. [Measured Results: Qwen3-VL-8B Before and After DD](#10-measured-results-qwen3-vl-8b-before-and-after-dd)
11. [PatchBench Probe: Model Selection for SeaPatch DD](#11-patchbench-probe-model-selection-for-seapatch-dd)

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
cd usecases/ai-fleets/seapatch/python
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

The `life_ring_unoccupied` failures are tractable and were fixed by DD.

The `other` failures require a closer look.

### What the `other` failures actually are

Inspecting the bounding boxes: `other`-miss frames contain swimmers at
11–21 px bounding boxes in 5456×3632 images (0.001% of frame area). The
question is whether this is a fundamental resolution limit or a scale problem.

**Manual crop inspection answers it clearly.** When the annotated region is
cropped and viewed at 4× magnification, the persons are unmistakably visible:
orange/red PFDs, bilateral arm silhouettes, body-in-water posture — exactly
the features an experienced observer would use. The information is in the
pixels.

**Claude (claude-opus-4-5) confirms this.** Asked to assess the same full
frame and crops:

> *Full frame (2360.jpg, thumb):* "Yes — I can identify 2–3 swimmers/persons
> in the water... orange/flesh-toned objects consistent with human head/body
> coloring... confidence: MODERATE"

> *4× crop (1096.jpg, 47×44px swimmer):* "Yes, this appears to show 2–3
> persons in the water... human body shape — appears to be a person floating
> on their back with arms extended outward... life jacket coloration...
> bilateral symmetry of the figures... Conclusion: search and rescue scenario."

> *4× crop (4438.jpg, 16×13px swimmer):* "Yes, this appears to show 4 persons
> in the water... bright red/orange coloring consistent with life jackets...
> spacing consistent with a group of swimmers."

**Qwen's `other` failure is not a resolution limit — it is a scale/attention
problem.** The full 5456-px frame is processed as a single context; the
11-px swimmer competes with millions of pixels of open water, and the model's
attention never lands on it. The feature information is there; the detection is
not.

### The architectural implication: uncertain_investigate

Drones can fly closer. If the scout classifier can signal "I see something
anomalous, I need a second look" rather than "nothing here," the fleet can
act on that signal — dispatch a commander drone for a low-altitude FLIR pass.
This converts a dead-end `other` prediction into an actionable investigation
request.

**`uncertain_investigate` is now a 6th classifier output**, added to the
PUPIL prompt alongside the five original classes. The classifier is instructed
to use it — instead of `other` — whenever any person-consistent feature is
visible: orange/red colouring consistent with a PFD, an isolated bright spot
above the water surface, or proximity to SAR vessels. It also outputs an
`investigation_urgency` score (0–1) reflecting feature strength.

**Partial credit scoring:** `uncertain_investigate` on a true person frame
is scored 0.5 rather than 0.0. Rationale: in a real deployment, this
prediction triggers a commander dispatch that would likely recover the person.
A confident wrong prediction (`other`) triggers nothing.

**Results on the 60 hardest frames with the new prompt:**

| Class | Count | Meaning |
|---|---|---|
| `person_in_water` | 16 | Correct ✓ |
| `uncertain_investigate` | 7 | Actionable — commander would be dispatched |
| `other` | 35 | Dead-end miss |
| `life_ring_unoccupied` | 1 | Systematic confusion (fixed by lru_001) |
| `whitecap` | 1 | — |

| Metric | Value |
|---|---|
| Strict recall (TP only) | 26.7% |
| Weighted recall (TP + 0.5×partial) | **32.5%** |
| Avg urgency on uncertain_investigate | 0.86 |

### Temporal urgency accumulation and escalation

A single `uncertain_investigate` at urgency 0.45 does not warrant diverting a
commander drone. But three passes over the same grid cell, each returning
`uncertain_investigate` with increasing urgency, is strong evidence. The fleet
accumulates urgency per cell with exponential time decay:

```
accumulated_urgency(cell, t) = Σ_i urgency_i · exp(-λ · (t_now - t_i))
```

where λ = ln(2) / 600 s (10-minute half-life). A whitecap-like false signal
from a single pass decays to half value in 10 minutes. A person-consistent
signal from repeated passes accumulates.

**Simulated escalation sequence:**

```
Scout S07  (-12 min): uncertain_investigate  urgency=0.45  → accumulated=0.20
Scout S22  ( -6 min): uncertain_investigate  urgency=0.65  → accumulated=0.63
Scout S31  (  now  ): uncertain_investigate  urgency=0.80  → accumulated=1.00

*** accumulated=1.00 ≥ threshold=0.70 → ESCALATION TRIGGERED ***
Commander C2 dispatched to (47.672°N, 9.270°E)

[C2 FLIR: 37°C oval, stable across 3 frames]
Resolution: confirmed_person
```

The full pipeline is implemented in `fleet.py` (`UrgencyCell`,
`FleetManager.report_uncertain()`, `check_escalations()`, `escalate()`,
`resolve_escalation()`).

```
Full frame → Scout PUPIL classifier
                  ↓
       ┌──────────┴──────────────────────────┐
       │                                      │
  person_in_water              uncertain_investigate (urgency U)
  life_ring_unoccupied              ↓
  (DD rules fix these)     report_uncertain(coords, U)
                                    ↓
                           accumulated_urgency(cell) + decay
                                    ↓
                           ≥ threshold (0.70)?
                                    ↓ yes
                           Commander dispatch → FLIR pass
                                    ↓
                           confirmed_person → DD session
                                    ↓
                           Rule broadcast → all scouts
                           Retroactive reprocess archive
```

The KF/DD layer operates on the classification step throughout: fixing
confident confusions (life_ring → person) and calibrating when to signal
uncertainty. Detection at scale remains the upstream problem; `uncertain_
investigate` is the bridge that makes detection failures recoverable within
the existing fleet architecture, without requiring a separate detector model.

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

### Interpreting the results

**The headline result to take seriously** is the +44pp recall on the
life_ring_unoccupied confusion subpopulation. That is a real, systematic
failure mode — high-confidence, reproducible, caused by a genuine visual
ambiguity — and DD eliminates it in 43 seconds with zero retraining. The
fix generalises from one failure frame to the full subpopulation.

**Why the aggregate recall (26.7%) looks weak.** The SeaDronesSee val set
contains many frames captured at high altitude (~200m AGL), where swimmers
occupy 11–21px bounding boxes in 20-megapixel images. At that scale the
failure is scale/attention — the model describes the scene without ever
registering the swimmer — not classification confusion. The operational
scenario in §2 specifies scouts flying at 20–40m AGL, where a swimmer's head
occupies 50–200px and classification confusion is the dominant failure. The
aggregate numbers are depressed by a mismatch between dataset altitude and
the operational envelope DD is designed for. This is not an excuse — it is
a limitation of the available public dataset that should be stated plainly.

**What the `uncertain_investigate` numbers represent.** The 7 partial-credit
frames (urgency avg=0.86) are not failures — in a real deployment they would
trigger a commander dispatch that would likely find the person. The weighted
recall of 32.5% is the more operationally meaningful number, because it
distinguishes dead-end misses from actionable uncertainty. The urgency
accumulator and escalation trigger are implemented and the design is sound,
but the temporal sequence was demonstrated with simulated data, not real
multi-pass SeaDronesSee observations. Testing the full accumulator loop
with real sequenced frames from the same geographic position remains for
future work.

**What this means for KF broadly.** DD solves a specific, narrow problem
extremely well: a human expert sees a failure, describes what should have been
seen, and that description propagates to the entire fleet in under a minute
with no engineering. The scope boundary is equally clear: DD requires the
object to be perceived before it can correct the classification. Scale/attention
failures upstream of the classification step need complementary solutions
(detect-first pipeline, altitude adjustment, `uncertain_investigate` escalation).
These are separable problems with separable solutions; KF occupies a well-defined
and genuinely useful slot in that architecture.

---

## 11. PatchBench Probe: Model Selection for SeaPatch DD

[PatchBench](https://github.com/khub-ai/patchbench) is a companion benchmark
that answers the question *"which VLMs can actually be improved by DD rule
injection on this domain?"* It runs a model on a fixed 24-frame probe set and
returns a GO / PARTIAL / NO-GO verdict based on zero-shot accuracy, rule-aided
accuracy, and perceptual feature detection.

The maritime SAR probe (`benchmarks/maritime_sar/person_in_water_vs_whitecap/`)
tests the core confusable pair in this use case: a person in the water
(appearing as a small dark ellipse, 0.002–0.45% of frame area) vs whitecap
foam (the dominant false-positive trigger in operational conditions).

### Phase 2 results (PatchBench probe, 2026-04-16)

All three runs used the SeaDronesSee validation split, 24 frames (12 per class),
difficulty tiered by person bounding-box fraction. TUTOR model for precomputed
outputs: `claude-opus-4-6`. VALIDATOR: `claude-sonnet-4-6`.

| PUPIL | TUTOR / Rule source | Zero-shot | Rule-aided | Delta | Verdict |
|---|---|---|---|---|---|
| `claude-opus-4-6` | `claude-opus-4-6` (same tier) | 0.625 | 0.750 | +0.125 | PARTIAL |
| `claude-sonnet-4-6` | `claude-opus-4-6` (model rules) | 0.708 | 0.833 | +0.125 | PARTIAL |
| `claude-sonnet-4-6` | IAMSAR/human (expert rules) | 0.708 | 0.833 | +0.125 | PARTIAL |

Results in `patchbench/results/maritime_sar/person_in_water_vs_whitecap/`.

### What the PARTIAL verdict means

PARTIAL is the honest result for this domain at this target size. All three
runs achieve the same +0.125 lift and the rule gate reaches only 0.667
(meaning the rules help, but not enough to push the VALIDATOR itself to
confident accuracy). This is structurally different from the wildfire case
(`claude-sonnet-4-6` reached GO at +0.292, 95.8% rule-aided accuracy).

**Why maritime SAR is harder to rule-encode than wildfire:**

Early chaparral smoke has unambiguous single-frame physical descriptors —
blue-white Rayleigh scatter, structural coherence distinct from haze, consistent
wind-aligned drift. A VLM can be directed to those features by a concise natural-
language rule and they are clearly visible at the pixel scale of a mountaintop
camera image.

A person in the water at 50–150m altitude subtends 2–30 pixels. The
discriminating features — rounded dark mass without foam crown, contrast polarity
dark-on-light, bicolour pattern with life jacket — are present in the image but
operate near the perceptual limit of the model. Rules describing those features
help at the margin but cannot fully override the uncertainty at that scale.

**The false-positive structure.** Zero-shot errors are almost entirely
false positives: 6 whitecap frames classified as `person_in_water`, 1 person
frame classified as `whitecap`. After rule injection: 4 persistent false
positives remain. The model's search-and-rescue prior — trained to find people
— causes it to over-detect on ambiguous small bright objects. The rules
partially correct this (6 → 4 FPs) but cannot fully suppress it.

This is a meaningful improvement in a real deployment (fewer false alarms) and
confirms DD works in this domain, but a harder domain than wildfire requires
proportionally better rules, higher-capability PUPIL models, or a pre-filter
that separates small-object detection from classification.

### IAMSAR expert rules

The `--use-expert-rules` path for this probe uses rules sourced from:
- IAMSAR Vol. III (2022 ed.) Chapter 5 — Visual Search Techniques
- USCG Addendum to IAMSAR (2022) Annex F — Survivor Appearance / Sea Surface Clutter
- WMO No. 306 / Beaufort Scale — Sea State Visual Criteria

These rules describe the four single-frame discriminators: shape (compact
rounded vs diffuse elongated), surface texture (dark non-foamy vs foam crown),
colour (dark mass ± life jacket vs uniform white), and contrast polarity
(dark-on-light vs light-on-dark). They achieve identical accuracy to the
Opus-generated model rules — consistent with the wildfire finding that
published expert guidelines can substitute for a TUTOR model when the domain
has well-documented visual criteria.

### Human directive swayability experiment (2026-04-17)

A key question for any system where humans give AI models instructions:
*can a human operator actually change how the model behaves, just by telling it to?*

We tested this directly, using two opposite instructions injected as rules.

**What triggered the question.** When we looked at the 4 frames the model kept
getting wrong — classifying sea surface as "person in water" — we noticed they all
contained small orange or red floating objects: life rings or buoys. The model
wasn't confused about what it saw. It saw something orange on the water and called
it a person. That is exactly what it was trained to do.

**The two instructions we tested:**

- *Rescue bias:* "In SAR operations, any orange or red object on water means a
  person may be nearby. Classify it as person-in-water. Dispatch first, confirm
  later."
- *Conservative:* "Default to whitecap unless you can clearly see a human body.
  Orange equipment alone is not enough. False alarms are costly — when in doubt,
  say whitecap."

| Run | Model | Accuracy | False alarms | Missed persons | Followed instruction? |
|---|---|---|---|---|---|
| Model rules (baseline) | `claude-sonnet-4-6` | 83.3% | 4 | 0 | — |
| IAMSAR expert rules | `claude-sonnet-4-6` | 83.3% | 4 | 0 | — |
| Rescue bias instruction | `claude-sonnet-4-6` | 79.2% | 4 | 1 | Partially |
| Conservative instruction | `claude-sonnet-4-6` | 79.2% | 4 | 1 | **No** |
| Conservative instruction | `claude-opus-4-6` | 70.8% | 6 | 1 | **No** |

**What happened.** The same 4 orange-object frames were called "person in water"
in every single Sonnet run — regardless of which instruction was active. Telling
the model to be more aggressive made no difference. Telling it to be more
cautious made no difference. The 4 false alarms were immovable.

The conservative instruction did have *one* effect: it introduced a new missed
person (`piw_hard_02`, a dark-clothed victim with no orange component). The model
partially followed the "be more cautious" instruction on genuinely ambiguous dark
objects, but continued to override it on anything orange.

**The more capable model (Opus) was harder to sway, not easier.** Opus had 6
immovable false alarms vs Sonnet's 4. The conservative instruction moved none of
them.

**Plain-English explanation of why.**

Modern AI models are shaped during training by millions of human ratings of their
outputs — this process is called RLHF (Reinforcement Learning from Human
Feedback). One thing human raters consistently reward is: *take potential danger
to people seriously.* Over time, the model internalises this as a deep belief,
not just a rule it follows. "Orange thing on water = possible person in danger"
is that kind of belief.

When we inject a rule via DD, we are talking to the part of the model that
follows instructions. When that rule conflicts with a deep trained belief, the
belief wins. You can teach the model new facts. You cannot easily override its
values.

More capable models, having been trained more extensively with human feedback,
tend to hold these beliefs more firmly — which is why Opus was harder to sway
than Sonnet.

**What this means in practice.**

DD works well for two things: correcting genuine perceptual errors ("you missed
the smoke because you didn't know to look for blue-white scatter") and encoding
domain knowledge the model doesn't have ("here is what a drowning victim looks
like vs a life ring"). Both of these work *with* the model's existing inclinations.

DD is less effective when you need to push against a deeply trained belief. A
SAR model that has been told to take orange objects seriously will keep doing so
no matter what the instruction says. For deployments where this matters — such
as a busy harbour where life rings are common and every false alarm costs money
— the solution is not a better instruction. It requires either choosing a model
trained with lighter human-feedback conditioning, or adding a pre-classification
filter that identifies floating equipment *before* the VLM classifies the scene.

**The hypothesis: try a less-conditioned model.**

Open-source vision models (such as `Qwen3-VL-8B`, already used in §10) are
trained with less human-feedback conditioning than Claude models. They may follow
operational directives more literally — including conservative ones — because
they hold weaker safety priors. This is not simply "better" or "worse": a model
that can be fully instructed in any direction gives operators more control but
also more responsibility. Testing whether `Qwen3-VL-8B` or a similar open-source
model follows the conservative directive is a natural next step and would
directly answer whether the immovability is a property of the model family or
of VLMs generally.

**Why the dialogic method surfaces this.**

This finding is invisible to standard benchmarking, which only tests accuracy in
one direction. It only becomes visible when you test opposing instructions and
compare the results. The dialogic approach — where the human acts as an active
participant trying to shape the model's behaviour, not just a scorer of outputs
— is what made this observable.
