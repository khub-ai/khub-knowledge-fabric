# PyroWatch: Fleet-Scale Early Ignition Detection Through Cross-Modal Knowledge Propagation

> **New to [Dialogic Distillation](../../dialogic-distillation/README.md) (DD)?**
> This use case involves three hardware tiers, cross-modal sensing (MWIR + optical),
> and environmental-context-conditional rules — it is among the more advanced
> applications in this repository. If you prefer to start with something simpler,
> the [bird species identification](../../image-classification/birds/README.md) and
> [dermatology](../../image-classification/dermatology/README.md) use cases
> demonstrate the same core DD mechanism on single-camera, single-domain tasks
> with results you can reproduce in under an hour.
>
> **Status**: Implementation complete (Phase 1). Measured results available
> in [§10](#10-measured-results). Python implementation in `python/` —
> domain config, three-tier agents, FIgLib pool builder, session runner, and
> end-to-end experiment script. Runs directly on the
> [FIgLib](https://github.com/brain-facens/FIgLib) public dataset of real wildfire
> ignition sequences from ALERTWildfire and HPWREN mountaintop cameras.
>
> **Also see**: [AI Fleets Use Case Overview](../README.md) for the broader
> pattern and the [SeaPatch](../drone-swarm/README.md) maritime SAR use case for
> an implemented and measured reference.

---

## Summary

**Motivating scenario**: a 164-agent wildfire detection fleet — 120 ground
sentinel cameras, 40 scout drones, 4 commander aircraft — where the commander
tier's MWIR sensor confirms ignitions that the optical tier misses. An expert
explains the distinguishing features; DD broadcasts the rule fleet-wide without
retraining. This scenario is a design pattern, not a system we built.

**What we actually implemented and measured**: a Dialogic Distillation session
between a TUTOR and a PUPIL model, tested on frames from the
[FIgLib](https://github.com/brain-facens/FIgLib) public dataset of real HPWREN
mountaintop camera ignition sequences. The TUTOR role can be filled by a
superior model, a human expert, or expert knowledge drawn from published domain
guidelines — all three are demonstrated below. FIgLib's ground-truth ignition
timestamps stand in for the MWIR commander confirmation that would trigger the
session in a real deployment.

**Key results** (full details in [§10](#10-measured-results)):

| What was measured | Result |
|---|---|
| Sonnet PUPIL accuracy before DD | 66.7% |
| Sonnet PUPIL accuracy after DD (Opus TUTOR) | **95.8%** (+29 pp) |
| Sonnet PUPIL accuracy after DD (NWCG human rules) | **95.8%** (+29 pp) |
| PatchBench verdict | **GO** |

---

## Contents

1. [Start Here: The Simple Version](#1-start-here-the-simple-version)
2. [The Full Scenario: Santa Ana Wildfire Ignition](#2-the-full-scenario-santa-ana-wildfire-ignition)
3. [What DD Does Here](#3-what-dd-does-here)
4. [Without DD vs With DD](#4-without-dd-vs-with-dd)
5. [Why This Is Hard Without DD](#5-why-this-is-hard-without-dd)
6. [Fleet Architecture](#6-fleet-architecture)
7. [The Three Roles of DD in This Context](#7-the-three-roles-of-dd-in-this-context)
8. [New DD Capabilities: Environmental Context and Temporal Reformulation](#8-new-dd-capabilities)
9. [Simulation Setup](#9-simulation-setup)
10. [Measured Results](#10-measured-results)
11. [Getting Started](#11-getting-started)

---

## 1. Start Here: The Simple Version

Before the drone fleet scenario, consider a simpler version of the same
problem.

A state forestry service operates 80 staffed fire lookout towers across a
mountainous region. Each tower has been upgraded with an AI camera system to
assist rangers during long overnight and afternoon shifts. The system performs
well on obvious cases — large smoke columns, visible flame, mature plumes
visible at 20 km. Management is satisfied.

One afternoon in peak fire season, a ranger at Tower 23 notices a faint blue
haze rising from a distant canyon, roughly 4 km out. The AI camera reviews the
same scene and returns: **heat_shimmer, confidence 0.93.** The alert is
suppressed. The ranger, with 14 years on the job, trusts her eyes, not the
system. She calls it in. Engines are dispatched. It is a real ignition — a
downed power line in dry chaparral, 0.1 acres. In 30 minutes at the measured
wind speed, it would have reached a subdivision.

The investigation asks the ranger to explain what she saw that the AI missed.
She says:

> *"Blue smoke. That is the thing nobody teaches people to look for. Big fires
> are gray — you're burning wood and carbon. But early fire, especially live
> chaparral, burns the plant oils first. Terpenes. That smoke is blue. Almost
> invisible in afternoon light, but it's a different color from shimmer —
> shimmer is transparent, you're seeing the hill through it. Real smoke has
> color. And it has a point source. It rises from one spot, then the wind takes
> it. Shimmer oscillates across a whole hillside simultaneously. Those are two
> completely different spatial patterns. The camera is seeing both as the same
> thing."*

That description becomes a candidate rule. The KF grounding check verifies
that each feature is observable by the tower cameras at 4 km range: blue color
shift — yes, in good light, at this resolution. Point source vs. distributed
pattern — yes, detectable across consecutive frames. Consistent wind-direction
drift — yes, measurable across a 10-second clip.

The rule is tested against a pool of 10 early-smoke frames and 10 confirmed
heat-shimmer frames drawn from the ALERTWildfire public archive. Precision
1.00, zero false positives.

**Rule accepted. Broadcast to all 80 towers. 41 seconds.**

Within the same session, the system retroactively reprocesses the past 6 hours
of footage from all 80 towers. It finds one prior incident — a false dismissal
from Tower 51, three days earlier — where the same blue-haze pattern appeared
for 9 minutes before a ranger spotted it manually. That incident is now in the
record.

**This is DD as an instant patch.** An expert describes what the AI should have
seen. The description becomes a rule. The rule reaches every camera in the
network immediately. The archived footage is re-examined with fresh eyes.

The drone fleet scenario below is the same phenomenon — the same early blue
smoke, the same point-source signature — now operating at greater scale, with
heterogeneous hardware tiers and one additional element: the expert knowledge
that unlocks the fix arrives from a *different sensor* (MWIR thermal) than the
cameras that must act on it (optical RGB).

---

## 2. The Full Scenario: Santa Ana Wildfire Ignition

### Setup

A red flag warning is in effect for the Santa Ana foothills: 95°F, 8% relative
humidity, 25–35 mph sustained northeast winds. A coordinated air-and-ground
detection fleet covers the region:

| Tier | Count | Sensor | Compute | Role |
|---|---|---|---|---|
| Ground Sentinel | 120 | RGB PTZ 4K + LWIR thermal (320×240, uncalibrated) | Edge TPU | Fixed mountaintop/ridgeline coverage, continuous 360° scan |
| Scout Drone | 40 | RGB 20MP + LWIR thermal (640×480) | Jetson Nano | Patrol routes over WUI corridors and high-risk terrain |
| Commander Aircraft | 4 | RGB 60MP + MWIR (640×512, temperature-calibrated) + anemometer | Jetson AGX Orin + Claude Sonnet | Long-loiter surveillance, TUTOR-capable, hot-spot confirmation |

The ground sentinels cover 100% of the monitored terrain continuously. The
scout drones cover their assigned corridors on 12-minute repeat cycles. The
commander aircraft operate on 45-minute sector sweeps and can be redirected on
demand.

### The failure

At **14:23**, a downed power line in a dense chaparral draw ignites a spot
fire approximately 0.1 acres in area. Ground Sentinel GS-047, positioned on
the ridgeline 350 m above, sweeps the draw as part of its regular rotation.
Its PTZ camera captures the scene at 14:23:18. The classifier returns:

**heat_shimmer_artifact, confidence 0.94**

At 14:29, Scout Drone SD-19 completes a pass over the same draw. The LWIR
reading shows a marginally elevated temperature patch — 12°C above ambient —
but the classifier's threshold for fire-anomaly is 40°C above ambient.
SD-19's return: **terrain_heat_differential, normal, confidence 0.91**

GS-047 sweeps the draw three more times between 14:23 and 14:41. Each pass
returns the same classification. No human alert is generated. The queue never
surfaces the frames for review.

At **14:41**, Commander Aircraft CA-2 makes a routine low pass over the
sector. Its MWIR sensor — calibrated for absolute temperature, not relative
contrast — returns a 380°C surface anomaly over a 6 m² area at the exact
coordinates GS-047 had flagged as shimmer 18 minutes earlier.

380°C is well above the 280°C chaparral combustion threshold. The fire is real.

CA-2's high-resolution RGB captures what GS-047 saw at lower resolution: a
faint blue-gray haze, 8–10 metres above the draw floor, drifting southeast at
a bearing consistent with the measured 28 mph wind. The column has a
well-defined base. The surrounding hillside shows no comparable pattern.

By 14:41 the fire is 0.8 acres. At measured wind speed, it will reach the
nearest WUI boundary in 22 minutes.

### Why the AI fails

**The LWIR calibration gap.** Low-cost LWIR thermal cameras (320×240,
uncalibrated) measure relative temperature contrast, not absolute temperature.
On a 95°F afternoon, sun-heated dark rock and chaparral can reach 160–200°C
surface temperature — producing a strong LWIR return that the classifier has
learned to label as normal terrain heat. A 280–380°C ignition adds only a
marginal increment in the uncalibrated LWIR signal. The threshold that would
catch the fire also produces dozens of false positives per day from hot rock
faces.

CA-2's MWIR camera operates in the 3–5 μm band. At that wavelength, Wien's
displacement law places the blackbody emission peak for fire (600–900°C) in
the MWIR band — fires are dramatically brighter relative to background in MWIR
than in LWIR. The temperature calibration allows absolute thresholding at
280°C. This is not available to the sentinel or scout tiers.

**The visual confusion.** Early chaparral smoke is blue because live
vegetation burns terpenes and volatile plant oils before cellulose — the
resulting particles have a Rayleigh scattering profile that produces a blue
cast at distance. In harsh afternoon backlight at 35 mph wind, this blue haze
is nearly transparent and covers only a few pixels at GS-047's range. The
classifier was trained predominantly on large, mature smoke plumes (gray,
opaque, billowing) and on heat shimmer drawn from hours of normal operation.
The early-ignition signature sits in neither training distribution.

**The wind distortion.** At 35 mph the smoke column does not rise vertically
before dispersing. It is pushed nearly horizontal from the point of origin,
creating a flat, drifting haze rather than an upright column. The temporal
signature the classifier would use to distinguish "rising smoke" from "drifting
shimmer" is absent or reversed.

**The spatial attention problem.** GS-047's PTZ camera covers a 120° sweep.
The ignition source occupies roughly 15×12 pixels in the full frame. The
classifier processes the full frame; at that spatial ratio, attention rarely
lands on the correct region.

### The DD loop

The incident commander on duty is shown GS-047's 14:23 RGB frame alongside
CA-2's MWIR confirmation at 14:41. She is a 19-year Cal Fire lookout and
aerial observer.

> *"Classic sleeper start. What you're looking at is terpene smoke — that
> blue-gray is the plant oils, not the wood. You almost never see it unless
> you've looked at hundreds of these. In 35-mph Santa Ana it doesn't go up,
> it goes sideways. But here's the thing: it always comes from a point. You
> will never see shimmer with a point source. Shimmer is a hillside phenomenon
> — it shimmers everywhere the rock face is hot. Real smoke has an origin, a
> single spot, and it fans from there. In this frame I can see the point. The
> column base is right there — look at where the blue is densest. And the
> drift is consistent: it's going southeast every frame, exactly the wind
> direction. Shimmer oscillates — it doesn't drift in one direction. Those two
> features, right there in the pixels."*

This description becomes a candidate rule. The KF grounding check runs
three validations:

1. **Blue color shift vs. transparent shimmer**: Observable at sentinel range
   (350 m, 4K RGB) under good lighting — retained for all tiers. Flagged
   "reduce confidence in backlit afternoon conditions."

2. **Point source vs. distributed spatial pattern**: Observable in a single
   RGB frame at this resolution — retained for all tiers.

3. **Consistent drift direction across frames**: Requires consecutive frames.
   Retained for sentinel tier (PTZ covers the same area continuously).
   Reformulated for scout drones as "asymmetric elongation along prevailing
   wind bearing" — a single-frame proxy detectable in a drone pass. Removed
   for commander tier (MWIR provides a direct confirmation and this rule is
   not needed there).

4. **Environmental context modifier** (added by the KF validator): Under red
   flag conditions (temperature > 90°F, humidity < 15%, wind > 20 mph), the
   base activation threshold for any blue-haze-with-point-source pattern is
   lowered. The validator confirms that RAWS (Remote Automated Weather Station)
   meteorological data is available in the ground station API and can be
   injected as rule context.

Pool validation: 8 early-smoke frames from FIgLib (real ignition sequences,
ground-truth labeled) and 8 confirmed heat-shimmer frames from ALERTWildfire
historical archive. Rule achieves precision 1.00, recall 7/8. One miss is
a nighttime frame where color information is unavailable — flagged as
out-of-scope for the optical rule; a separate thermal rule applies after dark.

**Rule accepted. Fleet broadcast. 38 seconds.**

### The outcome

All 120 ground sentinels and 40 scout drones retroactively reprocess their
18-minute archive against the new rule. GS-047's 14:23:18 frame is
reclassified as **early_smoke_signature, investigate_required, urgency 0.87**.
The RAWS data confirms red flag conditions were active at 14:23.

Incident report filed with coordinates. The earliest confirmed detection is
now 14:23 — 18 minutes earlier than CA-2's MWIR pass. Engines are rerouted.
The fire is contained at 0.1 acres.

In the following two weeks of deployment, the rule triggers on three further
events before a human observer called them in, including one nighttime
smoldering event caught by an independently developed thermal variant of the
rule.

---

## 3. What DD Does Here

This scenario extends the SeaPatch maritime case in two directions. DD plays
its three standard roles — Patch, Synthesizer, Propagator — and two new ones
unique to this domain.

**Patch** — the same role as in all AI Fleets use cases. A deployed classifier
has a blind spot for early terpene smoke. The fix reaches 160 agents instantly,
without retraining.

**Synthesizer** — the expert's rule is not a translation of MWIR knowledge
into optical terms alone. It integrates: what the MWIR calibrated reading
reveals (confirmed ignition temperature), what physical reality that corresponds
to in an optical RGB frame (blue haze with point source), what the ambient
meteorological context is (red flag conditions lower the threshold), and what
each sensor tier can actually resolve (color shift available to sentinel and
scout; drift direction available only to sentinel or reformulated for scout).
No single agent held all of this.

**Propagator** — the rule reaches 160 heterogeneous agents across three
hardware tiers in a single broadcast. MobileNetV3 edge classifiers on sentinels,
YOLOv8-based detectors on scouts, and Qwen3-VL-8B on commanders all receive
and apply the same natural-language rule without separate retraining pipelines.

**Environmental context injector** *(new)* — the rule includes a
meteorological precondition: RAWS-reported red flag conditions lower the
activation threshold. This is a DD capability not present in SeaPatch: rules
can be conditioned on real-time external state (weather, time of day, season)
injected at inference time. No retraining is needed to make the classifier
more aggressive during fire season and more conservative in spring.

**Tier differentiator** *(new)* — the rule is not uniform across tiers.
One grounded rule specification produces three tier variants: one for sentinels
(multi-frame drift), one for scouts (single-frame wind-elongation proxy), and
one for commanders (rule not needed; MWIR is definitive). SeaPatch had two
tiers; three tiers introduces the question of whether a feature present in the
expert's description is reformulable for each tier's specific constraints, or
must be dropped. The grounding check makes this explicit per tier.

---

## 4. Without DD vs With DD

| Dimension | Without DD | With DD |
|---|---|---|
| Early-stage smoke detection | Dependent on human observer calling it in | Automated rule covers the visual signature fleet-wide |
| Novel presentation (sleeper start) | Never queued; invisible to review | Caught — rule fixes it in under a minute |
| Time to fleet-wide update | 8–24 hours (retrain, validate, deploy per tier) | Under 2 minutes |
| Retroactive reprocessing | Not possible | Immediate, same session |
| Expert knowledge reach | One lookout's shift | All 160 agents, every future scan |
| Heterogeneous fleet update | Three separate pipelines (edge MCU, Jetson Nano, Jetson AGX) | Single rule broadcast |
| Environmental context | Baked into weights at training time; cannot change by season | Injected at inference time via RAWS API; adjustable without retraining |
| Tier-specific rule adaptation | Requires three separate retraining jobs | Handled in grounding check; single DD session |
| Auditability | Confidence score only | Full rule trace + meteorological context per classification |
| Rule revocation | Full retrain required | Delete from rule pool |
| Data requirement for fix | Hundreds of labeled early-smoke frames per tier | 16–20 pool frames total |

---

## 5. Why This Is Hard Without DD

**The rarity problem.** A genuine early ignition — sub-0.5 acres, first
5 minutes, terpene-stage smoke — is vanishingly rare in any labeled training
dataset. The ALERTWildfire archive contains tens of thousands of hours of
normal-terrain footage and hundreds of mature fire events. Pre-ignition blue
haze sequences are measured in tens of examples. Standard retraining would
require systematic data collection under controlled conditions before a single
weight update could be made.

**The confidence trap.** GS-047 returns 0.94 on "heat_shimmer." Frames above
the review threshold are never shown to a human analyst. The failure mode
looks exactly like a correctly classified non-event: high confidence, familiar
category, appropriate to the environmental context. There is no signal in the
output that would trigger review.

**The calibration gap.** LWIR and MWIR cameras are not interchangeable. The
key diagnostic feature — absolute surface temperature above combustion
threshold — is only available from a calibrated MWIR instrument. No amount of
additional LWIR training data teaches the system to make absolute temperature
distinctions the sensor cannot provide. The information does not exist in the
scout/sentinel data stream. It becomes available only when the commander tier
makes a pass — and only if the commander's knowledge can be transferred back
to the cheaper sensors that cover the terrain continuously.

**The temporal gap.** "Smoke drifts consistently; shimmer oscillates" is a
multi-frame temporal feature. Classifiers that process single frames cannot
access it directly. Getting this feature into a single-frame classifier by
retraining requires temporal supervision — labeling sequences, not frames —
multiplying the data collection burden. DD bypasses this by asking the expert
for single-frame proxies that carry the same discriminating information:
asymmetric elongation in the wind direction, color saturation at the point
source. These are reformulations of a temporal insight into an observable
spatial signal.

**The environmental context problem.** The same blue haze that warrants an
alert under red flag conditions (hot, dry, windy) might be fog or pollution
under normal spring conditions. The classifier cannot change its prior based
on current meteorological state without a retrain that bakes the condition
into the weights. DD rules can include meteorological preconditions that are
evaluated at runtime from the RAWS API — the same rule behaves differently
in fire season versus spring without any weight change.

**The three-tier retraining problem.** The fleet has three hardware tiers with
three different model architectures and three different sensor specifications.
A weight update that fixes the sentinel classifier does not touch the scout
classifier or the commander's language model. Three separate retraining
pipelines, three validation cycles, three deployment procedures. Under red
flag conditions, the fire does not wait.

---

## 6. Fleet Architecture

```
Ground Station
┌──────────────────────────────────────────────────────────────┐
│  Incident Commander / Cal Fire Lookout (TUTOR)               │
│  KF Grounding Validator (Claude Sonnet)                      │
│  RAWS Meteorological API (real-time weather context)         │
│  Rule Pool (tier-differentiated; broadcast over mesh)        │
│  Sector Heat Map (coordinates → detection class + urgency)   │
└──────────────┬───────────────────────────────────────────────┘
               │ mesh network / LTE
        ┌──────┴──────┐
        │  Commander  │  ×4   RGB 60MP + MWIR (temp-calibrated)
        │  Aircraft   │       + anemometer
        │  (TUTOR-    │       Jetson AGX Orin
        │   capable)  │       Qwen3-VL-8B
        │             │       Long-loiter, MWIR confirms ignition
        └──────┬──────┘
               │ rule broadcast (38 seconds fleet-wide)
        ┌──────┴──────┐
        │ Scout Drone │  ×40  RGB 20MP + LWIR (640×480)
        │   (PUPIL)   │       Jetson Nano
        │             │       Single-frame proxy rules
        │             │       12-min corridor repeat cycle
        └──────┬──────┘
               │
        ┌──────┴──────┐
        │   Ground    │  ×120 RGB PTZ 4K + LWIR (320×240, uncal)
        │  Sentinel   │       Edge TPU
        │   (PUPIL)   │       Multi-frame drift rules
        │             │       Continuous 360° scan
        └─────────────┘
```

The commander tier detects novel ignition presentations via MWIR absolute
temperature thresholding — the definitive sensor. The expert's description
creates rules for the two PUPIL tiers, which are the agents with continuous
coverage. Each PUPIL tier receives a rule variant adapted to its sensor and
processing constraints. Rules from any tier can trigger retroactive reprocessing
across all tiers.

---

## 7. The Three Roles of DD in This Context

| Role | What it does | SeaPatch equivalent |
|---|---|---|
| **Patch** | Fixes a blind spot in deployed classifiers, instantly, fleet-wide | Yes — identical |
| **Synthesizer** | Integrates MWIR confirmation, expert optical reasoning, meteorological state, and per-tier sensor constraints into tier-differentiated rules — producing single-frame proxies for multi-frame temporal features | Partial — SeaPatch synthesizes thermal → optical; PyroWatch additionally integrates environmental context and produces three tier variants from one session |
| **Propagator** | Broadcasts tier-differentiated rules to three heterogeneous hardware tiers; applies retroactively to archived frames | Partial — SeaPatch has two tiers; three tiers adds explicit tier-routing logic |

Two capabilities that are new relative to SeaPatch:

| New Capability | What it does |
|---|---|
| **Environmental context injection** | Rules include meteorological preconditions (RAWS data) evaluated at inference time. The same rule activates aggressively under red flag conditions and conservatively in spring — without any weight change |
| **Tier differentiator** | A single DD session produces a distinct rule variant per tier, each adapted to that tier's observable features. The grounding check handles feature-by-feature observability per tier |

---

## 8. New DD Capabilities

### Environmental context injection

In SeaPatch, rules operate on image features alone. In PyroWatch, the
expert's description implicitly depends on ambient conditions: "this blue
haze is concerning *in 35-mph Santa Ana conditions*." The same visual
pattern in fog season may be atmospheric haze, not smoke.

DD handles this through a **context precondition block** added to accepted
rules:

```json
{
  "rule_id": "terpene_smoke_001",
  "context_preconditions": {
    "raws_wind_speed_mph": "> 20",
    "raws_relative_humidity_pct": "< 15",
    "raws_temperature_f": "> 85",
    "fire_weather_watch": true
  },
  "features": [
    "blue or blue-gray haze visible in frame",
    "haze originates from single identifiable point source",
    "haze color is distinct from surrounding terrain (not transparent shimmer)",
    "for sentinel tier: haze drift direction consistent across consecutive frames",
    "for scout tier: haze elongation axis aligns with measured wind bearing ±20°"
  ],
  "target_class": "early_smoke_signature",
  "action": "investigate_required",
  "urgency_base": 0.85
}
```

RAWS data is queried from the ground station API at classification time. If
the meteorological preconditions are not met, the rule does not activate.
This eliminates a class of false positives without any threshold tuning on
image features.

### Temporal feature reformulation

The expert's primary discriminator — "smoke drifts; shimmer oscillates" — is
a temporal pattern not accessible to a single-frame classifier. The grounding
check explicitly flags features as:

- **Multi-frame** (sentinel tier: consecutive PTZ frames from fixed mount)
- **Single-frame reformulable** (scout tier: asymmetric elongation is a
  single-frame proxy for consistent drift)
- **Not reformulable** (removed from the rule for that tier)

This produces different rule variants per tier from a single expert session:

| Feature | Sentinel rule | Scout rule |
|---|---|---|
| Blue/blue-gray color | ✓ | ✓ |
| Point source | ✓ | ✓ |
| Consistent drift (multi-frame) | ✓ (fixed PTZ, consecutive frames) | — (removed; single-pass drone) |
| Elongation along wind bearing | — (not needed; drift available) | ✓ (single-frame proxy) |
| Red flag RAWS precondition | ✓ | ✓ |

The temporal feature is not lost — it is preserved in the tier where it is
observable, and replaced by a geometrically equivalent single-frame proxy in
the tier where it is not.

---

## 9. Simulation Setup

**Phase 1 requires no simulator.** The
[FIgLib](https://github.com/brain-facens/FIgLib) (Fire Ignition image Library)
dataset provides 8,138 labeled images from real wildfire ignition sequences
captured by ALERTWildfire and HPWREN mountaintop cameras. Each sequence covers
the period before and after a confirmed ignition, with ground-truth timestamps.
This is the direct equivalent of SeaDronesSee for this use case — real cameras,
real events, publicly available, no data collection required.

Additional public datasets for pool validation and negative sampling:

| Dataset | Contents | Use |
|---|---|---|
| [FIgLib](https://github.com/brain-facens/FIgLib) | 8,138 frames, 244 sequences, real ignitions | Primary pool: early smoke positive frames |
| [ALERTWildfire](https://alertwildfire.org) | 1,000+ cameras, real-time and historical | Pool: heat-shimmer negative frames (non-fire periods) |
| [HPWREN](https://hpwren.ucsd.edu) | 200+ mountaintop cameras, SoCal | Additional negative frames, same camera type as sentinels |
| [FLAME Dataset](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs) | UAV thermal + RGB of controlled burns | Drone-tier positive frames; thermal calibration reference |
| [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov) | Real-time + archive satellite fire detections | Ground truth for retrospective incident matching |

For fleet dynamics and the full broadcast demonstration:

**For flight dynamics and patrol routing:**
- **Gazebo + PX4 SITL** — multi-drone simulation with ROS2; 40-drone swarm
  feasible on a single workstation with reduced rendering
- **AirSim** — Unreal Engine forest and terrain environments for visual
  fidelity testing; drone camera simulation at configurable altitudes

**For fire behavior simulation:**
- **FARSITE / FlamMap** (USFS, free) — physics-based fire spread under
  measured weather and terrain inputs; produces time-stamped spread maps
  that can be used to generate synthetic thermal signatures for commander
  aircraft simulation

**For the full integrated demo:**
- FIgLib frames injected as simulated sentinel camera feeds
- FLAME UAV frames injected as scout drone feeds
- FARSITE spread model provides fire evolution timeline
- DD loop runs on host machine, consumes frames, broadcasts rules
- Ground station exposed as MCP server: `get_camera_frame()`,
  `get_raws_conditions()`, `broadcast_rule()`, `reprocess_archive()`,
  `update_sector_map()`

---

## 10. Measured Results

**Dataset**: [FIgLib](https://github.com/brain-facens/FIgLib) — real mountaintop
camera sequences from HPWREN/ALERTWildfire. Each tarball contains a full ignition
event (~80 frames at 120 s intervals, ±2400 s from first confirmed flame).

Two measurement phases are reported here.

---

### Phase 1 — DD mechanism validation (Haiku PUPIL)

**Eval set**: 40 frames from 4 held-out sequences — 20 positives (offset 0–+600 s,
early smoke window) + 20 negatives (offset < −600 s, clean terrain).

**Setup**: TUTOR = `claude-haiku-4-5-20251001`; PUPIL = `claude-haiku-4-5-20251001`.
MWIR oracle confirms ignition at the failure frame coordinates; rule broadcast
to all ground sentinels with per-tier adaptation.

#### Baseline (no DD rules)

| Metric | Value |
|---|---|
| Recall — early smoke detected | **0.0%** (0 / 20 frames) |
| Precision | 0.0% |
| Accuracy | 50.0% |
| Confident misses (conf ≥ 0.70) | **20 / 20** |

Failure mode breakdown — all 20 early-smoke frames misclassified:
`no_fire` (11, conf 0.95–0.98) and `atmospheric_haze` (9, conf 0.70–0.90).

#### After single DD session

**Rule produced by TUTOR** (ground-sentinel tier):
> When a blue-gray haze is visibly concentrated at a specific ground location,
> distinct in color from the brown-tan chaparral terrain and the surrounding
> clear sky, with the haze originating from a point source and transitioning to
> progressively fainter opacity upward, and the plume exhibiting asymmetric
> elongation along a preferential wind direction — classify as
> `early_smoke_signature`.

| Metric | Before DD | After DD | Delta |
|---|---|---|---|
| Recall — early smoke | 0.0% | **10.0%** | **+10.0 pp** |
| Precision | 0.0% | **100.0%** | +100.0 pp |
| Accuracy | 50.0% | 55.0% | +5.0 pp |

---

### Phase 2 — PatchBench probe (model selection)

**Probe set**: 24 frames from 4 sequences — 12 positives (offset 0–+360 s) +
12 near-ignition negatives (offset −60 to −360 s, same scene, no smoke).
Difficulty labels: hard = 0 s / −60 s, medium = 180 s, easy = 360 s.
Full benchmark at [khub-ai/patchbench](https://github.com/khub-ai/patchbench):
`benchmarks/wildfire/early_smoke_vs_terrain/probe_v1/`.

The PatchBench probe measures four things per model: vocabulary overlap,
feature detection by difficulty, zero-shot accuracy, and rule-aided accuracy
after DD rule injection. Results below.

| PUPIL | TUTOR | Zero-shot | Rule-aided | Delta | Verdict |
|---|---|---|---|---|---|
| `claude-opus-4-6` | `claude-opus-4-6` | 0.667 | 0.792 | +0.125 | PARTIAL |
| `claude-sonnet-4-6` | `claude-opus-4-6` (model rules) | 0.667 | **0.958** | **+0.292** | **GO** |
| `claude-sonnet-4-6` | NWCG/human (expert rules) | 0.667 | **0.958** | **+0.292** | **GO** |

The Opus row uses TUTOR = PUPIL (same tier), which is architecturally
constrained — the TUTOR cannot describe features the PUPIL does not already
detect, explaining the weaker lift. The Sonnet rows use the correct hierarchy
(Opus TUTOR > Sonnet PUPIL) and achieve a GO verdict.

The expert-rules row replaces the TUTOR model with published NWCG fire lookout
guidelines (IRPG 2025, S-190, S-290) as the rule source. Identical lift to the
Opus-generated rules confirms that domain expertise encoded as rules is
sufficient — the TUTOR model is not required when authoritative guidelines exist.

---

### What this shows

**DD mechanism validated.** Phase 1 (Haiku tier): 0% → 10% recall from one
expert explanation, zero false alarms. Phase 2 (Sonnet tier, PatchBench):
66.7% → 95.8% accuracy after rule injection, GO verdict, perfect consistency.

**TUTOR ≥ PUPIL is a hard requirement.** Opus-as-TUTOR for a Sonnet-PUPIL
produces +29 pp lift (GO). Opus-as-both-TUTOR-and-PUPIL produces only +12 pp
(PARTIAL). The TUTOR must know more than the PUPIL.

**Human expertise substitutes for a TUTOR model.** Published fire lookout
guidelines (NWCG) injected directly as rules achieve the same +29 pp lift as
Opus-generated rules. This removes the TUTOR API dependency for domains with
well-documented expert knowledge.

**0–6 minute ignition window is the hardest optical subset.** The hard-difficulty
frames (offset 0 s / −60 s) score 0.5 even for Sonnet — ignition onset is
genuinely ambiguous optically. PyroWatch's design intent is cross-modal MWIR
confirmation for this window; rule injection extends the optical detection
horizon at the margin.

---

## 11. Getting Started

The DD loop uses the same library as the birds, dermatology, and SeaPatch
experiments:

```bash
# Prerequisites
pip install anthropic           # TUTOR and KF validator
pip install transformers        # Qwen3-VL-8B (commander PUPIL)

# Download FIgLib (no account required)
git clone https://github.com/brain-facens/FIgLib data/figlib

# Run a standalone DD session (no simulator required)
cd usecases/ai-fleets/wildfire-detection/python
python run_dd_session.py \
    --failure-image path/to/sentinel_frame_gs047.jpg \
    --confirmation "MWIR camera confirmed 380°C surface anomaly at same coordinates" \
    --raws-conditions '{"wind_mph": 28, "humidity_pct": 8, "temp_f": 95}' \
    --pool-dir data/figlib/labeled_pool/ \
    --tier sentinel \
    --tutor-model claude-opus-4-6 \
    --validator-model claude-sonnet-4-6 \
    --pupil-model qwen/qwen3-vl-8b-instruct
```

The `--tier` argument controls which feature variants the grounding check
evaluates. Run the session once per tier to generate tier-differentiated rules.

For the full three-tier fleet simulation, see `DESIGN.md` for integration
architecture and setup instructions.

The domain configuration for this use case is in
[`python/domain_config.py`](python/domain_config.py).
