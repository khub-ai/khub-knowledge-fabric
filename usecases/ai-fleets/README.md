# Knowledge Fabric Use Case: AI Fleets
## Instant Knowledge Propagation Across Hierarchical Agent Networks

---

> **Status**: SeaPatch maritime SAR and PyroWatch wildfire detection both
> implemented and measured. **Theme**: [Knowledge Fabric (KF)](../../docs/what-is-kf.md)
> as a zero-retraining update mechanism for large-scale deployments of heterogeneous
> AI agents organised in capability tiers.
> **Last updated**: 2026-04-14

---

## The Pattern

Every large-scale AI deployment eventually faces the same problem. You have many
agents running continuously — cameras, sensors, drones, monitors, classifiers —
and one of them encounters a failure mode that none of them have seen before. A
human expert looks at the failure and immediately understands what went wrong.
But getting that understanding back into the fleet takes hours at minimum:
retrain, validate, deploy. Meanwhile the fleet keeps making the same mistake.

**AI Fleets** is the name for this family of use cases in this repository. The
common structure:

| Element | Description |
|---|---|
| **Tier architecture** | A small number of high-capability agents (expensive sensors, powerful models, hover-capable drones, ICU monitors) and a large number of low-capability agents (cheap cameras, edge classifiers, scout drones, ward sensors) |
| **Capability gap** | The high-capability tier can identify failures the low-capability tier cannot, often because it has access to a different sensor modality or higher resolution |
| **Knowledge bottleneck** | An expert can describe what the low-capability agent should have seen — but getting that description into the fleet currently requires retraining |
| **KF solution** | [Dialogic Distillation (DD)](../dialogic-distillation/README.md) turns the expert's description into a grounded, validated natural-language rule and broadcasts it to the entire fleet in seconds, without touching model weights |

---

## Why This Is a Different Category from Robotics

The [robotics](../robotics/README.md) use cases in this repository are about
physical embodiment: manipulation, locomotion, task planning for warehouse or
lab robots, correction governance for embodied agents navigating physical
environments.

AI Fleets is a different axis entirely:

| | Robotics | AI Fleets |
|---|---|---|
| **Core challenge** | Physical action and task execution | Classification and sensing at scale |
| **Update mechanism** | Correction persistence for single-agent sessions | Instant broadcast across heterogeneous multi-agent fleets |
| **Hardware** | Actuators, grippers, locomotion | Cameras, sensors, classifiers |
| **Key KF contribution** | Governed, scoped, expirable corrections | Zero-retraining fleet-wide rule propagation + retroactive reprocessing |
| **Heterogeneity** | One robot, multiple tasks | Many agents, multiple hardware tiers |

The SeaPatch drone swarm could superficially be labelled "robotics" because it
involves drones. But the contribution has nothing to do with flight control or
physical manipulation. It is entirely about the knowledge layer: how expert
understanding of a failure propagates to 38 heterogeneous classifiers in 52
seconds, including retroactive reprocessing of 45 minutes of already-captured
footage.

---

## The General Template

Any deployment fitting this shape is an AI Fleets use case:

```
High-capability tier (few agents)
  └── Rich sensor / expensive model
  └── Can confirm failure ground truth
  └── TUTOR role: explain what low tier should have seen

Low-capability tier (many agents)
  └── Cheap sensor / lightweight classifier
  └── Continuous coverage, high volume
  └── PUPIL role: apply injected rules at inference time

Knowledge propagation path
  └── Expert describes failure in natural language
  └── DD validates and tightens the rule against a labeled pool
  └── Rule broadcast to entire fleet (seconds, not hours)
  └── Archived frames retroactively reprocessed
  └── No retraining. No model weight changes.
```

---

## Example Domains

| Domain | High-capability tier | Low-capability tier | Knowledge gap |
|---|---|---|---|
| **Maritime SAR** (implemented) | Commander drone: thermal FLIR + stabilised RGB | Scout drones: fixed-mount RGB | Thermal confirms person; optical scout missed it |
| **Wildfire detection** (specified) | Commander aircraft: MWIR + calibrated temp + anemometer | Sentinels: RGB PTZ; Scout drones: RGB + LWIR | MWIR confirms ignition; optical tiers miss early terpene smoke |
| **Hospital monitoring** | ICU: full ECG, arterial line, continuous nursing | Ward: pulse-ox wearables, periodic checks | ICU recognises early deterioration pattern wearable misses |
| **Industrial IoT** | Cloud: full sensor fusion, ML inference | Edge: lightweight anomaly detector | Cloud confirms equipment failure mode edge never saw |
| **Agricultural drones** | Research UAV: hyperspectral + LiDAR | Scout fleet: RGB cameras | Hyperspectral identifies disease; RGB scouts miss early stage |
| **Security camera networks** | Analyst workstation: human review, audio | Fixed cameras: video-only classifiers | Analyst explains what cameras should flag in ambiguous scenes |
| **Satellite constellations** | High-resolution tasking satellites | Medium/low-res continuous coverage | High-res confirms target class that low-res fleet is missing |

The knowledge propagation problem is structurally identical across all of these.
The DD mechanism — TUTOR explains, pool validates, rule broadcasts — applies
without domain-specific engineering.

---

## Use Cases in This Category

### 1. SeaPatch — Maritime SAR Drone Fleet

**[drone-swarm/README.md](drone-swarm/README.md)**

A 40-drone maritime search-and-rescue fleet (38 scouts + 2 commanders). A
commander drone's FLIR thermal camera confirms a person in the water at the
same coordinates where 38 optical scouts classified "whitecap, 0.91 confidence"
18 minutes earlier.

A rescue swimmer explains what the optical scout should have seen. DD turns that
explanation into a grounded rule in 43 seconds. The rule broadcasts to the
entire fleet and retroactively reclassifies 45 minutes of archived footage.

**Implemented and measured.** Key result: Qwen3-VL-8B PUPIL classifier on the
SeaDronesSee dataset. Frame `71.jpg`: `life_ring_unoccupied` (0.95 confidence)
→ `person_in_water` (0.85 confidence) after one DD session. On the 25 hardest
person-in-water frames from the val split, recall improves from 8% to 52%
(+44 pp) with a single injected rule, zero retraining.

**Distinctive elements**: cross-modal knowledge transfer (thermal → optical),
temporal feature reformulation (temporal stability → single-frame proxies),
architecture-agnostic broadcast (MobileNetV3 scouts + Qwen3-VL-8B commander
updated by the same natural-language rule).

---

### 2. PyroWatch — Wildfire Early Ignition Detection Fleet

**[wildfire-detection/README.md](wildfire-detection/README.md)**

A 164-agent fleet covering fire-prone terrain during red flag conditions:
120 ground sentinel cameras (RGB PTZ + LWIR), 40 scout drones (RGB + LWIR),
and 4 commander aircraft (calibrated MWIR + high-res RGB). A commander
aircraft's MWIR sensor confirmed a 380°C ignition at coordinates where 160
sentinels and scouts had logged "heat_shimmer, 0.94 confidence" for 18 minutes.

A Cal Fire lookout explained that early chaparral smoke is blue (terpene
combustion), originates from a single point source, and drifts with consistent
wind direction — all distinguishable from shimmer in the optical frames. DD
turns that explanation into tier-differentiated rules in 38 seconds. The
fleet retroactively identifies the ignition 18 minutes earlier than the MWIR
pass that confirmed it.

**Implementation complete (Phase 1). Measured results available** — see [§10](wildfire-detection/README.md#10-measured-results). Python in `wildfire-detection/python/`.

**Distinctive elements**: three-tier fleet (sentinels + scout drones +
commander aircraft), environmental context injection (RAWS meteorological
preconditions evaluated at inference time without retraining), temporal feature
reformulation (multi-frame drift → single-frame wind-elongation proxy for
drone tier), tier-differentiated rule variants from a single DD session.

---

## Relationship to Dialogic Distillation

All AI Fleets use cases use [Dialogic Distillation](../dialogic-distillation/README.md)
as the knowledge-extraction mechanism. DD is the general protocol; AI Fleets is
one application context that adds:

- **Tier-aware grounding**: preconditions are validated against each hardware
  tier's actual sensor capabilities (resolution, frame rate, modality)
- **Cross-modal TUTOR**: the expert's knowledge originates in a different sensor
  than the classifier that must act on it
- **Fleet broadcast**: accepted rules are distributed to all agents in the fleet,
  not just the one that encountered the failure
- **Retroactive reprocessing**: previously captured frames are re-evaluated
  against newly accepted rules
