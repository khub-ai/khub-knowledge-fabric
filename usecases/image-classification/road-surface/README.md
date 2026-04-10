# Knowledge Fabric for Road Surface Condition Classification

> **For**: Autonomous driving engineers, ADAS developers, and road safety researchers interested in how domain expertise can improve small-model accuracy on safety-critical surface classification without retraining.
>
> **Status**: Domain setup complete — dataset identified, domain config authored, confusable pairs defined. Baseline experiments pending.
>
> **Dataset**: RSCD (Road Surface Classification Dataset), Tsinghua University — 1M images, 27 classes across friction, material, and roughness.
>
> **Also see**: [Image Classification Overview](../README.md) for the broader Knowledge Fabric context, including the dermatology and ornithology use cases.

---

## The One-Line Summary

A small vision model confuses wet road with black ice — both look like a dark reflective surface. A pavement engineer explains, in plain language, the visual indicators that distinguish them (texture visibility through the film, sheen uniformity, crystalline patterning at edges). The system turns those explanations into explicit rules, validates them against labeled images, and injects them at inference time. The fix is a written explanation. No retraining required.

---

## Contents

1. [Why Road Surface Conditions](#1-why-road-surface-conditions)
2. [The Dataset](#2-the-dataset)
3. [Confusable Pairs](#3-confusable-pairs)
4. [The Vocabulary Gap](#4-the-vocabulary-gap)
5. [Edge Deployment Context](#5-edge-deployment-context)
6. [Getting Started](#6-getting-started)

---

## 1. Why Road Surface Conditions

Road surface condition classification is an ideal domain for dialogic distillation because it combines four properties:

**The classification problem is genuinely hard.** Multi-class SOTA on RSCD's 27-class taxonomy (friction x material x roughness) is 80-90%. The safety-critical confusions — wet vs. ice, damp vs. wet — remain poorly resolved because the visual differences are subtle and context-dependent.

**Expert reasoning adds knowledge that pixels alone cannot provide.** A pavement engineer reasons about surface-appearance relationships that go beyond pattern matching: "A mirror-like reflective sheen with no visible surface texture means the surface film is thick enough to obscure aggregate — this indicates standing water or ice, not mere dampness where texture remains visible." This reasoning is transferable as explicit rules.

**The market demand is massive.** ADAS software is a $10B+ market growing at 21% CAGR. Every Level 2+ vehicle needs surface condition awareness for traction control, braking distance estimation, and route planning. Thousands of OEMs and suppliers are working on this.

**Edge deployment is the only viable architecture.** Surface condition classification must run on vehicle hardware (ECUs, Jetson-class devices) with millisecond latency. A small VLM enhanced with expert rules is a practical deployment target.

---

## 2. The Dataset

**RSCD — Road Surface Classification Dataset** (Tsinghua University)

| Property | Value |
|---|---|
| Images | ~1,000,000 (960k train / 20k val / 50k test) |
| Resolution | 240 x 360 pixels (cropped from 2MP monocular camera) |
| Classes | 27 (6 friction x 4 material x 3 roughness, with exclusions) |
| License | CC BY-NC (non-commercial) |
| Download | [Figshare (~14 GB)](https://thu-rsxd.com/dxhdiefb/) or [Kaggle](https://www.kaggle.com/datasets/cristvollerei/rscd-dataset-1million) |
| Paper | Tsinghua RSCD, multi-task classification |

### Class Taxonomy

**Friction (6 classes):**
| Class | Description |
|---|---|
| Dry | No moisture on surface |
| Wet | Visible moisture film, surface texture partially obscured |
| Water | Standing water, puddles, or continuous water film |
| Fresh snow | Uncompacted snow covering surface |
| Melted snow | Partially melted snow, slush, wet snow residue |
| Ice | Frozen surface film — may appear similar to wet |

**Material (4 classes):**
Asphalt, Concrete, Dirt/mud, Gravel
*(Not annotated when friction is fresh snow, melted snow, or ice)*

**Roughness (3 classes):**
Smooth, Slight unevenness, Severe unevenness
*(Not annotated for dirt/mud, gravel, or snow/ice surfaces)*

### Supplementary Dataset

**StreetSurfaceVis** — 9,122 street-level images from Germany with surface type and quality labels. Multi-resolution (up to original), CC BY-SA 4.0, [Zenodo download](https://zenodo.org/records/11449977). Higher resolution than RSCD; useful for cross-dataset validation.

---

## 3. Confusable Pairs

The following pairs represent the safety-critical confusions where dialogic distillation should have the most impact:

### Priority 1 — Safety-Critical Friction Confusions

| Pair | Why it's confusable | Why it matters |
|---|---|---|
| **Wet vs. Ice** | Both produce a dark, reflective surface sheen. RGB alone cannot distinguish a water film from a frozen film. | Misclassifying ice as wet could cause a vehicle to brake normally when it should engage ABS/ESC preemptively. |
| **Damp/Wet vs. Water** | A thin water film vs. standing water are on a continuum. The boundary is subjective even for humans. | Aquaplaning risk depends on water depth, not just presence. |

### Priority 2 — Material Confusions Under Degraded Conditions

| Pair | Why it's confusable | Why it matters |
|---|---|---|
| **Fresh snow vs. Melted snow** | Partial melt can look like fresh snow with shadows. Slush has variable appearance. | Traction characteristics differ substantially. |
| **Wet asphalt vs. Wet concrete** | Both darken when wet. Under water film, surface texture (the primary material discriminator) is obscured. | Friction coefficients differ by material even under identical moisture conditions. |

### Priority 3 — Roughness Assessment

| Pair | Why it's confusable | Why it matters |
|---|---|---|
| **Slight unevenness vs. Severe unevenness** | Severity is a continuous scale. Camera angle and lighting affect apparent depth of surface irregularities. | Ride quality estimation, suspension adaptation, speed recommendations. |

---

## 4. The Vocabulary Gap

This domain has a strong vocabulary gap between pavement engineering expertise and plain visual description — exactly the gap that dialogic distillation is designed to bridge.

### What the expert says vs. what the VLM sees

| Expert description | VLM visual description |
|---|---|
| "Critically wet pavement near freezing — the water film is about to transition to ice" | "Dark shiny road surface" |
| "Category 3 alligator cracking with 10-20mm block sizes indicating subbase fatigue" | "Road with lots of connected cracks in a grid pattern" |
| "Chemically wet surface from recent CaCl2 application — lower friction than plain wet" | "Wet-looking dark road" |
| "Raveling on aged asphalt — aggregate loss exposing binder, not surface contamination" | "Rough dark surface with loose stones" |
| "Black ice — note absence of visible texture through the film, unlike wet where aggregate remains faintly visible" | "Dark reflective road surface" |

The last example is the key one. The expert's rule — "if reflective sheen is present AND surface texture is completely invisible through the film (not just dimmed), suspect ice rather than wet" — is a concrete, testable precondition that a small VLM can evaluate when told what to look for. Without the rule, the VLM has no basis for the distinction.

---

## 5. Edge Deployment Context

Road surface classification is a canonical edge deployment scenario:

- **Latency requirement**: Safety-critical, sub-100ms decision loop
- **Compute budget**: Vehicle ECU or Jetson-class device (Orin Nano to AGX Orin)
- **Connectivity**: Cannot depend on cloud — must work in tunnels, rural areas, dead zones
- **Deployment scale**: Millions of vehicles, each with slightly different camera placement and optics

Current VLM throughput on Jetson hardware (2025-2026 benchmarks):

| Hardware | Model | Throughput |
|---|---|---|
| Orin Nano 8GB | PaliGemma2-3B (FP4) | ~22 tok/s |
| AGX Orin 64GB | Qwen3-VL-4B (W4A16) | ~47 tok/s |
| AGX Thor | Qwen2.5-VL-3B | ~72 tok/s |

VLMs do not achieve 30 FPS for per-frame analysis. The practical architecture is **hybrid**: a lightweight detector (YOLO-class, 30+ FPS) handles routine frames, while the VLM with injected expert rules handles ambiguous frames flagged for deeper analysis at 1-5 second intervals. This is the pattern NVIDIA promotes for Jetson VLM deployment.

Dialogic distillation's value in this architecture: the expert rules make the VLM's infrequent but high-stakes classifications as accurate as possible.

---

## 6. Getting Started

### Prerequisites

- Python 3.10+
- RSCD dataset downloaded and extracted
- Anthropic API key (for TUTOR model)

### Domain Configuration

The domain config is defined in `python/domain_config.py`:

```python
from road_surface.python.domain_config import ROAD_SURFACE_CONFIG
```

This provides the vocabulary mapping (expert role, feature nouns, observation guidance, vocabulary examples) that the core dialogic distillation library uses to generate domain-appropriate prompts.

### Recommended First Experiment

Start with the **Wet vs. Ice** confusable pair:
1. Filter RSCD for friction labels `wet` and `ice` on asphalt material
2. Run zero-shot baseline with a small VLM (e.g., Qwen2.5-VL-3B or Qwen3-VL-4B)
3. Identify failure cases where the model confuses wet for ice or vice versa
4. Run dialogic distillation to author corrective rules
5. Re-test with rules injected

This pair has the strongest safety motivation, the clearest vocabulary gap, and should produce the most compelling demonstration of DD's value.
