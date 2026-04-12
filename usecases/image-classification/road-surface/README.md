# Knowledge Fabric for Road Surface Condition Classification

> **For**: Autonomous driving engineers, ADAS developers, and road safety researchers interested in how domain expertise can improve small-model accuracy on safety-critical surface classification without retraining.
>
> **Status**: Active experiments — benchmark manifests committed, first DD session run (dry vs wet, Qwen3-VL-8B). PUPIL Domain Readiness Probe implemented.
>
> **Dataset**: RSCD (Road Surface Classification Dataset), Tsinghua University — ~600K images, labels encoded in filenames across friction, material, and roughness.
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
- RSCD dataset zip (see [Dataset Download](#dataset-download) below)
- `ANTHROPIC_API_KEY` for TUTOR/VALIDATOR models (Claude Opus / Sonnet)
- `OPENROUTER_API_KEY` for PUPIL model (Qwen3-VL-8B via OpenRouter)

### Dataset Download

The RSCD zip (~14 GB) is not included in this repository. Download once and
keep it locally:

```bash
# Via Kaggle CLI (recommended)
pip install kaggle
kaggle datasets download cristvollerei/rscd-dataset-1million
# Place the zip at: C:\_backup\ml\data\rscd-dataset-1million.zip

# Or direct download from Figshare (~14 GB):
# https://thu-rsxd.com/dxhdiefb/
```

The code reads directly from the zip — no extraction step needed.
Default zip path is `C:\_backup\ml\data\rscd-dataset-1million.zip`.
Override with `--data-dir /your/path` on any script.

> **Note on actual class coverage**: This RSCD release contains only
> **dry, wet, and water** friction classes (~600K images). Ice, snow, and
> slush classes referenced in the Tsinghua paper are absent from this release.
> The primary DD pair is **dry vs wet** (subtle, genuine visual confusion).

---

### Step 1 — Generate benchmark manifests (maintainer, run once)

Benchmark manifests are fixed sets of image IDs committed to git. They ensure
every run — yours, a collaborator's, a reviewer's — uses the exact same images.

```bash
cd usecases/image-classification/road-surface/python

# Generate probe manifest (24 images) and pool manifest (40 images).
# No API calls, no cost. Completes in ~30 seconds (zip scanning only).
python create_benchmark.py --pair dry_vs_wet --types probe,pool

# Optional: call TUTOR (Claude Opus) to annotate visual difficulty per image.
# Costs ~$0.10–0.15, takes ~3 minutes.
python create_benchmark.py --pair dry_vs_wet --types probe,pool --annotate-difficulty

# Optional: discover which images Qwen3-VL-8B gets wrong (failure manifest).
# Costs ~$0.01, takes ~5 minutes.
python create_benchmark.py --pair dry_vs_wet --types failures \
    --pupil-model qwen/qwen3-vl-8b-instruct --n-failures 8
```

Commit the resulting JSON files — they are the reproducible benchmark:

```bash
git add usecases/image-classification/road-surface/benchmarks/*.json
git commit -m "Add dry_vs_wet benchmark manifests v1"
git push
```

See [`benchmarks/README.md`](benchmarks/README.md) for the manifest format,
versioning policy, and full options reference.

---

### Step 2 — Check PUPIL readiness (optional but recommended)

Before running a full DD session, check whether the PUPIL model has sufficient
visual and verbal capability for this domain:

```bash
# Not yet a standalone script — see core/dialogic_distillation/probe.py
# and docs/probe.md for the API. A probe_rscd.py driver is planned.
```

The probe runs five steps (TUTOR descriptions → PUPIL vocabulary → feature
detection → rule comprehension delta → consistency) and returns a
`go / partial / no-go` verdict. TUTOR and VALIDATOR outputs are cached so
testing a second PUPIL model costs only PUPIL API calls.

---

### Step 3 — Run a DD experiment

```bash
cd usecases/image-classification/road-surface/python

# Auto-discover failures and run distillation (uses fixed pool from benchmark)
python distill_dialogic.py --pair dry_vs_wet \
    --val-per-class 20 --max-rounds 4

# Use a fixed failure manifest instead of auto-discovery
# (once dry_vs_wet_failures_qwen3_v1.json is generated)
python distill_dialogic.py --pair dry_vs_wet \
    --failure-ids 20220321182055148,2022021018461116,...
```

The session is saved to `distill_dialogic_session.json`. Key outputs per failure:

| Field | Description |
|---|---|
| `grounded_at_round` | Round where the rule's preconditions fired on the trigger image |
| `pool_result.precision` | Fraction of rule activations that were true positives |
| `pool_result.accepted` | Whether rule passed the precision gate (≥0.90, max 0 FP) |

---

### Step 4 — Interpret results

**What to look for:**

- `grounded=True, accepted=True` — rule is usable; add to injection library
- `grounded=True, accepted=False` — rule fires on trigger image but overfires on pool; needs tightening
- `grounded=False` — PUPIL cannot observe the described feature; try simpler vocabulary

**First experiment results (dry vs wet, 2026-04-12):**

| Metric | Value |
|---|---|
| PUPIL zero-shot error rate | ~57% (essentially random) |
| Rules grounded | 4/4 |
| Rules accepted (precision ≥ 0.90) | 1/4 |
| Best rule | High-key white-to-gray tonal range with individual aggregates visible |
| Best rule precision | 1.0 on small pool |

The 57% error rate confirms this is a genuinely hard confusable pair — a good
DD target. The low acceptance rate (1/4) reflects that dry vs wet on asphalt
is highly context-dependent; most visual features that indicate "dry" are also
occasionally present on lightly-wet surfaces.

---

### Files

```
usecases/image-classification/road-surface/
  README.md                     ← this file
  benchmarks/
    README.md                   ← manifest format, versioning policy
    dry_vs_wet_probe_v1.json    ← 24 probe images (committed)
    dry_vs_wet_pool_v1.json     ← 40 pool images (committed)
  python/
    domain_config.py            ← DomainConfig for road surface domain
    dataset.py                  ← RSCD loader (zip-native, no extraction needed)
    benchmark.py                ← load_benchmark(), to_probe_images(), to_pool_images()
    create_benchmark.py         ← one-time manifest generator (run by maintainer)
    distill_dialogic.py         ← three-party DD session runner
    agents.py                   ← model backend wiring
```
