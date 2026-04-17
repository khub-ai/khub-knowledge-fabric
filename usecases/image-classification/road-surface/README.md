# Knowledge Fabric for Road Surface Condition Classification

> **For**: Autonomous driving engineers, ADAS developers, and road safety researchers interested in how domain expertise can improve small-model accuracy on safety-critical surface classification without retraining.
>
> **Status**: Active experiments — dry_vs_wet run (4/4 grounded, 0/4 accepted — pair is a continuum); wet_vs_water pair set up (expected to be easier); visual similarity router and canonical reference curation added.
>
> **Dataset**: RSCD (Road Surface Classification Dataset), Tsinghua University — ~600K images, labels encoded in filenames across friction, material, and roughness.
>
> **Also see**: [Image Classification Overview](../README.md) for the broader Knowledge Fabric context, including the bird and dermatology use cases.

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
7. [Lessons Carried Over from Dermatology](#7-lessons-carried-over-from-dermatology)

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
- API keys for the models you plan to use as TUTOR, VALIDATOR, and PUPIL

The default models are Anthropic (TUTOR: `claude-opus-4-6`, VALIDATOR:
`claude-sonnet-4-6`) and OpenRouter (PUPIL: `qwen/qwen3-vl-8b-instruct`).
Any model can be substituted via `--tutor-model`, `--validator-model`,
`--pupil-model`. Local models served on an OpenAI-compatible endpoint are
also supported.

The runner reads API keys from environment variables:
- `ANTHROPIC_API_KEY` — for Anthropic models
- `OPENROUTER_API_KEY` — for OpenRouter models

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
# No model API calls needed.
python create_benchmark.py --pair dry_vs_wet --types probe,pool

# Optional: call TUTOR model to annotate visual difficulty per image.
python create_benchmark.py --pair dry_vs_wet --types probe,pool --annotate-difficulty

# Optional: discover which images the PUPIL gets wrong (failure manifest).
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
cd usecases/image-classification/road-surface/python

# Run probe with default models
python probe_rscd.py --pair dry_vs_wet

# Run with a different PUPIL model
python probe_rscd.py --pair dry_vs_wet --pupil-model llava-hf/llava-1.5-7b-hf

# Override TUTOR/VALIDATOR models
python probe_rscd.py --pair dry_vs_wet \
    --tutor-model     <your-tutor-model> \
    --validator-model <your-validator-model>

# List available probe manifests
python probe_rscd.py --list-manifests
```

The probe runs four steps (TUTOR expert descriptions → PUPIL vocabulary probe →
feature detection → rule comprehension delta + consistency) and returns a
`go / partial / no-go` verdict. TUTOR and VALIDATOR outputs are cached after
the first run — testing a second PUPIL model skips those calls entirely.

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

**Experiment 1 results (dry vs wet, 2026-04-13):**

| Metric | Value |
|---|---|
| PUPIL | qwen/qwen3-vl-8b-instruct |
| TUTOR / VALIDATOR | claude-opus-4-6 / claude-sonnet-4-6 |
| PUPIL zero-shot error rate | 60% (36/60 test images) |
| Failures processed | 4 |
| Rules grounded | 4/4 — all fired on trigger image at round 1 |
| Rules accepted (precision ≥ 0.90) | 0/4 |
| Best precision reached | 0.60 (3 TP, 2 FP on 20-image pool) |

Session JSON: [`benchmarks/sessions/distill_dialogic_dry_vs_wet_claude_opus_4_6.json`](benchmarks/sessions/distill_dialogic_dry_vs_wet_claude_opus_4_6.json)

**Interpretation:** The 60% error rate confirms this is a genuinely hard confusable
pair — a good DD target. All 4 rules grounded (the TUTOR's described features
are visible in each trigger image), but none passed the pool gate at precision
≥ 0.90. The core difficulty: features that indicate "dry" (matte granular
texture, no specular highlights, visible aggregate) are also intermittently
present in lightly-wet asphalt images. The visual boundary is a continuum, not
a sharp threshold.

**Why 0/4 accepted:** The dry/wet visual boundary is a continuum — lightly-wet
asphalt can look indistinguishable from dry in certain lighting, and the RSCD
pool inevitably contains borderline cases where even human annotators might
disagree. Rules that correctly identify clearly-wet images also fire on these
borderline dry images, capping precision below 0.90.

**Next steps based on dermatology lessons:** Switch to the **wet_vs_water** pair
(standing water vs. thin moisture film) where the visual signal is much stronger
and the decision boundary is sharper. Curate canonical reference images for the
visual similarity router. Process more failures with `--n-failures 8` and a
larger pool with `--val-per-class 20`. See §7 below for the full carry-over
plan.

---

---

### Step 5 — Run DD on the wet_vs_water pair

Benchmark manifests for wet_vs_water are committed. Run DD on this pair next —
standing water is visually far more distinct from wet than wet is from dry,
giving rules a better chance of passing the precision gate:

```bash
cd usecases/image-classification/road-surface/python

# Generate wet_vs_water benchmark manifests (already committed — skip if present)
# python create_benchmark.py --pair wet_vs_water --types probe,pool

# Run three-party DD on wet_vs_water
python distill_dialogic.py --pair wet_vs_water \
    --val-per-class 20 --max-rounds 4 --n-failures 8
```

---

### Step 6 — Build the visual similarity router (multi-class)

Once binary-pair DD has produced accepted rules for each pair, the next step is
the full multi-class L1 router. Instead of the schema-based Observer pipeline
(which suffers from feature denial on road surface imagery), the router uses
curated canonical reference images to answer "which friction class does this
image most resemble?":

```bash
cd usecases/image-classification/road-surface/python

# 1. Curate one canonical reference image per L1 friction group (Dry, Wet, Water)
python curate_references.py --n-candidates 8

# The script saves curated_references.json in the benchmarks/ directory.
# Review and commit the selected images.

# 2. The router is then available for evaluation scripts:
#    from router import run_l1_router, load_curated_refs
#    refs  = load_curated_refs()
#    group, conf, ms = await run_l1_router(image_path, refs, verbose=True)
```

---

## 7. Lessons Carried Over from Dermatology

The dermatology hierarchical classification experiments produced several
findings that apply directly here.

### Start with the easier pair

In dermatology, the most confusable pairs (BCC vs Keratosis) required more
rules and more dialogue rounds than the cleaner pairs (Melanoma vs Nevus).
The same principle applies here: **dry_vs_wet** is the hardest road surface pair
(continuum boundary, high borderline-image rate in the pool). **wet_vs_water**
has a much sharper visual boundary and should yield rules that pass the pool
gate more easily. Build the rule library on the easier pair first, then return
to dry_vs_wet with more evidence and a better-understood vocabulary.

### Feature denial: use visual similarity routing at L1

In dermatology, the VLM consistently reported diagnostic features (pigment
network, arborizing vessels, milia cysts) as **absent** even when clearly
visible — making the schema-based Observer pipeline unreliable for coarse
group routing. The same pattern will occur here: ask a general-purpose VLM
"is there aggregate texture visible through a water film?" and it will report
"no" regardless of the image content.

The fix is `router.py`: show the model a canonical reference image per group
and ask "does this look like THAT?" instead of "is feature X present?" For the
multi-class problem (all 27 RSCD classes), the visual similarity router should
be the L1 routing mechanism, with the DD rule engine applied at L2 within each
friction group.

### Vocabulary bridging is already in place — but refine the examples

The `domain_config.py` already includes the three meta-learnings from
dermatology:
1. Frame the task for the validator's vocabulary (visual colors and textures,
   not engineering terms)
2. GOOD/BAD vocabulary examples are provided
3. Precondition count is limited

The first DD session confirmed this works: 4/4 rules grounded at round 1,
meaning the TUTOR's vocabulary was compatible with the VALIDATOR's. The failure
was at the pool gate (precision), not the grounding check. The vocabulary
bridging is working; the challenge is pair difficulty.

### Multiple canonical exemplars per friction group

In dermatology, groups with visually diverse subtypes (Keratosis-type: warty vs
erythematous; Melanocytic: NV vs Melanoma) needed two canonical references in
the router. Road surface has the same problem:

| Group | Subtypes needing separate exemplars |
|---|---|
| **Wet** | Overcast (uniform dark surface) vs. bright sun (faint specular sheen) |
| **Water** | Shallow puddles (partial texture visible at edges) vs. deep film (full mirror reflection) |
| **Dry** | Fresh dark asphalt vs. bleached pale-gray aged surface |

Plan for 2 canonical references per group when running `curate_references.py`
with `--n-candidates 10`. The router already supports multi-exemplar groups.

### Published standards can replace the TUTOR model

From the wildfire and maritime SAR experiments: injecting NWCG fire-spotting
rules and IAMSAR search criteria directly — without running the dialogic TUTOR
loop — produced the same accuracy as model-generated rules. Road surface has
directly applicable published standards:

- **PIARC** (World Road Association) friction classification criteria
- **ASTM E1845** pavement condition assessment
- **Highway patrol and winter road maintenance operational thresholds** —
  concrete operational criteria for what counts as "icy road" or "standing water
  risk" in practice

These can be formatted as DD rules and tested directly on the pool before
running the full dialogic loop — providing a baseline and a sanity check.

---

### Files

```
usecases/image-classification/road-surface/
  README.md                          ← this file
  curated_references.json            ← canonical reference images per L1 group (after running curate_references.py)
  benchmarks/
    README.md                        ← manifest format, versioning policy
    dry_vs_wet_probe_v1.json         ← 24 probe images (committed)
    dry_vs_wet_pool_v1.json          ← 40 pool images (committed)
    wet_vs_water_probe_v1.json       ← 24 probe images (committed)
    wet_vs_water_pool_v1.json        ← 40 pool images (committed)
    sessions/
      distill_dialogic_dry_vs_wet_claude_opus_4_6.json  ← first DD session (4 grounded, 0 accepted)
    reports/                         ← probe reports saved here (gitignored)
  knowledge_base/                    ← accepted rules committed here after DD sessions (empty — no rules accepted yet)
  python/
    domain_config.py                 ← DomainConfig for road surface (with vocabulary bridging)
    dataset.py                       ← RSCD loader (zip-native, no extraction needed)
    benchmark.py                     ← load_benchmark(), to_probe_images(), to_pool_images()
    create_benchmark.py              ← one-time manifest generator (run by maintainer)
    probe_rscd.py                    ← PUPIL domain readiness probe driver
    distill_dialogic.py              ← three-party DD session runner
    curate_references.py             ← curate canonical reference images per L1 group (for router)
    router.py                        ← visual similarity L1 router (bypasses feature-denial problem)
    agents.py                        ← model backend wiring
```
