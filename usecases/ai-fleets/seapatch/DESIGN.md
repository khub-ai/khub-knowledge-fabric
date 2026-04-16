# SeaPatch Design Specification

This is the developer-facing design specification for the maritime
person-overboard fleet patching use case. The user-facing scenario and
motivation are in [README.md](README.md).

---

## 1. Core Contribution

This use case demonstrates a novel application of Dialogic Distillation (DD)
where:

1. **The TUTOR's knowledge originates in a different sensor modality** than the
   PUPIL's perception. A thermal FLIR camera on the commander tier identifies
   the person; the expert must bridge thermal knowledge to optical observables
   that RGB-only scout cameras can act on.

2. **The PUPIL is a heterogeneous fleet**, not a single model. Rules must be
   grounded against multiple hardware tiers with different sensors, resolutions,
   and compute budgets simultaneously.

3. **Temporal features must be reformulated as within-frame proxies.** The
   strongest discriminating feature (spatial stability over time) is unavailable
   to single-frame classifiers. The grounding check identifies this constraint
   and the TUTOR is prompted to produce within-frame proxies that approximate the
   temporal signal. This is a synthesis step with no equivalent in standard DD.

4. **Retroactive reprocessing** — applying a newly minted rule to the swarm's
   archived frame buffer — is a first-class operation rather than an optional
   bonus.

5. **The rule is the unit of fleet synchronisation.** Instead of distributing
   weight updates (architecture-specific, multi-megabyte, slow), the system
   distributes natural language rules (architecture-agnostic, ~200 tokens,
   sub-minute).

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        GROUND STATION                           │
│                                                                 │
│   Rescue Swimmer ──── TUTOR dialogue ──── Rule candidate       │
│                                               │                 │
│   KF Grounding Validator (Claude Sonnet) ─────┤                 │
│     • Per-tier observability check            │                 │
│     • Temporal feature → within-frame proxy   │                 │
│     • Removes unobservable criteria           │                 │
│                                               ↓                 │
│   Pool Validator (parallel asyncio) ──── Rule accepted?        │
│     • Precision gate: ≥ 0.90                │                  │
│     • Max FP: 0                              │                  │
│     • Contrastive tightening if needed       │                  │
│                                               ↓                 │
│   Rule Pool (KF artifact store) ◄──── Rule registered          │
│     • Version-controlled                     │                  │
│     • Tier-specific variants                 │                  │
│     • Revocable                              │                  │
│                                               ↓                 │
│   Broadcast Engine ───────────────── Mesh network              │
│   Archive Reprocessor ────────────── Frame buffer query        │
│                                                                 │
│   Semantic Track Map                                            │
│     (coordinates → detection_class, confidence, rule_id)       │
└─────────────────────────────────────┬───────────────────────────┘
                                      │
                   ┌──────────────────┼──────────────────────┐
                   │                 mesh                     │
        ┌──────────▼──────┐                    ┌─────────────▼──────┐
        │  COMMANDER ×2   │                    │    SCOUT ×38        │
        │                 │                    │                     │
        │ RGB 20MP        │                    │ RGB 12MP            │
        │ FLIR thermal    │                    │ Fixed wide-angle    │
        │ Jetson Orin NX  │                    │ Cortex-M MCU        │
        │ Qwen3-VL-8B     │                    │ Lightweight PUPIL   │
        │ Hover-capable   │                    │ Rule-injected ctx   │
        │ 10-20m AGL      │                    │ 20-40m AGL          │
        │                 │                    │                     │
        │ Detects novel   │                    │ Covers 100% area    │
        │ presentations   │                    │ Continuous sweep    │
        │ via thermal     │                    │ Reports low-conf    │
        └─────────────────┘                    └─────────────────────┘
```

---

## 3. DD Protocol Adaptations

The core DD protocol (from `core/dialogic_distillation/`) requires three
adaptations for this use case:

### 3.1 Cross-modal TUTOR prompt

Standard DD: *"The PUPIL misclassified this image. What visual features should
it have attended to?"*

Cross-modal adaptation: *"The PUPIL's RGB camera classified this region as
[class]. The commander drone's thermal camera confirmed the ground truth is
[class]. The RGB image is shown below. What features visible in the RGB image
correlate with what the thermal camera reveals, such that a camera-only system
could have reached the correct classification?"*

The TUTOR is not asked to describe thermal features — they are asked to bridge
from thermal knowledge to RGB observables. This is a distinct cognitive task
that the prompt must frame explicitly.

### 3.2 Temporal feature detection and reformulation

The grounding check includes an additional pass specifically for temporal
features — criteria that require multiple frames to evaluate. When a temporal
feature is identified:

1. It is flagged as `temporal: true` in the candidate rule
2. The TUTOR is prompted with a follow-up: *"This criterion requires observing
   change over time. The scout classifier processes single frames only. What
   within-frame optical proxy in this image approximates the same signal?"*
3. The proxy criterion replaces the temporal one for the scout-tier rule variant

Example from this scenario:
- Temporal criterion: *"the oval region maintains its position and shape across
  10–20 consecutive frames"*
- Within-frame proxy: *"the oval has uniform brightness across its full extent,
  unlike a whitecap which is bright at the crest and fades radially — the
  brightness gradient within a whitecap is absent from a floating head"*

The commander tier, which runs Qwen3-VL-8B on a video buffer, receives the
temporal criterion directly.

### 3.3 Multi-tier grounding check

Standard DD runs one grounding check against the validator model.

Multi-tier adaptation: the grounding check runs once per hardware tier, with
tier-specific context injected:

```python
TIER_CONTEXTS = {
    "scout": (
        "You are evaluating whether a 12MP RGB camera with fixed wide-angle "
        "mount, operating at 20-40m AGL with no stabilisation, can observe "
        "the described feature. Pixel footprint is approximately 1-2 cm at "
        "nadir. Motion blur is present at higher wind speeds and wave chop. "
        "Features requiring sub-10cm resolution, precise colour calibration, "
        "or multi-frame temporal comparison are NOT observable on this tier."
    ),
    "commander": (
        "You are evaluating whether a 20MP RGB camera with a stabilised "
        "3-axis gimbal, operating at 10-20m AGL with hover capability, can "
        "observe the described feature. Pixel footprint is approximately "
        "0.5-1cm at nadir. Multi-frame temporal features are available via "
        "the onboard frame buffer."
    ),
}
```

A rule criterion that fails grounding for the scout tier but passes for the
commander tier is retained in a tier-specific variant. The scout receives the
within-frame proxy version; the commander receives the temporal version.

### 3.4 Retroactive reprocessing

After a rule is accepted, the archive reprocessor runs the rule against the
swarm's frame buffer:

```python
async def reprocess_archive(
    rule: Rule,
    frame_buffer: FrameBuffer,
    lookback_seconds: int = 2700,  # 45 minutes
    tier: str = "scout",
) -> list[ReclassifiedFrame]:
    frames = frame_buffer.query(
        max_age_seconds=lookback_seconds,
        tier=tier,
        original_class="no_person",           # only re-examine negatives
        confidence_min=0.70,                  # confident negatives only
    )
    results = await asyncio.gather(*[
        apply_rule_to_frame(rule, frame, tier=tier)
        for frame in frames
    ])
    return [r for r in results if r.reclassified]
```

This is a first-class operation in the protocol. Retroactive results are
reported to the ground station and added to the semantic track map immediately.

---

## 4. Domain Configuration

```python
# python/domain_config.py  (abbreviated; see file for full content)

MARITIME_SAR_CONFIG = DomainConfig(
    expert_role=(
        "coast guard rescue swimmer and maritime search-and-rescue coordinator "
        "with operational experience in open-water person-recovery across "
        "multiple sea states"
    ),
    item_noun="UAV sea-surface surveillance frame",
    classification_noun="person-in-water assessment",
    feature_noun=(
        "optical sea-surface feature, floating silhouette characteristic, "
        "or water-disturbance pattern"
    ),
    observation_guidance=(
        "floating object silhouette geometry and bilateral symmetry, "
        "head-and-shoulder oval profile above the waterline, "
        "arm-induced V-shaped water disturbance flanking the floating object, "
        "brightness uniformity within a floating object compared with whitecap "
        "diffusion pattern, colour contrast between skin tone or life vest and "
        "surrounding water, object size relative to wave features"
    ),
    non_visual_exclusions=(
        "thermal infrared temperature readings (not available on scout tier), "
        "AIS transponder position data, acoustic pinger signals, GPS beacon "
        "coordinates, water temperature, time since person entered water, "
        "current or tide data"
    ),
    precision_gate=0.90,
    max_fp=0,
)
```

See [`python/domain_config.py`](python/domain_config.py) for the full
configuration including tier observability contexts and confusable pairs.

---

## 5. Simulation Stack

### 5.1 Phase 1: Real data, no simulator (recommended starting point)

**SeaDronesSee** provides real UAV footage of persons in water with ground-truth
bounding-box labels across multiple sea states. This is sufficient for a
complete DD session — including pool validation — without any data collection,
staging, or simulation:

```
SeaDronesSee frames
  ├── person_in_water/          ← PUPIL failure frames + pool positives
  ├── life_ring_unoccupied/     ← pool negatives (measured primary confusable)
  └── whitecap/ floating_debris/ ← pool negatives (secondary confusables)
        ↓
  Select failure frame (person, misclassified by baseline model)
  + commander thermal confirmation (ground truth label)
        ↓
  DD session (TUTOR dialogue → grounding check → pool validation)
        ↓
  Rule accepted → validate on held-out SeaDronesSee split
```

This is the critical data advantage over the original earthquake scenario,
which required simulation or staged photography to produce labeled
person-under-panel examples.

### 5.2 Option A: Gazebo + PX4 SITL (fleet dynamics)

For demonstrating fleet broadcast and retroactive reprocessing at swarm scale:

- Gazebo + PX4 SITL handles flight dynamics and swarm coordination (ROS2)
- SeaDronesSee frames are injected as simulated camera feeds — no rendering
  engine required
- 40-drone swarm feasible on a single workstation
- Mesh network simulation between drone nodes

```
SeaDronesSee frame injection          Gazebo + PX4 SITL
  ├── Frames served per drone ID  ←→  ├── Flight dynamics
  ├── Ground truth oracle             ├── 40-drone swarm
  └── Commander thermal labels        ├── Mesh network sim
        ↓                             └── ROS2 bridge
  PUPIL classifier per drone
  + DD rule injection
        ↓
  Classification output per drone
        ↓
  Semantic track map update
        ↓
  Broadcast engine → all scouts
```

### 5.3 Option B: Photorealistic rendering (optional)

If visual fidelity for VLM evaluation is required:
- **MarineVerse** (Unreal Engine ocean simulation) — photorealistic sea-surface
  rendering, configurable wave height and sea state
- Alternatively, **AirSim** ocean scene with water material applied to ground
- Increases setup complexity substantially; recommended only after Phase 2
  validates the DD loop with SeaDronesSee frames

### 5.4 Thermal simulation

The commander tier's thermal camera output is used only in the TUTOR dialogue
— the scout PUPIL never sees thermal. For the demo:

1. **SeaDronesSee ground truth labels** replace thermal output for the TUTOR
   input — simplest approach
2. **FLIR Free Thermal Dataset** (paired RGB + thermal frames) — real thermal
   imagery for TUTOR grounding if stronger realism is needed
3. **Gazebo thermal plugin** — simulated thermal for the full integrated demo

---

## 6. MCP Server Interface

The ground station exposes an MCP server for natural language control from
a Claude Code session:

```python
# Available MCP tools

get_camera_frame(
    drone_id: str,              # "S22", "C2"
    timestamp: str = "latest",  # "-18m", "2024-11-14T09:22:00Z"
    channel: str = "rgb",       # "rgb" | "thermal" (commander only)
) -> ImageFrame

broadcast_rule(
    rule: Rule,
    tiers: list[str],           # ["scout", "commander"] or ["scout"]
) -> BroadcastResult           # per-drone acknowledgement, latency

reprocess_archive(
    rule_id: str,
    lookback_seconds: int = 2700,
    tier: str = "scout",
) -> list[ReclassifiedFrame]

update_track_map(
    coordinates: tuple[float, float],
    detection_class: str,       # "person_in_water", "whitecap", etc.
    confidence: float,
    rule_id: str,
) -> None

get_swarm_state() -> SwarmState  # positions, coverage map, current rules
```

**Example Claude Code session**:

```
User: "Run DD on S22's frame from 18 minutes ago against C2's thermal
       confirmation at sector 4. If a rule is accepted, broadcast to all
       scouts and reprocess the last 45 minutes."

Claude calls:
  1. get_camera_frame("S22", timestamp="-18m", channel="rgb")
  2. get_camera_frame("C2", timestamp="-5m", channel="thermal")
  3. run_dd_session(rgb_frame, confirmation="Thermal confirmed 37°C human
                   heat signature at sector 4 coordinates")
  4. [if rule accepted] broadcast_rule(rule, tiers=["scout"])
  5. [if rule accepted] reprocess_archive(rule.id, lookback_seconds=2700)
  6. get_swarm_state()  → reports person-in-water candidate from S22 archive
```

---

## 7. Evaluation Framework

### 7.1 Primary metrics

| Metric | Measurement | Target |
|---|---|---|
| Time to fleet-wide rule deployment | Wall-clock from rule acceptance to last drone acknowledgement | < 60 seconds |
| Retroactive recall | Persons found in archive / total persons present in archive | > 0.80 |
| Rule precision on pool | TP / (TP + FP) on held-out SeaDronesSee split | ≥ 0.90 |
| Cross-tier rule consistency | Fraction of scout classifications matching commander on same frame | > 0.85 |
| Conventional pipeline latency | Time to equivalent update via retrain/validate/deploy | Baseline |

### 7.2 Ablation conditions

| Condition | What it removes | Purpose |
|---|---|---|
| No DD, no retraining | Baseline classifier only | Shows structural blind spot |
| No DD, human review queue | Analysts review flagged frames | Shows confident-miss problem |
| DD without temporal reformulation | Temporal rules applied verbatim to single-frame scouts | Shows within-frame proxy value |
| DD without grounding check | Rules deployed without tier filter | Shows unobservable criteria harm precision |
| DD without retroactive | Rule applies forward only | Quantifies retroactive value |
| DD single-tier rule | All scouts use commander-tier rule verbatim | Shows value of tier-specific variants |

### 7.3 Confusable pairs

> **Empirical update (2026-04-13):** Initial design assumed whitecap as the
> primary Qwen failure mode. Measurement on 120 SeaDronesSee val person frames
> showed Qwen never predicts `whitecap` for a swimmer. The measured primary
> tractable confusion is `life_ring_unoccupied`. See README §10 for details.

**Measured primary (Qwen3-VL-8B)**: `person_in_water` vs `life_ring_unoccupied`
— small bright oval near SAR vessels misread as thrown life ring. Systematic,
high-confidence (0.95–0.97), fixed by DD session_003 (lru_001/lru_002).

**Designed primary (general)**: `person_in_water` vs `whitecap` — both present
as small pale ovals at 30m altitude. Ground truth via thermal or manual
recovery. Relevant for other model architectures and closer-altitude frames.

**Secondary**:
- `person_in_water` vs `floating_debris` (similar size, non-uniform shape)
- `person_in_water` vs `seabird_on_water` (at lower altitudes, similar oval geometry)
- `person_in_water` vs `other` (scale/attention failure at high altitude — addressed
  by `uncertain_investigate` class + urgency accumulator, not by DD rules directly)

---

## 8. Data Strategy

### 8.1 SeaDronesSee (primary, no collection required)

SeaDronesSee is purpose-built for UAV maritime person-in-water detection.

| Property | Value |
|---|---|
| Total frames | ~5,000 annotated UAV frames |
| Classes | person, life vest, swimmer, boat, buoy, nothing |
| Sea states | Calm, moderate, rough |
| Altitude range | 5–100m AGL |
| Camera | DJI Phantom 4 Pro (20MP) |
| License | Academic use |
| Access | https://github.com/Ben93kie/SeaDronesSee |

For pool validation, the relevant split is:
- Positives: `person` class, confident-negative frames only (where a baseline
  MobileNetV3 scores ≥ 0.70 "no_person")
- Negatives: `nothing` class frames containing whitecaps or floating debris

Targeting 20–40 pool frames is achievable from SeaDronesSee without augmentation.

### 8.2 Supplementary datasets

| Dataset | Content | Relevance | Access |
|---|---|---|---|
| MODD2 | Maritime obstacle detection, UAV perspective | Floating debris negatives | University of Ljubljana |
| FLIR Free Thermal Dataset | Paired RGB + thermal frames | Real thermal for TUTOR grounding | FLIR (registration) |
| SeaShips | Open water imagery at drone altitude | Additional scene variety | Public |

### 8.3 Minimum viable dataset for a DD session

- 1 confirmed failure frame (person-in-water misclassified — measured primary:
  `life_ring_unoccupied`; designed primary: `whitecap` for other architectures)
- 1 ground truth confirmation (thermal label or SeaDronesSee ground truth)
- 20–40 labeled pool frames from SeaDronesSee (include LSA-only frames as
  life_ring negatives — only 10 exist in val split; supplement from train split)

All three are available from SeaDronesSee without any additional data collection.

---

## 9. Repository Layout

```
usecases/ai-fleets/seapatch/
├── README.md                    ← User-facing scenario, motivation, measured results
├── DESIGN.md                    ← This file
├── POSITIONING.md               ← Marketing narrative and claim framing
├── knowledge_base/
│   ├── person_in_water_vs_whitecap.json          ← Seed + session_001 rules
│   └── person_in_water_vs_life_ring_unoccupied.json  ← session_003 rules (lru_001/002)
├── python/
│   ├── domain_config.py         ← DomainConfig + TIER_OBSERVABILITY + CROSS_MODAL_TUTOR_PROMPT
│   ├── agents.py                ← Wraps core DD agents with maritime config
│   ├── fleet.py                 ← Swarm state, rule broadcast, urgency accumulator,
│   │                               escalation trigger (UrgencyCell, EscalationEvent)
│   ├── archive.py               ← Frame buffer + retroactive reprocessing
│   ├── pool_builder.py          ← SeaDronesSee → labeled pool construction
│   ├── qwen_pupil_eval.py       ← Qwen3-VL-8B PUPIL evaluation via OpenRouter;
│   │                               uncertain_investigate class + partial-credit scoring
│   ├── run_dd_session.py        ← Standalone DD session runner (CLI)
│   ├── run_pupil_dd_experiment.py ← End-to-end: baseline eval → DD → re-eval
│   ├── mcp_server.py            ← MCP tool definitions for Claude Code control
│   ├── simulation/
│   │   ├── seadronessee_bridge.py   ← SeaDronesSee frame injection (no renderer)
│   │   ├── gazebo_bridge.py         ← Gazebo + PX4 SITL interface
│   │   └── thermal_oracle.py        ← Ground truth → simulated thermal output
│   └── tests/
│       ├── test_agents_grounding.py
│       ├── test_fleet_archive.py
│       └── test_pool_builder.py
└── assets/
    └── README.md
```

---

## 10. Implementation Sequence

Recommended build order to reach a demonstrable result with minimum effort:

**Phase 1 (standalone DD, real data, no simulator)**
- Implement `domain_config.py` with cross-modal TUTOR prompt and temporal
  feature reformulation
- Implement multi-tier grounding check in `agents.py` including temporal
  feature detection pass
- Download SeaDronesSee; build a 30-frame labeled pool (6 person, 24 negative)
  from frames where baseline classifier is confidently wrong
- Run one complete DD session end-to-end
- Validate rule precision on held-out SeaDronesSee split

**Phase 2 (fleet simulation, no renderer)**
- Integrate Gazebo + PX4 SITL with 10-drone reduced swarm
- Implement `fleet.py` with rule broadcast and retroactive reprocessing
- Inject SeaDronesSee frames as simulated camera feeds via `seadronessee_bridge.py`
- Use ground truth labels in place of live thermal for TUTOR input
- Demonstrate full loop: fleet misses person → DD session → fleet patched →
  archive reprocessed → person found

**Phase 3 (MCP interface + full swarm)**
- Implement `mcp_server.py` exposing ground station tools
- Scale swarm to 40 drones
- Add Claude Code control loop via MCP session (see §6 example)

**Phase 4 (real thermal + evaluation)**
- Substitute FLIR paired thermal imagery for TUTOR dialogue
- Evaluate rule precision and retroactive recall against held-out SeaDronesSee
- Quantify conventional pipeline latency as baseline comparison

---

## 11. Connection to Broader KF Architecture

This use case extends DD in four directions not previously explored:

| Extension | Prior DD work | This use case |
|---|---|---|
| Sensor modality | Single modality throughout | Cross-modal: TUTOR in thermal, PUPIL in RGB |
| Fleet scale | Single PUPIL model | 40 heterogeneous PUPIL instances |
| Temporal scope | Forward-only rule application | Retroactive archive reprocessing |
| Feature reformulation | Rule features directly observable | Temporal features reformulated as within-frame proxies for single-frame tiers |

The rule pool is a KF artifact. The semantic track map is a KF artifact. The
fleet broadcast protocol is a KF distribution mechanism. This use case is
a demonstration of KF as a runtime knowledge infrastructure for multi-agent
embodied systems operating under strict time-to-fix constraints.

The MCP server interface means the entire system is controllable through natural
language from a Claude Code session, establishing a clean separation between
the cognitive layer (Claude + KF) and the physical layer (drone fleet +
physics simulation).
