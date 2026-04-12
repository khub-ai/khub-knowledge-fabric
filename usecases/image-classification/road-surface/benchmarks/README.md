# Road Surface Benchmark Manifests

Fixed, versioned image sets for reproducible DD experiments on the RSCD dataset.

---

## What is a benchmark manifest?

A JSON file that specifies exactly which RSCD images to use for a given
experiment. Image IDs and metadata are committed to git; the actual JPEG bytes
are not (the dataset is ~14 GB). Any developer with the RSCD zip can reproduce
the same image set by pointing the loader at their local copy.

Three types:

| Type | Purpose | Model-agnostic? |
|---|---|---|
| `probe` | PUPIL Domain Readiness Probe | Yes |
| `pool` | DD pool validation gate | Yes |
| `failures` | Known PUPIL error cases for DD sessions | No — per-model |

---

## Current manifests

| File | Type | Images | Pair | Created |
|---|---|---|---|---|
| `dry_vs_wet_probe_v1.json` | probe | 24 (12 dry + 12 wet) | dry_vs_wet | 2026-04-12 |
| `dry_vs_wet_pool_v1.json` | pool | 40 (20 dry + 20 wet) | dry_vs_wet | 2026-04-12 |

Failure manifests (per-PUPIL) to be added after first probe run.

---

## Manifest format

```jsonc
{
  "benchmark_id":    "road_surface_dry_vs_wet_probe_v1",
  "version":         "1.0.0",
  "benchmark_type":  "probe",          // "probe" | "pool" | "failures"
  "created":         "2026-04-12",
  "pair_id":         "dry_vs_wet",
  "class_a":         "Dry Road",
  "class_b":         "Wet Road",
  "material_filter": "asphalt",        // null = all materials
  "rscd_split":      "train",          // "train" | "test" | "val"
  "selection_seed":  7,                // for reproducible re-sampling
  "pupil_model":     null,             // null for model-agnostic; set for failures
  "n_per_class":     12,
  "description":     "...",
  "images": [
    {
      "image_id":   "20220321182055148",
      "filename":   "20220321182055148-dry-asphalt-smooth.jpg",
      "friction":   "dry",
      "material":   "asphalt",
      "roughness":  "smooth",
      "true_class": "Dry Road",
      "difficulty": "easy",            // "easy" | "medium" | "hard"
      "notes":      ""
    }
    // ...
  ]
}
```

**`difficulty` values:**

| Value | How assigned in v1 | How to upgrade |
|---|---|---|
| `easy` | roughness = smooth | Run `--annotate-difficulty` (TUTOR rates each image) |
| `medium` | roughness = slight | Same |
| `hard` | roughness = severe | Same |

The roughness proxy is structural, not visual. It is a reasonable starting point
but will be replaced with TUTOR-rated visual ambiguity scores in v2 manifests.

---

## Generating manifests (maintainer procedure)

Run once when creating or updating a benchmark version. Requires the RSCD zip.

### Prerequisites

```bash
# RSCD zip at default location (or pass --data-dir)
C:\_backup\ml\data\rscd-dataset-1million.zip

# Or download via Kaggle:
pip install kaggle
kaggle datasets download cristvollerei/rscd-dataset-1million

# API keys (only needed for --annotate-difficulty or --types failures)
export ANTHROPIC_API_KEY=sk-ant-...
export OPENROUTER_API_KEY=sk-or-...
```

### Commands

**Probe + pool manifests — no API calls, ~30 seconds:**

```bash
cd usecases/image-classification/road-surface/python
python create_benchmark.py --pair dry_vs_wet --types probe,pool
```

**With TUTOR difficulty annotation — ~$0.10, ~3 minutes:**

```bash
python create_benchmark.py --pair dry_vs_wet --types probe,pool \
    --annotate-difficulty --tutor-model claude-opus-4-6
```

**Failure manifest for Qwen3-VL-8B — ~$0.01, ~5 minutes:**

```bash
python create_benchmark.py --pair dry_vs_wet --types failures \
    --pupil-model qwen/qwen3-vl-8b-instruct --n-failures 8
```

**Full regeneration (probe + pool + failures + difficulty annotation):**

```bash
python create_benchmark.py --pair dry_vs_wet \
    --types probe,pool,failures \
    --annotate-difficulty \
    --pupil-model qwen/qwen3-vl-8b-instruct \
    --n-failures 8
```

**Dry run — print what would be generated without writing anything:**

```bash
python create_benchmark.py --pair dry_vs_wet --types probe,pool --dry-run
```

### Full CLI reference

```
--pair          Confusable pair to benchmark  [dry_vs_wet | wet_vs_water | ...]
--types         Comma-separated: probe, pool, failures  [default: probe,pool]
--n-probe       Images per class for probe manifest  [default: 12]
--n-pool        Images per class for pool manifest  [default: 20]
--n-failures    Number of failure cases to discover  [default: 8]
--pupil-model   Model for failure discovery  [default: qwen/qwen3-vl-8b-instruct]
--tutor-model   Model for difficulty annotation  [default: claude-opus-4-6]
--annotate-difficulty  Call TUTOR to rate visual difficulty per image
--data-dir      Path to directory containing RSCD zip  [default: C:\_backup\ml\data]
--output-dir    Where to write manifests  [default: benchmarks/]
--dry-run       Preview output without writing files or calling APIs
```

### After generating: commit and push

```bash
# From repo root
git add usecases/image-classification/road-surface/benchmarks/*.json
git commit -m "Add dry_vs_wet benchmark manifests v1"
git push
```

**Never commit extracted image files.** Only commit the `.json` manifest files.
Extracted images go in `.tmp/` which is gitignored.

---

## Using manifests in code

```python
from pathlib import Path
from dataset import load as load_rscd
from benchmark import load_benchmark, to_probe_images, to_pool_images

ds      = load_rscd("C:/_backup/ml/data")
tmp_dir = Path(".tmp/rscd_session")

# --- For the PUPIL Domain Readiness Probe ---
from core.dialogic_distillation import probe, reset_probe_costs

probe_images = to_probe_images("dry_vs_wet_probe_v1.json", ds, tmp_dir)
# Returns list[ProbeImage], ready to pass to probe()

report = asyncio.run(probe(
    pupil_model     = "qwen/qwen3-vl-8b-instruct",
    tutor_model     = "claude-opus-4-6",
    validator_model = "claude-sonnet-4-6",
    domain_config   = ROAD_SURFACE_CONFIG,
    probe_images    = probe_images,
    pair_info       = {"pair_id": "dry_vs_wet",
                       "class_a": "Dry Road", "class_b": "Wet Road"},
))
print(report["verdict"])   # "go" | "partial" | "no-go"

# --- For DD pool validation ---
pool_images = to_pool_images("dry_vs_wet_pool_v1.json", ds, tmp_dir)
# Returns list[(image_path, true_class)], ready for run_dialogic_distillation()

# --- Load full BenchmarkSet (includes metadata) ---
bset = load_benchmark("dry_vs_wet_probe_v1.json", ds, tmp_dir)
print(bset.benchmark_id)           # "road_surface_dry_vs_wet_probe_v1"
print(bset.n_images)               # 24
print(bset.images_for_class("Dry Road"))   # [(path, "Dry Road", "easy"), ...]
```

Manifest paths can be:
- Absolute: `/full/path/to/dry_vs_wet_probe_v1.json`
- Filename only: `"dry_vs_wet_probe_v1.json"` — loader searches `benchmarks/` dir automatically

---

## Versioning policy

- **Never modify** a committed manifest's `images` list in place.
- **Create a new version** (`v2`, `v3`, ...) when images are added, removed,
  or difficulty labels are reannotated.
- **Keep all versions** — published results reference specific version IDs
  (e.g. `road_surface_dry_vs_wet_probe_v1`).
- **Increment version** in the `"version"` field and in the filename.

Example version history:

```
dry_vs_wet_probe_v1.json   — roughness-based difficulty (structural proxy)
dry_vs_wet_probe_v2.json   — TUTOR-rated visual difficulty (after annotation)
dry_vs_wet_probe_v3.json   — expanded to 20 images per class
```

---

## Seed assignment

Different seeds are used for probe and pool so the two image sets are non-overlapping:

| Manifest type | Split | Seed |
|---|---|---|
| probe | train | 7 |
| pool | train | 42 |
| failures | test | 7 |

Pool seed 42 matches the original `distill_dialogic.py` experiment so pool
results are comparable to the first-run baseline.
