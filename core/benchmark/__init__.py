"""
core.benchmark — Cross-domain benchmark manifest registry and normalised types.

Provides a unified interface for loading, listing, and inspecting DD benchmark
manifests across all domains (road surface, dermatology, birds, drone swarm).

Manifests are JSON files committed to git in each domain's benchmarks/
directory. Images are NOT stored here — only image IDs and metadata. Callers
provide a domain-specific dataset loader to resolve IDs to real file paths.

Public API
----------
BenchmarkManifest       normalised manifest dataclass (load/save/from_dict)
BenchmarkImage          normalised per-image record with metadata{} dict
BenchmarkRegistry       scans repo and looks up manifests by benchmark_id
registry                module-level singleton — use this directly

BENCHMARK_TYPE_PROBE    "probe"
BENCHMARK_TYPE_POOL     "pool"
BENCHMARK_TYPE_FAILURES "failures"

DIFFICULTY_EASY / MEDIUM / HARD
DOMAIN_ROAD_SURFACE / DERMATOLOGY / BIRDS / DRONE_SWARM

Quick start
-----------
    from core.benchmark import registry

    # List everything
    for entry in registry.list_all():
        print(entry["benchmark_id"], entry["domain"], entry["n_images"])

    # Load by ID
    manifest = registry.load("road_surface_dry_vs_wet_probe_v1")
    print(manifest.n_images)               # 24
    print(manifest.domain)                 # "road_surface"
    print(manifest.images[0].metadata)     # {"friction": "dry", ...}

    # After generating new manifests at runtime
    registry.refresh()
"""

from .schema import (
    BenchmarkManifest,
    BenchmarkImage,
    BENCHMARK_TYPE_PROBE,
    BENCHMARK_TYPE_POOL,
    BENCHMARK_TYPE_FAILURES,
    DIFFICULTY_EASY,
    DIFFICULTY_MEDIUM,
    DIFFICULTY_HARD,
    DOMAIN_ROAD_SURFACE,
    DOMAIN_DERMATOLOGY,
    DOMAIN_BIRDS,
    DOMAIN_DRONE_SWARM,
)

from .registry import BenchmarkRegistry, registry
