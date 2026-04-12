"""
schema.py — Normalised benchmark manifest types for cross-domain DD benchmarks.

All domain-specific per-image fields (friction/material for road surface,
lesion_id/diagnosis for dermatology, species_id for birds) are stored in a
generic `metadata` dict so that the top-level schema is identical across
domains. Old manifests that have flat domain fields are normalised on load.

The schema is the stable contract for publication: benchmark IDs and image
lists committed here can be cited in papers and reproduced by any developer
with a copy of the underlying source dataset.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BENCHMARK_TYPE_PROBE    = "probe"
BENCHMARK_TYPE_POOL     = "pool"
BENCHMARK_TYPE_FAILURES = "failures"

DIFFICULTY_EASY   = "easy"
DIFFICULTY_MEDIUM = "medium"
DIFFICULTY_HARD   = "hard"

DOMAIN_ROAD_SURFACE = "road_surface"
DOMAIN_DERMATOLOGY  = "dermatology"
DOMAIN_BIRDS        = "birds"
DOMAIN_DRONE_SWARM  = "drone_swarm"

# Path component → domain identifier (handles both hyphenated and underscored)
_PATH_TO_DOMAIN: Dict[str, str] = {
    "road-surface":  DOMAIN_ROAD_SURFACE,
    "road_surface":  DOMAIN_ROAD_SURFACE,
    "dermatology":   DOMAIN_DERMATOLOGY,
    "birds":         DOMAIN_BIRDS,
    "drone-swarm":   DOMAIN_DRONE_SWARM,
    "drone_swarm":   DOMAIN_DRONE_SWARM,
}

# Per-image fields that should be promoted into metadata{} on old-format load
_LEGACY_IMAGE_FIELDS = (
    # road surface
    "friction", "material", "roughness",
    # dermatology
    "lesion_id", "dx", "diagnosis", "localization", "age", "sex",
    # birds
    "species_id", "subspecies", "part",
)

# Top-level manifest fields — everything else is stored in extra{}
_KNOWN_TOP_LEVEL = {
    "benchmark_id", "version", "benchmark_type", "domain", "created",
    "pair_id", "class_a", "class_b", "pupil_model", "description",
    "images", "source_dataset", "n_per_class", "selection_seed",
    "material_filter", "rscd_split",
}


# ---------------------------------------------------------------------------
# Per-image record
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkImage:
    """One image entry — uniform schema across all domains."""
    image_id:       str
    filename:       str
    true_class:     str
    difficulty:     str  = DIFFICULTY_MEDIUM
    notes:          str  = ""
    source_dataset: str  = ""      # e.g. "rscd-dataset-1million", "HAM10000"
    metadata:       Dict = field(default_factory=dict)  # domain-specific fields

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkManifest:
    """Normalised benchmark manifest — uniform across all DD domains.

    Load with BenchmarkManifest.load(path) or via the registry singleton.
    """
    benchmark_id:   str
    version:        str
    benchmark_type: str
    domain:         str
    created:        str
    pair_id:        str
    class_a:        str
    class_b:        str
    pupil_model:    Optional[str]
    description:    str
    images:         List[BenchmarkImage]      = field(default_factory=list)
    manifest_path:  Optional[Path]            = field(default=None, compare=False)
    extra:          Dict                      = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_images(self) -> int:
        return len(self.images)

    def images_for_class(self, true_class: str) -> List[BenchmarkImage]:
        return [img for img in self.images if img.true_class == true_class]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        d = {
            "benchmark_id":   self.benchmark_id,
            "version":        self.version,
            "benchmark_type": self.benchmark_type,
            "domain":         self.domain,
            "created":        self.created,
            "pair_id":        self.pair_id,
            "class_a":        self.class_a,
            "class_b":        self.class_b,
            "pupil_model":    self.pupil_model,
            "description":    self.description,
            "images":         [img.to_dict() for img in self.images],
        }
        d.update(self.extra)
        return d

    def save(self, path: str | Path) -> None:
        """Write manifest to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Loading — normalises old flat-field format on the fly
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(
        cls,
        data: dict,
        path: Optional[Path] = None,
        domain: str = "",
    ) -> "BenchmarkManifest":
        """Construct from raw manifest dict, normalising legacy field layouts."""
        # Domain: prefer explicit field, then caller arg, then infer from path
        resolved_domain = data.get("domain", domain) or (
            _infer_domain(path) if path else ""
        )

        # Per-image source_dataset may be set at manifest level
        manifest_source = data.get("source_dataset", "")

        images = []
        for entry in data.get("images", []):
            # Promote flat legacy domain fields into metadata{}
            meta = dict(entry.get("metadata", {}))
            for fname in _LEGACY_IMAGE_FIELDS:
                if fname in entry and fname not in meta:
                    meta[fname] = entry[fname]

            images.append(BenchmarkImage(
                image_id       = entry["image_id"],
                filename       = entry.get("filename", ""),
                true_class     = entry["true_class"],
                difficulty     = entry.get("difficulty", DIFFICULTY_MEDIUM),
                notes          = entry.get("notes", ""),
                source_dataset = entry.get("source_dataset", manifest_source),
                metadata       = meta,
            ))

        extra = {k: v for k, v in data.items() if k not in _KNOWN_TOP_LEVEL}

        return cls(
            benchmark_id   = data["benchmark_id"],
            version        = data.get("version", "1.0.0"),
            benchmark_type = data.get("benchmark_type", BENCHMARK_TYPE_PROBE),
            domain         = resolved_domain,
            created        = data.get("created", ""),
            pair_id        = data["pair_id"],
            class_a        = data["class_a"],
            class_b        = data["class_b"],
            pupil_model    = data.get("pupil_model"),
            description    = data.get("description", ""),
            images         = images,
            manifest_path  = path,
            extra          = extra,
        )

    @classmethod
    def load(cls, path: str | Path) -> "BenchmarkManifest":
        """Load and normalise a manifest from a JSON file."""
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data, path=path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_domain(path: Path) -> str:
    """Infer domain identifier from manifest path components."""
    for part in path.parts:
        key = part.lower()
        if key in _PATH_TO_DOMAIN:
            return _PATH_TO_DOMAIN[key]
        key_under = key.replace("-", "_")
        if key_under in _PATH_TO_DOMAIN:
            return _PATH_TO_DOMAIN[key_under]
    return ""
