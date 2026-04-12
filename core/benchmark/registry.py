"""
registry.py — Cross-domain benchmark manifest registry.

Scans all benchmarks/ directories under the KF repo and provides lookup
by benchmark_id. Import the module-level `registry` singleton directly:

    from core.benchmark import registry
    manifest = registry.load("road_surface_dry_vs_wet_probe_v1")
    for entry in registry.list_all():
        print(entry["benchmark_id"], entry["domain"], entry["n_images"])
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from .schema import BenchmarkManifest

_KF_ROOT = Path(__file__).resolve().parents[2]


class BenchmarkRegistry:
    """Scans all benchmark manifests under the KF repo and provides lookup by ID.

    The registry is lazy — scanning happens on first use.  Call refresh() to
    pick up newly generated manifests without restarting the process.
    """

    def __init__(self, kf_root: Optional[Path] = None):
        self._kf_root  = Path(kf_root) if kf_root else _KF_ROOT
        self._index:   Dict[str, Path] = {}
        self._scanned: bool            = False

    # ------------------------------------------------------------------
    # Internal scan
    # ------------------------------------------------------------------

    def _scan(self) -> None:
        if self._scanned:
            return
        # Glob all benchmarks/ directories anywhere under usecases/
        for json_path in self._kf_root.glob("usecases/**/benchmarks/*.json"):
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                bid = data.get("benchmark_id")
                if bid:
                    self._index[bid] = json_path
            except Exception:
                pass
        self._scanned = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find(self, benchmark_id: str) -> Path:
        """Return the filesystem path to a manifest JSON by benchmark_id.

        Raises:
            KeyError: if benchmark_id is not in the registry.
        """
        self._scan()
        if benchmark_id not in self._index:
            raise KeyError(
                f"Benchmark '{benchmark_id}' not found in registry.\n"
                f"Known IDs ({len(self._index)}): "
                + ", ".join(sorted(self._index))
            )
        return self._index[benchmark_id]

    def load(self, benchmark_id: str) -> BenchmarkManifest:
        """Load and return a normalised BenchmarkManifest by benchmark_id."""
        path = self.find(benchmark_id)
        return BenchmarkManifest.load(path)

    def list_all(self) -> List[dict]:
        """Return a summary list of all known benchmarks, sorted by ID."""
        self._scan()
        results = []
        for bid, path in sorted(self._index.items()):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                results.append({
                    "benchmark_id":   bid,
                    "version":        data.get("version", "?"),
                    "benchmark_type": data.get("benchmark_type", "?"),
                    "domain":         data.get("domain", ""),
                    "pair_id":        data.get("pair_id", "?"),
                    "n_images":       len(data.get("images", [])),
                    "pupil_model":    data.get("pupil_model"),
                    "path":           str(path.relative_to(self._kf_root)),
                })
            except Exception:
                pass
        return results

    def refresh(self) -> None:
        """Force re-scan (call after generating new manifests at runtime)."""
        self._index.clear()
        self._scanned = False
        self._scan()

    def __repr__(self) -> str:
        self._scan()
        return f"BenchmarkRegistry({len(self._index)} manifests, root={self._kf_root})"


# ---------------------------------------------------------------------------
# Module-level singleton — import this directly
# ---------------------------------------------------------------------------
registry = BenchmarkRegistry()
