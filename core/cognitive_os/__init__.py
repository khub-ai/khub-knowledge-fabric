"""core.cognitive_os — Cognitive OS namespace.

All COS work lives under this package, clearly separated from non-COS
KF framework code (``core.knowledge``, ``core.pipeline``,
``core.benchmark``, ``core.dialogic_distillation``).

Current contents:

* ``core.cognitive_os.engine``  — the second-generation cognitive
                                   engine.  Domain-agnostic symbolic
                                   reasoning substrate shared by ARC
                                   and robotics.  See
                                   ``core/cognitive_os/engine/DESIGN.md``.

Archived elsewhere (``.private/_archive/cognitive_os_v1/``):

* The first-generation COS implementation (StateStore, Environment,
  Observation, Hypothesis, Perception, Tick/StreamRecorder, miners,
  safety, causal, resources, similarity).  Kept for reference only.
  Was imported exclusively by ``usecases/arc-agi-3/`` (now legacy).

This top-level package intentionally exports nothing at module level —
import from the specific sub-package you need (currently
``core.cognitive_os.engine``).
"""
