"""
state_store.py — StateStore for the Cognitive OS layer.

A schemaless, evidence-tracked, scope-managed, change-emitting fact store.
Replaces hardcoded game-specific data structures with a uniform representation
where every piece of knowledge is a (key, value, confidence, source, scope)
tuple. Rules, the OBSERVER, and the planner all read and write the same store.

Design principles (from DESIGN.md):
  - Everything is a fact in a typed store.
  - Keys are open-ended tuples: ("world", "step_size"), ("obj", 3, "role"), etc.
  - Every write emits a Delta so rules can pattern-match on changes.
  - Scope lifecycle (step < level < episode < game) controls fact retention.
  - Relation facts (RelFact) are first-class, not second-class annotations.
  - The store is domain-agnostic — ARC colors and robot temperatures are both
    just Any-typed values keyed by open tuples.
"""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Core data classes
# ---------------------------------------------------------------------------

@dataclass
class StateFact:
    """A single piece of knowledge in the store."""
    value:      Any
    confidence: float = 1.0        # 0.0–1.0
    source:     str   = "observed"  # "prior" | "inferred" | "observed" | "rule" | "told"
    scope:      str   = "level"    # "step" | "level" | "episode" | "game"
    timestamp:  float = 0.0        # step index (int) or Unix epoch (float)
    evidence:   int   = 1          # observation count supporting this value


@dataclass
class RelFact:
    """A relationship between two or more entities."""
    rel_type:   str                # e.g. "adjacent", "same_shape", "blocks"
    subjects:   tuple              # entity IDs (ordered where meaningful)
    properties: dict  = field(default_factory=dict)
    confidence: float = 1.0
    scope:      str   = "level"
    timestamp:  float = 0.0
    evidence:   int   = 1


@dataclass
class Delta:
    """A change to a fact, emitted on every write."""
    key:        tuple
    old_value:  Any     # None if fact is new
    new_value:  Any
    timestamp:  float


@dataclass
class EventFact:
    """An immutable event record (append-only log)."""
    event_type: str                # "action_taken", "level_advanced", "contact", ...
    subjects:   tuple = ()         # entity IDs involved
    properties: dict  = field(default_factory=dict)
    timestamp:  float = 0.0
    source:     str   = "observed"
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# Scope ordering for clear_scope cascading
# ---------------------------------------------------------------------------

_SCOPE_ORDER = {"step": 0, "level": 1, "episode": 2, "game": 3}


# ---------------------------------------------------------------------------
# StateStore
# ---------------------------------------------------------------------------

class StateStore:
    """
    Schemaless fact store with evidence tracking, scope management, and
    change notification.

    Usage
    -----
    >>> store = StateStore()
    >>> store.set(("world", "step_size"), 5, source="inferred", scope="game")
    >>> store.get(("world", "step_size"))
    5
    >>> store.set(("obj", 3, "role"), "wall", confidence=0.8, source="rule")
    >>> store.query(("obj", 3))
    {("obj", 3, "role"): StateFact(value="wall", ...)}
    """

    def __init__(self) -> None:
        self._facts:    dict[tuple, StateFact] = {}
        self._rels:     dict[int, RelFact]     = {}
        self._deltas:   list[Delta]            = []   # current step's deltas
        self._events:   list[EventFact]        = []   # append-only event log
        self._next_rel: int = 0
        self._step:     int = 0

    # ------------------------------------------------------------------
    # Fact read
    # ------------------------------------------------------------------

    def get(self, key: tuple, default: Any = None) -> Any:
        """Return the value of a fact, or *default* if not present."""
        f = self._facts.get(key)
        return f.value if f is not None else default

    def get_fact(self, key: tuple) -> Optional[StateFact]:
        """Return the full StateFact, or None."""
        return self._facts.get(key)

    def has(self, key: tuple) -> bool:
        return key in self._facts

    def query(self, prefix: tuple) -> dict[tuple, StateFact]:
        """Return all facts whose key starts with *prefix*."""
        n = len(prefix)
        return {
            k: v for k, v in self._facts.items()
            if k[:n] == prefix
        }

    def all_facts(self) -> dict[tuple, StateFact]:
        """Return a shallow copy of all facts."""
        return dict(self._facts)

    # ------------------------------------------------------------------
    # Fact write
    # ------------------------------------------------------------------

    def set(
        self,
        key:        tuple,
        value:      Any,
        confidence: float = 1.0,
        source:     str   = "observed",
        scope:      str   = "level",
        evidence:   int   = 1,
    ) -> Delta:
        """
        Write a fact. If the key already exists, the old value is overwritten.
        A Delta is emitted regardless of whether the value changed.
        """
        old = self._facts.get(key)
        old_value = old.value if old is not None else None

        self._facts[key] = StateFact(
            value=value,
            confidence=confidence,
            source=source,
            scope=scope,
            timestamp=self._step,
            evidence=evidence,
        )

        delta = Delta(
            key=key,
            old_value=old_value,
            new_value=value,
            timestamp=self._step,
        )
        self._deltas.append(delta)
        return delta

    def strengthen(self, key: tuple, additional_evidence: int = 1,
                   new_confidence: Optional[float] = None) -> None:
        """Increment evidence count on an existing fact (no delta emitted)."""
        f = self._facts.get(key)
        if f is not None:
            f.evidence += additional_evidence
            if new_confidence is not None:
                f.confidence = new_confidence
            f.timestamp = self._step

    def remove(self, key: tuple) -> Optional[Delta]:
        """Remove a fact and emit a delta. Returns None if key didn't exist."""
        old = self._facts.pop(key, None)
        if old is None:
            return None
        delta = Delta(key=key, old_value=old.value, new_value=None,
                      timestamp=self._step)
        self._deltas.append(delta)
        return delta

    # ------------------------------------------------------------------
    # Relation read / write
    # ------------------------------------------------------------------

    def add_relation(
        self,
        rel_type:   str,
        subjects:   tuple,
        properties: Optional[dict] = None,
        confidence: float = 1.0,
        scope:      str   = "level",
        evidence:   int   = 1,
    ) -> int:
        """Add a relation fact. Returns the relation ID."""
        rid = self._next_rel
        self._next_rel += 1
        self._rels[rid] = RelFact(
            rel_type=rel_type,
            subjects=subjects,
            properties=properties or {},
            confidence=confidence,
            scope=scope,
            timestamp=self._step,
            evidence=evidence,
        )
        # Also emit a delta for the relation creation
        delta = Delta(
            key=("rel", rid),
            old_value=None,
            new_value=rel_type,
            timestamp=self._step,
        )
        self._deltas.append(delta)
        return rid

    def get_relations(
        self,
        rel_type:  Optional[str] = None,
        subject:   Optional[int] = None,
    ) -> list[tuple[int, RelFact]]:
        """Query relations by type and/or subject involvement."""
        results = []
        for rid, rf in self._rels.items():
            if rel_type is not None and rf.rel_type != rel_type:
                continue
            if subject is not None and subject not in rf.subjects:
                continue
            results.append((rid, rf))
        return results

    def remove_relation(self, rid: int) -> Optional[RelFact]:
        """Remove a relation by ID."""
        rf = self._rels.pop(rid, None)
        if rf is not None:
            self._deltas.append(Delta(
                key=("rel", rid), old_value=rf.rel_type, new_value=None,
                timestamp=self._step,
            ))
        return rf

    # ------------------------------------------------------------------
    # Event log (append-only)
    # ------------------------------------------------------------------

    def log_event(
        self,
        event_type: str,
        subjects:   tuple = (),
        properties: Optional[dict] = None,
        source:     str = "observed",
        confidence: float = 1.0,
    ) -> EventFact:
        """Record an immutable event."""
        evt = EventFact(
            event_type=event_type,
            subjects=subjects,
            properties=properties or {},
            timestamp=self._step,
            source=source,
            confidence=confidence,
        )
        self._events.append(evt)
        return evt

    def events(self, event_type: Optional[str] = None,
               since: Optional[float] = None) -> list[EventFact]:
        """Query events by type and/or time range."""
        results = []
        for e in self._events:
            if event_type is not None and e.event_type != event_type:
                continue
            if since is not None and e.timestamp < since:
                continue
            results.append(e)
        return results

    # ------------------------------------------------------------------
    # Scope lifecycle
    # ------------------------------------------------------------------

    def clear_scope(self, scope: str) -> int:
        """
        Remove all facts and relations with the given scope.
        Returns the number of facts removed.
        """
        removed = 0
        keys_to_remove = [k for k, f in self._facts.items() if f.scope == scope]
        for k in keys_to_remove:
            self.remove(k)
            removed += 1
        rids_to_remove = [rid for rid, rf in self._rels.items() if rf.scope == scope]
        for rid in rids_to_remove:
            self.remove_relation(rid)
            removed += 1
        return removed

    # ------------------------------------------------------------------
    # Step management
    # ------------------------------------------------------------------

    def advance_step(self) -> list[Delta]:
        """
        Advance the step counter. Clears step-scoped facts and returns
        the deltas from the completed step, then resets the delta buffer.
        """
        completed_deltas = list(self._deltas)
        self._deltas.clear()
        self.clear_scope("step")
        self._step += 1
        return completed_deltas

    @property
    def step(self) -> int:
        return self._step

    def pending_deltas(self) -> list[Delta]:
        """Return deltas accumulated in the current step (not consumed)."""
        return list(self._deltas)

    # ------------------------------------------------------------------
    # Concept facts (LLM-as-ontology-engine support)
    # ------------------------------------------------------------------

    def set_concept(
        self,
        name: str,
        parent: Optional[str] = None,
        discriminator: str = "",
        confidence: float = 0.8,
    ) -> None:
        """Write a concept fact into the store (used by OBSERVER/MEDIATOR)."""
        self.set(("concept", name, "discriminator"), discriminator,
                 confidence=confidence, source="observed", scope="game")
        if parent:
            self.set(("concept", name, "parent"), parent,
                     confidence=confidence, source="observed", scope="game")

    def get_concept_children(self, parent: str) -> list[str]:
        """Return all concept names whose parent is *parent*."""
        children = []
        for k, f in self._facts.items():
            if (len(k) == 3 and k[0] == "concept" and k[2] == "parent"
                    and f.value == parent):
                children.append(k[1])
        return children

    # ------------------------------------------------------------------
    # Prompt serialization
    # ------------------------------------------------------------------

    def format_for_prompt(self, max_lines: int = 60) -> str:
        """
        Produce a concise, human-readable summary of current facts
        suitable for injection into LLM prompts.
        """
        if not self._facts and not self._rels:
            return ""

        lines: list[str] = ["## StateStore snapshot"]
        line_count = 1

        # Group facts by namespace
        namespaces: dict[str, list[tuple[tuple, StateFact]]] = {}
        for k, f in sorted(self._facts.items(), key=lambda x: x[0]):
            ns = k[0] if k else "?"
            namespaces.setdefault(ns, []).append((k, f))

        for ns, facts in namespaces.items():
            if line_count >= max_lines:
                lines.append(f"  ... ({len(self._facts) - line_count} more facts)")
                break
            lines.append(f"\n**{ns}:**")
            line_count += 1
            for k, f in facts:
                if line_count >= max_lines:
                    break
                # Format key without namespace prefix for readability
                key_str = ".".join(str(p) for p in k[1:]) if len(k) > 1 else str(k)
                val_str = str(f.value)
                if len(val_str) > 80:
                    val_str = val_str[:77] + "..."
                conf_str = f" [{f.confidence:.0%}]" if f.confidence < 1.0 else ""
                src_str = f" ({f.source})" if f.source not in ("observed",) else ""
                lines.append(f"  {key_str} = {val_str}{conf_str}{src_str}")
                line_count += 1

        # Relations
        if self._rels and line_count < max_lines:
            lines.append("\n**relations:**")
            line_count += 1
            for rid, rf in sorted(self._rels.items()):
                if line_count >= max_lines:
                    lines.append(f"  ... ({len(self._rels)} total relations)")
                    break
                subj_str = ", ".join(str(s) for s in rf.subjects)
                props_str = f" {rf.properties}" if rf.properties else ""
                conf_str = f" [{rf.confidence:.0%}]" if rf.confidence < 1.0 else ""
                lines.append(f"  {rf.rel_type}({subj_str}){props_str}{conf_str}")
                line_count += 1

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Bulk import from legacy structures
    # ------------------------------------------------------------------

    def import_concept_bindings(self, bindings: dict, step: Optional[int] = None) -> None:
        """
        Import concept_bindings dict from the legacy state_manager format.

        Handles both formats:
          {color_int: {"role": str, "confidence": float, ...}}
          {"wall_colors": [c1, c2, ...]}
        """
        if step is not None:
            self._step = step

        for k, v in bindings.items():
            if k == "wall_colors":
                # Convert wall_colors list to individual role facts
                for wc in (v if isinstance(v, list) else []):
                    self.set(("obj", f"color_{wc}", "role"), "wall",
                             confidence=0.7, source="inferred", scope="level")
                continue

            if not isinstance(k, int):
                continue

            if isinstance(v, dict):
                role = v.get("role", "unknown")
                conf = v.get("confidence", 0.5)
                obs_count = v.get("total_obs", v.get("observations", 1))
            elif isinstance(v, str):
                role = v
                conf = 0.5
                obs_count = 1
            else:
                continue

            self.set(("obj", f"color_{k}", "role"), role,
                     confidence=conf, source="observed", scope="level",
                     evidence=obs_count)

    def import_action_effects(self, effects: dict) -> None:
        """
        Import action_effects dict from the legacy state_manager format.

        Extracts key facts: action direction, movement magnitude, call counts.
        """
        for action_name, data in effects.items():
            total = data.get("total_calls", 0)
            nonzero = data.get("nonzero_calls", 0)

            self.set(("action", action_name, "total_calls"), total,
                     source="observed", scope="episode")
            self.set(("action", action_name, "nonzero_calls"), nonzero,
                     source="observed", scope="episode")

            # Extract dominant direction from observations
            observations = data.get("observations", [])
            if observations:
                # Find most common direction
                dr_sum, dc_sum, n = 0, 0, 0
                for obs_entry in observations:
                    moves = obs_entry if isinstance(obs_entry, dict) else {}
                    dr = moves.get("dr", 0)
                    dc = moves.get("dc", 0)
                    if dr != 0 or dc != 0:
                        dr_sum += dr
                        dc_sum += dc
                        n += 1
                if n > 0:
                    avg_dr = dr_sum / n
                    avg_dc = dc_sum / n
                    self.set(("action", action_name, "avg_displacement"),
                             (round(avg_dr, 1), round(avg_dc, 1)),
                             source="inferred", scope="episode",
                             evidence=n)

    def import_env_hypothesis(self, hyp: Any) -> None:
        """
        Import an environment hypothesis object (duck-typed).

        Expected attributes on ``hyp`` (all optional via getattr):
          effective_step_size      / obs_step_size
          effective_action_map     / obs_action_map
          effective_player_colors  / obs_player_colors
          effective_walkable_colors / obs_walkable_colors

        ARC-AGI uses GameHypothesis from dynamic_discovery.py.
        Robotics adapters should produce objects with the same attribute names
        (or a thin wrapper) to reuse this import path.
        """
        # Step size
        ss = getattr(hyp, 'effective_step_size', None)
        if ss is not None:
            source = "inferred" if hyp.obs_step_size is not None else "prior"
            conf = 0.9 if source == "inferred" else 0.3
            self.set(("world", "step_size"), ss,
                     confidence=conf, source=source, scope="game")

        # Action map
        am = getattr(hyp, 'effective_action_map', None)
        if am:
            source = "inferred" if hyp.obs_action_map is not None else "prior"
            conf = 0.9 if source == "inferred" else 0.3
            self.set(("world", "action_map"), am,
                     confidence=conf, source=source, scope="game")

        # Player colors
        pc = getattr(hyp, 'effective_player_colors', None)
        if pc:
            source = "inferred" if hyp.obs_player_colors is not None else "prior"
            conf = 0.9 if source == "inferred" else 0.3
            self.set(("world", "player_colors"), pc,
                     confidence=conf, source=source, scope="episode")

        # Walkable colors
        wc = getattr(hyp, 'effective_walkable_colors', None)
        if wc:
            source = "inferred" if hyp.obs_walkable_colors is not None else "prior"
            conf = 0.9 if source == "inferred" else 0.3
            self.set(("world", "walkable_colors"), wc,
                     confidence=conf, source=source, scope="game")

    # ------------------------------------------------------------------
    # Snapshot / serialization
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """Return a JSON-serializable snapshot of all facts."""
        facts = {}
        for k, f in self._facts.items():
            key_str = ".".join(str(p) for p in k)
            facts[key_str] = {
                "value": _json_safe(f.value),
                "confidence": f.confidence,
                "source": f.source,
                "scope": f.scope,
                "timestamp": f.timestamp,
                "evidence": f.evidence,
            }
        rels = {}
        for rid, rf in self._rels.items():
            rels[str(rid)] = {
                "rel_type": rf.rel_type,
                "subjects": list(rf.subjects),
                "properties": _json_safe(rf.properties),
                "confidence": rf.confidence,
                "scope": rf.scope,
            }
        return {"facts": facts, "relations": rels, "step": self._step}

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return summary counts."""
        scopes: dict[str, int] = {}
        for f in self._facts.values():
            scopes[f.scope] = scopes.get(f.scope, 0) + 1
        return {
            "total_facts": len(self._facts),
            "total_relations": len(self._rels),
            "total_events": len(self._events),
            "pending_deltas": len(self._deltas),
            "step": self._step,
            "by_scope": scopes,
        }

    def __repr__(self) -> str:
        s = self.stats()
        return (f"StateStore(facts={s['total_facts']}, "
                f"rels={s['total_relations']}, "
                f"events={s['total_events']}, "
                f"step={s['step']})")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json_safe(v: Any) -> Any:
    """Convert a value to something JSON-serializable."""
    if isinstance(v, set):
        return sorted(v)
    if isinstance(v, tuple):
        return list(v)
    if isinstance(v, dict):
        return {str(k): _json_safe(val) for k, val in v.items()}
    return v
