"""
rules.py — Production rule system for the ARC-AGI ensemble.

Each rule has a natural-language condition (identifies puzzle type) and action
(guidance for solving). Rules accumulate across puzzle runs, forming a growing
knowledge base that improves ensemble performance over time.

Rule lifecycle:
  1. New puzzle arrives → match conditions against demo pairs (one LLM call)
  2. Top-ranked matching rules fire → their actions are injected into agent prompts
  3. After solving:
     - Success → fired rules get stats.succeeded++
     - Failure → fired rules get stats.failed++
     - MEDIATOR may generalize/specialize rules or create new ones

Rule lineage tracks how each rule was created (from scratch, generalized, or
specialized from a parent), forming a derivation tree.
"""

from __future__ import annotations
import json
import os
import re
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Observability filter
# ---------------------------------------------------------------------------

# Keywords that indicate a rule references non-visual cues — evidence that
# cannot be observed in a single static image or grid frame.  Used by
# add_rule() when observability_filter=True to reject non-visual rules
# before they enter the knowledge base.
_NON_VISUAL_PATTERNS = re.compile(
    r"\b(call|calls|song|songs|voice|voices|vocaliz\w*|sound|sounds"
    r"|habitat|habitats|range|ranges|season|seasons|migration|migratory"
    r"|size\s+(?:relative|compared|without|unless)|smell|touch"
    r"|behavior|behaviour|flock|flocking|diet|feeding|nest|nesting)\b",
    re.IGNORECASE,
)


def is_visually_observable(text: str) -> bool:
    """Return True if the rule text contains no non-visual cues.

    Checks the rule's condition and action text against a list of keywords
    that indicate evidence not available in a single static image or frame
    (vocalizations, habitat, range, size-relative-to-another-bird, etc.).

    Use this via add_rule(observability_filter=True) to reject non-visual
    rules at extraction time rather than letting them degrade classification.
    """
    return not bool(_NON_VISUAL_PATTERNS.search(text))


# ---------------------------------------------------------------------------
# File locking (cross-platform, no external deps)
# ---------------------------------------------------------------------------

def _acquire_lock(lock_path: Path, timeout: float = 15.0) -> bool:
    """Spin-acquire an exclusive lock file. Returns True on success."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return True
        except FileExistsError:
            if time.monotonic() > deadline:
                return False
            time.sleep(0.05)

def _release_lock(lock_path: Path) -> None:
    try:
        os.unlink(str(lock_path))
    except OSError:
        pass

DEFAULT_PATH = Path(__file__).parent / "rules.json"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RuleMatch:
    """Result of evaluating a single rule's condition against a puzzle."""
    rule_id: str
    confidence: str          # "high" | "medium" | "low"
    score: float             # 0.0–1.0 combined (match_conf × success_rate)
    rule: dict               # the full rule dict


@dataclass
class FiringResult:
    """Outcome of firing rules on a task."""
    task_id: str
    matched: list[RuleMatch]
    injected_ids: list[str]   # IDs of rules whose actions were injected


# ---------------------------------------------------------------------------
# Rule engine
# ---------------------------------------------------------------------------

class RuleEngine:
    def __init__(self, path: str | Path | None = None,
                 dataset_tag: str = "arc-agi-legacy"):
        self.path = Path(path or os.environ.get("RULES_FILE", DEFAULT_PATH))
        self.dataset_tag = dataset_tag   # namespace tag for this run
        self._data: dict[str, Any] = self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> dict[str, Any]:
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"version": 2, "rules": []}

    def save(self) -> None:
        """Write rules to disk with file locking and merge-on-save.

        Concurrent processes each hold their own in-memory state. On save:
        1. Acquire exclusive lock
        2. Re-read current disk state (may have rules added by other processes)
        3. Merge: add any rules we have that disk doesn't, update stats for shared rules
        4. Atomic write (temp file + os.replace)
        5. Update our in-memory state to the merged result
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(".lock")
        _acquire_lock(lock_path)
        try:
            # Re-read disk to pick up concurrent writes
            disk: dict[str, Any] = self._load()
            disk_index: dict[str, int] = {r["id"]: i for i, r in enumerate(disk["rules"])}

            for rule in self.rules:
                if rule["id"] not in disk_index:
                    # New rule we created — append it
                    disk["rules"].append(rule)
                else:
                    # Rule exists on disk — our in-memory copy has the freshest stats
                    disk["rules"][disk_index[rule["id"]]] = rule

            # Atomic write: write to .tmp then rename
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(disk, f, indent=2)
            os.replace(tmp, self.path)

            # Sync our in-memory state to what we wrote
            self._data = disk
        finally:
            _release_lock(lock_path)

    def reload(self) -> None:
        self._data = self._load()

    # ------------------------------------------------------------------
    # Namespace helpers
    # ------------------------------------------------------------------

    def _ns_stats(self, rule: dict) -> dict:
        """Return the stats dict for the current namespace.

        Prefers stats_by_ns[dataset_tag] when present; falls back to legacy
        flat stats so old data continues to work during migration.
        """
        if "stats_by_ns" in rule and self.dataset_tag:
            return rule["stats_by_ns"].get(
                self.dataset_tag, {"fires": 0, "successes": 0, "failures": 0}
            )
        # Legacy flat stats
        s = rule.get("stats", {})
        return {
            "fires":     s.get("fired", 0),
            "successes": s.get("succeeded", 0),
            "failures":  s.get("failed", 0),
        }

    def _rule_in_ns(self, rule: dict) -> bool:
        """True if this rule should be active in the current namespace."""
        if rule.get("scope", "dataset") == "global":
            return True
        return self.dataset_tag in rule.get("tags", [])

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    @property
    def rules(self) -> list[dict]:
        return self._data["rules"]

    def get(self, rule_id: str) -> Optional[dict]:
        for r in self.rules:
            if r["id"] == rule_id:
                return r
        return None

    def active_rules(self) -> list[dict]:
        """Return rules that are usable in the current namespace.

        Includes 'active', 'candidate', and 'flagged' rules whose scope/tags
        match the current dataset_tag (or scope == 'global').
        Excludes 'deprecated' and 'archived' rules.
        """
        return [
            r for r in self.rules
            if r.get("status", "active") in ("active", "candidate", "flagged")
            and self._rule_in_ns(r)
        ]

    def active_task_rules(self) -> list[dict]:
        """Return active + candidate task rules (matched per-puzzle in Round 0)."""
        return [r for r in self.active_rules() if r.get("rule_type", "task") == "task"]

    def active_preference_rules(self) -> list[dict]:
        """Return active preference rules (soft priors injected for every puzzle)."""
        return [r for r in self.active_rules() if r.get("rule_type") == "preference"]

    def candidate_rules(self) -> list[dict]:
        """Return candidate rules (generalized; awaiting first independent confirmation)."""
        return [r for r in self.rules if r.get("status") == "candidate"]

    def stats_summary(self) -> dict[str, Any]:
        all_rules = self.rules
        total = len(all_rules)
        confirmed  = [r for r in all_rules if r.get("status", "active") == "active"]
        candidates = self.candidate_rules()
        flagged    = [r for r in all_rules if r.get("status") == "flagged"]
        deprecated = [r for r in all_rules if r.get("status") == "deprecated"]
        archived   = [r for r in all_rules if r.get("status") == "archived"]
        fired     = sum(self._ns_stats(r)["fires"]     for r in confirmed + candidates)
        succeeded = sum(self._ns_stats(r)["successes"] for r in confirmed + candidates)
        return {
            "total":           total,
            "active":          len(confirmed),
            "candidates":      len(candidates),
            "flagged":         len(flagged),
            "deprecated":      len(deprecated),
            "archived":        len(archived),
            "total_fired":     fired,
            "total_succeeded": succeeded,
        }

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def _next_id(self) -> str:
        if not self.rules:
            return "r_001"
        nums = []
        for r in self.rules:
            try:
                nums.append(int(r["id"].split("_")[1]))
            except (ValueError, IndexError):
                pass
        return f"r_{(max(nums) + 1 if nums else 1):03d}"

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    def add_rule(
        self,
        condition: str,
        action: str,
        source: str = "mediator",
        source_task: str = "",
        tags: list[str] | None = None,
        lineage: dict | None = None,
        rule_type: str = "task",
        status: str = "active",
        scope: str = "dataset",
        observability_filter: bool = False,
    ) -> Optional[dict]:
        """
        Create a new rule and persist it.

        Args:
            condition:   Natural language condition (identifies puzzle type)
            action:      Natural language action (guidance for solving)
            source:      Who created this rule: "mediator", "human", "system"
            source_task: Task ID that triggered creation
            tags:        Optional category tags
            lineage:     Optional derivation info:
                         {"type": "new" | "generalized" | "specialized" | "merged",
                          "parent_ids": ["r_001", ...],
                          "reason": "why this derivation was needed"}
            rule_type:   "task" (default) or "preference".
                         - "task" rules encode how to solve a category of puzzle.
                           They are matched per-puzzle in Round 0 and injected as prior
                           knowledge when the condition matches.
                         - "preference" rules encode *which hypothesis property to prefer*
                           when multiple plausible hypotheses exist. They are learned from
                           correction events (wrong hypothesis → human insight → success)
                           and injected as soft priors for every puzzle regardless of match.
            status:      "active" (default), "candidate", or "deprecated".
                         - "active": fully vetted; used in Round 0 matching.
                         - "candidate": generalized/inferred; included in matching but
                           labeled as unconfirmed. Promoted to "active" on first independent
                           success. Deprecated after 1 failure on an unrelated task.
                         - "deprecated": excluded from all matching.
            observability_filter: When True, reject rules whose condition or
                         action text contains non-visual cues (vocalizations,
                         habitat, range, season, etc.) that cannot be observed
                         in a single static image or grid frame.  Returns None
                         if the rule is rejected.  Intended for vision-based
                         use cases such as image classification (UC200) where
                         non-visual rules cause systematic misclassification.

        Returns:
            The newly created rule dict, or None if rejected by the filter.
        """
        # Observability filter: reject non-visual rules before storage
        if observability_filter:
            if not is_visually_observable(condition) or not is_visually_observable(action):
                return None

        # Auto-tag with the current dataset namespace
        tags_list = list(tags or [])
        if self.dataset_tag and self.dataset_tag not in tags_list:
            tags_list.append(self.dataset_tag)

        rule = {
            "id": self._next_id(),
            "condition": condition,
            "action": action,
            "rule_type": rule_type,
            "scope": scope,
            "stats_by_ns": {
                self.dataset_tag: {"fires": 0, "successes": 0, "failures": 0}
            } if self.dataset_tag else {},
            "source": source,
            "source_task": source_task,
            "tags": tags_list,
            "lineage": lineage or {"type": "new", "parent_ids": [], "reason": ""},
            "status": status,
            "tasks_seen": 0,
            "created": self._now_iso(),
            "last_fired": None,
        }
        self.rules.append(rule)
        self.save()
        return rule

    def generalize_rule(
        self,
        parent_id: str,
        new_condition: str,
        new_action: str,
        reason: str,
        source_task: str = "",
        tags: list[str] | None = None,
    ) -> dict:
        """Create a more general rule derived from an existing one.

        Starts as 'candidate' — promoted to 'active' on first independent success.
        """
        parent = self.get(parent_id)
        merged_tags = list(set((tags or []) + (parent.get("tags", []) if parent else [])))
        return self.add_rule(
            condition=new_condition,
            action=new_action,
            source="mediator",
            source_task=source_task,
            tags=merged_tags,
            lineage={
                "type": "generalized",
                "parent_ids": [parent_id],
                "reason": reason,
            },
            status="candidate",
        )

    def specialize_rule(
        self,
        parent_id: str,
        new_condition: str,
        new_action: str,
        reason: str,
        source_task: str = "",
        tags: list[str] | None = None,
    ) -> dict:
        """Create a more specific rule derived from an existing one."""
        parent = self.get(parent_id)
        merged_tags = list(set((tags or []) + (parent.get("tags", []) if parent else [])))
        return self.add_rule(
            condition=new_condition,
            action=new_action,
            source="mediator",
            source_task=source_task,
            tags=merged_tags,
            lineage={
                "type": "specialized",
                "parent_ids": [parent_id],
                "reason": reason,
            },
        )

    def merge_rules(
        self,
        parent_ids: list[str],
        new_condition: str,
        new_action: str,
        reason: str,
        source_task: str = "",
        tags: list[str] | None = None,
    ) -> dict:
        """Create a rule that combines insights from multiple parent rules."""
        all_tags = list(tags or [])
        for pid in parent_ids:
            p = self.get(pid)
            if p:
                all_tags.extend(p.get("tags", []))
        return self.add_rule(
            condition=new_condition,
            action=new_action,
            source="mediator",
            source_task=source_task,
            tags=list(set(all_tags)),
            lineage={
                "type": "merged",
                "parent_ids": parent_ids,
                "reason": reason,
            },
        )

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def record_success(self, rule_id: str, task_id: str) -> None:
        r = self.get(rule_id)
        if r:
            ns = r.setdefault("stats_by_ns", {}).setdefault(
                self.dataset_tag, {"fires": 0, "successes": 0, "failures": 0}
            )
            ns["fires"] += 1
            ns["successes"] += 1
            r["last_fired"] = self._now_iso()
            # Track which tasks fired this rule (capped at 200 to avoid unbounded growth)
            fired_on: list = r.setdefault("fired_on", [])
            if task_id not in fired_on:
                fired_on.append(task_id)
                if len(fired_on) > 200:
                    r["fired_on"] = fired_on[-200:]
            # Promote candidate on first independent success
            if r.get("status") == "candidate":
                r["status"] = "active"
            # Unflag on success
            elif r.get("status") == "flagged":
                r["status"] = "active"
                r.pop("flagged_reason", None)
            self.save()

    def record_failure(self, rule_id: str, task_id: str) -> None:
        r = self.get(rule_id)
        if r:
            ns = r.setdefault("stats_by_ns", {}).setdefault(
                self.dataset_tag, {"fires": 0, "successes": 0, "failures": 0}
            )
            ns["fires"] += 1
            ns["failures"] += 1
            r["last_fired"] = self._now_iso()
            # Track which tasks fired this rule
            fired_on: list = r.setdefault("fired_on", [])
            if task_id not in fired_on:
                fired_on.append(task_id)
                if len(fired_on) > 200:
                    r["fired_on"] = fired_on[-200:]
            self.save()

    def increment_tasks_seen(self, fired_ids: Optional[set] = None) -> None:
        """Increment tasks_seen for every active rule in this namespace.

        Called once per task run (after stats are recorded) so that the
        staleness pruner can tell how long each rule has gone without firing.
        Rules that fired this task get their tasks_seen incremented too — the
        counter just means 'this rule has been eligible N times.'

        Args:
            fired_ids: set of rule IDs that fired on this task (not used for
                       staleness; kept as a parameter for future per-rule
                       breakdowns).
        """
        changed = False
        for r in self.active_rules():
            r["tasks_seen"] = r.get("tasks_seen", 0) + 1
            changed = True
        if changed:
            self.save()

    def deprecate_rule(self, rule_id: str, reason: str = "") -> None:
        r = self.get(rule_id)
        if r:
            r["status"] = "deprecated"
            r["deprecated_reason"] = reason
            self.save()

    def archive_rule(self, rule_id: str, reason: str = "") -> None:
        r = self.get(rule_id)
        if r:
            r["status"] = "archived"
            r["archived_reason"] = reason
            self.save()

    def flag_rule(self, rule_id: str, reason: str = "") -> None:
        """Mark a rule as flagged (poor performance, under review)."""
        r = self.get(rule_id)
        if r and r.get("status") == "active":
            r["status"] = "flagged"
            r["flagged_reason"] = reason
            self.save()

    def unflag_rule(self, rule_id: str) -> None:
        """Manually restore a flagged rule to active."""
        r = self.get(rule_id)
        if r and r.get("status") == "flagged":
            r["status"] = "active"
            r.pop("flagged_reason", None)
            self.save()

    def reactivate_rule(self, rule_id: str) -> None:
        r = self.get(rule_id)
        if r:
            r["status"] = "active"
            r.pop("deprecated_reason", None)
            r.pop("flagged_reason", None)
            self.save()

    def promote_candidate(self, rule_id: str) -> bool:
        """Promote a candidate rule to active after its first independent success.

        Returns True if promoted, False if rule not found or not a candidate.
        """
        r = self.get(rule_id)
        if r and r.get("status") == "candidate":
            r["status"] = "active"
            self.save()
            return True
        return False

    def auto_deprecate(self, min_fired: int = 10,
                       stale_flag: int = 50, stale_deprecate: int = 100,
                       min_candidate_fired: int = 1) -> list[str]:
        """Flag or deprecate rules based on namespace performance and staleness.

        Performance pruning (per-namespace stats):
          - Candidates: deprecated after min_candidate_fired failures with 0 successes.
            Default is 1 (original behavior). Set higher (e.g. 5) for exploratory
            domains where rules need multiple episodes to be confirmed or refuted.
          - Active: deprecated if fires >= min_fired and 0 successes.
          - Active: flagged  if fires >= 5 and success_rate < 20%.

        Staleness pruning (tasks_seen counter):
          - tasks_seen >= stale_deprecate and 0 fires → deprecate.
          - tasks_seen >= stale_flag      and 0 fires → flag.

        Returns list of rule IDs that were deprecated or newly flagged.
        """
        changed: list[str] = []
        for r in self.active_rules():   # namespace-filtered already
            ns        = self._ns_stats(r)
            fires     = ns["fires"]
            successes = ns["successes"]
            seen      = r.get("tasks_seen", 0)
            is_candidate = r.get("status") == "candidate"

            if is_candidate:
                if fires >= min_candidate_fired and successes == 0:
                    self.deprecate_rule(
                        r["id"],
                        reason=f"auto: candidate fired {fires}x, 0 successes",
                    )
                    changed.append(r["id"])
                    continue

            # Staleness checks (skip candidates — they're new by definition)
            if not is_candidate and fires == 0:
                if seen >= stale_deprecate:
                    self.deprecate_rule(
                        r["id"],
                        reason=f"auto: stale — 0 fires after {seen} tasks seen",
                    )
                    changed.append(r["id"])
                    continue
                elif seen >= stale_flag and r.get("status") == "active":
                    self.flag_rule(
                        r["id"],
                        reason=f"auto: stale — 0 fires after {seen} tasks seen",
                    )
                    changed.append(r["id"])
                    continue

            # Performance checks
            if not is_candidate:
                sr = successes / fires if fires > 0 else 0.5
                if fires >= min_fired and successes == 0:
                    self.deprecate_rule(
                        r["id"],
                        reason=f"auto: fired {fires}x, 0 successes",
                    )
                    changed.append(r["id"])
                elif fires >= 5 and sr < 0.20 and r.get("status") == "active":
                    self.flag_rule(
                        r["id"],
                        reason=f"auto: fired {fires}x, success rate {sr:.0%}",
                    )
                    changed.append(r["id"])
        return changed

    def edit_rule(self, rule_id: str, condition: str = "", action: str = "") -> None:
        """Human-driven edit of a rule's condition or action."""
        r = self.get(rule_id)
        if r:
            if condition:
                r["condition"] = condition
            if action:
                r["action"] = action
            self.save()

    # ------------------------------------------------------------------
    # Redundancy detection (Jaccard overlap on fired_on lists)
    # ------------------------------------------------------------------

    def find_redundant_pairs(self, threshold: float = 0.5) -> list[dict]:
        """Find pairs of active task rules that fire on nearly the same tasks.

        Uses Jaccard similarity on the fired_on task-ID lists.  A pair with
        J >= threshold is a merge/deprecation candidate.

        Returns a list of dicts:
            {"rule_a": id, "rule_b": id, "jaccard": float,
             "shared": N, "union": N, "suggestion": str}
        """
        active = [r for r in self.active_task_rules()
                  if len(r.get("fired_on", [])) >= 3]   # need enough data
        pairs = []
        for i, a in enumerate(active):
            set_a = set(a.get("fired_on", []))
            for b in active[i + 1:]:
                set_b = set(b.get("fired_on", []))
                union = set_a | set_b
                if not union:
                    continue
                shared = set_a & set_b
                j = len(shared) / len(union)
                if j >= threshold:
                    sr_a = self._success_rate(a)
                    sr_b = self._success_rate(b)
                    suggestion = (
                        f"merge into one rule"
                        if abs(sr_a - sr_b) < 0.15
                        else f"deprecate {a['id'] if sr_a < sr_b else b['id']} "
                             f"(lower success rate)"
                    )
                    pairs.append({
                        "rule_a":   a["id"],
                        "rule_b":   b["id"],
                        "jaccard":  round(j, 3),
                        "shared":   len(shared),
                        "union":    len(union),
                        "suggestion": suggestion,
                    })
        return sorted(pairs, key=lambda p: p["jaccard"], reverse=True)

    # ------------------------------------------------------------------
    # Two-stage matching helpers
    # ------------------------------------------------------------------

    _CATEGORY_RE = re.compile(r"^\[([^\]]+)\]", re.IGNORECASE)

    def _rule_categories(self) -> dict[str, list[str]]:
        """Return {category: [rule_id, ...]} from active task rules."""
        mapping: dict[str, list[str]] = {}
        for r in self.active_task_rules():
            m = self._CATEGORY_RE.match(r.get("condition", ""))
            cat = m.group(1).lower().strip() if m else "other"
            mapping.setdefault(cat, []).append(r["id"])
        return mapping

    def build_category_filter_prompt(self, task_text: str) -> str:
        """Stage-1 prompt: ask the LLM which categories are relevant.

        Returns a compact prompt (much cheaper than the full rule list).
        """
        cat_map = self._rule_categories()
        categories = sorted(cat_map.keys())
        if not categories:
            return ""
        cat_line = ", ".join(f'"{c}" ({len(cat_map[c])} rules)' for c in categories)
        return (
            "You are a rule pre-filter. Given the ARC-AGI puzzle below, "
            "select which transformation categories are most likely relevant.\n\n"
            f"Available categories: {cat_line}\n\n"
            f"{task_text}\n\n"
            "Reply with a JSON block listing the relevant categories (≤5):\n"
            "```json\n"
            '{"categories": ["gravity", "path-drawing"]}\n'
            "```\n"
            "If none clearly match, return an empty list."
        )

    def filter_rules_by_categories(self, category_response: str,
                                    max_rules: int = 25) -> list[dict]:
        """Parse the stage-1 LLM response and return the filtered rule subset.

        Falls back to top-N rules by success rate if parsing fails or the
        filtered set is empty.
        """
        cat_map = self._rule_categories()
        block_re = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
        chosen: set[str] = set()
        for raw in block_re.findall(category_response):
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict) and "categories" in obj:
                    for c in obj["categories"]:
                        chosen.update(cat_map.get(c.lower().strip(), []))
                    break
            except (json.JSONDecodeError, Exception):
                continue

        if chosen:
            rules = [r for r in self.active_task_rules() if r["id"] in chosen]
        else:
            # Fallback: top-N by success rate (prefer rules with a track record)
            rules = sorted(self.active_task_rules(),
                           key=lambda r: (self._success_rate(r), self._ns_stats(r)["fires"]),
                           reverse=True)

        return rules[:max_rules]

    # ------------------------------------------------------------------
    # Match (LLM-based)
    # ------------------------------------------------------------------

    def format_rules_for_matching(self) -> str:
        """Build a prompt fragment listing active task rules for the LLM to evaluate."""
        active = self.active_task_rules()
        if not active:
            return "(no rules in the rule base)"
        return self._format_rules_list(active)

    def format_preference_rules_for_solver(self) -> str:
        """
        Build a prompt section listing preference rules as soft priors for solvers.

        Preference rules are learned from correction events: when a solver's initial
        hypothesis was wrong, a human provided an insight, and the corrected approach
        succeeded. They encode *which hypothesis property to prefer* when multiple
        plausible interpretations exist — not how to solve a specific puzzle type.

        Returned section is injected into every solver prompt regardless of puzzle match.
        Solvers are explicitly told these are suggestions, not mandates.
        """
        prefs = self.active_preference_rules()
        if not prefs:
            return ""
        lines = [
            "## Reasoning Preferences (soft priors — treat as suggestions, not rules)",
            "The system has learned these preferences from past correction events.",
            "When multiple hypotheses seem equally valid based on demos alone, prefer the one",
            "that aligns with these priors. Demo evidence always overrides a prior.\n",
        ]
        for r in prefs:
            lines.append(f"- **[{r['id']}]** {r['action']}")
            if r.get("source_task"):
                lines.append(f"  *(learned from task {r['source_task']})*")
        return "\n".join(lines)

    def format_fired_rules_for_prompt(self, matches: list[RuleMatch],
                                       max_rules: int = 5) -> str:
        """
        Build a prompt section with the top-N fired rules' actions,
        suitable for injection into solver/MEDIATOR prompts.
        """
        top = sorted(matches, key=lambda m: m.score, reverse=True)[:max_rules]
        if not top:
            return ""
        lines = ["## Applicable Rules (from prior experience)\n"]
        for m in top:
            sr = self._success_rate(m.rule)
            fires = self._ns_stats(m.rule)["fires"]
            lines.append(
                f"- **{m.rule['id']}** (confidence: {m.confidence}, "
                f"success rate: {sr:.0%}, fired {fires}x)\n"
                f"  {m.rule['action']}"
            )
        return "\n".join(lines)

    def _success_rate(self, rule: dict) -> float:
        ns = self._ns_stats(rule)
        fires = ns["fires"]
        if fires == 0:
            return 0.5  # neutral prior for untested rules
        return ns["successes"] / fires

    def rank_matches(self, matches: list[RuleMatch]) -> list[RuleMatch]:
        """Sort matches by combined score (match confidence × success rate)."""
        return sorted(matches, key=lambda m: m.score, reverse=True)

    # ------------------------------------------------------------------
    # Parse LLM rule-matching response
    # ------------------------------------------------------------------

    def parse_match_response(self, llm_text: str) -> list[RuleMatch]:
        """
        Parse the LLM's rule-matching response.

        Expected format in a JSON code block:
        ```json
        {
          "matches": [
            {"rule_id": "r_001", "confidence": "high"},
            {"rule_id": "r_003", "confidence": "medium"}
          ]
        }
        ```
        """
        block_re = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
        for raw in block_re.findall(llm_text):
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict) and "matches" in obj:
                    results = []
                    for m in obj["matches"]:
                        rid = m.get("rule_id", "")
                        conf = m.get("confidence", "medium")
                        rule = self.get(rid)
                        if rule:
                            conf_score = {"high": 0.9, "medium": 0.6, "low": 0.3}.get(conf, 0.5)
                            sr = self._success_rate(rule)
                            results.append(RuleMatch(
                                rule_id=rid,
                                confidence=conf,
                                score=conf_score * sr,
                                rule=rule,
                            ))
                    return self.rank_matches(results)
            except (json.JSONDecodeError, Exception):
                continue
        return []

    # ------------------------------------------------------------------
    # Parse MEDIATOR rule-creation/evolution response
    # ------------------------------------------------------------------

    def parse_mediator_rule_updates(self, mediator_text: str,
                                     task_id: str) -> list[dict]:
        """
        Parse MEDIATOR output for rule creation/evolution instructions.

        Expected JSON block:
        ```json
        {
          "rule_updates": [
            {
              "action": "new",
              "condition": "...",
              "rule_action": "...",
              "tags": ["gravity", "spatial"]
            },
            {
              "action": "generalize",
              "parent_id": "r_001",
              "condition": "...",
              "rule_action": "...",
              "reason": "original was too specific to downward gravity"
            },
            {
              "action": "specialize",
              "parent_id": "r_002",
              "condition": "...",
              "rule_action": "...",
              "reason": "fails on grids with border cells"
            }
          ]
        }
        ```

        Returns list of created/modified rule dicts.
        """
        block_re = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
        created: list[dict] = []
        for raw in block_re.findall(mediator_text):
            try:
                obj = json.loads(raw)
                if not isinstance(obj, dict) or "rule_updates" not in obj:
                    continue
                for upd in obj["rule_updates"]:
                    act = upd.get("action", "new")
                    cond = upd.get("condition", "")
                    ract = upd.get("rule_action", "")
                    tags = upd.get("tags", [])
                    reason = upd.get("reason", "")
                    parent = upd.get("parent_id", "")

                    if not cond or not ract:
                        continue

                    rule_type = upd.get("rule_type", "task")

                    if act == "generalize" and parent:
                        r = self.generalize_rule(parent, cond, ract, reason,
                                                  source_task=task_id, tags=tags)
                    elif act == "specialize" and parent:
                        r = self.specialize_rule(parent, cond, ract, reason,
                                                  source_task=task_id, tags=tags)
                    elif act == "merge":
                        pids = upd.get("parent_ids", [parent] if parent else [])
                        r = self.merge_rules(pids, cond, ract, reason,
                                              source_task=task_id, tags=tags)
                    else:
                        r = self.add_rule(cond, ract, source="mediator",
                                           source_task=task_id, tags=tags,
                                           rule_type=rule_type)
                    if r is not None:
                        created.append(r)
            except (json.JSONDecodeError, Exception):
                continue
        return created

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def build_match_prompt(self, task_text: str,
                            rules_subset: list[dict] | None = None) -> str:
        """Build the user message for the rule-matching LLM call.

        Args:
            task_text:     Formatted puzzle description.
            rules_subset:  If given, match only against these rules instead of
                           the full active set (used in two-stage retrieval).
        """
        if rules_subset is not None:
            rules_listing = self._format_rules_list(rules_subset)
        else:
            rules_listing = self.format_rules_for_matching()
        return (
            "You are a rule matcher. Given the ARC-AGI puzzle below and a list of "
            "rules, determine which rules' conditions match this puzzle.\n\n"
            "For each matching rule, rate confidence as high/medium/low.\n"
            "Only include rules whose conditions genuinely apply — do not force matches.\n\n"
            f"## Available Rules\n\n{rules_listing}\n\n"
            f"{task_text}\n\n"
            "Respond with a JSON block:\n"
            "```json\n"
            '{"matches": [{"rule_id": "r_001", "confidence": "high"}, ...]}\n'
            "```\n"
            "If no rules match, return an empty matches array."
        )

    def _format_rules_list(self, rules: list[dict]) -> str:
        """Format an arbitrary list of rules for a matching prompt."""
        if not rules:
            return "(no rules)"
        lines = []
        for r in rules:
            sr    = self._success_rate(r)
            fires = self._ns_stats(r)["fires"]
            status = r.get("status", "active")
            if status == "candidate":
                label = " [CANDIDATE - unconfirmed]"
            elif status == "flagged":
                label = " [FLAGGED - low success rate]"
            else:
                label = ""
            lines.append(
                f"- [{r['id']}]{label} (success {sr:.0%}, fired {fires}x)\n"
                f"  CONDITION: {r['condition']}\n"
                f"  ACTION: {r['action']}"
            )
        return "\n".join(lines)

    def build_mediator_rule_section(self, fired: list[RuleMatch],
                                     success: bool) -> str:
        """
        Build a prompt section instructing MEDIATOR to update rules.
        Shows all existing rules so MEDIATOR can merge/generalize instead of duplicating.
        """
        parts = ["\n## Rule System\n"]

        # Show ALL active rules so MEDIATOR can detect duplicates before creating new ones
        all_active = self.active_rules()
        if all_active:
            parts.append("### Existing rules (check before creating new ones)\n")
            for r in all_active:
                sr = self._success_rate(r)
                fires = self._ns_stats(r)["fires"]
                rtype = r.get("rule_type", "task")
                parts.append(
                    f"- [{r['id']}] type={rtype} tags={r.get('tags',[])} success={sr:.0%} fired={fires}x\n"
                    f"  CONDITION: {r['condition']}\n"
                    f"  ACTION: {r['action'][:120]}"
                )

        if fired:
            parts.append("\n### Rules matched for this task\n")
            for m in fired:
                parts.append(f"- {m.rule_id}: {m.rule['condition'][:80]}")
            outcome = "SUCCEEDED" if success else "FAILED"
            parts.append(f"\nThe ensemble {outcome} on this task.")

        if not success:
            parts.append(
                "\nSince the task failed, consider:\n"
                "- Were fired rules too broad? → **specialize** them\n"
                "- Were fired rules misleading? → create a **new** rule with better guidance\n"
                "- Was a correct rule missing? → create a **new** rule for this puzzle type"
            )
        else:
            parts.append(
                "\nSince the task succeeded, consider:\n"
                "- Can the successful approach be captured as a rule?\n"
                "- Can any existing rule be **generalized** to cover this puzzle type too?"
            )

        parts.append(
            "\n### Rule writing guidelines\n"
            "**DEDUPLICATION**: Before writing `action: new`, scan the existing rules above. "
            "If an existing rule covers the same transformation category, prefer `generalize` or `merge` over `new`. "
            "Do NOT create a new rule if one already exists with a very similar condition.\n\n"
            "**CONDITION FORMAT**: Start every condition with a category tag in brackets, e.g.:\n"
            "  `[gravity] Grid contains objects that need to be sorted by type...`\n"
            "  `[path-drawing] Grid has exactly 3 colored points on a black background...`\n"
            "  `[proximity] Grid has two boundary lines and scattered trigger cells...`\n"
            "Categories: gravity, path-drawing, proximity, flood-fill, object-manipulation, "
            "sorting, completion, extraction, scaling, rule-induction\n\n"
            "**GENERALITY**: Conditions should describe a *category* of puzzles, not a specific puzzle. "
            "Avoid encoding specific colors, specific grid sizes, or specific coordinates.\n\n"
            "To update rules, include a JSON block:\n"
            "```json\n"
            '{"rule_updates": [\n'
            '  {"action": "new", "condition": "[category] ...", "rule_action": "...", "tags": [...]},\n'
            '  {"action": "generalize", "parent_id": "r_001", "condition": "[category] ...", "rule_action": "...", "reason": "..."},\n'
            '  {"action": "merge", "parent_ids": ["r_003", "r_005"], "condition": "[category] ...", "rule_action": "...", "reason": "..."}\n'
            "]}\n"
            "```\n"
            "Omit the rule_updates block if no changes are needed."
        )
        return "\n".join(parts)
