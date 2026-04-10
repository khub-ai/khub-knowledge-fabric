"""
protocols.py — Domain configuration and type definitions for dialogic distillation.

DomainConfig captures all domain-specific vocabulary so that the core library
can generate domain-appropriate prompts without importing any domain code.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, TypedDict


@dataclass
class DomainConfig:
    """Domain-specific vocabulary and configuration for dialogic distillation.

    Each domain (dermatology, ornithology, etc.) creates one instance of this
    and passes it to every core library function.
    """
    # Role description for the expert TUTOR
    expert_role: str  # e.g. "senior dermoscopy expert and knowledge engineer"

    # Nouns for items being classified
    item_noun: str           # e.g. "dermoscopic image", "bird photograph"
    item_noun_plural: str    # e.g. "dermoscopic images", "bird photographs"

    # Classification vocabulary
    classification_noun: str  # e.g. "diagnosis", "species identification"
    class_noun: str           # e.g. "class", "species"

    # Feature vocabulary
    feature_noun: str         # e.g. "dermoscopic feature", "field mark or plumage feature"

    # Domain-specific observation guidance (what to look for)
    observation_guidance: str  # e.g. "pigment network, globules, dots, vessels..."

    # What NOT to include in rules (non-visual exclusions)
    non_visual_exclusions: str  # e.g. "patient history, symptoms, age, body location"

    # Optional: vocabulary examples for KF meta-learning (GOOD/BAD examples)
    good_vocabulary_examples: list[str] = field(default_factory=list)
    bad_vocabulary_examples: list[str] = field(default_factory=list)

    # Optional: per-domain pool gate overrides.
    # If None, the global defaults from constants.py are used.
    # Safety-critical domains (e.g. road surface) may set stricter thresholds.
    precision_gate: float | None = None   # e.g. 0.90 for safety-critical domains
    max_fp: int | None = None             # e.g. 0 to require zero false positives


class PairInfo(TypedDict):
    """Standardized confusable pair information."""
    class_a: str
    class_b: str
    pair_id: str


class FailureRecord(TypedDict, total=False):
    """A single failure case to be processed by the patching loop."""
    task_id: str
    pair_id: str
    image_path: str
    correct_label: str
    predicted_label: str
    reasoning: str
