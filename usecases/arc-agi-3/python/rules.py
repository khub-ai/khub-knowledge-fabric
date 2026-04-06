"""
rules.py (arc-agi-3 shim) — re-exports RuleEngine from core/knowledge/rules.py
and overrides DEFAULT_PATH to point at this use case's rules.json.
"""
import sys
from pathlib import Path

_KF_ROOT = Path(__file__).resolve().parents[3]
if str(_KF_ROOT) not in sys.path:
    sys.path.insert(0, str(_KF_ROOT))

import core.knowledge.rules as _rules_mod  # noqa: E402

_rules_mod.DEFAULT_PATH = Path(__file__).parent.parent / ".tmp" / "rules.json"

from core.knowledge.rules import (  # noqa: F401, E402
    RuleEngine,
    RuleMatch,
    FiringResult,
)

DEFAULT_PATH = _rules_mod.DEFAULT_PATH

