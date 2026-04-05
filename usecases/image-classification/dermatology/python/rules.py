"""
rules.py (derm-ham10000 shim) — re-exports RuleEngine from core/knowledge/rules.py
and overrides DEFAULT_PATH to point at this use case's rules.json.
"""
import sys
from pathlib import Path

_KF_ROOT = Path(__file__).resolve().parents[4]
if str(_KF_ROOT) not in sys.path:
    sys.path.insert(0, str(_KF_ROOT))

import core.knowledge.rules as _rules_mod
_rules_mod.DEFAULT_PATH = Path(__file__).parent / "rules.json"
from core.knowledge.rules import RuleEngine, RuleMatch, FiringResult, is_visually_observable
DEFAULT_PATH = _rules_mod.DEFAULT_PATH
