"""
tools.py (bird-uc200 shim) — re-exports ToolRegistry from core/knowledge/tools.py
and overrides DEFAULT_PATH to point at this use case's tools.json.

For UC200, the registry stores tool_type="schema" entries — per-pair feature
observation forms retrieved by OBSERVER at inference time.
"""
import sys
from pathlib import Path

# Ensure KF repo root is on sys.path for core/ imports
_KF_ROOT = Path(__file__).resolve().parents[3]
if str(_KF_ROOT) not in sys.path:
    sys.path.insert(0, str(_KF_ROOT))

import core.knowledge.tools as _tools_mod  # noqa: E402

# Point the default path at this use case's data directory, not core/knowledge/
_tools_mod.DEFAULT_PATH = Path(__file__).parent / "tools.json"

from core.knowledge.tools import (  # noqa: F401, E402
    ToolRegistry,
)

DEFAULT_PATH = _tools_mod.DEFAULT_PATH
