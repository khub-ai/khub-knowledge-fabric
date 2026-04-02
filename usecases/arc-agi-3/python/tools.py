"""
tools.py (arc-agi-3 shim) — re-exports ToolRegistry from core/knowledge/tools.py
and overrides DEFAULT_PATH to point at this use case's tools.json.
"""
import sys
from pathlib import Path

_KF_ROOT = Path(__file__).resolve().parents[3]
if str(_KF_ROOT) not in sys.path:
    sys.path.insert(0, str(_KF_ROOT))

import core.knowledge.tools as _tools_mod  # noqa: E402

_tools_mod.DEFAULT_PATH = Path(__file__).parent / "tools.json"

from core.knowledge.tools import (  # noqa: F401, E402
    ToolRegistry,
)

DEFAULT_PATH = _tools_mod.DEFAULT_PATH
