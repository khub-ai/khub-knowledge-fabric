#!/usr/bin/env bash
# Shortcut to run the Python ARC-AGI ensemble harness from the repo root.
# Usage:
#   bash usecases/arc-agi-ensemble/run-python.sh --human --task-id 1e0a9b12
#   npm run arc:py -- --human --task-id 1e0a9b12

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="C:/Users/kaihu/AppData/Local/pypoetry/Cache/virtualenvs/01os-TLh4bqwo-py3.10/Scripts/python.exe"
KEY_FILE="P:/_access/Security/api_keys.env"

# Load API key if not already set
if [ -z "$ANTHROPIC_API_KEY" ] && [ -f "$KEY_FILE" ]; then
    export ANTHROPIC_API_KEY=$(sed -n 's/^ANTHROPIC_API_KEY=//p' "$KEY_FILE")
fi

export PYTHONUTF8=1

# Route --stats to the stats reporter instead of the harness
if [[ "$*" == *"--stats"* ]]; then
    # Strip --stats from args and pass remainder to stats.py
    STATS_ARGS=()
    for arg in "$@"; do
        [[ "$arg" == "--stats" ]] && continue
        STATS_ARGS+=("$arg")
    done
    exec "$PYTHON" "$SCRIPT_DIR/python/stats.py" "${STATS_ARGS[@]}"
fi

exec "$PYTHON" "$SCRIPT_DIR/python/harness.py" "$@"
