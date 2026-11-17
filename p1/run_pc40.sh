#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -x /opt/dda5001/venv/bin/python ]]; then
    PYTHON=/opt/dda5001/venv/bin/python
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
else
    echo "python3 was not found" >&2
    exit 127
fi

exec "$PYTHON" run_pc40.py "$@"
