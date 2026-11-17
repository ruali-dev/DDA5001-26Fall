#!/usr/bin/env bash
set -euo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
if [[ -x /opt/dda5001/venv/bin/python ]]; then
    BASE_PYTHON=/opt/dda5001/venv/bin/python
elif command -v python3 >/dev/null 2>&1; then
    BASE_PYTHON="$(command -v python3)"
else
    echo "Python 3 was not found" >&2
    exit 127
fi

CACHE="$SCRIPT_DIR/.cache/hf-backup"
VENV="$CACHE/venv"
mkdir -p "$CACHE"
chmod 700 "$CACHE"
if [[ ! -x "$VENV/bin/python" ]]; then
    if ! "$BASE_PYTHON" -m venv --without-pip "$VENV"; then
        echo "Could not create isolated backup venv; check that Python venv support is available" >&2
        exit 1
    fi
fi
PYTHON="$VENV/bin/python"
if ! "$PYTHON" -c 'import huggingface_hub' >/dev/null 2>&1; then
    if command -v uv >/dev/null 2>&1; then
        uv pip install --python "$PYTHON" 'huggingface_hub==1.33.0'
    elif "$PYTHON" -m pip --version >/dev/null 2>&1; then
        "$PYTHON" -m pip install 'huggingface_hub==1.33.0'
    elif "$BASE_PYTHON" -m pip --version >/dev/null 2>&1; then
        "$BASE_PYTHON" -m pip --python "$PYTHON" install 'huggingface_hub==1.33.0'
    elif "$PYTHON" -m ensurepip --upgrade; then
        "$PYTHON" -m pip install 'huggingface_hub==1.33.0'
    else
        echo "pip is unavailable; cannot install huggingface_hub into the isolated venv" >&2
        exit 1
    fi
fi
export PC40_BACKUP_PYTHON="$BASE_PYTHON"
exec "$PYTHON" backup_hf.py "$@"
