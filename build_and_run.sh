#!/usr/bin/env bash
# Train the project models, then serve the frontend and prediction API.
# Usage: ./build_and_run.sh [--host 127.0.0.1] [--port 8000]
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$PROJECT_DIR"

if [[ -n "${PROJECT_PYTHON:-}" ]]; then
    PYTHON_BIN="$PROJECT_PYTHON"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
elif [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
else
    PYTHON_BIN="python3"
fi

# Let server.py handle help and argument validation before training starts.
"$PYTHON_BIN" - "$@" <<'PY'
import argparse

parser = argparse.ArgumentParser(description='Train models and start the dashboard server.')
parser.add_argument('--host', default='127.0.0.1')
parser.add_argument('--port', type=int, default=8000)
args = parser.parse_args()
if not 1 <= args.port <= 65535:
    parser.error('--port must be between 1 and 65535')
PY
for argument in "$@"; do
    if [[ "$argument" == "--help" || "$argument" == "-h" ]]; then
        exit 0
    fi
done

echo "Checking Python dependencies using $PYTHON_BIN ..."
if ! "$PYTHON_BIN" -c 'import pandas, numpy, sklearn, joblib, xgboost'; then
    echo "Install dependencies in the selected Python environment:" >&2
    echo "  $PYTHON_BIN -m pip install pandas numpy scikit-learn joblib xgboost" >&2
    exit 1
fi

echo "Building models using parameters.json ..."
"$PYTHON_BIN" src/data_ingestion_pipeline_xgboost.py

echo "Starting the dashboard and API. Press Ctrl+C to stop."
exec "$PYTHON_BIN" src/server.py "$@"
