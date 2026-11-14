#!/usr/bin/env bash
# Regenerate the submitted possession file from the provided data, then validate it.
#
#   MTAH_DATA_DIR=/path/to/data scripts/reproduce.sh [team_name]
#
# Needs only the competition data bundle: the learned model (possession/learned_model_v11.joblib), its
# decoder settings and the other tuned values are committed in possession/. No external data is downloaded.
set -euo pipefail

cd "$(dirname "$0")/.."
NAME="${1:-saad_yaqine}"
PYTHON="${PYTHON:-.venv/bin/python}"

if [ -z "${MTAH_DATA_DIR:-}" ] && [ ! -f data/tactical/BAR-ATH_tactical_atd.txt ]; then
    echo "Set MTAH_DATA_DIR to the unpacked data bundle (or unpack it into ./data)." >&2
    exit 1
fi

"$PYTHON" -m possession.run --method v11 --name "$NAME"
"$PYTHON" evaluation/validate_submission.py "evaluation/results/$NAME.csv"
shasum "evaluation/results/$NAME.csv" 2>/dev/null || sha1sum "evaluation/results/$NAME.csv"
