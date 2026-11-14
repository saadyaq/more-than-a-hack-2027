#!/usr/bin/env bash
# Download the external training data: Metrica Sports' public sample games 1-3.
#
#   scripts/fetch_external_data.sh [destination]      # default: ~/ext_data/metrica
#
# Only needed to re-run the tuning (possession/tune.py) and the team-shape training
# (possession/train_shape.py). Source: https://github.com/metrica-sports/sample-data
# (credit: Metrica Sports).
set -euo pipefail

DEST="${1:-${EXT_DATA_DIR:-$HOME/ext_data/metrica}}"
BASE="https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"
FILES=(
    Sample_Game_1/Sample_Game_1_RawEventsData.csv
    Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv
    Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv
    Sample_Game_2/Sample_Game_2_RawEventsData.csv
    Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv
    Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv
    Sample_Game_3/Sample_Game_3_events.json
    Sample_Game_3/Sample_Game_3_metadata.xml
    Sample_Game_3/Sample_Game_3_tracking.txt
)

for file in "${FILES[@]}"; do
    mkdir -p "$DEST/$(dirname "$file")"
    curl -sfL "$BASE/$file" -o "$DEST/$file"
    echo "fetched $file"
done

# DFL open data, Bassek, Rein, Weber & Memmert, Scientific Data 2025, doi
# 10.1038/s41597-025-04505-y, CC BY 4.0.
# Discover the 21 original XML names from the mirror rather than assume competition IDs.
DFL_DEST="${DFL_DATA_DIR:-$HOME/ext_data/dfl}"
DFL_BASE="https://huggingface.co/datasets/pysport/idsse-data/resolve/main"
DFL_MANIFEST="$(mktemp)"
trap 'rm -f "$DFL_MANIFEST"' EXIT
curl -sfL 'https://huggingface.co/api/datasets/pysport/idsse-data/tree/main' -o "$DFL_MANIFEST"
DFL_FILES="$(.venv/bin/python - "$DFL_MANIFEST" <<'PY'
import json
import sys

from possession.dfl import DFL_GAMES

files = sorted(item['path'] for item in json.load(open(sys.argv[1]))
               if item['path'].endswith('.xml')
               and any(item['path'].endswith('DFL-MAT-' + game + '.xml') for game in DFL_GAMES))
if len(files) != 21:
    raise SystemExit('Expected 21 DFL XML files in the mirror')
print('\n'.join(files))
PY
)"
mkdir -p "$DFL_DEST"
while IFS= read -r file; do
    if [[ -f "$DFL_DEST/$file" ]]; then
        continue
    fi
    curl -sfL "$DFL_BASE/$file" -o "$DFL_DEST/$file.part"
    mv "$DFL_DEST/$file.part" "$DFL_DEST/$file"
    echo "fetched $file"
done <<< "$DFL_FILES"
