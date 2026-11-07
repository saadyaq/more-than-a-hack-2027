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
