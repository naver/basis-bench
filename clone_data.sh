#!/bin/bash
# BASIS-Bench
#     Copyright (C) 2026 NAVER Corporation.
#     Apache-2.0

#   ./clone_data.sh            -> ./data/{BASIS_health_product,ELIP_Pair,SpokenSS_plus}
#   ./clone_data.sh /some/dir  -> /some/dir/...

set -euo pipefail

DATA_DIR="${1:-data}"

BASIS_HP_URL="https://github.com/naver/BASIS-data.git"
ELIP_PAIR_URL="https://github.com/naver/ELIP-Pair.git"
SPOKEN_SS_URL="https://github.com/naver/SpokenStereosetPlus.git"

mkdir -p "$DATA_DIR"

clone_or_update () {
    local url="$1" name="$2" dest="$DATA_DIR/$2"
    if [[ -d "$dest/.git" ]]; then
        echo "--- $name: already cloned, pulling ---"
        git -C "$dest" pull --ff-only
        git -C "$dest" lfs pull
    else
        echo "--- $name: cloning ---"
        git clone "$url" "$dest"
        git -C "$dest" lfs pull
    fi
}

clone_or_update "$BASIS_HP_URL"   "BASIS_health_product"
clone_or_update "$ELIP_PAIR_URL"  "ELIP_Pair"
clone_or_update "$SPOKEN_SS_URL"  "SpokenSS_plus"

echo
echo "Done. Data is in $DATA_DIR:"
du -sh "$DATA_DIR"/* 2>/dev/null || true
