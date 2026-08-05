#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$ROOT/scripts/research"
RECEIPTS="$SCRIPT_DIR/receipts/cs6_v7b_target23_arb_tm2r_hset_covering_v1"
DEPS="${CS6_PYTHONPATH:-/tmp/sounio-cs6-arb-full-leaf-deps}"
SUPPORT_WORKER="$SCRIPT_DIR/cs6_v7b_target23_arb_tm2r_hset_covering_carrier_worker.py"
FACE_WORKER="$SCRIPT_DIR/cs6_v7b_target23_arb_tm2r_hset_covering_face_worker.py"
ANALYZER="$SCRIPT_DIR/cs6_v7b_target23_arb_tm2r_hset_covering_analyze.py"

support_files=(
  support_XLEL.json support_XLEH.json support_XHEL.json support_XHEH.json
)
face_specs=(
  "LEFT XLEL L face_LEFT_XLEL_L.json"
  "LEFT XLEL H face_LEFT_XLEL_H.json"
  "LEFT XLEH ROOT face_LEFT_XLEH_ROOT.json"
  "RIGHT XHEL ROOT face_RIGHT_XHEL_ROOT.json"
  "RIGHT XHEH L face_RIGHT_XHEH_L.json"
  "RIGHT XHEH H face_RIGHT_XHEH_H.json"
)

if [[ "${CS6_REGENERATE:-0}" == "1" ]]; then
  for index in 0 1 2 3; do
    tile="${support_files[$index]#support_}"
    tile="${tile%.json}"
    CS6_SOURCE_TILE="$tile" PYTHONPATH="$DEPS" python3 "$SUPPORT_WORKER" \
      > "$RECEIPTS/${support_files[$index]}"
  done
  for spec in "${face_specs[@]}"; do
    read -r face tile refinement filename <<< "$spec"
    CS6_SOURCE_FACE="$face" CS6_SOURCE_TILE="$tile" \
      CS6_FACE_ETA_REFINEMENT="$refinement" PYTHONPATH="$DEPS" \
      python3 "$FACE_WORKER" > "$RECEIPTS/$filename"
  done
fi

inputs=()
for filename in "${support_files[@]}"; do
  inputs+=("$RECEIPTS/$filename")
done
for spec in "${face_specs[@]}"; do
  read -r _face _tile _refinement filename <<< "$spec"
  inputs+=("$RECEIPTS/$filename")
done

fresh="$(mktemp)"
trap 'rm -f "$fresh"' EXIT
PYTHONPATH="$DEPS" python3 "$ANALYZER" "${inputs[@]}" > "$fresh"
cmp "$fresh" "$RECEIPTS/aggregate.txt"
python3 "$SCRIPT_DIR/cs6_v7b_target23_arb_tm2r_hset_covering_verify.py" \
  --receipts "$RECEIPTS"
python3 "$SCRIPT_DIR/cs6_v7b_target23_arb_tm2r_hset_covering_mutations.py"

echo "CS6_V7B_TARGET23_ARB_TM2R_HSET_COVERING_GATE=true"
