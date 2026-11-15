#!/usr/bin/env bash
# Build the .olean files of the bootstrap model into <dir>, after checking that
# formal/Bootstrap/SeedData.lean matches the seed files.
#   bootstrap/model/build_model.sh <dir>
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
out="${1:?usage: build_model.sh <dir>}"
lean="${LEAN_BIN:+$LEAN_BIN/}lean"
mkdir -p "$out/Bootstrap"

fresh="$(mktemp)"
trap 'rm -f "$fresh"' EXIT
"$root/bootstrap/model/gen_seed_data.sh" "$fresh"
if ! cmp -s "$fresh" "$root/formal/Bootstrap/SeedData.lean"; then
  echo "formal/Bootstrap/SeedData.lean is stale: run bootstrap/model/gen_seed_data.sh" >&2
  exit 1
fi

cd "$root/formal"
for m in X86Min SeedData; do
  LEAN_PATH="$out" "$lean" "Bootstrap/$m.lean" -o "$out/Bootstrap/$m.olean"
done
