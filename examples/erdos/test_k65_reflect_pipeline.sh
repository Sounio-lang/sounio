#!/usr/bin/env bash
# End-to-end smoke gate for the finite K6/no-5 reflected-LRAT pipeline.
#
# This tests the finite K6/no-5-colouring certificate path only. It does not
# certify a Euclidean unit-distance chi>=6 witness.
#
# Set K65_REPRO_CHECK=1 to run a second independent generation and compare the
# generated Lean module byte-for-byte. The default keeps the smoke gate short.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="${WORK:-$(mktemp -d)}"
CONVERTER_WORK="$WORK/converter"
K65_WORK="$WORK/k65"
REPRO_WORK="$WORK/k65-repro"
OUT_LEAN="$ROOT/formal/lean4/SounioSatK65Reflect.lean"
REPRO_LEAN="$REPRO_WORK/SounioSatK65Reflect.lean"
K65_REPRO_CHECK="${K65_REPRO_CHECK:-0}"
LAKE="${LAKE:-$(command -v lake || true)}"
if [[ -z "$LAKE" && -x /workspace/.home/openvscode-server/.elan/bin/lake ]]; then
  LAKE=/workspace/.home/openvscode-server/.elan/bin/lake
fi
if [[ -z "$LAKE" ]]; then
  echo "error: lake not found; set LAKE=/path/to/lake" >&2
  exit 127
fi

echo "k65_reflect_pipeline: workdir=$WORK"
"$LAKE" --version

WORK="$CONVERTER_WORK" "$ROOT/examples/erdos/test_drup_to_lrat_rup.sh"

WORK="$K65_WORK" "$ROOT/examples/erdos/make_k65_reflect_smoke.sh" "$OUT_LEAN"

if [[ "$K65_REPRO_CHECK" == 1 ]]; then
  mkdir -p "$REPRO_WORK"
  WORK="$REPRO_WORK" "$ROOT/examples/erdos/make_k65_reflect_smoke.sh" "$REPRO_LEAN"
  cmp -s "$OUT_LEAN" "$REPRO_LEAN"
  sha256sum "$OUT_LEAN" "$REPRO_LEAN"
else
  sha256sum "$OUT_LEAN"
fi

rg -q '^theorem k65_unsat : k65_cnf\.Unsat :=' "$OUT_LEAN"
rg -q '^theorem k65_not_colourable :' "$OUT_LEAN"

(
  cd "$ROOT/formal/lean4"
  "$LAKE" build SounioSatK65Reflect SounioFiniteUnitDistanceWitnessSmoke DeGreyChi5Vitrine
)

if rg -q '\b(sorry|admit)\b' \
  "$OUT_LEAN" \
  "$ROOT/formal/lean4/SounioFiniteUnitDistanceWitnessSmoke.lean" \
  "$ROOT/formal/lean4/DeGreyChi5Vitrine.lean"
then
  echo "error: sorry/admit found in reflected K6 smoke surface" >&2
  exit 1
fi

echo "k65_reflect_pipeline: PASS"
