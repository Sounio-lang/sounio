#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_SPARK_PAIR_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_SPARK_PAIR_ENGINE:-madaros}"
AUTHORITY="$ROOT_DIR/stdlib/coordination/spark_pair_historical_provenance.sio"
ENTRYPOINT="$ROOT_DIR/tools/cluster/spark_pair_historical_provenance_main.sio"
OUTPUT="${SOUNIO_SPARK_PAIR_HISTORICAL_PROVENANCE_OUTPUT:-$ROOT_DIR/tools/cluster/_build/default/sounio-spark-pair-historical-provenance-plan}"

fail() {
  printf 'build-sounio-spark-pair-historical-provenance: FAIL: %s\n' "$*" >&2
  exit 1
}

[[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
[[ -f "$AUTHORITY" ]] || fail "historical provenance authority is missing: $AUTHORITY"
[[ -f "$ENTRYPOINT" ]] || fail "historical provenance entrypoint is missing: $ENTRYPOINT"

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-spark-pair-historical-provenance-build.XXXXXX")"
trap 'rm -rf "$work"' EXIT
combined="$work/spark_pair_historical_provenance.sio"
compiled="$work/sounio-spark-pair-historical-provenance-plan"

mkdir -p "$(dirname "$OUTPUT")"
sed -n '1,$p' "$AUTHORITY" "$ENTRYPOINT" > "$combined"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined" -o "$compiled"
[[ -x "$compiled" ]] || fail 'compiler omitted the native Sounio executable'
install -m 0755 "$compiled" "$OUTPUT"

zero=0000000000000000000000000000000000000000000000000000000000000000
source=1111111111111111111111111111111111111111111111111111111111111111
node0=2222222222222222222222222222222222222222222222222222222222222222
node1=3333333333333333333333333333333333333333333333333333333333333333
pair=4444444444444444444444444444444444444444444444444444444444444444
anchor=5555555555555555555555555555555555555555555555555555555555555555
parent=d1d67253355be3deab0b3faf05fb345497b1c98dfc15f1194b787830e632fb50
probe="$($OUTPUT 9028 55 30 1 "$source" "$node0" "$node1" "$pair" "$anchor" \
  "$parent" "$zero" 473 255 0 0)"
case "$probe" in
  SOUNIO_SPARK_PAIR_HISTORICAL_PROVENANCE_PLAN_ALLOW*' effect=NONE material_dispatch=false '*) ;;
  *) fail "native historical provenance authority failed its effect-free probe: $probe" ;;
esac

printf 'BUILT_SPARK_PAIR_HISTORICAL_PROVENANCE_PLAN path=%s language=Sounio engine=%s effect=NONE material_dispatch=false\n' \
  "$OUTPUT" "$ENGINE"
