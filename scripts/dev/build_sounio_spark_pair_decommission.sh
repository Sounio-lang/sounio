#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_SPARK_PAIR_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_SPARK_PAIR_ENGINE:-madaros}"
BASE="$ROOT_DIR/stdlib/coordination/spark_pair_arbiter.sio"
EXTENSION="$ROOT_DIR/stdlib/coordination/spark_pair_decommission.sio"
ENTRYPOINT="$ROOT_DIR/tools/cluster/spark_pair_decommission_main.sio"
OUTPUT="${SOUNIO_SPARK_PAIR_DECOMMISSION_OUTPUT:-$ROOT_DIR/tools/cluster/_build/default/sounio-spark-pair-decommission-plan}"

fail() {
  printf 'build-sounio-spark-pair-decommission: FAIL: %s\n' "$*" >&2
  exit 1
}

[[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
[[ -f "$BASE" ]] || fail "base Sounio authority is missing: $BASE"
[[ -f "$EXTENSION" ]] || fail "decommission Sounio authority is missing: $EXTENSION"
[[ -f "$ENTRYPOINT" ]] || fail "decommission Sounio entrypoint is missing: $ENTRYPOINT"

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-spark-pair-decommission-build.XXXXXX")"
trap 'rm -rf "$work"' EXIT
combined="$work/spark_pair_decommission.sio"
compiled="$work/sounio-spark-pair-decommission-plan"

mkdir -p "$(dirname "$OUTPUT")"
sed -n '1,$p' "$BASE" "$EXTENSION" "$ENTRYPOINT" > "$combined"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined" -o "$compiled"
[[ -x "$compiled" ]] || fail 'compiler omitted the native Sounio executable'
install -m 0755 "$compiled" "$OUTPUT"

probe="$($OUTPUT 9026 33 1 1 1 2 1017 219 1009 131071 7)"
case "$probe" in
  SOUNIO_SPARK_PAIR_DECOMMISSION_PLAN_ALLOW*' effect=NONE '*) ;;
  *) fail "native decommission authority failed its effect-free probe: $probe" ;;
esac

printf 'BUILT_SPARK_PAIR_DECOMMISSION_PLAN path=%s language=Sounio engine=%s effect=NONE\n' \
  "$OUTPUT" "$ENGINE"
