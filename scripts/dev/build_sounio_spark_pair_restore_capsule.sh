#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_SPARK_PAIR_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_SPARK_PAIR_ENGINE:-madaros}"
AUTHORITY="$ROOT_DIR/stdlib/coordination/spark_pair_restore_capsule.sio"
ENTRYPOINT="$ROOT_DIR/tools/cluster/spark_pair_restore_capsule_main.sio"
OUTPUT="${SOUNIO_SPARK_PAIR_RESTORE_CAPSULE_OUTPUT:-$ROOT_DIR/tools/cluster/_build/default/sounio-spark-pair-restore-capsule-plan}"

fail() {
  printf 'build-sounio-spark-pair-restore-capsule: FAIL: %s\n' "$*" >&2
  exit 1
}

[[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
[[ -f "$AUTHORITY" ]] || fail "restore capsule authority is missing: $AUTHORITY"
[[ -f "$ENTRYPOINT" ]] || fail "restore capsule entrypoint is missing: $ENTRYPOINT"

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-spark-pair-restore-capsule-build.XXXXXX")"
trap 'rm -rf "$work"' EXIT
combined="$work/spark_pair_restore_capsule.sio"
compiled="$work/sounio-spark-pair-restore-capsule-plan"

mkdir -p "$(dirname "$OUTPUT")"
sed -n '1,$p' "$AUTHORITY" "$ENTRYPOINT" > "$combined"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined" -o "$compiled"
[[ -x "$compiled" ]] || fail 'compiler omitted the native Sounio executable'
install -m 0755 "$compiled" "$OUTPUT"

zero=0000000000000000000000000000000000000000000000000000000000000000
capsule=1111111111111111111111111111111111111111111111111111111111111111
node0=2222222222222222222222222222222222222222222222222222222222222222
node1=3333333333333333333333333333333333333333333333333333333333333333
parent=77dde525618f0f7a683aee1c0744163db84306d4a0eef9ab45d3f79ce1eb4d8e
probe="$($OUTPUT 9027 51 20 "$capsule" "$node0" "$node1" "$zero" "$parent" "$zero" 473 524287 0)"
case "$probe" in
  SOUNIO_SPARK_PAIR_RESTORE_CAPSULE_PLAN_ALLOW*' effect=NONE material_dispatch=false '*) ;;
  *) fail "native restore capsule authority failed its effect-free probe: $probe" ;;
esac

printf 'BUILT_SPARK_PAIR_RESTORE_CAPSULE_PLAN path=%s language=Sounio engine=%s effect=NONE material_dispatch=false\n' \
  "$OUTPUT" "$ENGINE"
