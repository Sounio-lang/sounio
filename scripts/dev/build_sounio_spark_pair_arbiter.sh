#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_SPARK_PAIR_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_SPARK_PAIR_ENGINE:-madaros}"
MODULE="${SOUNIO_SPARK_PAIR_MODULE:-$ROOT_DIR/stdlib/coordination/spark_pair_arbiter.sio}"
ENTRYPOINT="${SOUNIO_SPARK_PAIR_MAIN:-$ROOT_DIR/tools/cluster/spark_pair_arbiter_main.sio}"
OUTPUT="${SOUNIO_SPARK_PAIR_OUTPUT:-$ROOT_DIR/tools/cluster/_build/default/sounio-spark-pair-arbiter}"

fail() {
  printf 'build-sounio-spark-pair-arbiter: FAIL: %s\n' "$*" >&2
  exit 1
}

[[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
[[ -f "$MODULE" ]] || fail "Sounio authority module is missing: $MODULE"
[[ -f "$ENTRYPOINT" ]] || fail "Sounio adapter entrypoint is missing: $ENTRYPOINT"

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-spark-pair-build.XXXXXX")"
trap 'rm -rf "$work"' EXIT
combined="$work/spark_pair_arbiter.sio"
compiled="$work/sounio-spark-pair-arbiter"

mkdir -p "$(dirname "$OUTPUT")"
sed -n '1,$p' "$MODULE" "$ENTRYPOINT" > "$combined"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined" -o "$compiled"
[[ -x "$compiled" ]] || fail 'compiler omitted the native Sounio executable'
install -m 0755 "$compiled" "$OUTPUT"

# STATUS over a canonical fact frame. Authority mask 249 has policy/holder/
# Lease/pair/NodeSet/plugin set and the policy-error/timeout bits clear.
probe="$($OUTPUT 9024 14 1 1 1 249 255 255)"
case "$probe" in
  SOUNIO_SPARK_PAIR_ALLOW*) ;;
  *) fail "native authority failed its probe: $probe" ;;
esac

printf 'BUILT_SPARK_PAIR_ARBITER path=%s language=Sounio engine=%s\n' "$OUTPUT" "$ENGINE"
