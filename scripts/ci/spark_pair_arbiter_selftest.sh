#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_SPARK_PAIR_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_SPARK_PAIR_ENGINE:-madaros}"
MODULE="$ROOT_DIR/stdlib/coordination/spark_pair_arbiter.sio"
VECTORS="$ROOT_DIR/tests/fixtures/spark_pair_arbiter/spark_pair_arbiter_vectors.sio"
BUILD="$ROOT_DIR/scripts/dev/build_sounio_spark_pair_arbiter.sh"
ADAPTER="$ROOT_DIR/tools/cluster/_build/default/sounio-spark-pair-arbiter"

fail() {
  printf 'spark-pair-arbiter-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-spark-pair-selftest.XXXXXX")"
trap 'rm -rf "$work"' EXIT
combined="$work/spark_pair_arbiter_selftest.sio"
executable="$work/spark_pair_arbiter_selftest"

sed -n '1,$p' "$MODULE" "$VECTORS" > "$combined"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined" -o "$executable"
result="$($executable)"
[[ "$result" == 'SOUNIO_SPARK_PAIR_SELFTEST_PASS vectors=41 authority=Sounio' ]] || \
  fail "Sounio vectors failed: $result"

SOUNIO_SPARK_PAIR_ENGINE="$ENGINE" "$BUILD" >/dev/null
set +e
malformed="$($ADAPTER 9024 14 1 1 1 249 255 2>&1)"
malformed_status=$?
set -e
[[ $malformed_status -eq 64 ]] || fail "malformed frame exited $malformed_status, expected 64"
[[ "$malformed" == *'reason=MALFORMED_FRAME code=104'* ]] || \
  fail "malformed frame did not preserve Sounio reason: $malformed"

printf '%s\n' "$result"
printf 'SPARK_PAIR_ADAPTER_NEGATIVE_PASS reason=MALFORMED_FRAME status=64\n'
