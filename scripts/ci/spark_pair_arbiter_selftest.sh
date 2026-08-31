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
ARBITER="$ROOT_DIR/scripts/dev/spark_pair_arbiter.sh"
MOCK_BACKEND="$ROOT_DIR/tests/fixtures/spark_pair_arbiter/mock_backend.sh"
POLICY="$ROOT_DIR/tools/cluster/spark_pair_arbiter.policy.v1"
FREEZE="$ROOT_DIR/tools/cluster/spark_pair_arbiter.freeze.v1"
TEST_FREEZE=''

fail() {
  printf 'spark-pair-arbiter-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-spark-pair-selftest.XXXXXX")"
trap 'rm -rf "$work"' EXIT
combined="$work/spark_pair_arbiter_selftest.sio"
executable="$work/spark_pair_arbiter_selftest"
ADAPTER="$work/sounio-spark-pair-arbiter"
MOCK_DIR="$work/mock"
RECEIPTS="$work/receipts"

sed -n '1,$p' "$MODULE" "$VECTORS" > "$combined"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined" -o "$executable"
result="$($executable)"
[[ "$result" == 'SOUNIO_SPARK_PAIR_SELFTEST_PASS vectors=59 authority=Sounio' ]] || \
  fail "Sounio vectors failed: $result"

SOUNIO_SPARK_PAIR_ENGINE="$ENGINE" SOUNIO_SPARK_PAIR_OUTPUT="$ADAPTER" "$BUILD" >/dev/null
"$ARBITER" verify >/dev/null
set +e
malformed="$($ADAPTER 9024 14 1 1 1 249 255 2>&1)"
malformed_status=$?
set -e
[[ $malformed_status -eq 64 ]] || fail "malformed frame exited $malformed_status, expected 64"
[[ "$malformed" == *'reason=MALFORMED_FRAME code=104'* ]] || \
  fail "malformed frame did not preserve Sounio reason: $malformed"

export SOUNIO_SPARK_PAIR_BACKEND="$MOCK_BACKEND"
export SOUNIO_SPARK_PAIR_POLICY="$POLICY"
TEST_FREEZE="$work/mock.freeze.v1"
adapter_hash="$(sha256sum "$ADAPTER" | cut -d ' ' -f 1)"
mock_hash="$(sha256sum "$MOCK_BACKEND" | cut -d ' ' -f 1)"
sed \
  -e "s|^native_executable_sha256=.*|native_executable_sha256=$adapter_hash|" \
  -e 's|^material_backend_source=.*|material_backend_source=tests/fixtures/spark_pair_arbiter/mock_backend.sh|' \
  -e "s|^material_backend_sha256=.*|material_backend_sha256=$mock_hash|" \
  "$FREEZE" > "$TEST_FREEZE"
export SOUNIO_SPARK_PAIR_FREEZE="$TEST_FREEZE"
export SOUNIO_SPARK_PAIR_AUTHORITY="$ADAPTER"
export SOUNIO_SPARK_PAIR_RECEIPT_DIR="$RECEIPTS"
export SOUNIO_SPARK_PAIR_MOCK_DIR="$MOCK_DIR"

reset_mock() {
  rm -rf "$MOCK_DIR"
  mkdir -p "$MOCK_DIR"
  "$MOCK_BACKEND" --policy "$POLICY" --freeze "$FREEZE" bootstrap-lease
}

expect_refusal() {
  local name="$1"
  shift
  local output status
  set +e
  output="$("$@" 2>&1)"
  status=$?
  set -e
  [[ $status -eq 42 ]] || fail "$name exited $status, expected 42: $output"
}

reset_mock
SOUNIO_SPARK_PAIR_HOLDER=holder-positive "$ARBITER" hold 1 >/dev/null
[[ "$(sed -n '1p' "$MOCK_DIR/state")" == SLURM_OWNED ]] || fail 'positive hold did not return the pair to Slurm'

reset_mock
SOUNIO_SPARK_PAIR_HOLDER=holder-first "$ARBITER" hold 4 >"$work/first-holder.log" 2>&1 &
first_pid=$!
for _ in 1 2 3 4 5 6 7 8 9 10; do
  [[ "$(sed -n '1p' "$MOCK_DIR/state")" == K8S_OWNED ]] && break
  sleep 1
done
[[ "$(sed -n '1p' "$MOCK_DIR/state")" == K8S_OWNED ]] || fail 'first holder did not reach K8S_OWNED'
expect_refusal concurrent-holder env SOUNIO_SPARK_PAIR_HOLDER=holder-second "$ARBITER" hold 1
wait "$first_pid" || fail "first holder failed: $(sed -n '1,120p' "$work/first-holder.log")"
[[ "$(sed -n '1p' "$MOCK_DIR/state")" == SLURM_OWNED ]] || fail 'concurrent-holder test did not restore Slurm'

reset_mock
expect_refusal forbidden-python-authority env \
  SOUNIO_SPARK_PAIR_AUTHORITY="$(command -v python3)" SOUNIO_SPARK_PAIR_HOLDER=python-oracle "$ARBITER" status

reset_mock
expect_refusal direct-backend-without-receipt "$MOCK_BACKEND" \
  --policy "$POLICY" --freeze "$TEST_FREEZE" drain-slurm \
  --holder unauthorized --epoch 1

reset_mock
expect_refusal stale-epoch env SOUNIO_SPARK_PAIR_MOCK_OBSERVED_EPOCH=999 \
  SOUNIO_SPARK_PAIR_HOLDER=holder-stale "$ARBITER" hold 1

reset_mock
expect_refusal dead-lease env SOUNIO_SPARK_PAIR_MOCK_LEASE_LIVE=0 \
  SOUNIO_SPARK_PAIR_HOLDER=holder-dead "$ARBITER" hold 1

reset_mock
expect_refusal drain-failure env SOUNIO_SPARK_PAIR_MOCK_FAIL=drain-slurm \
  SOUNIO_SPARK_PAIR_HOLDER=holder-drain "$ARBITER" hold 1
[[ "$(sed -n '1p' "$MOCK_DIR/state")" == SLURM_OWNED ]] || fail 'drain rollback did not restore Slurm'

reset_mock
expect_refusal partial-reservation env SOUNIO_SPARK_PAIR_MOCK_PARTIAL_RESERVATION=1 \
  SOUNIO_SPARK_PAIR_HOLDER=holder-partial "$ARBITER" hold 1
[[ "$(sed -n '1p' "$MOCK_DIR/state")" == SLURM_OWNED ]] || fail 'partial reservation rollback did not restore Slurm'

reset_mock
expect_refusal heartbeat-loss env SOUNIO_SPARK_PAIR_MOCK_FAIL=lease-renew \
  SOUNIO_SPARK_PAIR_HOLDER=holder-heartbeat "$ARBITER" hold 12
[[ "$(sed -n '1p' "$MOCK_DIR/state")" == SLURM_OWNED ]] || fail 'heartbeat rollback did not restore Slurm'

reset_mock
expect_refusal sticky-workload env SOUNIO_SPARK_PAIR_MOCK_STICKY_WORKLOAD=1 \
  SOUNIO_SPARK_PAIR_HOLDER=holder-sticky "$ARBITER" hold 1
[[ "$(sed -n '1p' "$MOCK_DIR/state")" == RECOVERY_REQUIRED ]] || fail 'sticky workload was not fenced in recovery'

reset_mock
expect_refusal observation-timeout env SOUNIO_SPARK_PAIR_COMMAND_TIMEOUT=1 \
  SOUNIO_SPARK_PAIR_MOCK_SLEEP_COMMAND=facts SOUNIO_SPARK_PAIR_MOCK_SLEEP_SECONDS=3 \
  SOUNIO_SPARK_PAIR_HOLDER=holder-timeout "$ARBITER" hold 1

reset_mock
SOUNIO_SPARK_PAIR_MOCK_LEASE_LIVE=0 SOUNIO_SPARK_PAIR_HOLDER=holder-recovery \
  "$ARBITER" recover >/dev/null
[[ "$(sed -n '1p' "$MOCK_DIR/state")" == SLURM_OWNED ]] || fail 'manual recovery did not prove Slurm ownership'

printf '%s\n' "$result"
printf 'SPARK_PAIR_ADAPTER_NEGATIVE_PASS reason=MALFORMED_FRAME status=64\n'
printf 'SPARK_PAIR_MATERIAL_SELFTEST_PASS positive=2 negative=10 python_oracle=DENY direct_backend=DENY concurrency=DENY\n'
