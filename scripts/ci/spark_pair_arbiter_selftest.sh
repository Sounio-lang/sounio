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
MATERIAL_BACKEND="$ROOT_DIR/scripts/dev/spark_pair_arbiter_k8s_backend.sh"
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
TEST_ARBITER="$work/spark-pair-arbiter-fixture"
MOCK_DIR="$work/mock"
RECEIPTS="$work/receipts"

sed -n '1,$p' "$MODULE" "$VECTORS" > "$combined"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined" -o "$executable"
result="$($executable)"
[[ "$result" == 'SOUNIO_SPARK_PAIR_SELFTEST_PASS vectors=64 authority=Sounio' ]] || \
  fail "Sounio vectors failed: $result"
[[ "$(grep -Fc "printf 'PIREUS_NVML_CLEAN node=%s epoch=%s uuid=%s product=%s driver=%s memory_observation=%s utilization_pct=%s" "$MATERIAL_BACKEND")" == 2 ]] || \
  fail 'initial and fresh NVML probes do not share the frozen evidence frame'
[[ "$(grep -Fc "memory=UNAVAILABLE_UNIFIED" "$MATERIAL_BACKEND")" == 2 ]] || \
  fail 'initial and fresh NVML probes do not normalize unified memory identically'
[[ "$(grep -Fc "pgrep -f '[n]vidia-cuda-mps'" "$MATERIAL_BACKEND")" == 2 ]] || \
  fail 'initial and fresh MPS probes are not protected against self-match'
grep -Fq "lease_is_live \"\$lease\" || fail 'Lease expired before material keepalive'" "$MATERIAL_BACKEND" || \
  fail 'material keepalive can revive an expired Lease'
grep -Fq "lease_is_live \"\$lease\" || fail 'Lease expired before recording NVML receipts'" "$MATERIAL_BACKEND" || \
  fail 'NVML receipt recording can revive an expired Lease'
[[ "$(grep -Fc 'verify_lease_freeze_binding "$lease"' "$MATERIAL_BACKEND")" -ge 10 ]] || \
  fail 'a Lease mutation path is not bound to the active semantics freeze'
[[ "$(grep -Fc 'verify_bootstrap_journal_binding' "$MATERIAL_BACKEND")" -ge 3 ]] || \
  fail 'bootstrap recovery does not bind the journal to the active semantics freeze'

SOUNIO_SPARK_PAIR_ENGINE="$ENGINE" SOUNIO_SPARK_PAIR_OUTPUT="$ADAPTER" "$BUILD" >/dev/null
install -m 0755 "$ARBITER" "$TEST_ARBITER"
"$ARBITER" verify >/dev/null
DRIFT_FREEZE="$work/drift.freeze.v1"
sed 's/^authority_sha256=.*/authority_sha256=0000000000000000000000000000000000000000000000000000000000000000/' \
  "$FREEZE" > "$DRIFT_FREEZE"
set +e
drift_output="$(SOUNIO_SPARK_PAIR_TEST_MODE=fixture-v1 SOUNIO_SOURCE_ROOT="$ROOT_DIR" \
  SOUNIO_SPARK_PAIR_FREEZE="$DRIFT_FREEZE" "$TEST_ARBITER" verify 2>&1)"
drift_status=$?
set -e
[[ $drift_status -eq 42 && "$drift_output" == *'frozen file drifted: authority_source'* ]] || \
  fail "semantic hash drift did not fail closed: status=$drift_status output=$drift_output"
set +e
override_output="$(SOUNIO_SPARK_PAIR_BACKEND=/bin/true "$ARBITER" verify 2>&1)"
override_status=$?
set -e
[[ $override_status -eq 42 && "$override_output" == *'runtime path overrides are forbidden'* ]] || \
  fail "production root override did not fail closed: status=$override_status output=$override_output"
set +e
fixture_output="$(SOUNIO_SPARK_PAIR_TEST_MODE=fixture-v1 SOUNIO_SOURCE_ROOT="$ROOT_DIR" "$ARBITER" verify 2>&1)"
fixture_status=$?
set -e
[[ $fixture_status -eq 42 && "$fixture_output" == *'fixture-v1 is forbidden in the canonical controller'* ]] || \
  fail "canonical fixture mode did not fail closed: status=$fixture_status output=$fixture_output"
set +e
malformed="$($ADAPTER 9024 14 1 1 1 249 255 2>&1)"
malformed_status=$?
set -e
[[ $malformed_status -eq 64 ]] || fail "malformed frame exited $malformed_status, expected 64"
[[ "$malformed" == *'reason=MALFORMED_FRAME code=104'* ]] || \
  fail "malformed frame did not preserve Sounio reason: $malformed"

export SOUNIO_SPARK_PAIR_BACKEND="$MOCK_BACKEND"
export SOUNIO_SPARK_PAIR_POLICY="$POLICY"
export SOUNIO_SPARK_PAIR_TEST_MODE=fixture-v1
export SOUNIO_SOURCE_ROOT="$ROOT_DIR"
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
ARBITER="$TEST_ARBITER"

reset_mock() {
  rm -rf "$MOCK_DIR"
  mkdir -p "$MOCK_DIR"
  "$MOCK_BACKEND" --policy "$POLICY" --freeze "$FREEZE" fixture-slurm-owned
}

reset_bootstrap() {
  rm -rf "$MOCK_DIR"
  mkdir -p "$MOCK_DIR"
  "$MOCK_BACKEND" --policy "$POLICY" --freeze "$FREEZE" fixture-uninitialized
}

reset_empty() {
  rm -rf "$MOCK_DIR"
  mkdir -p "$MOCK_DIR"
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

reset_empty
SOUNIO_SPARK_PAIR_HOLDER=bootstrap-old "$ARBITER" bootstrap-init >/dev/null
[[ "$(sed -n '1p' "$MOCK_DIR/state")" == SLURM_OWNED ]] || fail 'happy bootstrap did not establish Slurm ownership'
[[ "$(sed -n '1p' "$MOCK_DIR/nodeset_generation")" == 2 ]] || fail 'happy bootstrap did not refresh NodeSet generation'
[[ "$(sed -n '1,2p' "$MOCK_DIR/effects")" == $'install-fence\ndrain-slurm' ]] || \
  fail 'bootstrap did not install the admission fence before draining Slurm'

reset_empty
expect_refusal action28-post-lease-crash env \
  SOUNIO_SPARK_PAIR_MOCK_FAIL_AFTER_LEASE=1 \
  SOUNIO_SPARK_PAIR_HOLDER=bootstrap-crash "$ARBITER" bootstrap-init
[[ "$(sed -n '1p' "$MOCK_DIR/state")" == UNINITIALIZED ]] || \
  fail 'action28 post-Lease crash did not preserve the UNINITIALIZED anchor'
[[ "$(sed -n '1p' "$MOCK_DIR/journal")" == 0 ]] || \
  fail 'action28 crash fixture unexpectedly created a journal'
SOUNIO_SPARK_PAIR_MOCK_LEASE_LIVE=0 \
  SOUNIO_SPARK_PAIR_HOLDER=bootstrap-crash-recovery \
  "$ARBITER" bootstrap-recover >/dev/null
[[ "$(sed -n '1p' "$MOCK_DIR/state")" == SLURM_OWNED ]] || \
  fail 'action28 post-Lease crash was not recovered to Slurm'
[[ "$(sed -n '1p' "$MOCK_DIR/journal")" == 1 ]] || \
  fail 'bootstrap takeover did not reconstruct the missing journal'

reset_bootstrap
expect_refusal live-foreign-bootstrap-takeover env \
  SOUNIO_SPARK_PAIR_HOLDER=bootstrap-foreign "$ARBITER" bootstrap-recover

reset_bootstrap
expect_refusal bootstrap-journal-freeze-drift env \
  SOUNIO_SPARK_PAIR_MOCK_LEASE_LIVE=0 \
  SOUNIO_SPARK_PAIR_MOCK_JOURNAL_BOUND=0 \
  SOUNIO_SPARK_PAIR_HOLDER=bootstrap-journal-drift "$ARBITER" bootstrap-recover

for bootstrap_failure in drain-slurm install-fence install-gpu-bound-slurmd resume-slurm; do
  reset_bootstrap
  expect_refusal "bootstrap-$bootstrap_failure" env \
    SOUNIO_SPARK_PAIR_MOCK_FAIL="$bootstrap_failure" \
    SOUNIO_SPARK_PAIR_HOLDER=bootstrap-old "$ARBITER" bootstrap
  [[ "$(sed -n '1p' "$MOCK_DIR/state")" == UNINITIALIZED ]] || \
    fail "$bootstrap_failure did not remain fenced in UNINITIALIZED"
  SOUNIO_SPARK_PAIR_MOCK_FAIL="$bootstrap_failure" \
    SOUNIO_SPARK_PAIR_MOCK_LEASE_LIVE=0 \
    SOUNIO_SPARK_PAIR_HOLDER="bootstrap-recovery-$bootstrap_failure" \
    "$ARBITER" bootstrap-recover >/dev/null
  [[ "$(sed -n '1p' "$MOCK_DIR/state")" == SLURM_OWNED ]] || \
    fail "$bootstrap_failure recovery did not prove Slurm ownership"
  [[ "$(sed -n '1p' "$MOCK_DIR/nodeset_generation")" == 2 ]] || \
    fail "$bootstrap_failure recovery did not refresh NodeSet generation"
done

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
expect_refusal persisted-freeze-drift env SOUNIO_SPARK_PAIR_MOCK_FREEZE_BOUND=0 \
  SOUNIO_SPARK_PAIR_HOLDER=holder-freeze-drift "$ARBITER" hold 1

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

compgen -G "$RECEIPTS/material-*.receipt" >/dev/null || fail 'material result receipts were not emitted'
grep -h -q '^result=PASS$' "$RECEIPTS"/material-*.receipt || fail 'material PASS receipt is missing'
grep -h -q '^result=FAIL$' "$RECEIPTS"/material-*.receipt || fail 'material FAIL receipt is missing'
awk -F= '$1 == "decision_receipt_sha256" && $2 ~ /^[0-9a-f]+$/ && length($2) == 64 { found=1 } END { exit found ? 0 : 1 }' \
  "$RECEIPTS"/material-*.receipt || fail 'material receipt is not linked to a Sounio decision'

printf '%s\n' "$result"
printf 'SPARK_PAIR_ADAPTER_NEGATIVE_PASS reason=MALFORMED_FRAME status=64\n'
printf 'SPARK_PAIR_MATERIAL_SELFTEST_PASS positive=8 negative=21 freeze_drift=DENY persisted_freeze=DENY journal_freeze=DENY root_override=DENY canonical_fixture=DENY python_oracle=DENY direct_backend=DENY concurrency=DENY bootstrap_recovery=PASS action28_crash_recovery=PASS bootstrap_fence_first=PASS material_keepalive_expiry=DENY material_receipts=PASS nvml_formats=PASS\n'
