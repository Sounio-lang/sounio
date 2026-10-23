#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
FREEZE="$ROOT_DIR/tools/cluster/spark_pair_decommission.freeze.v1"
PARITY_OPEN="$ROOT_DIR/tools/cluster/spark_pair_decommission.parity-open.v1"
LEAN_SOURCE="$ROOT_DIR/formal/lean4/SounioSparkPairDecommissionParity.lean"
KOKA_SOURCE="$ROOT_DIR/tests/parity/spark_pair_decommission_effect_parity.kk"
CPP_SOURCE="$ROOT_DIR/tools/cluster/spark_pair_decommission_noop_material_parity.cpp"
LEAN_RECEIPT="$ROOT_DIR/tools/cluster/spark_pair_decommission.formal-parity.v1"
KOKA_RECEIPT="$ROOT_DIR/tools/cluster/spark_pair_decommission.effect-parity.v1"
CPP_RECEIPT="$ROOT_DIR/tools/cluster/spark_pair_decommission.material-parity.v1"
LEAN_TOOLCHAIN="leanprover/lean4:v4.33.0"

fail() {
  printf 'spark-pair-decommission-parity: FAIL: %s\n' "$*" >&2
  exit 1
}

field() {
  local path="$1"
  local key="$2"
  sed -n "s/^${key}=//p" "$path"
}

sha256() {
  local digest
  digest="$(sha256sum "$1")"
  printf '%s\n' "${digest%% *}"
}

check_receipt() {
  local receipt="$1"
  local hash_field="$2"
  local status="$3"
  local role="$4"
  local source="$5"
  local expected_receipt_hash actual_receipt_hash expected_source_hash actual_source_hash
  expected_receipt_hash="$(field "$PARITY_OPEN" "$hash_field")"
  actual_receipt_hash="$(sha256 "$receipt")"
  [[ -n "$expected_receipt_hash" && "$actual_receipt_hash" == "$expected_receipt_hash" ]] ||
    fail "parity receipt drift: $receipt"
  grep -qx "status=$status" "$receipt" || fail "unexpected receipt status: $receipt"
  grep -qx "language_role=$role" "$receipt" || fail "unexpected language role: $receipt"
  grep -qx 'semantic_authority=Sounio' "$receipt" || fail "Sounio authority missing: $receipt"
  grep -qx 'semantic_authority_role=false' "$receipt" || fail "parity promoted to authority: $receipt"
  grep -qx 'authority_promotion=false' "$receipt" || fail "authority promotion permitted: $receipt"
  grep -qx 'effect=NONE' "$receipt" || fail "non-empty effect receipt: $receipt"
  grep -qx 'material_dispatch=false' "$receipt" || fail "material dispatch receipt: $receipt"
  [[ "$(field "$receipt" semantics_freeze_sha256)" == "$(sha256 "$FREEZE")" ]] ||
    fail "receipt freeze drift: $receipt"
  expected_source_hash="$(field "$receipt" source_sha256)"
  actual_source_hash="$(sha256 "$source")"
  [[ -n "$expected_source_hash" && "$actual_source_hash" == "$expected_source_hash" ]] ||
    fail "parity source drift: $source"
}

for command in elan koka c++ sha256sum; do
  command -v "$command" >/dev/null 2>&1 || fail "missing toolchain command: $command"
done

for source in "$FREEZE" "$PARITY_OPEN" "$LEAN_SOURCE" "$KOKA_SOURCE" "$CPP_SOURCE" \
  "$LEAN_RECEIPT" "$KOKA_RECEIPT" "$CPP_RECEIPT"; do
  [[ -f "$source" ]] || fail "missing parity input: $source"
done

grep -qx 'status=SEMANTICS_FROZEN' "$FREEZE" || fail 'Sounio semantics are not frozen'
grep -qx 'frame_schema=9026' "$FREEZE" || fail 'unexpected Sounio semantic frame'
grep -qx 'material_dispatch=false' "$FREEZE" || fail 'frozen frame permits material dispatch'
grep -qx 'claim_ready=false' "$PARITY_OPEN" || fail 'parity receipt promoted the claim'
check_receipt "$LEAN_RECEIPT" formal_parity_receipt_sha256 FORMAL_PARITY FORMAL_PARITY "$LEAN_SOURCE"
check_receipt "$KOKA_RECEIPT" effect_parity_receipt_sha256 EFFECT_PARITY EFFECT_PARITY "$KOKA_SOURCE"
check_receipt "$CPP_RECEIPT" material_parity_receipt_sha256 MATERIAL_PARITY MATERIAL_PARITY "$CPP_SOURCE"

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-spark-pair-parity.XXXXXX")"
trap 'rm -rf "$work"' EXIT

elan toolchain list | grep -qx "$LEAN_TOOLCHAIN" ||
  fail "required Lean toolchain is not installed: $LEAN_TOOLCHAIN"
lean_version="$(elan run "$LEAN_TOOLCHAIN" lean --version)"
[[ "$lean_version" == Lean\ \(version\ 4.33.0,* ]] ||
  fail "unexpected Lean toolchain: $lean_version"
lean_output="$(elan run "$LEAN_TOOLCHAIN" lean --threads=1 "$LEAN_SOURCE")"
[[ "$lean_output" == *'SOUNIO_SPARK_PAIR_DECOMMISSION_LEAN_PARITY_PASS frame=9026 scope=STRUCTURAL effect=NONE'* ]] ||
  fail "Lean structural parity witness did not pass: $lean_output"
[[ "$lean_output" != *'sorryAx'* ]] || fail 'Lean parity depends on sorryAx'
if grep -Eq '(^|[^[:alpha:]])(sorry|axiom)([^[:alpha:]]|$)' "$LEAN_SOURCE"; then
  fail 'Lean parity contains an unproved declaration'
fi

koka_version="$(koka --version | sed -n '1p')"
[[ "$koka_version" == 'Koka 3.2.3,'* ]] || fail "unexpected Koka toolchain: $koka_version"
koka_types="$(koka -c --showtypesigs --console=raw \
  --builddir="$work/koka-types-build" --outputdir="$work/koka-types-out" \
  "$KOKA_SOURCE")"
[[ "$koka_types" == *'/planned-effect: (action : int) -> pure string'* ]] ||
  fail 'Koka compiler did not infer planned-effect as pure'
[[ "$koka_types" == *'/all-actions-effect-none: () -> pure bool'* ]] ||
  fail 'Koka compiler did not infer the aggregate witness as pure'
koka -O2 --builddir="$work/koka-build" -o "$work/koka-parity" "$KOKA_SOURCE" >/dev/null
chmod 0700 "$work/koka-parity"
koka_output="$("$work/koka-parity")"
[[ "$koka_output" == 'SOUNIO_SPARK_PAIR_DECOMMISSION_KOKA_PARITY_PASS frame=9026 actions=17 effect=NONE' ]] ||
  fail "Koka effect parity witness did not pass: $koka_output"
if grep -Eq '#include <(filesystem|fstream|net|sys/|unistd)|\b(system|exec|fork|popen|socket|bpf)\s*\(' "$CPP_SOURCE"; then
  fail 'C++ no-op consumer references a material API'
fi
c++ -std=c++20 -O2 -Wall -Wextra -Werror -pedantic -fno-exceptions -fno-rtti \
  "$CPP_SOURCE" -o "$work/cpp-parity"
cpp_output="$("$work/cpp-parity" --selftest)"
[[ "$cpp_output" == 'SOUNIO_SPARK_PAIR_DECOMMISSION_CPP_PARITY_PASS frame=9026 material_effect=NONE' ]] ||
  fail "C++ no-op material parity witness did not pass: $cpp_output"

SOUNIO_SPARK_PAIR_DECOMMISSION_OUTPUT="$work/sounio-plan" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_spark_pair_decommission.sh" >/dev/null
sounio_allow="$("$work/sounio-plan" 9026 34 1 1 1 2 1017 219 1009 131071 7)"
[[ "$("$work/cpp-parity" --classify "$sounio_allow")" == 'NONE' ]] ||
  fail 'C++ consumer did not preserve the frozen Sounio no-op result'

if "$work/cpp-parity" --classify \
    'SOUNIO_SPARK_PAIR_ALLOW schema=sounio-spark-pair-plan-v1 effect=NONE' >/dev/null 2>&1; then
  fail 'C++ consumer accepted the parent material prefix'
fi
if "$work/cpp-parity" --classify \
    'SOUNIO_SPARK_PAIR_DECOMMISSION_PLAN_ALLOW schema=sounio-spark-pair-decommission-plan-v1 effect=EXEC' >/dev/null 2>&1; then
  fail 'C++ consumer accepted a material effect'
fi

printf 'SPARK_PAIR_DECOMMISSION_PARITY_PASS frame=9026 lean=STRUCTURAL_PROVEN koka=PURE_EFFECT_NONE cpp=NOOP_CONSUMER material_dispatch=false\n'
