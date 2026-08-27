#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
SOUC="${SOUNIO_LOOM_WITNESS_EPOCH_HANDOFF_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_WITNESS_EPOCH_HANDOFF_ENGINE:-lean_single}"
STDLIB="$ROOT_DIR/stdlib"
MODULE="$STDLIB/coordination/loom_witness_epoch_handoff.sio"
WITNESS="$ROOT_DIR/tests/compiler/loom_witness_epoch_handoff_typestate_check.sio"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-witness-epoch-typestate.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

fail() {
  printf 'loom-witness-epoch-handoff-typestate: FAIL: %s\n' "$*" >&2
  exit 1
}

compose_module_source() {
  local module="$1" source="$2" output="$3"
  {
    sed -n '1,$p' "$module"
    awk '
      /^use coordination::loom_witness_epoch_handoff::\{/ { skip=1; next }
      skip && /^\}/ { skip=0; next }
      !skip { print }
    ' "$source"
  } > "$output"
}

compose_source() {
  compose_module_source "$MODULE" "$1" "$2"
}

expect_rejection() {
  local label="$1" source="$2" code="$3" rc=0
  local log="$WORK/$label.log"
  set +e
  SOUNIO_STDLIB_PATH="$STDLIB" "$SOUC" check "$source" >"$log" 2>&1
  rc=$?
  set -e
  [[ "$rc" -eq 1 ]] || { sed -n '1,240p' "$log" >&2; fail "$label rc=$rc"; }
  rg -q "error\[$code" "$log" || {
    sed -n '1,240p' "$log" >&2
    fail "$label omitted $code"
  }
}

expect_runtime_refusal() {
  local label="$1" source="$2" rc=0
  local log="$WORK/$label.log"
  set +e
  SOUNIO_STDLIB_PATH="$STDLIB" SOUNIO_SOUC_ENGINE="$ENGINE" \
    "$SOUC" run "$source" >"$log" 2>&1
  rc=$?
  set -e
  [[ "$rc" -eq 1 ]] || { sed -n '1,240p' "$log" >&2; fail "$label rc=$rc"; }
}

sabotage_rule() {
  local rule="$1" output="$2"
  awk -v target="fn $rule(" '
    BEGIN { in_rule=0; in_body=0; changed=0 }
    $0 == target { in_rule=1; print; next }
    in_rule && $0 == ") -> bool {" {
      print
      print "    true"
      in_body=1
      next
    }
    in_body && $0 == "}" {
      print
      in_rule=0
      in_body=0
      changed++
      next
    }
    in_body { next }
    { print }
    END { if (changed != 1) exit 42 }
  ' "$MODULE" > "$output" || fail "could not sabotage named rule $rule"
}

[[ -x "$SOUC" ]] || fail "compiler is missing: $SOUC"
combined="$WORK/loom_witness_epoch_handoff_typestate.sio"
compose_source "$WITNESS" "$combined"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" run "$combined" \
  >"$WORK/runtime.log" 2>&1 || {
  sed -n '1,280p' "$WORK/runtime.log" >&2
  fail 'linear witness epoch handoff proof did not run'
}
rg -Fxq \
  'loom-witness-epoch-handoff-typestate: PASS linear=1 old_quorum=1 new_quorum=1 handoff=1 host_receipt=1' \
  "$WORK/runtime.log" || {
  sed -n '1,280p' "$WORK/runtime.log" >&2
  fail 'linear witness epoch handoff proof omitted its exact receipt'
}

expect_rejection private-host-receipt \
  "$ROOT_DIR/tests/compile-fail/loom_witness_epoch_handoff_host_receipt_private.sio" E176

reuse="$WORK/reuse.sio"
compose_source \
  "$ROOT_DIR/tests/compile-fail/loom_witness_epoch_handoff_quorum_reuse.sio" \
  "$reuse"
expect_rejection quorum-reuse "$reuse" E039

certificate_reuse_source="$ROOT_DIR/tests/compiler/loom_witness_epoch_handoff_join_certificate_reuse.sio"
expect_runtime_refusal certificate-reuse "$certificate_reuse_source"
epoch_rebinding_source="$ROOT_DIR/tests/compiler/loom_witness_epoch_handoff_join_epoch_rebinding.sio"
expect_runtime_refusal epoch-rebinding "$epoch_rebinding_source"
unverified_summary_source="$ROOT_DIR/tests/compiler/loom_witness_epoch_handoff_unverified_summary.sio"
expect_runtime_refusal unverified-summary "$unverified_summary_source"
sabotaged_module="$WORK/certificate-rule-sabotaged.sio"
sabotaged_case="$WORK/certificate-reuse-sabotaged.sio"
sabotage_rule witness_epoch_certificates_are_bound "$sabotaged_module"
compose_module_source "$sabotaged_module" "$certificate_reuse_source" \
  "$sabotaged_case"
SOUNIO_STDLIB_PATH="$STDLIB" SOUNIO_SOUC_ENGINE="$ENGINE" \
  "$SOUC" run "$sabotaged_case" >"$WORK/certificate-sabotage.log" 2>&1 || {
  sed -n '1,240p' "$WORK/certificate-sabotage.log" >&2
  fail 'certificate-rule sabotage did not admit the exact reused-certificate control'
}

printf 'loom-witness-epoch-handoff-typestate: PASS privacy=E176 linear=E039 certificate_reuse=REFUSED certificate_rule_sabotage=ADMITS epoch_rebinding=REFUSED unverified_summary=REFUSED joint_quorum=3/4+3/4 frame=9015\n'
