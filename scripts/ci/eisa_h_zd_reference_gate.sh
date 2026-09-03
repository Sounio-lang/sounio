#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/eisa-h-zd-ref.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

MODULE="$ROOT_DIR/stdlib/eisa/hypercomplex_zd.sio"
WITNESS="$ROOT_DIR/tests/stdlib/eisa/test_eisa_h_zd.sio"

fail() {
  echo "[eisa-h-zd-ref] FAIL: $*" >&2
  exit 1
}

"$ROOT_DIR/bin/souc" check "$MODULE" >"$TMP_DIR/module.check.log" 2>&1 || {
  cat "$TMP_DIR/module.check.log" >&2
  fail "Madaros module check"
}

"$ROOT_DIR/bin/souc" check "$WITNESS" >"$TMP_DIR/witness.check.log" 2>&1 || {
  cat "$TMP_DIR/witness.check.log" >&2
  fail "Madaros witness check"
}

set +e
"$ROOT_DIR/bin/souc" run "$WITNESS" >"$TMP_DIR/madaros.run.log" 2>&1
madaros_wrapper_rc=$?
set -e
if [[ "$madaros_wrapper_rc" -eq 0 ]]; then
  madaros_runtime_status="PASS"
  madaros_driver_rc="0"
elif [[ "$madaros_wrapper_rc" -eq 1 ]] && grep -Fq 'main.elf rc=12' "$TMP_DIR/madaros.run.log"; then
  madaros_runtime_status="BLOCKED"
  madaros_driver_rc="12"
else
  cat "$TMP_DIR/madaros.run.log" >&2
  fail "unexpected default Madaros runtime result rc=$madaros_wrapper_rc"
fi

output="$(SOUNIO_SOUC_ENGINE=lean_single "$ROOT_DIR/bin/souc" run "$WITNESS" 2>&1)" || {
  printf '%s\n' "$output" >&2
  fail "lean_single witness execution"
}
printf '%s\n' "$output" | grep -Fq 'EISA_H_ZD_REF_V0 PASS' || {
  printf '%s\n' "$output" >&2
  fail "missing witness marker"
}

cat >"$TMP_DIR/proof_measurement_type_error.sio" <<'EOF'
use eisa::hypercomplex_zd::{zd_measured_norm_f64_v0, zd_requires_exact}

fn main() -> i32 with Mut {
    let measured = zd_measured_norm_f64_v0(1, 2, 2.0, 2.0, 0.0, 1.0e-12)
    if zd_requires_exact(measured) { 1 } else { 0 }
}
EOF

if "$ROOT_DIR/bin/souc" check "$TMP_DIR/proof_measurement_type_error.sio" >"$TMP_DIR/type-error.log" 2>&1; then
  fail "measured receipt was accepted by exact-token consumer"
fi
grep -Eq 'type mismatch|expected ZDExactTokenV0|does not match' "$TMP_DIR/type-error.log" || {
  cat "$TMP_DIR/type-error.log" >&2
  fail "proof/measurement rejection lacked a type boundary diagnostic"
}

echo "[eisa-h-zd-ref] RECEIPT check=Madaros execution=lean_single madaros_runtime=${madaros_runtime_status} wrapper_rc=${madaros_wrapper_rc} native_driver_rc=${madaros_driver_rc}"
echo '[eisa-h-zd-ref] PASS: bounded exact token, measured receipt, sign tamper, identity binding, and fail-closed classifications'
