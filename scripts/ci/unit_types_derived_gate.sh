#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

CURRENT_SOUC="$TMP_DIR/sounio-current-source"
BUILD_LOG="$TMP_DIR/build-current-source.log"
AST_PROBE="self-hosted/compiler/k2_unit_derived_ast_probe.sio"
AST_PROBE_WITNESS="tests/frontend/unit_derived_acceleration_chain_current_source.sio"
CHECKER_PROBE="self-hosted/compiler/k2_unit_derived_checker_probe.sio"

AST_PROBE_LOG="$TMP_DIR/unit-derived-ast-probe.log"
if bin/souc run "$AST_PROBE" -- "$AST_PROBE_WITNESS" >"$AST_PROBE_LOG" 2>&1 &&
   grep -q 'unit_expr_factors=3' "$AST_PROBE_LOG" &&
   grep -q 'unit_expr_binaries=2' "$AST_PROBE_LOG" &&
   grep -q 'unit_expr_has_left=1' "$AST_PROBE_LOG" &&
   grep -q 'unit_derived_ast_verdict=0' "$AST_PROBE_LOG"; then
  printf 'PASS  %s retained chained derived unit AST payload\n' "$AST_PROBE"
else
  printf 'FAIL  %s did not retain the chained derived unit AST payload\n' "$AST_PROBE" >&2
  cat "$AST_PROBE_LOG" >&2
  exit 1
fi

CHECKER_PROBE_LOG="$TMP_DIR/unit-derived-checker-probe.log"
if bin/souc run "$CHECKER_PROBE" -- "$AST_PROBE_WITNESS" >"$CHECKER_PROBE_LOG" 2>&1 &&
   grep -q 'checker_unit_derived_semantic_probe=ok' "$CHECKER_PROBE_LOG" &&
   grep -q 'unit_derived_checker_verdict=0' "$CHECKER_PROBE_LOG"; then
  printf 'PASS  %s validated chained derived unit dimensions through check::knowledge_context\n' "$CHECKER_PROBE"
else
  printf 'FAIL  %s did not validate chained derived unit dimensions\n' "$CHECKER_PROBE" >&2
  cat "$CHECKER_PROBE_LOG" >&2
  exit 1
fi

if bin/souc-linux-x86_64 self-hosted/compiler/lean_single.sio "$CURRENT_SOUC" >"$BUILD_LOG" 2>&1; then
  chmod +x "$CURRENT_SOUC"
  printf 'PASS  built current-source lean_single compiler for derived unit gate\n'
else
  printf 'FAIL  could not build current-source lean_single compiler\n' >&2
  cat "$BUILD_LOG" >&2
  exit 1
fi

DERIVED_OK="tests/frontend/unit_derived_velocity_decl_current_source.sio"
DERIVED_REJECT="tests/compile-fail/unit_derived_velocity_reject_length.sio"
DERIVED_CHAIN_OK="tests/frontend/unit_derived_acceleration_chain_current_source.sio"
DERIVED_CHAIN_REJECT="tests/compile-fail/unit_derived_acceleration_reject_velocity.sio"
F64_UNIT_EXPR_OK="tests/frontend/unit_f64_unit_expr_velocity_current_source.sio"
F64_UNIT_EXPR_REJECT="tests/compile-fail/unit_f64_unit_expr_reject_length.sio"
F64_UNIT_EXPR_UNKNOWN_REJECT="tests/compile-fail/unit_f64_unit_expr_unknown_reject.sio"
LITERAL_SUFFIX_OK="tests/frontend/unit_literal_suffix_current_source.sio"
LITERAL_SUFFIX_REJECT="tests/compile-fail/unit_literal_suffix_reject_length_as_mass.sio"
CLINICAL_LITERAL_OK="tests/frontend/unit_literal_clinical_current_source.sio"
CLINICAL_LITERAL_REJECT="tests/compile-fail/unit_literal_clinical_reject_mass_as_amount_concentration.sio"
ENERGY_EXPLICIT_OK="tests/run-pass/unit_energy_explicit_conversion.sio"
ENERGY_REQUIRES_EXPLICIT="tests/compile-fail/unit_energy_requires_explicit_conversion.sio"

DERIVED_OK_LOG="$TMP_DIR/unit-derived-velocity.log"
if SOUNIO_SOUC_BIN="$CURRENT_SOUC" bin/souc run "$DERIVED_OK" >"$DERIVED_OK_LOG" 2>&1 &&
   grep -q 'unit derived velocity: PASS' "$DERIVED_OK_LOG"; then
  printf 'PASS  %s accepted m/s as derived velocity\n' "$DERIVED_OK"
else
  printf 'FAIL  %s did not accept the derived velocity witness\n' "$DERIVED_OK" >&2
  cat "$DERIVED_OK_LOG" >&2
  exit 1
fi

DERIVED_REJECT_LOG="$TMP_DIR/unit-derived-velocity-reject.log"
if SOUNIO_SOUC_BIN="$CURRENT_SOUC" bin/souc check "$DERIVED_REJECT" >"$DERIVED_REJECT_LOG" 2>&1; then
  printf 'FAIL  %s unexpectedly accepted length as velocity\n' "$DERIVED_REJECT" >&2
  cat "$DERIVED_REJECT_LOG" >&2
  exit 1
fi
if grep -q 'unit mismatch in call argument' "$DERIVED_REJECT_LOG"; then
  printf 'PASS  %s rejected length where velocity is required\n' "$DERIVED_REJECT"
else
  printf 'FAIL  %s failed without the expected unit mismatch diagnostic\n' "$DERIVED_REJECT" >&2
  cat "$DERIVED_REJECT_LOG" >&2
  exit 1
fi

DERIVED_CHAIN_OK_LOG="$TMP_DIR/unit-derived-acceleration.log"
if SOUNIO_SOUC_BIN="$CURRENT_SOUC" bin/souc run "$DERIVED_CHAIN_OK" >"$DERIVED_CHAIN_OK_LOG" 2>&1 &&
   grep -q 'unit derived acceleration: PASS' "$DERIVED_CHAIN_OK_LOG"; then
  printf 'PASS  %s accepted m/s/s as derived acceleration\n' "$DERIVED_CHAIN_OK"
else
  printf 'FAIL  %s did not accept the chained derived acceleration witness\n' "$DERIVED_CHAIN_OK" >&2
  cat "$DERIVED_CHAIN_OK_LOG" >&2
  exit 1
fi

DERIVED_CHAIN_REJECT_LOG="$TMP_DIR/unit-derived-acceleration-reject.log"
if SOUNIO_SOUC_BIN="$CURRENT_SOUC" bin/souc check "$DERIVED_CHAIN_REJECT" >"$DERIVED_CHAIN_REJECT_LOG" 2>&1; then
  printf 'FAIL  %s unexpectedly accepted velocity as acceleration\n' "$DERIVED_CHAIN_REJECT" >&2
  cat "$DERIVED_CHAIN_REJECT_LOG" >&2
  exit 1
fi
if grep -q 'unit mismatch in call argument' "$DERIVED_CHAIN_REJECT_LOG"; then
  printf 'PASS  %s rejected velocity where acceleration is required\n' "$DERIVED_CHAIN_REJECT"
else
  printf 'FAIL  %s failed without the expected unit mismatch diagnostic\n' "$DERIVED_CHAIN_REJECT" >&2
  cat "$DERIVED_CHAIN_REJECT_LOG" >&2
  exit 1
fi

F64_UNIT_EXPR_OK_LOG="$TMP_DIR/unit-f64-unit-expr.log"
if SOUNIO_SOUC_BIN="$CURRENT_SOUC" bin/souc run "$F64_UNIT_EXPR_OK" >"$F64_UNIT_EXPR_OK_LOG" 2>&1 &&
   grep -q 'unit f64 unit expr: PASS' "$F64_UNIT_EXPR_OK_LOG"; then
  printf 'PASS  %s accepted f64<UnitExpr> derived dimensions\n' "$F64_UNIT_EXPR_OK"
else
  printf 'FAIL  %s did not accept f64<UnitExpr> derived dimensions\n' "$F64_UNIT_EXPR_OK" >&2
  cat "$F64_UNIT_EXPR_OK_LOG" >&2
  exit 1
fi

F64_UNIT_EXPR_REJECT_LOG="$TMP_DIR/unit-f64-unit-expr-reject.log"
if SOUNIO_SOUC_BIN="$CURRENT_SOUC" bin/souc check "$F64_UNIT_EXPR_REJECT" >"$F64_UNIT_EXPR_REJECT_LOG" 2>&1; then
  printf 'FAIL  %s unexpectedly accepted length as f64<m/s>\n' "$F64_UNIT_EXPR_REJECT" >&2
  cat "$F64_UNIT_EXPR_REJECT_LOG" >&2
  exit 1
fi
if grep -q 'unit mismatch in call argument' "$F64_UNIT_EXPR_REJECT_LOG"; then
  printf 'PASS  %s rejected length where f64<m/s> is required\n' "$F64_UNIT_EXPR_REJECT"
else
  printf 'FAIL  %s failed without the expected f64<UnitExpr> diagnostic\n' "$F64_UNIT_EXPR_REJECT" >&2
  cat "$F64_UNIT_EXPR_REJECT_LOG" >&2
  exit 1
fi

F64_UNIT_EXPR_UNKNOWN_LOG="$TMP_DIR/unit-f64-unit-expr-unknown.log"
if SOUNIO_SOUC_BIN="$CURRENT_SOUC" bin/souc check "$F64_UNIT_EXPR_UNKNOWN_REJECT" >"$F64_UNIT_EXPR_UNKNOWN_LOG" 2>&1; then
  printf 'FAIL  %s unexpectedly accepted an unknown f64<UnitExpr> unit\n' "$F64_UNIT_EXPR_UNKNOWN_REJECT" >&2
  cat "$F64_UNIT_EXPR_UNKNOWN_LOG" >&2
  exit 1
fi
if grep -q 'unknown unit in f64<UnitExpr> annotation' "$F64_UNIT_EXPR_UNKNOWN_LOG"; then
  printf 'PASS  %s rejected an unknown f64<UnitExpr> unit\n' "$F64_UNIT_EXPR_UNKNOWN_REJECT"
else
  printf 'FAIL  %s failed without the expected unknown-unit diagnostic\n' "$F64_UNIT_EXPR_UNKNOWN_REJECT" >&2
  cat "$F64_UNIT_EXPR_UNKNOWN_LOG" >&2
  exit 1
fi

LITERAL_SUFFIX_OK_LOG="$TMP_DIR/unit-literal-suffix.log"
if SOUNIO_SOUC_BIN="$CURRENT_SOUC" bin/souc run "$LITERAL_SUFFIX_OK" >"$LITERAL_SUFFIX_OK_LOG" 2>&1 &&
   grep -q 'unit literal suffix: PASS' "$LITERAL_SUFFIX_OK_LOG"; then
  printf 'PASS  %s accepted numeric literal unit suffixes\n' "$LITERAL_SUFFIX_OK"
else
  printf 'FAIL  %s did not accept numeric literal unit suffixes\n' "$LITERAL_SUFFIX_OK" >&2
  cat "$LITERAL_SUFFIX_OK_LOG" >&2
  exit 1
fi

LITERAL_SUFFIX_REJECT_LOG="$TMP_DIR/unit-literal-suffix-reject.log"
if SOUNIO_SOUC_BIN="$CURRENT_SOUC" bin/souc check "$LITERAL_SUFFIX_REJECT" >"$LITERAL_SUFFIX_REJECT_LOG" 2>&1; then
  printf 'FAIL  %s unexpectedly accepted a length literal where mass was required\n' "$LITERAL_SUFFIX_REJECT" >&2
  cat "$LITERAL_SUFFIX_REJECT_LOG" >&2
  exit 1
fi
if grep -q 'unit mismatch in call argument' "$LITERAL_SUFFIX_REJECT_LOG"; then
  printf 'PASS  %s rejected numeric literal unit suffix with incompatible dimension\n' "$LITERAL_SUFFIX_REJECT"
else
  printf 'FAIL  %s failed without the expected unit mismatch diagnostic\n' "$LITERAL_SUFFIX_REJECT" >&2
  cat "$LITERAL_SUFFIX_REJECT_LOG" >&2
  exit 1
fi

CLINICAL_LITERAL_OK_LOG="$TMP_DIR/unit-literal-clinical.log"
if SOUNIO_SOUC_BIN="$CURRENT_SOUC" bin/souc run "$CLINICAL_LITERAL_OK" >"$CLINICAL_LITERAL_OK_LOG" 2>&1 &&
   grep -q 'unit literal clinical: PASS' "$CLINICAL_LITERAL_OK_LOG"; then
  printf 'PASS  %s accepted clinical numeric literal unit suffixes\n' "$CLINICAL_LITERAL_OK"
else
  printf 'FAIL  %s did not accept clinical numeric literal unit suffixes\n' "$CLINICAL_LITERAL_OK" >&2
  cat "$CLINICAL_LITERAL_OK_LOG" >&2
  exit 1
fi

CLINICAL_LITERAL_REJECT_LOG="$TMP_DIR/unit-literal-clinical-reject.log"
if SOUNIO_SOUC_BIN="$CURRENT_SOUC" bin/souc check "$CLINICAL_LITERAL_REJECT" >"$CLINICAL_LITERAL_REJECT_LOG" 2>&1; then
  printf 'FAIL  %s unexpectedly accepted mass concentration where amount concentration was required\n' "$CLINICAL_LITERAL_REJECT" >&2
  cat "$CLINICAL_LITERAL_REJECT_LOG" >&2
  exit 1
fi
if grep -q 'unit mismatch in call argument' "$CLINICAL_LITERAL_REJECT_LOG"; then
  printf 'PASS  %s rejected incompatible internal label dimension\n' "$CLINICAL_LITERAL_REJECT"
else
  printf 'FAIL  %s failed without the expected unit mismatch diagnostic\n' "$CLINICAL_LITERAL_REJECT" >&2
  cat "$CLINICAL_LITERAL_REJECT_LOG" >&2
  exit 1
fi

ENERGY_EXPLICIT_OK_LOG="$TMP_DIR/unit-energy-explicit.log"
if SOUNIO_SOUC_BIN="$CURRENT_SOUC" bin/souc run "$ENERGY_EXPLICIT_OK" >"$ENERGY_EXPLICIT_OK_LOG" 2>&1 &&
   grep -q 'unit energy explicit conversion: PASS' "$ENERGY_EXPLICIT_OK_LOG"; then
  printf 'PASS  %s accepted explicit conversion between compatible energy units\n' "$ENERGY_EXPLICIT_OK"
else
  printf 'FAIL  %s did not complete explicit energy-unit conversion witness\n' "$ENERGY_EXPLICIT_OK" >&2
  cat "$ENERGY_EXPLICIT_OK_LOG" >&2
  exit 1
fi

ENERGY_REQUIRES_EXPLICIT_LOG="$TMP_DIR/unit-energy-requires-explicit.log"
if SOUNIO_SOUC_BIN="$CURRENT_SOUC" bin/souc check "$ENERGY_REQUIRES_EXPLICIT" >"$ENERGY_REQUIRES_EXPLICIT_LOG" 2>&1; then
  printf 'FAIL  %s unexpectedly accepted energy-unit arithmetic without explicit conversion\n' "$ENERGY_REQUIRES_EXPLICIT" >&2
  cat "$ENERGY_REQUIRES_EXPLICIT_LOG" >&2
  exit 1
fi
if grep -q 'unit mismatch' "$ENERGY_REQUIRES_EXPLICIT_LOG"; then
  printf 'PASS  %s rejected compatible-dimension energy arithmetic without explicit conversion\n' "$ENERGY_REQUIRES_EXPLICIT"
else
  printf 'FAIL  %s failed without the expected unit mismatch diagnostic\n' "$ENERGY_REQUIRES_EXPLICIT" >&2
  cat "$ENERGY_REQUIRES_EXPLICIT_LOG" >&2
  exit 1
fi

cat <<'MSG'
Derived unit gate passed.
This uses a compiler rebuilt from the current self-hosted lean_single source,
not the checked-in default binary. It also runs modular parser AST and
check::knowledge_context semantic probes for the chained unit payload. It proves
`unit velocity = m / s;` registers a derived dimension, `m / s` can flow into a
velocity-typed boundary, and plain length cannot. It also proves a chained
derived unit, `unit acceleration = m / s / s;`, accepts m/s/s and rejects m/s.
It also proves current-source numeric literal unit suffixes such as `200.0<mg>`
and `300<mg>` lower into the same dimensional call-boundary checks. It also
shows declared internal dimension labels such as `mg_dL`, `mmol_L`, `U_L`, and
`mm_h` participate as internal current-source unit identifiers with no built-in
conversion factors. This is not a clinical-correctness, terminology-conformance,
conversion-safety, or dosing-safety claim. Finally, it proves same-dimension
energy units such as J and eV require explicit conversion before arithmetic.
MSG
