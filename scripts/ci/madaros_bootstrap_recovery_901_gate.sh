#!/usr/bin/env bash
# Recover a self-hosted Madaros seed when the tracked operational seed cannot
# parse the current compiler closure. This is an audited bridge, never the
# normal operational bootstrap path.

set -euo pipefail
export LC_ALL=C

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_SCRIPT="$ROOT_DIR/scripts/ci/build_modular_madaros.sh"
ACCEPTANCE_GATE="$ROOT_DIR/scripts/ci/madaros_imported_runtime_acceptance_gate.sh"
CAPACITY_GATE="$ROOT_DIR/scripts/ci/madaros_struct_layout_capacity_gate.sh"
SCOPE_GATE="$ROOT_DIR/scripts/ci/madaros_scope_contextual_binding_gate.sh"
LEGACY_BOOTSTRAP="${SOUNIO_MADAROS_RECOVERY_LEGACY_BOOTSTRAP:-$ROOT_DIR/bin/souc-linux-x86_64}"
KEEP_WORK="${SOUNIO_MADAROS_RECOVERY_KEEP:-0}"
SOURCE_COMMIT="${SOUNIO_MADAROS_RECOVERY_SOURCE_COMMIT:-unrecorded}"

fail() {
  echo "[madaros-bootstrap-recovery] FAIL: $*" >&2
  exit 1
}

portable_sha256() {
  sha256sum "$1" 2>/dev/null | awk '{print $1}' || shasum -a 256 "$1" | awk '{print $1}'
}

require_raw_elf() {
  local path="$1"
  local label="$2"

  [[ -x "$path" && -s "$path" ]] || fail "$label is missing, empty, or not executable: $path"
  [[ "$(head -c4 "$path" 2>/dev/null)" == $'\x7fELF' ]] || fail "$label is not a raw ELF: $path"
}

if [[ "${1:-}" == '--structural-only' ]]; then
  [[ $# -eq 1 ]] || fail 'usage: madaros_bootstrap_recovery_901_gate.sh [--structural-only]'
  [[ -x "$BUILD_SCRIPT" ]] || fail "missing modular build script: $BUILD_SCRIPT"
  [[ -x "$ACCEPTANCE_GATE" ]] || fail "missing nominal-layout gate: $ACCEPTANCE_GATE"
  [[ -x "$CAPACITY_GATE" ]] || fail "missing layout-capacity gate: $CAPACITY_GATE"
  [[ -x "$SCOPE_GATE" ]] || fail "missing contextual-scope gate: $SCOPE_GATE"
  [[ -x "$LEGACY_BOOTSTRAP" ]] || fail "missing declared legacy recovery bootstrap: $LEGACY_BOOTSTRAP"
  echo '[madaros-bootstrap-recovery] PASS: audited bridge and fixed-point wiring is present'
  exit 0
fi
[[ $# -eq 0 ]] || fail 'usage: madaros_bootstrap_recovery_901_gate.sh [--structural-only]'

require_raw_elf "$LEGACY_BOOTSTRAP" 'legacy recovery bootstrap'
[[ -f "$ROOT_DIR/self-hosted/compiler/main.sio" ]] || fail 'missing modular compiler source'
[[ -f "$ROOT_DIR/self-hosted/compiler/lean_single.sio" ]] || fail 'missing audited legacy seed source'

if [[ -n "${SOUNIO_MADAROS_RECOVERY_DIR:-}" ]]; then
  WORK="$SOUNIO_MADAROS_RECOVERY_DIR"
  [[ ! -e "$WORK" ]] || fail "refusing existing recovery work directory: $WORK"
  mkdir -p "$WORK"
else
  WORK="$(mktemp -d /tmp/sounio-madaros-bootstrap-recovery.XXXXXX)"
fi
if [[ "$KEEP_WORK" != '1' ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

BRIDGE="$WORK/madaros-bridge"
STAGE1="$WORK/madaros-stage1"
STAGE2="$WORK/madaros-stage2"
RECEIPT="$WORK/bootstrap-recovery-receipt.tsv"

if ! env \
  -u MADAROS_RAW_BIN \
  -u SOUNIO_MADAROS_BIN \
  -u SOUNIO_MADAROS_SEED \
  SOUC_BIN="$LEGACY_BOOTSTRAP" \
  SOUNIO_MADAROS_BOOTSTRAP_MODE=lean-audit \
  SOUNIO_BUILD_LOCK="$WORK/souc-build.lock" \
  SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" \
  bash "$BUILD_SCRIPT" "$BRIDGE" >"$WORK/bridge-build.log" 2>&1; then
  tail -n 120 "$WORK/bridge-build.log" >&2 || true
  fail 'audited legacy bridge could not build Madaros from current source'
fi
require_raw_elf "$BRIDGE" 'bridge Madaros'

"$BRIDGE" --science-boundary-closure "$ROOT_DIR/self-hosted/compiler/main.sio" >"$WORK/bridge-closure.log" 2>&1 || {
  cat "$WORK/bridge-closure.log" >&2
  fail 'bridge Madaros could not emit the compiler closure report'
}
grep -Fxq $'status\tcomplete' "$WORK/bridge-closure.log" || {
  cat "$WORK/bridge-closure.log" >&2
  fail 'bridge Madaros still sees an incomplete compiler AST closure'
}
grep -Fxq $'parse_failed\tfalse' "$WORK/bridge-closure.log" || {
  cat "$WORK/bridge-closure.log" >&2
  fail 'bridge Madaros still reports parser failure in the compiler closure'
}

if ! env \
  -u MADAROS_RAW_BIN \
  -u SOUNIO_MADAROS_BIN \
  -u SOUC_BIN \
  -u SOUNIO_SOUC_BIN \
  SOUNIO_MADAROS_BOOTSTRAP_MODE=madaros-seed \
  SOUNIO_MADAROS_SEED="$BRIDGE" \
  SOUNIO_BUILD_LOCK="$WORK/souc-build.lock" \
  SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" \
  bash "$BUILD_SCRIPT" "$STAGE1" >"$WORK/stage1-build.log" 2>&1; then
  tail -n 120 "$WORK/stage1-build.log" >&2 || true
  fail 'bridge Madaros could not produce the first operational generation'
fi
require_raw_elf "$STAGE1" 'first operational Madaros generation'

if ! env \
  -u MADAROS_RAW_BIN \
  -u SOUNIO_MADAROS_BIN \
  -u SOUC_BIN \
  -u SOUNIO_SOUC_BIN \
  SOUNIO_MADAROS_BOOTSTRAP_MODE=madaros-seed \
  SOUNIO_MADAROS_SEED="$STAGE1" \
  SOUNIO_BUILD_LOCK="$WORK/souc-build.lock" \
  SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" \
  bash "$BUILD_SCRIPT" "$STAGE2" >"$WORK/stage2-build.log" 2>&1; then
  tail -n 120 "$WORK/stage2-build.log" >&2 || true
  fail 'first operational Madaros generation could not reproduce current source'
fi
require_raw_elf "$STAGE2" 'second operational Madaros generation'

LEGACY_SHA256="$(portable_sha256 "$LEGACY_BOOTSTRAP")"
BRIDGE_SHA256="$(portable_sha256 "$BRIDGE")"
STAGE1_SHA256="$(portable_sha256 "$STAGE1")"
STAGE2_SHA256="$(portable_sha256 "$STAGE2")"
[[ "$STAGE1_SHA256" == "$STAGE2_SHA256" ]] || fail "operational fixed point diverged: stage1=$STAGE1_SHA256 stage2=$STAGE2_SHA256"

MADAROS_RAW_BIN="$STAGE2" \
SOUNIO_MADAROS_IMPORTED_RUNTIME_ACCEPTANCE_EXPECTED_SHA256="$STAGE2_SHA256" \
SOUNIO_MADAROS_IMPORTED_RUNTIME_ACCEPTANCE_DIR="$WORK/nominal-layout" \
SOUNIO_MADAROS_IMPORTED_RUNTIME_ACCEPTANCE_KEEP=1 \
  bash "$ACCEPTANCE_GATE" >"$WORK/nominal-layout.log" 2>&1 || {
    cat "$WORK/nominal-layout.log" >&2
    fail 'recovered fixed point did not preserve nominal imported layouts'
  }

MADAROS_RAW_BIN="$STAGE2" \
SOUNIO_MADAROS_STRUCT_LAYOUT_CAPACITY_EXPECT=resolved \
SOUNIO_MADAROS_STRUCT_LAYOUT_CAPACITY_DIR="$WORK/layout-capacity" \
SOUNIO_MADAROS_STRUCT_LAYOUT_CAPACITY_KEEP=1 \
  bash "$CAPACITY_GATE" >"$WORK/layout-capacity.log" 2>&1 || {
    cat "$WORK/layout-capacity.log" >&2
  fail 'recovered fixed point did not cross the 256/257 layout boundary'
}

MADAROS_RAW_BIN="$STAGE2" \
SOUNIO_MADAROS_SCOPE_CONTEXTUAL_EXPECTED_SHA256="$STAGE2_SHA256" \
SOUNIO_MADAROS_SCOPE_CONTEXTUAL_DIR="$WORK/contextual-scope" \
SOUNIO_MADAROS_SCOPE_CONTEXTUAL_KEEP=1 \
  bash "$SCOPE_GATE" >"$WORK/contextual-scope.log" 2>&1 || {
    cat "$WORK/contextual-scope.log" >&2
    fail 'recovered fixed point did not preserve contextual scope bindings'
  }

MADAROS_RAW_BIN="$STAGE2" \
SOUNIO_MADAROS_CONTEXTUAL_BINDING_KIND=policy \
SOUNIO_MADAROS_SCOPE_CONTEXTUAL_EXPECTED_SHA256="$STAGE2_SHA256" \
SOUNIO_MADAROS_SCOPE_CONTEXTUAL_DIR="$WORK/contextual-policy" \
SOUNIO_MADAROS_SCOPE_CONTEXTUAL_KEEP=1 \
  bash "$SCOPE_GATE" >"$WORK/contextual-policy.log" 2>&1 || {
    cat "$WORK/contextual-policy.log" >&2
    fail 'recovered fixed point did not preserve contextual policy bindings'
  }

MADAROS_RAW_BIN="$STAGE2" \
SOUNIO_MADAROS_CONTEXTUAL_BINDING_KIND=is \
SOUNIO_MADAROS_SCOPE_CONTEXTUAL_EXPECTED_SHA256="$STAGE2_SHA256" \
SOUNIO_MADAROS_SCOPE_CONTEXTUAL_DIR="$WORK/contextual-is" \
SOUNIO_MADAROS_SCOPE_CONTEXTUAL_KEEP=1 \
  bash "$SCOPE_GATE" >"$WORK/contextual-is.log" 2>&1 || {
    cat "$WORK/contextual-is.log" >&2
    fail 'recovered fixed point did not preserve contextual is bindings'
  }

MADAROS_RAW_BIN="$STAGE2" \
SOUNIO_MADAROS_CONTEXTUAL_BINDING_KIND=study \
SOUNIO_MADAROS_SCOPE_CONTEXTUAL_EXPECTED_SHA256="$STAGE2_SHA256" \
SOUNIO_MADAROS_SCOPE_CONTEXTUAL_DIR="$WORK/contextual-study" \
SOUNIO_MADAROS_SCOPE_CONTEXTUAL_KEEP=1 \
  bash "$SCOPE_GATE" >"$WORK/contextual-study.log" 2>&1 || {
    cat "$WORK/contextual-study.log" >&2
    fail 'recovered fixed point did not preserve contextual study bindings'
  }

printf 'receipt_version\tmadaros-bootstrap-recovery-901-v1\n' >"$RECEIPT"
printf 'source_commit\t%s\n' "$SOURCE_COMMIT" >>"$RECEIPT"
printf 'bootstrap_mode\tquarantined-lean-audit-bridge\n' >>"$RECEIPT"
printf 'legacy_bootstrap_sha256\t%s\n' "$LEGACY_SHA256" >>"$RECEIPT"
printf 'bridge_madaros_sha256\t%s\n' "$BRIDGE_SHA256" >>"$RECEIPT"
printf 'stage1_madaros_sha256\t%s\n' "$STAGE1_SHA256" >>"$RECEIPT"
printf 'stage2_madaros_sha256\t%s\n' "$STAGE2_SHA256" >>"$RECEIPT"
printf 'operational_fixed_point\tsha256-stage1-equals-stage2\n' >>"$RECEIPT"
printf 'nominal_layout_gate\tdirect-raw-elf-pass\n' >>"$RECEIPT"
printf 'layout_capacity_gate\tdirect-raw-elf-resolved-pass\n' >>"$RECEIPT"
printf 'contextual_scope_gate\tdirect-raw-elf-pass\n' >>"$RECEIPT"
printf 'contextual_policy_gate\tdirect-raw-elf-pass\n' >>"$RECEIPT"
printf 'contextual_is_gate\tdirect-raw-elf-pass\n' >>"$RECEIPT"
printf 'contextual_study_gate\tdirect-raw-elf-pass\n' >>"$RECEIPT"
printf 'promotion\trequires-separate-reviewed-tracked-seed-update\n' >>"$RECEIPT"

cat "$RECEIPT"
echo "[madaros-bootstrap-recovery] PASS: bridge=$BRIDGE_SHA256 fixed_point=$STAGE2_SHA256 receipt=$RECEIPT"
