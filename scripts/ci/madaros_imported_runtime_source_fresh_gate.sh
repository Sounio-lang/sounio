#!/usr/bin/env bash
# #901: bind imported-layout runtime acceptance to a compiler built in this tree.

set -euo pipefail
export LC_ALL=C

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_SCRIPT="$ROOT_DIR/scripts/ci/build_modular_madaros.sh"
ACCEPTANCE_GATE="$ROOT_DIR/scripts/ci/madaros_imported_runtime_acceptance_gate.sh"
CAPACITY_GATE="$ROOT_DIR/scripts/ci/madaros_struct_layout_capacity_gate.sh"
SCOPE_GATE="$ROOT_DIR/scripts/ci/madaros_scope_contextual_binding_gate.sh"
KEEP_WORK="${SOUNIO_MADAROS_ISSUE901_SOURCE_FRESH_KEEP:-0}"
OPERATIONAL_SEED="${SOUNIO_MADAROS_ISSUE901_OPERATIONAL_SEED:-$ROOT_DIR/bin/madaros-linux-x86_64}"

fail() {
  echo "[madaros-issue901-source-fresh] FAIL: $*" >&2
  exit 1
}

portable_sha256() {
  sha256sum "$1" 2>/dev/null | awk '{print $1}' || shasum -a 256 "$1" | awk '{print $1}'
}

require_raw_elf() {
  local path="$1"
  [[ -x "$path" && -s "$path" ]] || fail "expected raw Madaros ELF is missing, empty, or not executable: $path"
  [[ "$(head -c4 "$path" 2>/dev/null)" == $'\x7fELF' ]] || fail "expected raw Madaros ELF is not an ELF: $path"
}

require_operational_madaros_seed() {
  local path="$1"
  local banner
  require_raw_elf "$path"
  banner="$("$path" --version 2>&1 || true)"
  [[ "$banner" == *"Madaros"* ]] || fail "operational seed does not identify as Madaros: $path"
}

if [[ "${1:-}" == '--structural-only' ]]; then
  [[ $# -eq 1 ]] || fail 'usage: madaros_imported_runtime_source_fresh_gate.sh [--structural-only]'
  [[ -x "$BUILD_SCRIPT" ]] || fail "missing source build script: $BUILD_SCRIPT"
  [[ -x "$ACCEPTANCE_GATE" ]] || fail "missing raw-ELF acceptance gate: $ACCEPTANCE_GATE"
  [[ -x "$CAPACITY_GATE" ]] || fail "missing direct raw layout-capacity gate: $CAPACITY_GATE"
  [[ -x "$SCOPE_GATE" ]] || fail "missing direct raw contextual-scope gate: $SCOPE_GATE"
  grep -Fq -- '--native-v2-compile "$SRC" "$OUT"' "$BUILD_SCRIPT" || fail 'operational Madaros build does not use its explicit native-v2 compile contract'
  [[ -f "$ROOT_DIR/self-hosted/compiler/main.sio" ]] || fail 'missing modular compiler entry source'
  [[ -x "$ROOT_DIR/bin/madaros-linux-x86_64" ]] || fail 'missing tracked operational Madaros seed'
  [[ -f "$ROOT_DIR/tests/compiler/madaros_imported_runtime_acceptance/issue_901_nested_field_chain_main.sio" ]] || fail 'missing #901 positive witness'
  [[ -f "$ROOT_DIR/tests/compiler/madaros_imported_runtime_acceptance/issue_901_known_layout_miss_main.sio" ]] || fail 'missing #901 negative witness'
  echo '[madaros-issue901-source-fresh] PASS: structural source-build and direct-raw acceptance wiring is present'
  exit 0
fi
[[ $# -eq 0 ]] || fail 'usage: madaros_imported_runtime_source_fresh_gate.sh [--structural-only]'

cd "$ROOT_DIR"
[[ -z "$(git status --porcelain)" ]] || fail 'source tree must be clean; commit the source under test before claiming source-fresh evidence'
require_operational_madaros_seed "$OPERATIONAL_SEED"
[[ "$OPERATIONAL_SEED" == "$ROOT_DIR/"* ]] || fail "operational Madaros seed must be tracked in the source tree: $OPERATIONAL_SEED"

if [[ -n "${SOUNIO_MADAROS_ISSUE901_SOURCE_FRESH_DIR:-}" ]]; then
  WORK="$SOUNIO_MADAROS_ISSUE901_SOURCE_FRESH_DIR"
  [[ ! -e "$WORK" ]] || fail "refusing existing gate directory: $WORK"
  mkdir -p "$WORK"
else
  WORK="$(mktemp -d /tmp/sounio-madaros-issue901-source-fresh.XXXXXX)"
fi
if [[ "$KEEP_WORK" != '1' ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

SOURCE_HEAD="$(git rev-parse HEAD)"
SOURCE_TREE="$(git rev-parse 'HEAD^{tree}')"
MAIN_SHA256="$(portable_sha256 self-hosted/compiler/main.sio)"
LEAN_SHA256="$(portable_sha256 self-hosted/compiler/lean_single.sio)"
BUILDER_SHA256="$(portable_sha256 "$BUILD_SCRIPT")"
SEED_SHA256="$(portable_sha256 "$OPERATIONAL_SEED")"
SEED_REPO_PATH="${OPERATIONAL_SEED#"$ROOT_DIR/"}"
SEED_GIT_BLOB="$(git ls-files -s -- "$SEED_REPO_PATH" | awk 'NR == 1 {print $2}')"
[[ -n "$SEED_GIT_BLOB" ]] || fail "operational Madaros seed is not tracked by this source tree: $SEED_REPO_PATH"
RAW_MADAROS="$WORK/madaros"
FIXED_POINT_MADAROS="$WORK/madaros-fixed-point"
BUILD_LOG="$WORK/source-build.log"
FIXED_POINT_BUILD_LOG="$WORK/fixed-point-build.log"
RECEIPT="$WORK/source-build-receipt.tsv"

if ! env \
  -u MADAROS_RAW_BIN \
  -u SOUNIO_MADAROS_BIN \
  -u SOUC_BIN \
  -u SOUNIO_SOUC_BIN \
  SOUNIO_MADAROS_BOOTSTRAP_MODE=madaros-seed \
  SOUNIO_MADAROS_SEED="$OPERATIONAL_SEED" \
  SOUNIO_BUILD_LOCK="$WORK/souc-build.lock" \
  SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" \
  bash "$BUILD_SCRIPT" "$RAW_MADAROS" >"$BUILD_LOG" 2>&1; then
  tail -n 120 "$BUILD_LOG" >&2 || true
  fail 'current-source Madaros build failed'
fi

[[ -z "$(git status --porcelain)" ]] || fail 'source tree changed while building the claimed current-source compiler'
[[ "$(git rev-parse HEAD)" == "$SOURCE_HEAD" ]] || fail 'source HEAD changed while building the claimed compiler'
[[ "$(git rev-parse 'HEAD^{tree}')" == "$SOURCE_TREE" ]] || fail 'source tree changed while building the claimed compiler'
require_raw_elf "$RAW_MADAROS"

# A source build is operationally useful, but it becomes a self-hosting claim
# only after the result can rebuild the same compiler without drift.
[[ -z "$(git status --porcelain)" ]] || fail 'source tree changed before operational fixed-point rebuild'
[[ "$(git rev-parse HEAD)" == "$SOURCE_HEAD" ]] || fail 'source HEAD changed before operational fixed-point rebuild'
[[ "$(git rev-parse 'HEAD^{tree}')" == "$SOURCE_TREE" ]] || fail 'source tree changed before operational fixed-point rebuild'
if ! env \
  -u MADAROS_RAW_BIN \
  -u SOUNIO_MADAROS_BIN \
  -u SOUC_BIN \
  -u SOUNIO_SOUC_BIN \
  SOUNIO_MADAROS_BOOTSTRAP_MODE=madaros-seed \
  SOUNIO_MADAROS_SEED="$RAW_MADAROS" \
  SOUNIO_BUILD_LOCK="$WORK/souc-build.lock" \
  SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" \
  bash "$BUILD_SCRIPT" "$FIXED_POINT_MADAROS" >"$FIXED_POINT_BUILD_LOG" 2>&1; then
  tail -n 120 "$FIXED_POINT_BUILD_LOG" >&2 || true
  fail 'first-generation Madaros could not rebuild current source'
fi

require_raw_elf "$FIXED_POINT_MADAROS"
STAGE1_SHA256="$(portable_sha256 "$RAW_MADAROS")"
STAGE2_SHA256="$(portable_sha256 "$FIXED_POINT_MADAROS")"
[[ "$STAGE1_SHA256" == "$STAGE2_SHA256" ]] || fail "operational Madaros fixed point diverged: stage1=$STAGE1_SHA256 stage2=$STAGE2_SHA256"

printf 'receipt_version\tissue901-source-fresh-v3\n' >"$RECEIPT"
printf 'source_head\t%s\n' "$SOURCE_HEAD" >>"$RECEIPT"
printf 'source_tree\t%s\n' "$SOURCE_TREE" >>"$RECEIPT"
printf 'main_sio_sha256\t%s\n' "$MAIN_SHA256" >>"$RECEIPT"
printf 'lean_single_sio_sha256\t%s\n' "$LEAN_SHA256" >>"$RECEIPT"
printf 'build_script_sha256\t%s\n' "$BUILDER_SHA256" >>"$RECEIPT"
printf 'bootstrap_mode\tmadaros-operational-seed\n' >>"$RECEIPT"
printf 'bootstrap_seed_repo_path\t%s\n' "$SEED_REPO_PATH" >>"$RECEIPT"
printf 'bootstrap_seed_git_blob\t%s\n' "$SEED_GIT_BLOB" >>"$RECEIPT"
printf 'bootstrap_seed_sha256\t%s\n' "$SEED_SHA256" >>"$RECEIPT"
printf 'stage1_madaros_sha256\t%s\n' "$STAGE1_SHA256" >>"$RECEIPT"
printf 'stage2_madaros_sha256\t%s\n' "$STAGE2_SHA256" >>"$RECEIPT"
printf 'operational_fixed_point\tsha256-stage1-equals-stage2\n' >>"$RECEIPT"
printf 'acceptance_mode\tdirect-raw-elf-no-wrapper\n' >>"$RECEIPT"

MADAROS_RAW_BIN="$FIXED_POINT_MADAROS" \
SOUNIO_MADAROS_IMPORTED_RUNTIME_ACCEPTANCE_EXPECTED_SHA256="$STAGE2_SHA256" \
SOUNIO_MADAROS_IMPORTED_RUNTIME_ACCEPTANCE_DIR="$WORK/acceptance" \
SOUNIO_MADAROS_IMPORTED_RUNTIME_ACCEPTANCE_KEEP=1 \
  bash "$ACCEPTANCE_GATE" >"$WORK/acceptance.log" 2>&1 || {
    cat "$WORK/acceptance.log" >&2
    fail 'source-built Madaros did not satisfy the direct raw #901 acceptance gate'
  }

MADAROS_RAW_BIN="$FIXED_POINT_MADAROS" \
SOUNIO_MADAROS_STRUCT_LAYOUT_CAPACITY_EXPECT=resolved \
SOUNIO_MADAROS_STRUCT_LAYOUT_CAPACITY_DIR="$WORK/capacity" \
SOUNIO_MADAROS_STRUCT_LAYOUT_CAPACITY_KEEP=1 \
  bash "$CAPACITY_GATE" >"$WORK/capacity.log" 2>&1 || {
    cat "$WORK/capacity.log" >&2
    fail 'source-built Madaros did not satisfy the direct raw layout-capacity gate'
}
printf 'layout_capacity_mode\tdirect-raw-elf-resolved\n' >>"$RECEIPT"

MADAROS_RAW_BIN="$FIXED_POINT_MADAROS" \
SOUNIO_MADAROS_SCOPE_CONTEXTUAL_EXPECTED_SHA256="$STAGE2_SHA256" \
SOUNIO_MADAROS_SCOPE_CONTEXTUAL_DIR="$WORK/contextual-scope" \
SOUNIO_MADAROS_SCOPE_CONTEXTUAL_KEEP=1 \
  bash "$SCOPE_GATE" >"$WORK/contextual-scope.log" 2>&1 || {
    cat "$WORK/contextual-scope.log" >&2
    fail 'source-built Madaros did not preserve contextual scope bindings'
  }
printf 'contextual_scope_mode\tdirect-raw-elf\n' >>"$RECEIPT"

MADAROS_RAW_BIN="$FIXED_POINT_MADAROS" \
SOUNIO_MADAROS_CONTEXTUAL_BINDING_KIND=policy \
SOUNIO_MADAROS_SCOPE_CONTEXTUAL_EXPECTED_SHA256="$STAGE2_SHA256" \
SOUNIO_MADAROS_SCOPE_CONTEXTUAL_DIR="$WORK/contextual-policy" \
SOUNIO_MADAROS_SCOPE_CONTEXTUAL_KEEP=1 \
  bash "$SCOPE_GATE" >"$WORK/contextual-policy.log" 2>&1 || {
    cat "$WORK/contextual-policy.log" >&2
    fail 'source-built Madaros did not preserve contextual policy bindings'
  }
printf 'contextual_policy_mode\tdirect-raw-elf\n' >>"$RECEIPT"

MADAROS_RAW_BIN="$FIXED_POINT_MADAROS" \
SOUNIO_MADAROS_CONTEXTUAL_BINDING_KIND=is \
SOUNIO_MADAROS_SCOPE_CONTEXTUAL_EXPECTED_SHA256="$STAGE2_SHA256" \
SOUNIO_MADAROS_SCOPE_CONTEXTUAL_DIR="$WORK/contextual-is" \
SOUNIO_MADAROS_SCOPE_CONTEXTUAL_KEEP=1 \
  bash "$SCOPE_GATE" >"$WORK/contextual-is.log" 2>&1 || {
    cat "$WORK/contextual-is.log" >&2
    fail 'source-built Madaros did not preserve contextual is bindings'
  }
printf 'contextual_is_mode\tdirect-raw-elf\n' >>"$RECEIPT"

MADAROS_RAW_BIN="$FIXED_POINT_MADAROS" \
SOUNIO_MADAROS_CONTEXTUAL_BINDING_KIND=study \
SOUNIO_MADAROS_SCOPE_CONTEXTUAL_EXPECTED_SHA256="$STAGE2_SHA256" \
SOUNIO_MADAROS_SCOPE_CONTEXTUAL_DIR="$WORK/contextual-study" \
SOUNIO_MADAROS_SCOPE_CONTEXTUAL_KEEP=1 \
  bash "$SCOPE_GATE" >"$WORK/contextual-study.log" 2>&1 || {
    cat "$WORK/contextual-study.log" >&2
    fail 'source-built Madaros did not preserve contextual study bindings'
  }
printf 'contextual_study_mode\tdirect-raw-elf\n' >>"$RECEIPT"

[[ -z "$(git status --porcelain)" ]] || fail 'source tree changed during direct raw #901 acceptance'
[[ "$(git rev-parse HEAD)" == "$SOURCE_HEAD" ]] || fail 'source HEAD changed during direct raw #901 acceptance'
[[ "$(git rev-parse 'HEAD^{tree}')" == "$SOURCE_TREE" ]] || fail 'source tree changed during direct raw #901 acceptance'

cat "$RECEIPT"
echo "[madaros-issue901-source-fresh] PASS: source_head=$SOURCE_HEAD source_tree=$SOURCE_TREE stage1_sha256=$STAGE1_SHA256 stage2_sha256=$STAGE2_SHA256 receipt=$RECEIPT"
