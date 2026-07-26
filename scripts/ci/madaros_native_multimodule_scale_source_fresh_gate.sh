#!/usr/bin/env bash
# Bind the filed #901 scale witness to a direct, current-source Madaros ELF.
#
# The historical scale gate intentionally exercises the public `bin/souc`
# route. This gate is stricter: it builds Madaros from this clean tree, reaches
# a Madaros-to-Madaros fixed point, and invokes that raw ELF directly.

set -euo pipefail
export LC_ALL=C

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_SCRIPT="$ROOT_DIR/scripts/ci/build_modular_madaros.sh"
KEEP_WORK="${SOUNIO_MADAROS_ISSUE901_SCALE_SOURCE_FRESH_KEEP:-0}"

fail() {
  echo "[madaros-issue901-scale-source-fresh] FAIL: $*" >&2
  exit 1
}

portable_sha256() {
  sha256sum "$1" 2>/dev/null | awk '{print $1}' || shasum -a 256 "$1" | awk '{print $1}'
}

require_raw_madaros() {
  local path="$1"
  local banner

  [[ -x "$path" && -s "$path" ]] || fail "Madaros ELF is missing, empty, or not executable: $path"
  [[ "$(head -c4 "$path" 2>/dev/null)" == $'\x7fELF' ]] || fail "Madaros output is not an ELF: $path"
  banner="$("$path" --version 2>&1 || true)"
  [[ "$banner" == *Madaros* ]] || fail "raw compiler does not identify as Madaros: $path"
}

require_clean_source() {
  [[ -z "$(git -C "$ROOT_DIR" status --porcelain)" ]] || fail 'source tree must be clean; commit the source under test before source-fresh evidence'
}

assert_direct_raw_log() {
  local log="$1"

  if grep -Eiq \
    'lean_single|falling back to the legacy|native_prebundle:|SELFHOST=fallback|compatibility fallback|multimodule native thin-link compilation failed|imported_simple_ir_emit_failed' \
    "$log"; then
    cat "$log" >&2
    fail "direct raw #901 compile observed a forbidden fallback or historical scale failure: $log"
  fi
}

if [[ "${1:-}" == '--structural-only' ]]; then
  [[ $# -eq 1 ]] || fail 'usage: madaros_native_multimodule_scale_source_fresh_gate.sh [--structural-only]'
  [[ -x "$BUILD_SCRIPT" ]] || fail "missing modular source-build script: $BUILD_SCRIPT"
  [[ -f "$ROOT_DIR/tests/run-pass/madaros_native_multimodule_scale_prob.sio" ]] || fail 'missing #901 acceptance probe'
  [[ -f "$ROOT_DIR/tests/run-pass/madaros_native_multimodule_scale_prob_textbook.sio" ]] || fail 'missing #901 textbook probe'
  [[ -f "$ROOT_DIR/tests/stdlib/prob/test_prob_stdlib.sio" ]] || fail 'missing #901 stdlib driver'
  echo '[madaros-issue901-scale-source-fresh] PASS: source-build and direct-raw scale wiring is present'
  exit 0
fi
[[ $# -eq 0 ]] || fail 'usage: madaros_native_multimodule_scale_source_fresh_gate.sh [--structural-only]'

cd "$ROOT_DIR"
require_clean_source

if [[ -n "${SOUNIO_MADAROS_ISSUE901_SCALE_SOURCE_FRESH_DIR:-}" ]]; then
  WORK="$SOUNIO_MADAROS_ISSUE901_SCALE_SOURCE_FRESH_DIR"
  [[ ! -e "$WORK" ]] || fail "refusing existing gate directory: $WORK"
  mkdir -p "$WORK"
else
  WORK="$(mktemp -d /tmp/sounio-madaros-issue901-scale-source-fresh.XXXXXX)"
fi
if [[ "$KEEP_WORK" != '1' ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

SOURCE_HEAD="$(git rev-parse HEAD)"
SOURCE_TREE="$(git rev-parse 'HEAD^{tree}')"
MAIN_SHA256="$(portable_sha256 self-hosted/compiler/main.sio)"
LEAN_SHA256="$(portable_sha256 self-hosted/compiler/lean_single.sio)"
BUILDER_SHA256="$(portable_sha256 "$BUILD_SCRIPT")"
BOOTSTRAP_PATH="$ROOT_DIR/bin/souc-linux-x86_64"
[[ -x "$BOOTSTRAP_PATH" ]] || fail "tracked initial bootstrap is missing: $BOOTSTRAP_PATH"
BOOTSTRAP_SHA256="$(portable_sha256 "$BOOTSTRAP_PATH")"
BOOTSTRAP_BLOB="$(git ls-files -s -- bin/souc-linux-x86_64 | awk 'NR == 1 {print $2}')"
[[ -n "$BOOTSTRAP_BLOB" ]] || fail 'initial bootstrap is not tracked by this source tree'

STAGE1="$WORK/madaros-stage1"
STAGE2="$WORK/madaros-stage2"
STAGE3="$WORK/madaros-stage3"
RECEIPT="$WORK/madaros_native_multimodule_scale_901_source_fresh_receipt.tsv"

# Stage 1 derives the current lean seed from the tracked initial bootstrap, then
# compiles main.sio. Stages 2 and 3 intentionally use Madaros as their seed.
if ! env \
  -u MADAROS_RAW_BIN \
  -u SOUNIO_MADAROS_BIN \
  -u SOUNIO_SOUC_ENGINE \
  -u SOUNIO_ENABLE_COMPACT_IMPORTED_IR \
  -u SOUNIO_MADAROS_DEP_MERGE \
  -u SOUNIO_INTO_ACC_NO_RESET \
  -u SOUC_BIN \
  -u SOUNIO_SOUC_BIN \
  SOUNIO_BUILD_LOCK="$WORK/souc-build.lock" \
  SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" \
  bash "$BUILD_SCRIPT" "$STAGE1" >"$WORK/stage1-build.log" 2>&1; then
  tail -n 120 "$WORK/stage1-build.log" >&2 || true
  fail 'stage1 source build failed'
fi
require_raw_madaros "$STAGE1"

if ! env \
  -u MADAROS_RAW_BIN \
  -u SOUNIO_MADAROS_BIN \
  -u SOUNIO_SOUC_ENGINE \
  -u SOUNIO_ENABLE_COMPACT_IMPORTED_IR \
  -u SOUNIO_MADAROS_DEP_MERGE \
  -u SOUNIO_INTO_ACC_NO_RESET \
  SOUC_BIN="$STAGE1" \
  SOUNIO_BUILD_LOCK="$WORK/souc-build.lock" \
  SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" \
  bash "$BUILD_SCRIPT" "$STAGE2" >"$WORK/stage2-build.log" 2>&1; then
  tail -n 120 "$WORK/stage2-build.log" >&2 || true
  fail 'Madaros stage1 could not rebuild current source'
fi
require_raw_madaros "$STAGE2"

if ! env \
  -u MADAROS_RAW_BIN \
  -u SOUNIO_MADAROS_BIN \
  -u SOUNIO_SOUC_ENGINE \
  -u SOUNIO_ENABLE_COMPACT_IMPORTED_IR \
  -u SOUNIO_MADAROS_DEP_MERGE \
  -u SOUNIO_INTO_ACC_NO_RESET \
  SOUC_BIN="$STAGE2" \
  SOUNIO_BUILD_LOCK="$WORK/souc-build.lock" \
  SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" \
  bash "$BUILD_SCRIPT" "$STAGE3" >"$WORK/stage3-build.log" 2>&1; then
  tail -n 120 "$WORK/stage3-build.log" >&2 || true
  fail 'Madaros stage2 could not rebuild current source'
fi
require_raw_madaros "$STAGE3"

require_clean_source
[[ "$(git rev-parse HEAD)" == "$SOURCE_HEAD" ]] || fail 'source HEAD changed during fixed-point build'
[[ "$(git rev-parse 'HEAD^{tree}')" == "$SOURCE_TREE" ]] || fail 'source tree changed during fixed-point build'

STAGE1_SHA256="$(portable_sha256 "$STAGE1")"
STAGE2_SHA256="$(portable_sha256 "$STAGE2")"
STAGE3_SHA256="$(portable_sha256 "$STAGE3")"
[[ "$STAGE2_SHA256" == "$STAGE3_SHA256" ]] || fail "Madaros fixed point diverged: stage2=$STAGE2_SHA256 stage3=$STAGE3_SHA256"

PROBE="$ROOT_DIR/tests/run-pass/madaros_native_multimodule_scale_prob.sio"
TEXTBOOK="$ROOT_DIR/tests/run-pass/madaros_native_multimodule_scale_prob_textbook.sio"
DRIVER="$ROOT_DIR/tests/stdlib/prob/test_prob_stdlib.sio"
for source in "$PROBE" "$TEXTBOOK" "$DRIVER"; do
  [[ -f "$source" ]] || fail "required #901 source is missing: $source"
done

CASE_MERGED=""
run_direct_case() {
  local label="$1"
  local source="$2"
  local expected="$3"
  local case_dir="$WORK/$label"
  local elf="$case_dir/$label.elf"
  local compile_log="$case_dir/compile.log"
  local run_log="$case_dir/run.log"

  mkdir -p "$case_dir"
  if ! (
    cd "$case_dir"
    exec env \
      -u MADAROS_RAW_BIN \
      -u SOUNIO_MADAROS_BIN \
      -u SOUNIO_SOUC_ENGINE \
      -u SOUNIO_ENABLE_COMPACT_IMPORTED_IR \
      -u SOUNIO_MADAROS_DEP_MERGE \
      -u SOUNIO_INTO_ACC_NO_RESET \
      -u SOUC_BIN \
      -u SOUNIO_SOUC_BIN \
      SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" \
      "$STAGE3" --native-v2-compile "$source" "$elf"
  ) >"$compile_log" 2>&1; then
    cat "$compile_log" >&2 || true
    fail "direct raw compile failed for $label"
  fi
  assert_direct_raw_log "$compile_log"
  [[ -s "$elf" ]] || fail "direct raw compile emitted no ELF for $label"
  [[ "$(head -c4 "$elf" 2>/dev/null)" == $'\x7fELF' ]] || fail "direct raw output is not an ELF for $label"
  chmod +x "$elf"
  if ! "$elf" >"$run_log" 2>&1; then
    cat "$run_log" >&2 || true
    fail "direct raw ELF exited nonzero for $label"
  fi
  grep -Eq "$expected" "$run_log" || {
    cat "$run_log" >&2 || true
    fail "direct raw ELF stdout missed expected witness for $label: /$expected/"
  }
  CASE_MERGED="$(awk '/Merged IR:/{n=$NF} END{print n}' "$compile_log" 2>/dev/null || true)"
  printf '[madaros-issue901-scale-source-fresh] PASS: %s merged_ir=%s\n' "$label" "${CASE_MERGED:-unknown}"
}

run_direct_case acceptance-probe "$PROBE" 'm=5(\.0+)?'
PROBE_MERGED="$CASE_MERGED"
run_direct_case textbook "$TEXTBOOK" 'PROB_TEXTBOOK_OK'
TEXTBOOK_MERGED="$CASE_MERGED"
run_direct_case stdlib-driver "$DRIVER" 'PROB_STDLIB_OK'
DRIVER_MERGED="$CASE_MERGED"

run_public_default_case() {
  local source="$1"
  local expected="$2"
  local case_dir="$WORK/public-default"
  local elf="$case_dir/public-default.elf"
  local compile_log="$case_dir/compile.log"
  local run_log="$case_dir/run.log"

  mkdir -p "$case_dir"
  if ! (
    cd "$case_dir"
    exec env \
      -u SOUNIO_MADAROS_BIN \
      -u SOUNIO_ENABLE_COMPACT_IMPORTED_IR \
      -u SOUNIO_MADAROS_DEP_MERGE \
      -u SOUNIO_INTO_ACC_NO_RESET \
      -u SOUC_BIN \
      -u SOUNIO_SOUC_BIN \
      MADAROS_RAW_BIN="$STAGE3" \
      SOUNIO_SOUC_ENGINE=madaros \
      SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" \
      "$ROOT_DIR/bin/souc" compile "$source" -o "$elf"
  ) >"$compile_log" 2>&1; then
    cat "$compile_log" >&2 || true
    fail 'public default Madaros compile failed for the #901 acceptance probe'
  fi
  assert_direct_raw_log "$compile_log"
  [[ -s "$elf" ]] || fail 'public default Madaros compile emitted no ELF for the #901 acceptance probe'
  [[ "$(head -c4 "$elf" 2>/dev/null)" == $'\x7fELF' ]] || fail 'public default Madaros output is not an ELF for the #901 acceptance probe'
  chmod +x "$elf"
  if ! "$elf" >"$run_log" 2>&1; then
    cat "$run_log" >&2 || true
    fail 'public default Madaros ELF exited nonzero for the #901 acceptance probe'
  fi
  grep -Eq "$expected" "$run_log" || {
    cat "$run_log" >&2 || true
    fail "public default Madaros stdout missed the #901 witness: /$expected/"
  }
  CASE_MERGED="$(awk '/Merged IR:/{n=$NF} END{print n}' "$compile_log" 2>/dev/null || true)"
  printf '[madaros-issue901-scale-source-fresh] PASS: public-default merged_ir=%s\n' "${CASE_MERGED:-unknown}"
}

run_public_default_case "$PROBE" 'm=5(\.0+)?'
PUBLIC_DEFAULT_MERGED="$CASE_MERGED"

require_clean_source
[[ "$(git rev-parse HEAD)" == "$SOURCE_HEAD" ]] || fail 'source HEAD changed during direct raw scale acceptance'
[[ "$(git rev-parse 'HEAD^{tree}')" == "$SOURCE_TREE" ]] || fail 'source tree changed during direct raw scale acceptance'

printf 'receipt_version\tissue901-scale-source-fresh-v1\n' >"$RECEIPT"
printf 'source_head\t%s\n' "$SOURCE_HEAD" >>"$RECEIPT"
printf 'source_tree\t%s\n' "$SOURCE_TREE" >>"$RECEIPT"
printf 'main_sio_sha256\t%s\n' "$MAIN_SHA256" >>"$RECEIPT"
printf 'lean_single_sio_sha256\t%s\n' "$LEAN_SHA256" >>"$RECEIPT"
printf 'build_script_sha256\t%s\n' "$BUILDER_SHA256" >>"$RECEIPT"
printf 'initial_bootstrap_repo_path\tbin/souc-linux-x86_64\n' >>"$RECEIPT"
printf 'initial_bootstrap_git_blob\t%s\n' "$BOOTSTRAP_BLOB" >>"$RECEIPT"
printf 'initial_bootstrap_sha256\t%s\n' "$BOOTSTRAP_SHA256" >>"$RECEIPT"
printf 'bootstrap_mode\tsource-tracking-lean-then-madaros-fixed-point\n' >>"$RECEIPT"
printf 'stage1_madaros_sha256\t%s\n' "$STAGE1_SHA256" >>"$RECEIPT"
printf 'stage2_madaros_sha256\t%s\n' "$STAGE2_SHA256" >>"$RECEIPT"
printf 'stage3_madaros_sha256\t%s\n' "$STAGE3_SHA256" >>"$RECEIPT"
printf 'operational_fixed_point\tsha256-stage2-equals-stage3\n' >>"$RECEIPT"
printf 'acceptance_mode\tdirect-raw-elf-no-wrapper\n' >>"$RECEIPT"
printf 'engine_fallback\t0\n' >>"$RECEIPT"
printf 'compact_imported_ir\t0\n' >>"$RECEIPT"
printf 'default_merge_mode\tinto-acc\n' >>"$RECEIPT"
printf 'target_resolution\tauto-x86_64-linux\n' >>"$RECEIPT"
printf 'acceptance_probe\tm=5.000000\n' >>"$RECEIPT"
printf 'acceptance_probe_merged_ir\t%s\n' "${PROBE_MERGED:-unknown}" >>"$RECEIPT"
printf 'textbook_probe\tPROB_TEXTBOOK_OK\n' >>"$RECEIPT"
printf 'textbook_probe_merged_ir\t%s\n' "${TEXTBOOK_MERGED:-unknown}" >>"$RECEIPT"
printf 'stdlib_driver\tPROB_STDLIB_OK\n' >>"$RECEIPT"
printf 'stdlib_driver_merged_ir\t%s\n' "${DRIVER_MERGED:-unknown}" >>"$RECEIPT"
printf 'public_default_route\tbin/souc-compile-pinned-to-stage3\n' >>"$RECEIPT"
printf 'public_default_probe\tm=5.000000\n' >>"$RECEIPT"
printf 'public_default_probe_merged_ir\t%s\n' "${PUBLIC_DEFAULT_MERGED:-unknown}" >>"$RECEIPT"

cat "$RECEIPT"
echo "[madaros-issue901-scale-source-fresh] PASS: source_head=$SOURCE_HEAD stage3_sha256=$STAGE3_SHA256 receipt=$RECEIPT"
