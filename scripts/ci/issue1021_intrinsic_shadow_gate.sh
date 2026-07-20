#!/usr/bin/env bash
# Focused acceptance gate for GitHub issue #1021.

set -euo pipefail
export LC_ALL=C

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE="$ROOT_DIR/self-hosted/compiler/lean_single.sio"
FIXTURE_DIR="$ROOT_DIR/tests/compiler/issue1021_intrinsic_shadow"
MODE="${1:-source-only}"
KEEP_WORK="${SOUNIO_ISSUE1021_KEEP_WORK:-0}"
COMPILE_TIMEOUT="${SOUNIO_ISSUE1021_COMPILE_TIMEOUT_SECONDS:-90}"
RUNTIME_TIMEOUT="${SOUNIO_ISSUE1021_RUNTIME_TIMEOUT_SECONDS:-15}"

case "$MODE" in
  source-only|--source-only) MODE=source-only ;;
  old-seed-negative|--old-seed-negative) MODE=old-seed-negative ;;
  source-fresh|--source-fresh) MODE=source-fresh ;;
  *) printf 'ISSUE1021_INTRINSIC_SHADOW_FAIL stage=setup reason=unknown_mode mode=%s\n' "$MODE" >&2; exit 1 ;;
esac
[[ $# -le 1 ]] || { printf 'ISSUE1021_INTRINSIC_SHADOW_FAIL stage=setup reason=unexpected_argument\n' >&2; exit 1; }

CASE_NAMES=(
  epistemic_exp
  epistemic_ln
  epistemic_pow
  epistemic_inverse_control
  f64_exp_control
  local_exp
  imported_exp
  extern_c_math
  extern_local_collision
  method_exp_overload
  scalar_free_math_control
  scalar_method_math_control
  forward_exp
  recursive_exp
  generic_exp
)
CASE_SOURCES=(
  "$FIXTURE_DIR/epistemic_exp.sio"
  "$FIXTURE_DIR/epistemic_ln.sio"
  "$FIXTURE_DIR/epistemic_pow.sio"
  "$FIXTURE_DIR/epistemic_inverse_control.sio"
  "$FIXTURE_DIR/f64_exp_control.sio"
  "$FIXTURE_DIR/local_exp.sio"
  "$FIXTURE_DIR/imported_exp_main.sio"
  "$FIXTURE_DIR/extern_c_math.sio"
  "$FIXTURE_DIR/extern_local_collision.sio"
  "$FIXTURE_DIR/method_exp_overload.sio"
  "$FIXTURE_DIR/scalar_free_math_control.sio"
  "$FIXTURE_DIR/scalar_method_math_control.sio"
  "$FIXTURE_DIR/forward_exp.sio"
  "$FIXTURE_DIR/recursive_exp.sio"
  "$FIXTURE_DIR/generic_exp.sio"
)
CASE_MARKERS=(
  ISSUE1021_EPISTEMIC_EXP_OK
  ISSUE1021_EPISTEMIC_LN_OK
  ISSUE1021_EPISTEMIC_POW_OK
  ISSUE1021_EPISTEMIC_INVERSE_CONTROL_OK
  ISSUE1021_F64_EXP_CONTROL_OK
  ISSUE1021_LOCAL_EXP_OK
  ISSUE1021_IMPORTED_EXP_OK
  ISSUE1021_EXTERN_C_MATH_OK
  ISSUE1021_EXTERN_LOCAL_COLLISION_OK
  ISSUE1021_METHOD_EXP_OVERLOAD_OK
  ISSUE1021_SCALAR_FREE_MATH_CONTROL_OK
  ISSUE1021_SCALAR_METHOD_MATH_CONTROL_OK
  ISSUE1021_FORWARD_EXP_OK
  ISSUE1021_RECURSIVE_EXP_OK
  ISSUE1021_GENERIC_EXP_OK
)
WRONG_NAMESPACE_SOURCE="$FIXTURE_DIR/wrong_namespace_exp.sio"
WRONG_NAMESPACE_DIAGNOSTIC='error: unknown identifier `wrong_namespace` at '

fail() {
  local stage="$1"
  local reason="$2"
  printf 'ISSUE1021_INTRINSIC_SHADOW_FAIL mode=%s stage=%s reason=%s\n' "$MODE" "$stage" "$reason" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail setup "missing_command_$1"
}

artifact_state() {
  local path="$1"
  if [[ -e "$path" || -L "$path" ]]; then
    printf 'present\n'
  else
    printf 'absent\n'
  fi
}

elf_magic() {
  od -An -tx1 -N4 "$1" 2>/dev/null | tr -d ' \n'
}

elf_u16() {
  od -An -tu2 -j "$2" -N2 "$1" 2>/dev/null | tr -d ' \n'
}

assert_elf() {
  local stage="$1"
  local path="$2"
  local machine="$3"

  [[ -f "$path" && -s "$path" ]] || fail "$stage" elf_missing
  [[ "$(elf_magic "$path")" == 7f454c46 ]] || fail "$stage" artifact_not_elf
  [[ "$(elf_u16 "$path" 16)" == 2 ]] || fail "$stage" elf_not_et_exec
  [[ "$(elf_u16 "$path" 18)" == "$machine" ]] || fail "$stage" elf_machine_mismatch
}

assert_no_fallback_log() {
  local stage="$1"
  local log="$2"
  if grep -Eiq 'falling back|fallback backend|fallback path' "$log"; then
    fail "$stage" fallback_observed
  fi
}

show_tail() {
  local log="$1"
  [[ -f "$log" ]] && tail -n 80 "$log" >&2 || true
}

assert_source_contract() {
  [[ -f "$SOURCE" ]] || fail source_contract compiler_source_missing
  [[ -d "$FIXTURE_DIR" ]] || fail source_contract fixture_directory_missing
  [[ -f "$FIXTURE_DIR/imported_exp_leaf.sio" ]] || fail source_contract imported_leaf_missing
  [[ -f "$WRONG_NAMESPACE_SOURCE" ]] || fail source_contract wrong_namespace_fixture_missing

  grep -Fq 'fn math_intrinsic_call_allowed(fi: i64) -> bool' "$SOURCE" || fail source_contract helper_missing
  grep -Fq 'EXTERN_STUB_GEN_NS[si as usize] == fn_ns' "$SOURCE" || fail source_contract extern_exact_start_identity_missing
  grep -Fq 'EXTERN_STUB_GEN_NE[si as usize] == fn_ne' "$SOURCE" || fail source_contract extern_exact_end_identity_missing
  grep -Fq 'EXTERN_STUB_IS_MATH[si as usize] == 1' "$SOURCE" || fail source_contract extern_math_kind_missing
  if grep -Fq 'extern_stub_matches_src' "$SOURCE"; then
    fail source_contract text_only_extern_match_present
  fi
  [[ "$(grep -Fc 'math_intrinsic_call_allowed(ident_fi)' "$SOURCE")" -eq 2 ]] || fail source_contract backend_helper_count_mismatch
  [[ "$(grep -Fc 'append_extern_math_stub_prefix(si)' "$SOURCE")" -eq 14 ]] || fail source_contract extern_math_stub_count_mismatch
  grep -Fq 'if math_intrinsic_allowed && src_match(ns, ne - ns, "atan2")' "$SOURCE" || fail source_contract x86_atan2_guard_missing
  grep -Fq 'if math_intrinsic_allowed && src_match(ns, ne - ns, "pow")' "$SOURCE" || fail source_contract x86_pow_guard_missing
  grep -Fq 'if math_intrinsic_allowed_a64 && src_match(ns, ne - ns, "abs")' "$SOURCE" || fail source_contract aarch64_abs_guard_missing
  [[ "$(grep -Fc 'declared_math_mm = fn_find_expr_method' "$SOURCE")" -eq 1 ]] || fail source_contract x86_method_shadow_guard_missing
  [[ "$(grep -Fc 'declared_math_mm_a = fn_find_expr_method' "$SOURCE")" -eq 1 ]] || fail source_contract aarch64_method_shadow_guard_missing
  grep -Fq 'fn fn_find_qualified(call_pos: i64, qns: i64, qne: i64, ns: i64, ne: i64)' "$SOURCE" || fail source_contract qualified_lookup_missing
  grep -Fq 'IMP_QUAL_NAME_BUF[(IMP_QUAL_NAME_USED + i) as usize] = SRC[(ns + i) as usize]' "$SOURCE" || fail source_contract stable_qualifier_sidecar_missing
  grep -Fq 'imp_register_qualifier_binding(use_from_source, target_source, use_qualifier_ns, use_qualifier_ne)' "$SOURCE" || fail source_contract qualified_import_registration_missing
  grep -Fq 'let math_intrinsic_allowed = call_is_qualified == 0 && math_intrinsic_call_allowed(ident_fi)' "$SOURCE" || fail source_contract x86_qualified_intrinsic_guard_missing
  grep -Fq 'let math_intrinsic_allowed_a64 = call_is_qualified_a64 == 0 && math_intrinsic_call_allowed(ident_fi)' "$SOURCE" || fail source_contract aarch64_qualified_intrinsic_guard_missing

  local i
  for i in "${!CASE_NAMES[@]}"; do
    [[ -f "${CASE_SOURCES[$i]}" ]] || fail source_contract "fixture_${CASE_NAMES[$i]}_missing"
    [[ "$(grep -Fc "${CASE_MARKERS[$i]}" "${CASE_SOURCES[$i]}")" -eq 1 ]] || \
      fail source_contract "fixture_${CASE_NAMES[$i]}_marker_mismatch"
  done
  grep -Fq 'imported_exp_leaf::exp(2.0)' "$FIXTURE_DIR/imported_exp_main.sio" || fail source_contract qualified_import_witness_missing
  grep -Fq 'fn exp(x: f64, shift: f64)' "$FIXTURE_DIR/extern_local_collision.sio" || fail source_contract extern_collision_witness_missing
  grep -Fq 'impl LeftValue' "$FIXTURE_DIR/method_exp_overload.sio" || fail source_contract method_left_overload_missing
  grep -Fq 'impl RightValue' "$FIXTURE_DIR/method_exp_overload.sio" || fail source_contract method_right_overload_missing
  grep -Fq 'wrong_namespace::exp(2.0)' "$WRONG_NAMESPACE_SOURCE" || fail source_contract wrong_namespace_collision_missing
  grep -Fq 'let value = exp::<i64>(42)' "$FIXTURE_DIR/generic_exp.sio" || fail source_contract generic_resolution_witness_missing
  grep -Fq 'n + exp(n - 1)' "$FIXTURE_DIR/recursive_exp.sio" || fail source_contract recursive_resolution_witness_missing
  grep -Fq 'let value = exp(2.0)' "$FIXTURE_DIR/forward_exp.sio" || fail source_contract forward_resolution_witness_missing
  grep -Fq 'let raised = one.exp()' "$FIXTURE_DIR/scalar_method_math_control.sio" || fail source_contract scalar_method_control_missing
}

make_workdir() {
  if [[ -n "${SOUNIO_ISSUE1021_WORK_DIR:-}" ]]; then
    WORK="$SOUNIO_ISSUE1021_WORK_DIR"
    [[ ! -e "$WORK" ]] || fail setup work_directory_already_exists
    mkdir -p "$WORK"
  else
    WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-issue1021.${MODE}.XXXXXX")"
  fi
  if [[ "$KEEP_WORK" != 1 ]]; then
    trap 'rm -rf "$WORK"' EXIT
  fi
}

resolve_compiler() {
  local path="$1"
  [[ -x "$path" ]] || fail compiler compiler_missing_or_not_executable
  COMPILER="$(cd "$(dirname "$path")" && pwd -P)/$(basename "$path")"
  assert_elf compiler "$COMPILER" 62
  COMPILER_SHA256="$(sha256sum "$COMPILER" | awk '{print $1}')"
}

run_compile() {
  local label="$1"
  local source="$2"
  local output="$3"
  local target="${4:-x86_64-linux}"
  local log="$WORK/$label.compile.log"

  rm -f "$output"
  set +e
  if [[ "$target" == x86_64-linux ]]; then
    timeout --signal=TERM --kill-after=5s "$COMPILE_TIMEOUT" \
      env -u SOUNIO_SOUC_ENGINE -u SOUNIO_SOUC_BIN \
      "$COMPILER" "$source" "$output" >"$log" 2>&1
  else
    timeout --signal=TERM --kill-after=5s "$COMPILE_TIMEOUT" \
      env -u SOUNIO_SOUC_ENGINE -u SOUNIO_SOUC_BIN \
      "$COMPILER" "$source" "$output" --target "$target" >"$log" 2>&1
  fi
  CASE_RC=$?
  set -e
  CASE_LOG="$log"
  CASE_ELF="$output"
}

assert_compile_success() {
  local stage="$1"
  local machine="$2"
  if [[ "$CASE_RC" -ne 0 ]]; then
    printf 'ISSUE1021_COMPILE_FAILURE mode=%s stage=%s rc=%s final_elf=%s\n' \
      "$MODE" "$stage" "$CASE_RC" "$(artifact_state "$CASE_ELF")" >&2
    show_tail "$CASE_LOG"
    exit 1
  fi
  assert_no_fallback_log "$stage" "$CASE_LOG"
  assert_elf "$stage" "$CASE_ELF" "$machine"
}

assert_compile_rejection() {
  local stage="$1"
  local expected_diagnostic="$2"

  [[ "$CASE_RC" -eq 1 ]] || fail "$stage" "expected_compile_exit_1_got_$CASE_RC"
  [[ "$(artifact_state "$CASE_ELF")" == absent ]] || fail "$stage" rejected_case_left_elf
  assert_no_fallback_log "$stage" "$CASE_LOG"
  [[ "$(grep -Ec '^error:' "$CASE_LOG" || true)" -eq 1 ]] || fail "$stage" unexpected_error_diagnostic_count
  grep -Fq "$expected_diagnostic" "$CASE_LOG" || {
    show_tail "$CASE_LOG"
    fail "$stage" expected_diagnostic_missing
  }
  if grep -Eiq 'segmentation fault|sigsegv|aborted|core dumped|internal compiler|timed out' "$CASE_LOG"; then
    fail "$stage" compiler_crash_or_internal_error_observed
  fi
}

run_runtime() {
  local label="$1"
  local elf="$2"
  local marker="$3"
  local stdout="$WORK/$label.runtime.stdout"
  local stderr="$WORK/$label.runtime.stderr"

  chmod +x "$elf"
  set +e
  timeout --signal=TERM --kill-after=5s "$RUNTIME_TIMEOUT" "$elf" >"$stdout" 2>"$stderr"
  RUNTIME_RC=$?
  set -e
  RUNTIME_STDOUT="$stdout"
  RUNTIME_STDERR="$stderr"
  RUNTIME_MARKER="$marker"
}

assert_runtime_success() {
  local stage="$1"
  if [[ "$RUNTIME_RC" -ne 0 ]]; then
    printf 'ISSUE1021_RUNTIME_FAILURE mode=%s stage=%s rc=%s\n' "$MODE" "$stage" "$RUNTIME_RC" >&2
    show_tail "$RUNTIME_STDOUT"
    show_tail "$RUNTIME_STDERR"
    exit 1
  fi
  printf '%s\n' "$RUNTIME_MARKER" | cmp -s - "$RUNTIME_STDOUT" || fail "$stage" stdout_marker_mismatch
  [[ ! -s "$RUNTIME_STDERR" ]] || fail "$stage" runtime_stderr_not_empty
}

run_positive_case() {
  local index="$1"
  local name="${CASE_NAMES[$index]}"
  local elf="$WORK/$name.x86_64.elf"
  run_compile "$name" "${CASE_SOURCES[$index]}" "$elf"
  assert_compile_success "${name}_compile" 62
  run_runtime "$name" "$elf" "${CASE_MARKERS[$index]}"
  assert_runtime_success "${name}_runtime"
  printf 'ISSUE1021_CASE_PASS mode=%s case=%s compile=pass runtime_exit=0 marker=%s elf_sha256=%s fallback=none\n' \
    "$MODE" "$name" "${CASE_MARKERS[$index]}" "$(sha256sum "$elf" | awk '{print $1}')"
}

run_old_sigsegv_case() {
  local index="$1"
  local name="${CASE_NAMES[$index]}"
  local elf="$WORK/$name.old-seed.elf"
  run_compile "$name" "${CASE_SOURCES[$index]}" "$elf"
  assert_compile_success "${name}_compile" 62
  run_runtime "$name" "$elf" "${CASE_MARKERS[$index]}"
  [[ "$RUNTIME_RC" -eq 139 ]] || fail "${name}_runtime" "expected_sigsegv_139_got_$RUNTIME_RC"
  [[ ! -s "$RUNTIME_STDOUT" ]] || fail "${name}_runtime" runtime_stdout_not_empty
  [[ ! -s "$RUNTIME_STDERR" ]] || fail "${name}_runtime" runtime_stderr_not_empty
  printf 'ISSUE1021_OLD_SEED_NEGATIVE_PASS case=%s compile=pass runtime_exit=139 signal=SIGSEGV marker=absent elf_sha256=%s\n' \
    "$name" "$(sha256sum "$elf" | awk '{print $1}')"
}

run_old_semantic_bug_case() {
  local index="$1"
  local name="${CASE_NAMES[$index]}"
  local elf="$WORK/$name.old-seed.elf"
  run_compile "$name" "${CASE_SOURCES[$index]}" "$elf"
  assert_compile_success "${name}_compile" 62
  run_runtime "$name" "$elf" "${CASE_MARKERS[$index]}"
  [[ "$RUNTIME_RC" -eq 1 ]] || fail "${name}_runtime" "expected_semantic_exit_1_got_$RUNTIME_RC"
  [[ ! -s "$RUNTIME_STDOUT" ]] || fail "${name}_runtime" runtime_stdout_not_empty
  [[ ! -s "$RUNTIME_STDERR" ]] || fail "${name}_runtime" runtime_stderr_not_empty
  printf 'ISSUE1021_OLD_SEED_NEGATIVE_PASS case=%s compile=pass runtime_exit=1 failure=wrong_intrinsic_dispatch marker=absent elf_sha256=%s\n' \
    "$name" "$(sha256sum "$elf" | awk '{print $1}')"
}

run_wrong_namespace_rejection() {
  local target="$1"
  local label="wrong_namespace_${target}"
  local elf="$WORK/$label.elf"
  run_compile "$label" "$WRONG_NAMESPACE_SOURCE" "$elf" "$target"
  assert_compile_rejection "${label}_compile" "$WRONG_NAMESPACE_DIAGNOSTIC"
  printf 'ISSUE1021_WRONG_NAMESPACE_REJECT_PASS target=%s compile_exit=1 diagnostic=unknown_qualified_identifier final_elf=absent fallback=none\n' "$target"
}

run_aarch64_compile_case() {
  local index="$1"
  local name="${CASE_NAMES[$index]}"
  local elf="$WORK/$name.aarch64.elf"
  run_compile "${name}_aarch64" "${CASE_SOURCES[$index]}" "$elf" aarch64-linux
  assert_compile_success "${name}_aarch64_compile" 183
  printf 'ISSUE1021_AARCH64_COMPILE_PASS case=%s runtime=unavailable elf_type=ET_EXEC elf_machine=183 elf_sha256=%s fallback=none\n' \
    "$name" "$(sha256sum "$elf" | awk '{print $1}')"
}

for command_name in awk basename chmod cmp dirname env find grep kill mkdir mktemp od rm sha256sum tail timeout tr uname; do
  require_command "$command_name"
done
[[ "$KEEP_WORK" == 0 || "$KEEP_WORK" == 1 ]] || fail setup invalid_keep_work
[[ "$COMPILE_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || fail setup invalid_compile_timeout
[[ "$RUNTIME_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || fail setup invalid_runtime_timeout

assert_source_contract
SOURCE_SHA256="$(sha256sum "$SOURCE" | awk '{print $1}')"
make_workdir

if [[ "$MODE" == source-only ]]; then
  if find "$WORK" -type f \( -name '*.elf' -o -name '*.out' \) -print -quit | grep -q .; then
    fail source_only unexpected_executable_artifact
  fi
  printf 'ISSUE1021_INTRINSIC_SHADOW_RECEIPT mode=source-only source_sha256=%s source_contract=pass positive_fixtures=%s rejection_fixtures=1 x86_guard=pass aarch64_guard=pass runtime=not_run aarch64_runtime=unavailable compiler=not_run compiler_sha256=not_applicable final_elf=absent fallback=none\n' "$SOURCE_SHA256" "${#CASE_NAMES[@]}"
  printf 'ISSUE1021_INTRINSIC_SHADOW_PASS mode=source-only\n'
  exit 0
fi

[[ "$(uname -s 2>/dev/null || true)" == Linux ]] || fail setup linux_required
case "$(uname -m 2>/dev/null || true)" in
  x86_64|amd64) ;;
  *) fail setup x86_64_required ;;
esac
ulimit -c 0 2>/dev/null || true
ulimit -s 1048576 2>/dev/null || true
cd "$ROOT_DIR"

if [[ "$MODE" == old-seed-negative ]]; then
  OLD_SEED="${SOUNIO_ISSUE1021_OLD_SEED_BIN:-$ROOT_DIR/bin/souc-lean-single-x86_64}"
  resolve_compiler "$OLD_SEED"
  if [[ -n "${SOUNIO_ISSUE1021_OLD_SEED_EXPECTED_SHA256:-}" ]]; then
    [[ "$COMPILER_SHA256" == "$SOUNIO_ISSUE1021_OLD_SEED_EXPECTED_SHA256" ]] || fail compiler old_seed_sha256_mismatch
  fi
  INITIAL_COMPILER_SHA256="$COMPILER_SHA256"
  printf 'ISSUE1021_COMPILER_IDENTITY mode=old-seed-negative compiler=%s compiler_sha256=%s source_sha256=%s authority=explicit_raw_seed fallback=none\n' \
    "$COMPILER" "$COMPILER_SHA256" "$SOURCE_SHA256"

  run_old_sigsegv_case 0
  run_old_sigsegv_case 1
  run_old_sigsegv_case 2
  run_positive_case 3
  run_positive_case 4
  run_old_semantic_bug_case 5
  run_old_semantic_bug_case 6
  run_positive_case 7
  run_old_semantic_bug_case 8
  run_old_semantic_bug_case 9

  [[ "$(sha256sum "$COMPILER" | awk '{print $1}')" == "$INITIAL_COMPILER_SHA256" ]] || fail final compiler_changed_during_gate
  printf 'ISSUE1021_INTRINSIC_SHADOW_RECEIPT mode=old-seed-negative compiler_sha256=%s authority=legacy_bug_witness_only abi_regressions=3xSIGSEGV139 semantic_regressions=4xexit1 controls=3xpass aarch64=not_run final_elf=case_dependent fallback=none\n' "$COMPILER_SHA256"
  printf 'ISSUE1021_INTRINSIC_SHADOW_PASS mode=old-seed-negative\n'
  exit 0
fi

SOURCE_FRESH_BIN="${SOUNIO_ISSUE1021_SOURCE_FRESH_BIN:-}"
EXPECTED_SOURCE_FRESH_SHA256="${SOUNIO_ISSUE1021_SOURCE_FRESH_EXPECTED_SHA256:-}"
[[ -n "$SOURCE_FRESH_BIN" ]] || fail compiler source_fresh_binary_required
[[ "$EXPECTED_SOURCE_FRESH_SHA256" =~ ^[0-9a-f]{64}$ ]] || fail compiler source_fresh_expected_sha256_required
resolve_compiler "$SOURCE_FRESH_BIN"
[[ "$COMPILER_SHA256" == "$EXPECTED_SOURCE_FRESH_SHA256" ]] || fail compiler source_fresh_sha256_mismatch
INITIAL_COMPILER_SHA256="$COMPILER_SHA256"
printf 'ISSUE1021_COMPILER_IDENTITY mode=source-fresh compiler=%s compiler_sha256=%s source_sha256=%s authority=external_build_plus_explicit_sha256 fallback=none\n' \
  "$COMPILER" "$COMPILER_SHA256" "$SOURCE_SHA256"

for i in "${!CASE_NAMES[@]}"; do
  run_positive_case "$i"
done
for i in "${!CASE_NAMES[@]}"; do
  run_aarch64_compile_case "$i"
done
run_wrong_namespace_rejection x86_64-linux
run_wrong_namespace_rejection aarch64-linux

[[ "$(sha256sum "$COMPILER" | awk '{print $1}')" == "$INITIAL_COMPILER_SHA256" ]] || fail final compiler_changed_during_gate
printf 'ISSUE1021_INTRINSIC_SHADOW_RECEIPT mode=source-fresh compiler_sha256=%s source_sha256=%s x86_runtime_cases=%s aarch64_compile_cases=%s qualified_rejections=2 aarch64_runtime=unavailable final_elf=case_dependent fallback=none\n' \
  "$COMPILER_SHA256" "$SOURCE_SHA256" "${#CASE_NAMES[@]}" "${#CASE_NAMES[@]}"
printf 'ISSUE1021_INTRINSIC_SHADOW_PASS mode=source-fresh\n'
