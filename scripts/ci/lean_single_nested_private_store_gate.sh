#!/usr/bin/env bash
# Issue #2363: nested explicit-deref stores must retain both privacy checks.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
KEEP_WORK="${SOUNIO_LEAN_NESTED_PRIVATE_KEEP:-0}"

fail() {
  echo "[lean-nested-private] FAIL: $*" >&2
  exit 1
}

if [[ -n "${SOUNIO_LEAN_NESTED_PRIVATE_DIR:-}" ]]; then
  WORK="$SOUNIO_LEAN_NESTED_PRIVATE_DIR"
  [[ ! -e "$WORK" ]] || fail "refusing existing gate directory: $WORK"
  mkdir -p "$WORK"
else
  WORK="$(mktemp -d /tmp/sounio-lean-nested-private.XXXXXX)"
fi

if [[ "$KEEP_WORK" != "1" ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

semantic_private_rejection() {
  local rc="$1"
  local log="$2"
  local expected_struct="$3"

  [[ "$rc" -eq 1 ]] || return 1
  grep -Fq "error: private struct field access [struct=$expected_struct]" "$log" || return 1
  grep -Fxq 'typecheck: failed' "$log" || return 1
  if grep -Eiq 'segmentation fault|core dumped|terminated by signal|fatal:|bus error|illegal instruction' "$log"; then
    return 1
  fi
  return 0
}

valid_log="$WORK/valid-semantic.log"
printf '%s\n' \
  'error: private struct field access [struct=HiddenInner] at <main>:1' \
  'typecheck: failed' >"$valid_log"
semantic_private_rejection 1 "$valid_log" HiddenInner || fail "valid privacy rejection was refused"

signal_log="$WORK/adversarial-signal.log"
printf '%s\n' \
  'error: private struct field access [struct=HiddenInner] at <main>:1' \
  'typecheck: failed' \
  'Segmentation fault (core dumped)' >"$signal_log"
if semantic_private_rejection 139 "$signal_log" HiddenInner; then
  fail "signal failure was accepted as a semantic rejection"
fi

fatal_log="$WORK/adversarial-fatal.log"
printf '%s\n' \
  'error: private struct field access [struct=HiddenInner] at <main>:1' \
  'typecheck: failed' \
  'fatal: synthetic compiler abort' >"$fatal_log"
if semantic_private_rejection 1 "$fatal_log" HiddenInner; then
  fail "fatal failure was accepted as a semantic rejection"
fi

echo "[lean-nested-private] PASS: semantic_rc1_accepted=1 adversarial_failures_rejected=2"

if [[ "${SOUNIO_LEAN_NESTED_PRIVATE_SELF_TEST_ONLY:-0}" == "1" ]]; then
  exit 0
fi

for fn_name in compile_deref_field_field_store_x86 compile_deref_field_field_array_store_x86; do
  definition_count="$(grep -cE "^fn ${fn_name}\\(" "$ROOT_DIR/self-hosted/compiler/lean_single.sio" || true)"
  [[ "$definition_count" -eq 1 ]] || fail "$fn_name definition_count=$definition_count expected=1"
done

SEED="${SOUNIO_LEAN_NESTED_PRIVATE_SEED:-$ROOT_DIR/bin/souc-lean-single-x86_64}"
[[ -x "$SEED" ]] || fail "lean_single seed is missing or not executable: $SEED"
[[ "$(head -c 2 "$SEED" 2>/dev/null || true)" != '#!' ]] || \
  fail "lean_single seed must be a raw ELF, not a script wrapper: $SEED"

seed_hello="$WORK/seed-hello.elf"
SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" "$SEED" \
  "$ROOT_DIR/examples/hello.sio" "$seed_hello" >"$WORK/seed-hello.compile.log" 2>&1 || {
    tail -n 80 "$WORK/seed-hello.compile.log" >&2 || true
    fail "raw seed could not compile the hello instrument control"
  }
chmod +x "$seed_hello"
"$seed_hello" >"$WORK/seed-hello.run.log" 2>&1 || fail "seed hello control did not execute"
grep -Fxq 'Hello, Sounio' "$WORK/seed-hello.run.log" || fail "seed hello exact output mismatch"

COMPILER="${SOUNIO_LEAN_NESTED_PRIVATE_BIN:-$WORK/lean_single.current}"
if [[ -z "${SOUNIO_LEAN_NESTED_PRIVATE_BIN:-}" ]]; then
  "$ROOT_DIR/scripts/dev/souc-build-lock.sh" \
    "$SEED" "$ROOT_DIR/self-hosted/compiler/lean_single.sio" "$COMPILER" \
    >"$WORK/build.log" 2>&1 || {
      tail -n 80 "$WORK/build.log" >&2 || true
      fail "current-source lean_single build failed"
    }
  chmod +x "$COMPILER"
fi
[[ -x "$COMPILER" ]] || fail "lean_single candidate is missing or not executable: $COMPILER"
[[ "$(head -c 2 "$COMPILER" 2>/dev/null || true)" != '#!' ]] || \
  fail "lean_single candidate must be a raw ELF, not a script wrapper: $COMPILER"

positive_elf="$WORK/public-store.elf"
SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" "$COMPILER" \
  "$ROOT_DIR/tests/run-pass/lean_nested_deref_store_public.sio" "$positive_elf" \
  >"$WORK/public-store.compile.log" 2>&1 || {
    cat "$WORK/public-store.compile.log" >&2
    fail "public nested-store control did not compile"
  }
chmod +x "$positive_elf"
"$positive_elf" >"$WORK/public-store.run.log" 2>&1 || {
  cat "$WORK/public-store.run.log" >&2
  fail "public nested-store control did not execute"
}
grep -Fxq 'PASS lean_nested_deref_store_public' "$WORK/public-store.run.log" || {
  cat "$WORK/public-store.run.log" >&2
  fail "public nested-store exact PASS marker missing"
}

compile_private_case() {
  local case_name="$1"
  local source="$2"
  local expected_struct="$3"
  local elf="$WORK/$case_name.elf"
  local log="$WORK/$case_name.compile.log"
  local rc=0

  set +e
  SOUNIO_STDLIB_PATH="$ROOT_DIR/tests/compiler/lean_nested_private_store" "$COMPILER" \
    "$source" "$elf" >"$log" 2>&1
  rc=$?
  set -e

  semantic_private_rejection "$rc" "$log" "$expected_struct" || {
    cat "$log" >&2
    fail "$case_name did not produce the required semantic privacy rejection (rc=$rc)"
  }
  [[ ! -s "$elf" ]] || fail "$case_name produced an ELF despite semantic rejection"
}

compile_private_case inner-scalar \
  "$ROOT_DIR/tests/compiler/lean_nested_private_store/inner_scalar.sio" HiddenInner
compile_private_case inner-array \
  "$ROOT_DIR/tests/compiler/lean_nested_private_store/inner_array.sio" HiddenInner
compile_private_case outer-scalar \
  "$ROOT_DIR/tests/compiler/lean_nested_private_store/outer_scalar.sio" HiddenOuter
compile_private_case outer-array \
  "$ROOT_DIR/tests/compiler/lean_nested_private_store/outer_array.sio" HiddenOuter

seed_sha="$(sha256sum "$SEED" | awk '{print $1}')"
compiler_sha="$(sha256sum "$COMPILER" | awk '{print $1}')"
echo "[lean-nested-private] receipt issue=2363 definitions=1 inner_scalar=REJECTED inner_array=REJECTED outer_scalar=REJECTED outer_array=REJECTED public=PASS seed_sha256=$seed_sha compiler_sha256=$compiler_sha"
echo "LEAN_SINGLE_NESTED_PRIVATE_STORE_GATE_OK"
