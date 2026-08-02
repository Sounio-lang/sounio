#!/usr/bin/env bash
# Prove bounded pub(crate) compatibility in a source-derived lean_single seed.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BOOTSTRAP_BIN="${SOUNIO_LEAN_SINGLE_BOOTSTRAP_BIN:-}"

fail() {
  echo "[lean-single-pub-crate-seed] FAIL: $*" >&2
  exit 1
}

assert_no_fatal_log() {
  local log="$1"
  if grep -Eiq 'segmentation fault|core dumped|terminated by signal|fatal:|bus error|illegal instruction' "$log"; then
    cat "$log" >&2
    fail "fatal process marker in $log"
  fi
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

[[ -n "$BOOTSTRAP_BIN" ]] || fail "SOUNIO_LEAN_SINGLE_BOOTSTRAP_BIN must name an explicit bootstrap ELF"
[[ -x "$BOOTSTRAP_BIN" ]] || fail "bootstrap ELF is missing or not executable: $BOOTSTRAP_BIN"
[[ "$(head -c2 "$BOOTSTRAP_BIN" 2>/dev/null)" != '#!' ]] || fail "bootstrap input must be an ELF, not a wrapper"

WORK="${SOUNIO_LEAN_SINGLE_PUB_CRATE_SEED_DIR:-}"
if [[ -n "$WORK" ]]; then
  [[ ! -e "$WORK" ]] || fail "refusing existing gate directory: $WORK"
  mkdir -p "$WORK"
else
  WORK="$(mktemp -d /tmp/sounio-lean-single-pub-crate.XXXXXX)"
fi
KEEP="${SOUNIO_LEAN_SINGLE_PUB_CRATE_SEED_KEEP:-0}"
[[ "$KEEP" == 1 ]] || trap 'rm -rf "$WORK"' EXIT

SEED="$WORK/lean-current.elf"
POSITIVE_ELF="$WORK/pub-crate.elf"
PUBLIC_ELF="$WORK/public.elf"
PRIVATE_FN_ELF="$WORK/private-fn.elf"
PRIVATE_STRUCT_ELF="$WORK/private-struct.elf"
AST_PROBE_ELF="$WORK/pub-crate-ast-probe.elf"

scripts_dev_lock="$ROOT_DIR/scripts/dev/souc-build-lock.sh"
"$scripts_dev_lock" "$BOOTSTRAP_BIN" \
  "$ROOT_DIR/self-hosted/compiler/lean_single.sio" "$SEED" \
  >"$WORK/seed-build.log" 2>&1 || {
    cat "$WORK/seed-build.log" >&2
    fail "current lean_single source did not derive a seed"
  }
[[ -s "$SEED" ]] || fail "derived seed is empty"
chmod +x "$SEED"

"$scripts_dev_lock" "$SEED" \
  "$ROOT_DIR/tests/compiler/madaros_visibility_context/pub_crate_facade_main.sio" "$POSITIVE_ELF" \
  >"$WORK/pub-crate-build.log" 2>&1 || {
    cat "$WORK/pub-crate-build.log" >&2
    fail "derived seed rejected the non-generic pub(crate) facade"
  }
chmod +x "$POSITIVE_ELF"
"$POSITIVE_ELF" >"$WORK/pub-crate-run.log" 2>&1 || {
  cat "$WORK/pub-crate-run.log" >&2
  fail "pub(crate) facade ELF failed at runtime"
}
grep -Fxq 'PASS pub_crate_facade_module_authority' "$WORK/pub-crate-run.log" || {
  cat "$WORK/pub-crate-run.log" >&2
  fail "pub(crate) facade ELF omitted its exact marker"
}

"$scripts_dev_lock" "$SEED" \
  "$ROOT_DIR/tests/compiler/madaros_visibility_context/public_facade_main.sio" "$PUBLIC_ELF" \
  >"$WORK/public-build.log" 2>&1 || {
    cat "$WORK/public-build.log" >&2
    fail "derived seed regressed plain pub visibility"
  }
chmod +x "$PUBLIC_ELF"
"$PUBLIC_ELF" >"$WORK/public-run.log" 2>&1 || {
  cat "$WORK/public-run.log" >&2
  fail "plain pub facade ELF failed at runtime"
}
grep -Fxq 'PASS public_facade_module_authority' "$WORK/public-run.log" || {
  cat "$WORK/public-run.log" >&2
  fail "plain pub facade ELF omitted its exact marker"
}

set +e
"$scripts_dev_lock" "$SEED" \
  "$ROOT_DIR/tests/multimodule/visibility_fn_private_main.sio" "$PRIVATE_FN_ELF" \
  >"$WORK/private-fn.log" 2>&1
private_fn_rc=$?
"$scripts_dev_lock" "$SEED" \
  "$ROOT_DIR/tests/multimodule/visibility_struct_private_main.sio" "$PRIVATE_STRUCT_ELF" \
  >"$WORK/private-struct.log" 2>&1
private_struct_rc=$?
set -e

assert_no_fatal_log "$WORK/private-fn.log"
assert_no_fatal_log "$WORK/private-struct.log"
private_fn_mode=unknown
if [[ "$private_fn_rc" -eq 1 && ! -s "$PRIVATE_FN_ELF" ]] \
  && grep -Fq 'error: cannot call non-pub function from imported module' "$WORK/private-fn.log" \
  && grep -Fxq 'typecheck: failed' "$WORK/private-fn.log"; then
  private_fn_mode=semantic_reject
elif [[ "$private_fn_rc" -eq 0 && -s "$PRIVATE_FN_ELF" ]] \
  && grep -Fq 'warning: cannot call non-pub function from imported module' "$WORK/private-fn.log" \
  && ! grep -Fxq 'typecheck: failed' "$WORK/private-fn.log"; then
  private_fn_mode=warning_only
else
  cat "$WORK/private-fn.log" >&2
  fail "private function visibility boundary returned an incoherent seed result"
fi
[[ "$private_struct_rc" -eq 1 && ! -s "$PRIVATE_STRUCT_ELF" ]] || {
  cat "$WORK/private-struct.log" >&2
  fail "derived seed did not fail closed on an imported private struct"
}
grep -Fq 'error: private struct literal' "$WORK/private-struct.log" || {
  cat "$WORK/private-struct.log" >&2
  fail "private struct rejection omitted its seed diagnostic"
}
grep -Fxq 'typecheck: failed' "$WORK/private-struct.log" || {
  cat "$WORK/private-struct.log" >&2
  fail "private struct rejection omitted the semantic failure receipt"
}

SOUNIO_STDLIB_PATH="$ROOT_DIR/self-hosted" "$scripts_dev_lock" "$SEED" \
  "$ROOT_DIR/tests/compiler/madaros_visibility_context/pub_crate_ast_probe.sio" "$AST_PROBE_ELF" \
  >"$WORK/pub-crate-ast-probe-build.log" 2>&1 || {
    cat "$WORK/pub-crate-ast-probe-build.log" >&2
    fail "derived seed could not build the modular pub(crate) AST probe"
  }
chmod +x "$AST_PROBE_ELF"
"$AST_PROBE_ELF" "$ROOT_DIR/tests/compiler/madaros_visibility_context/pub_crate_facade_leaf.sio" \
  >"$WORK/pub-crate-ast-probe-run.log" 2>&1 || {
    cat "$WORK/pub-crate-ast-probe-run.log" >&2
    fail "modular parser did not preserve pub(crate) visibility"
  }
grep -Fxq 'PASS pub_crate_ast_visibility kind=2 items=3 parse_errors=0' "$WORK/pub-crate-ast-probe-run.log" || {
  cat "$WORK/pub-crate-ast-probe-run.log" >&2
  fail "modular pub(crate) AST probe omitted its exact marker"
}

seed_sha="$(sha256_file "$SEED")"
bootstrap_sha="$(sha256_file "$BOOTSTRAP_BIN")"
lean_source_sha="$(sha256_file "$ROOT_DIR/self-hosted/compiler/lean_single.sio")"
echo "[lean-single-pub-crate-seed] receipt bootstrap_sha256=$bootstrap_sha lean_source_sha256=$lean_source_sha seed_sha256=$seed_sha pub_crate=runtime-pass public=runtime-pass lean_private_fn_mode=$private_fn_mode lean_private_struct_mode=semantic_reject modular_ast_kind=2 modular_ast_parse_errors=0 privacy_acceptance_dependency=madaros_external_gate privacy_acceptance_status=not_run pub_super=not-claimed generic_restricted_visibility=not-claimed"
echo '[lean-single-pub-crate-seed] PASS: bounded non-generic pub(crate) authority survives source-derived bootstrap'
