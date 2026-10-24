#!/usr/bin/env bash
# madaros_ref_alias_bind_as_ref_gate.sh — regression gate for the bare-identifier
# ref-alias is_ref propagation bug.
#
# Bug: `let zs = a` / `var zs = a`, where `a` is itself an already
# reference-typed parameter (no explicit `&T` annotation on the alias, RHS is
# a bare identifier rather than a literal `&x`/`&!x` unary expression), left
# `zs` without the is_ref bit. Every lookup_local_is_ref call site in
# self-hosted/ir/lower.sio (IndexGet/Set, FieldGet/Set, .len(), &self
# auto-ref, for-in-array, struct_local_is_value_copyable) then treated the
# address `zs` holds as an ordinary GC handle instead of dereferencing it
# first, silently reading/writing through the wrong indirection level.
#
# Fixed in lower_let_stmt_finalize_ref's bind_as_ref computation
# (self-hosted/ir/lower.sio): a bare-identifier RHS that is itself
# already ref-typed (lookup_local_is_ref) now also sets bind_as_ref.
#
# See docs/audit/REF_ALIAS_BIND_AS_REF_MISS_2026-09-22.md for the full
# investigation and PR #2615's Copilot review thread that surfaced it.
#
# Usage:
#   bash scripts/ci/madaros_ref_alias_bind_as_ref_gate.sh
#
# Environment:
#   MADAROS_RAW_BIN — path to an already-built Madaros ELF to reuse instead of
#                     building one from current source (mirrors the convention
#                     used by scripts/ci/madaros_gum_fo_trust_gate.sh and
#                     scripts/ci/madaros_fixed_array_call_boundary_alias_gate.sh).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOURCE="$ROOT_DIR/tests/run-pass/ref_alias_bind_as_ref.sio"
KEEP_WORK="${SOUNIO_MADAROS_REF_ALIAS_GATE_KEEP:-0}"

fail() {
  echo "[madaros-ref-alias-bind-as-ref] FAIL: $*" >&2
  exit 1
}

[[ -f "$SOURCE" ]] || fail "missing regression source: $SOURCE"

if [[ -n "${SOUNIO_MADAROS_REF_ALIAS_GATE_DIR:-}" ]]; then
  WORK="$SOUNIO_MADAROS_REF_ALIAS_GATE_DIR"
  [[ ! -e "$WORK" ]] || fail "refusing existing gate directory: $WORK"
  mkdir -p "$WORK"
else
  WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-madaros-ref-alias.XXXXXX")"
fi

if [[ "$KEEP_WORK" != "1" ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

MADAROS_ELF="${MADAROS_RAW_BIN:-$WORK/madaros}"

if [[ -z "${MADAROS_RAW_BIN:-}" ]]; then
  COMPILER_SOURCE="current_source"
  if ! bash "$ROOT_DIR/scripts/ci/build_modular_madaros.sh" "$MADAROS_ELF" >"$WORK/build.log" 2>&1; then
    tail -n 80 "$WORK/build.log" >&2 || true
    fail "current-source Madaros build failed"
  fi
else
  COMPILER_SOURCE="override"
fi
[[ -x "$MADAROS_ELF" ]] || fail "Madaros is missing or not executable: $MADAROS_ELF"

ELF="$WORK/ref_alias_bind_as_ref"
CLOG="$WORK/check.log"
BLOG="$WORK/compile.log"
RLOG="$WORK/run.log"

compiler_sha="$(sha256sum "$MADAROS_ELF" | awk '{print $1}')"
source_sha="$(sha256sum "$SOURCE" | awk '{print $1}')"
echo "[madaros-ref-alias-bind-as-ref] compiler_source=$COMPILER_SOURCE"
echo "[madaros-ref-alias-bind-as-ref] compiler_sha256=$compiler_sha"
echo "[madaros-ref-alias-bind-as-ref] source_sha256=$source_sha"

if ! MADAROS_RAW_BIN="$MADAROS_ELF" "$ROOT_DIR/bin/madaros" check "$SOURCE" >"$CLOG" 2>&1; then
  cat "$CLOG" >&2
  fail "regression source did not check"
fi

if ! MADAROS_RAW_BIN="$MADAROS_ELF" "$ROOT_DIR/bin/madaros" compile "$SOURCE" -o "$ELF" >"$BLOG" 2>&1; then
  cat "$BLOG" >&2
  fail "regression source did not compile"
fi
[[ -x "$ELF" ]] || fail "compile did not produce an executable"

set +e
"$ELF" >"$RLOG" 2>&1
run_rc=$?
set -e

cat "$RLOG"

if [[ "$run_rc" -ne 0 ]]; then
  fail "regression binary exited rc=$run_rc (expected 0)"
fi

if ! grep -Fq 'ref_alias_bind_as_ref: PASS' "$RLOG"; then
  fail "expected 'ref_alias_bind_as_ref: PASS' in output, got:"
fi

echo "[madaros-ref-alias-bind-as-ref] PASS: read/write through a bare-identifier alias of a &T / &!T / &![T;N] param stays reference-typed (IndexGet/Set + FieldGet/Set)"
